"""Provider point ROIs -> class + subtype 128 px patch dataset (provider_subtype_point_dataset v1).

Usage::

    python -m sic_xrt_data_tools.provider_subtype_dataset build --source "<...>/2D XRT" --output <new folder>
    python -m sic_xrt_data_tools.provider_subtype_dataset validate <dataset folder>

Labels are the provider's ROI file names. They are not expert-verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from . import __version__
from .annotations.provider_roi import (
    CLASSES,
    FINE_LABELS,
    SUBTYPES,
    RoiFile,
    discover,
    sha256_file,
)
from .tiff.orientation import (
    TRANSFORMS,
    apply_transform,
    contrast_ratio,
    exact_row_agreement,
    image_relation,
    transform_points,
)

SCHEMA = "provider_subtype_point_dataset"
SCHEMA_VERSION = 1
PATCH_SIZE = 128
CENTER_BOX = 32
# Same wafer-level split as the earlier data-tools research datasets (#17, #25):
# every image of an Area (Before and After annealing) shares one split.
DEFAULT_SPLITS = {"1": "train", "9": "train", "2": "val", "8": "test"}
SPLITS = ("train", "val", "test", "unassigned")
# Orientation probe thresholds (see docs/provider-subtype-dataset-v1.md).
MIN_PROBE_POINTS = 15
MIN_SIGNAL_RATIO = 1.15
MIN_PREFERENCE = 1.10
MIN_EXACT_ROWS_FOR_SAME_IMAGE = 0.01
MAX_DIFF_SAMPLE = 5000
CLASS_COLORS = {"BPD": (244, 63, 94), "TED": (34, 197, 94), "TSD": (56, 189, 248)}

Image.MAX_IMAGE_PIXELS = None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def code_revision() -> dict:
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"package": "sic-xrt-data-tools", "version": __version__, "git_commit": commit, "dirty": dirty}


def _inside_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in [path, *path.parents])


def load_tiff(path: Path) -> tuple[np.ndarray, str, dict]:
    """Read once sequentially (slow disks), hash, and decode a single-page TIFF."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        if len(tif.pages) != 1:
            raise ValueError(f"expected single-page TIFF: {path}")
        page = tif.pages[0]
        info = {"shape": list(page.shape), "dtype": str(page.dtype), "axes": tif.series[0].axes,
                "compression": str(page.compression.name), "photometric": str(page.photometric.name)}
        if page.dtype != np.uint8 or page.ndim != 3 or page.shape[2] != 3:
            raise ValueError(f"expected uint8 RGB TIFF, got {page.dtype} {page.shape}: {path}")
        offsets, counts = page.dataoffsets, page.databytecounts
        expected = int(np.prod(page.shape))
        contiguous = page.compression == 1 and all(
            offsets[i] + counts[i] == offsets[i + 1] for i in range(len(offsets) - 1))
        if contiguous and sum(counts) == expected and page.planarconfig == 1:
            array = np.frombuffer(data, np.uint8, expected, offsets[0]).reshape(page.shape)
        else:
            array = page.asarray()
    return array, digest, info


def load_jpeg(path: Path) -> tuple[np.ndarray, dict]:
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        record = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "width": image.size[0],
                  "height": image.size[1], "mode": image.mode, "exif_orientation": image.getexif().get(274),
                  "decoder": f"Pillow {Image.__version__}"}
        array = np.asarray(image.convert("RGB"))
    return array, record


def _tiff_differences(tiff: np.ndarray, rebuilt: np.ndarray, block: int = 256) -> dict:
    """Describe how a damaged TIFF differs (e.g. isolated burned-in white marks)."""
    count, white, sample = 0, 0, []
    for start in range(0, tiff.shape[0], block):
        a, b = tiff[start:start + block], rebuilt[start:start + block]
        differs = np.any(a != b, axis=-1)
        ys, xs = np.nonzero(differs)
        count += len(ys)
        white += int(np.all(a[ys, xs] == 255, axis=-1).sum())
        if len(sample) < MAX_DIFF_SAMPLE:
            sample.extend(zip((xs + 0.5).tolist(), (ys + start + 0.5).tolist(), strict=True))
    return {"tiff_differing_pixels": count,
            "tiff_differing_pixel_fraction": round(count / (tiff.shape[0] * tiff.shape[1]), 8),
            "tiff_differing_pixels_pure_white": white,
            "_diff_xy": np.asarray(sample[:MAX_DIFF_SAMPLE], dtype=np.float64).reshape(-1, 2)}


def _nearest_distance(points: np.ndarray, targets: np.ndarray, chunk: int = 2048) -> np.ndarray:
    result = np.empty(len(points))
    for start in range(0, len(points), chunk):
        delta = points[start:start + chunk, None, :] - targets[None, :, :]
        result[start:start + chunk] = np.sqrt((delta ** 2).sum(-1)).min(axis=1)
    return result


def resolve_pixels(tiff_path: Path, source: Path) -> tuple[np.ndarray, dict]:
    """Choose the pixel source for one labelled image and record the evidence.

    The provider TIFFs are bit-identical copies of the same-named JPEG after a
    fixed transform (a horizontal mirror in the real data). The TIFF is used when
    that holds for every row. When only some rows agree, the TIFF copy is
    damaged and the image is rebuilt from the JPEG with the verified transform,
    which reproduces what an intact TIFF would contain.
    """
    tiff, tiff_hash, info = load_tiff(tiff_path)
    record = {"relpath": tiff_path.relative_to(source).as_posix(), "sha256": tiff_hash,
              "bytes": tiff_path.stat().st_size, "width": tiff.shape[1], "height": tiff.shape[0], **info}
    jpeg_path = tiff_path.with_suffix(".jpg")
    if not jpeg_path.exists():
        record.update(jpeg_pair={"present": False}, jpeg_to_tiff="none", pixel_source="tiff",
                      pixel_check="no_jpeg_pair_unverified")
        return tiff, record
    jpeg, jpeg_record = load_jpeg(jpeg_path)
    jpeg_record.update(present=True, relpath=jpeg_path.relative_to(source).as_posix(),
                       size_matches_tiff=jpeg.shape == tiff.shape)
    record["jpeg_pair"] = jpeg_record
    agreement = {}
    for name in TRANSFORMS:
        agreement[name] = round(exact_row_agreement(tiff, apply_transform(jpeg, name)), 6)
        if agreement[name] == 1.0:
            break
    best = max(agreement, key=agreement.get)
    jpeg_record["exact_row_agreement"] = agreement
    if agreement[best] == 1.0:
        record.update(jpeg_to_tiff=best, pixel_source="tiff",
                      pixel_check=f"tiff_bit_identical_to_jpeg_{best}")
        return tiff, record
    if agreement[best] >= MIN_EXACT_ROWS_FOR_SAME_IMAGE:
        rebuilt = np.ascontiguousarray(apply_transform(jpeg, best))
        del jpeg
        record.update(jpeg_to_tiff=best, pixel_source=f"jpeg_{best}",
                      pixel_check="tiff_damaged_rebuilt_from_jpeg",
                      tiff_damaged_row_fraction=round(1 - agreement[best], 6),
                      **_tiff_differences(tiff, rebuilt))
        del tiff
        return rebuilt, record
    if jpeg.shape == tiff.shape:
        small = image_relation(tiff[::8, ::8].astype(np.float32).mean(axis=-1),
                               jpeg[::8, ::8].astype(np.float32).mean(axis=-1))
    else:
        small = {"best": "undetermined", "reason": "size_mismatch"}
    jpeg_record["low_resolution_relation"] = small
    record.update(jpeg_to_tiff=small["best"], pixel_source="tiff",
                  pixel_check="tiff_not_an_exact_copy_of_jpeg_used_as_is")
    return tiff, record


def decide_transform(probes: dict, relation: str, n_points: int) -> tuple[str | None, str]:
    """Pick the pixel grid of one ROI file from contrast evidence.

    Candidates are the TIFF grid (``identity``) and, when the JPEG differs from
    the TIFF by ``relation``, the raw JPEG grid. Returns ``(transform, basis)``;
    ``transform`` is ``None`` when undetermined.
    """
    if relation not in TRANSFORMS or relation == "identity":
        return "identity", "tiff_grid_only_candidate"
    identity, other = probes["identity"].get("ratio"), probes[relation].get("ratio")
    if n_points >= MIN_PROBE_POINTS and identity is not None and other is not None:
        if identity >= MIN_SIGNAL_RATIO and identity >= other * MIN_PREFERENCE:
            return "identity", "contrast_probe_tiff_grid"
        if other >= MIN_SIGNAL_RATIO and other >= identity * MIN_PREFERENCE:
            return relation, "contrast_probe_raw_jpeg_grid"
    return None, "contrast_probe_inconclusive"


def _neighbor_counts(xy: np.ndarray, labels: np.ndarray, index: int, half: float) -> tuple[int, int]:
    dx = np.abs(xy[:, 0] - xy[index, 0])
    dy = np.abs(xy[:, 1] - xy[index, 1])
    inside = (dx < half) & (dy < half)
    inside[index] = False
    same = labels == labels[index]
    return int(np.sum(inside & same)), int(np.sum(inside & ~same))


def _overview(image: np.ndarray, points: list[dict], stride: int = 8) -> Image.Image:
    small = Image.fromarray(np.ascontiguousarray(image[::stride, ::stride]))
    draw = ImageDraw.Draw(small)
    for point in points:
        x, y = point["x"] / stride, point["y"] / stride
        color = CLASS_COLORS[point["label"]] if point["status"] == "included" else (250, 204, 21)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color)
    return small


def build(source: Path, output: Path, split_map: dict[str, str] | None = None,
          overrides: dict[str, str] | None = None, preview_per_label: int = 8) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    split_map = dict(DEFAULT_SPLITS if split_map is None else split_map)
    overrides = dict(overrides or {})
    if any(value not in SPLITS for value in split_map.values()):
        raise ValueError(f"split names must be one of {SPLITS}")
    if any(value not in {"identity", "hflip"} for value in overrides.values()):
        raise ValueError("orientation overrides must be identity or hflip")
    if not source.is_dir():
        raise ValueError(f"missing source folder: {source}")
    if output == source or source in output.parents:
        raise ValueError("output must be outside the source data")
    if output.exists():
        raise ValueError("output already exists; choose a new version folder")
    if _inside_git_worktree(output.parent):
        raise ValueError("output must not be inside a git work tree (real data stays out of git)")

    started = datetime.now(UTC).isoformat()
    roi_files = discover(source)
    by_image: dict[str, list[RoiFile]] = defaultdict(list)
    for roi in roi_files:
        if roi.status == "mapped":
            by_image[roi.image_relpath].append(roi)
    if not by_image:
        raise ValueError("no mapped provider ROI files found")

    output.mkdir(parents=True)
    try:
        return _build(source, output, split_map, overrides, preview_per_label, started, roi_files, by_image)
    except BaseException:
        write_json(output / "BUILD_FAILED.json", {"status": "incomplete_do_not_use",
                                                  "failed_at": datetime.now(UTC).isoformat()})
        raise


def _build(source, output, split_map, overrides, preview_per_label, started, roi_files, by_image):
    preview = output / "preview"
    preview.mkdir()
    images, samples, point_rows = [], [], []
    stats = Counter()
    for image_rel, rois in sorted(by_image.items()):
        image_path = source / image_rel
        before_stat = image_path.stat()
        print(f"[{datetime.now():%H:%M:%S}] reading {image_rel}", flush=True)
        array, image_record = resolve_pixels(image_path, source)
        height, width = array.shape[:2]
        relation = image_record["jpeg_to_tiff"]
        area = rois[0].area
        split = split_map.get(area, "unassigned")
        image_id = hashlib.sha256(image_rel.encode("utf-8")).hexdigest()[:12]

        # 1) coordinate grid per ROI file
        for roi in rois:
            roi.probes = {"identity": contrast_ratio(array, roi.points)}
            if relation in TRANSFORMS and relation != "identity":
                roi.probes[relation] = contrast_ratio(array, transform_points(roi.points, relation, width, height))
            if roi.relpath in overrides:
                roi.transform, roi.transform_basis = overrides[roi.relpath], "manual_override"
            else:
                roi.transform, roi.transform_basis = decide_transform(roi.probes, relation, len(roi.points))
        decided = defaultdict(set)
        for roi in rois:
            if roi.transform and roi.transform_basis.startswith("contrast_probe"):
                decided[roi.cls].add(roi.transform)
        for roi in rois:
            if roi.transform is None and len(decided[roi.cls]) == 1:
                roi.transform = next(iter(decided[roi.cls]))
                roi.transform_basis = "inherited_from_same_image_same_class_files"
            if roi.transform is None:
                roi.status, roi.reason = "excluded", "coordinate_grid_undetermined"

        # 2) points in TIFF grid, dedupe, conflicts
        unique: dict[tuple, dict] = {}
        for roi in rois:
            if roi.status != "mapped":
                continue
            grid = transform_points(roi.points, roi.transform, width, height)
            for index, ((x, y), (rx, ry)) in enumerate(zip(grid, roi.points, strict=True)):
                key = (roi.fine_label, round(float(x), 4), round(float(y), 4))
                ref = {"roi": roi.relpath, "index": index}
                if key in unique:
                    unique[key]["refs"].append(ref)
                    stats["duplicate_point_refs"] += 1
                    continue
                point_id = hashlib.sha256(f"{image_rel}|{key[0]}|{key[1]:.4f}|{key[2]:.4f}".encode()).hexdigest()[:20]
                unique[key] = {"point_id": point_id, "image_id": image_id, "image": image_rel, "area": area,
                               "stage": roi.stage, "split": split, "label": roi.cls,
                               "subtype": roi.subtype or "", "fine_label": roi.fine_label,
                               "x": float(x), "y": float(y), "roi_x": float(rx), "roi_y": float(ry),
                               "coordinate_transform": roi.transform,
                               "pixel_source": image_record["pixel_source"], "refs": [ref],
                               "status": "included", "reason": ""}
        points = list(unique.values())
        diff_xy = image_record.pop("_diff_xy", None)
        if diff_xy is not None and len(diff_xy) and points:
            distance = _nearest_distance(diff_xy, np.array([[p["x"], p["y"]] for p in points]))
            image_record["tiff_differing_pixels_within_2px_of_roi_point"] = round(float(np.mean(distance <= 2)), 4)
        if points:
            xy = np.array([[p["x"], p["y"]] for p in points])
            labels = np.array([p["fine_label"] for p in points])
            location = Counter((round(p["x"], 4), round(p["y"], 4)) for p in points)
        for index, point in enumerate(points):
            same_128, other_128 = _neighbor_counts(xy, labels, index, PATCH_SIZE / 2)
            _, other_center = _neighbor_counts(xy, labels, index, CENTER_BOX / 2)
            point.update(neighbors_same_label_in_patch=same_128, neighbors_other_label_in_patch=other_128,
                         neighbors_other_label_in_center=other_center)
            cx, cy = math.floor(point["x"]), math.floor(point["y"])
            left, top = cx - PATCH_SIZE // 2, cy - PATCH_SIZE // 2
            point.update(center_x=cx, center_y=cy, left=left, top=top)
            if not (0 <= point["x"] < width and 0 <= point["y"] < height):
                point.update(status="excluded", reason="point_outside_image")
            elif location[(round(point["x"], 4), round(point["y"], 4))] > 1:
                point.update(status="excluded", reason="same_coordinate_conflicting_labels")
            elif other_center:
                point.update(status="excluded", reason=f"other_label_within_center_{CENTER_BOX}px")
            elif left < 0 or top < 0 or left + PATCH_SIZE > width or top + PATCH_SIZE > height:
                point.update(status="excluded", reason="patch_crosses_image_boundary")

        # 3) patches (raw TIFF pixels, no resize / normalisation)
        for point in points:
            if point["status"] != "included":
                continue
            left, top = point["left"], point["top"]
            patch = np.ascontiguousarray(array[top:top + PATCH_SIZE, left:left + PATCH_SIZE])
            relative = Path(f"patches_{PATCH_SIZE}") / split / point["fine_label"] / f"{point['point_id']}.tif"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(target, patch, photometric="rgb", metadata=None)
            samples.append({"patch_id": point["point_id"], "path": relative.as_posix(),
                            "sha256": sha256_file(target), **{k: v for k, v in point.items() if k not in
                                                              {"refs", "status", "reason"}},
                            "wafer": area, "size": PATCH_SIZE, "dtype": "uint8", "channels": "RGB",
                            "roi_files": len({r["roi"] for r in point["refs"]}),
                            "roi_refs": json.dumps(point["refs"], ensure_ascii=False),
                            "labels_source": "provider", "expert_verified": False,
                            "review_status": "provider_label_unverified"})
        for point in points:
            point_rows.append({**{k: v for k, v in point.items() if k != "refs"},
                               "roi_refs": json.dumps(point["refs"], ensure_ascii=False)})
        _overview(array, points).save(preview / f"overview_{image_id}.png")

        after_stat = image_path.stat()
        if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
            raise RuntimeError(f"source changed during build: {image_rel}")
        images.append({"image_id": image_id, **image_record, "area": area, "stage": rois[0].stage,
                       "split": split, "roi_files": sorted(r.relpath for r in rois),
                       "overview": f"preview/overview_{image_id}.png"})
        print(f"  {sum(p['status'] == 'included' for p in points)} patches / {len(points)} unique points",
              flush=True)
        del array

    return _finish(source, output, split_map, overrides, preview_per_label, started, roi_files,
                   images, samples, point_rows, stats)


def _roi_record(roi: RoiFile) -> dict:
    return {"relpath": roi.relpath, "sha256": roi.sha256, "bytes": roi.size, "area": roi.area,
            "stage": roi.stage, "label": roi.cls, "subtype": roi.subtype, "fine_label": roi.fine_label,
            "status": "used" if roi.status == "mapped" else "excluded", "reason": roi.reason,
            "duplicate_of": roi.duplicate_of,
            "point_count": None if roi.points is None else len(roi.points),
            "subpixel": roi.subpixel, "image": roi.image_relpath, "mapping_basis": roi.mapping_basis,
            "image_hint_in_name": roi.image_hint, "hint_matches_image": roi.hint_matches_image,
            "coordinate_transform": roi.transform,
            "transform_basis": roi.transform_basis,
            "contrast_probe": roi.probes, "warnings": roi.warnings}


def _nested_counts(rows: list[dict]) -> dict:
    counts: dict = {}
    for row in rows:
        split = counts.setdefault(row["split"], {})
        cls = split.setdefault(row["label"], {})
        key = row["subtype"] or "-"
        cls[key] = cls.get(key, 0) + 1
    return {split: {cls: dict(sorted(subs.items())) for cls, subs in sorted(by.items())}
            for split, by in sorted(counts.items())}


def _contact_sheet(output: Path, samples: list[dict], per_label: int) -> tuple[str, str]:
    tile, gap, label_width = PATCH_SIZE, 6, 92
    by_label = defaultdict(list)
    for row in samples:
        by_label[row["fine_label"]].append(row)
    labels = [name for name in FINE_LABELS if by_label[name]]
    columns = per_label + 1
    sheet = Image.new("RGB", (label_width + columns * (tile + gap), len(labels) * (tile + gap + 14) + 24),
                      (17, 24, 39))
    draw = ImageDraw.Draw(sheet)
    draw.text((label_width, 4), f"provider labels (not expert verified) | {per_label} samples + mean of all "
              "| ticks mark patch centre", fill=(229, 231, 235))
    for row_index, name in enumerate(labels):
        rows = sorted(by_label[name], key=lambda r: hashlib.sha256(r["patch_id"].encode()).hexdigest())
        chosen = rows[:per_label]
        y0 = 24 + row_index * (tile + gap + 14)
        draw.text((4, y0 + tile // 2 - 12), f"{name}\nn={len(rows)}", fill=(229, 231, 235))
        for column, row in enumerate(chosen):
            x0 = label_width + column * (tile + gap)
            sheet.paste(Image.fromarray(tifffile.imread(output / row["path"])), (x0, y0))
            draw.text((x0, y0 + tile + 1), f"{row['split']} W{row['area']} {row['stage'] or ''}"[:22],
                      fill=(156, 163, 175))
        mean = np.zeros((tile, tile, 3), np.float64)
        for row in rows:
            mean += tifffile.imread(output / row["path"])
        mean /= len(rows)
        low, high = np.percentile(mean, [1, 99])
        stretched = np.clip((mean - low) * 255.0 / max(high - low, 1e-6), 0, 255).astype(np.uint8)
        x0 = label_width + per_label * (tile + gap)
        sheet.paste(Image.fromarray(stretched), (x0, y0))
        draw.text((x0, y0 + tile + 1), "mean (stretched)", fill=(250, 204, 21))
        for column in range(columns):
            x0 = label_width + column * (tile + gap)
            cx, cy = x0 + tile // 2, y0 + tile // 2
            for a, b in (((cx, y0), (cx, y0 + 6)), ((cx, y0 + tile - 7), (cx, y0 + tile - 1)),
                         ((x0, cy), (x0 + 6, cy)), ((x0 + tile - 7, cy), (x0 + tile - 1, cy))):
                draw.line((a, b), fill=(250, 204, 21))
    sheet_name = "preview/contact_sheet.png"
    sheet.save(output / sheet_name)
    return sheet_name, "display only: tiles are raw patches; the mean tile is contrast-stretched"


def _finish(source, output, split_map, overrides, preview_per_label, started, roi_files, images,
            samples, point_rows, stats):
    sample_fields = ["patch_id", "path", "sha256", "point_id", "image_id", "image", "area", "wafer", "stage",
                     "split", "label", "subtype", "fine_label", "x", "y", "roi_x", "roi_y",
                     "coordinate_transform", "pixel_source", "center_x", "center_y", "left", "top",
                     "size", "dtype",
                     "channels", "neighbors_same_label_in_patch", "neighbors_other_label_in_patch",
                     "neighbors_other_label_in_center", "roi_files", "roi_refs", "labels_source",
                     "expert_verified", "review_status"]
    point_fields = ["point_id", "image_id", "image", "area", "stage", "split", "label", "subtype",
                    "fine_label", "x", "y", "roi_x", "roi_y", "coordinate_transform", "center_x",
                    "center_y", "left", "top", "status", "reason", "neighbors_same_label_in_patch",
                    "neighbors_other_label_in_patch", "neighbors_other_label_in_center", "roi_refs"]
    samples.sort(key=lambda r: r["path"])
    write_csv(output / "samples.csv", samples, sample_fields)
    write_csv(output / "points.csv", point_rows, point_fields)
    roi_records = [_roi_record(r) for r in roi_files]
    write_csv(output / "roi_files.csv", [{**r, "contrast_probe": json.dumps(r["contrast_probe"]),
                                          "warnings": ";".join(r["warnings"])} for r in roi_records],
              list(roi_records[0]))
    write_json(output / "images.json", images)

    sheet, sheet_note = _contact_sheet(output, samples, preview_per_label)
    excluded_points = Counter(r["reason"] for r in point_rows if r["status"] != "included")
    excluded_rois = [{"relpath": r["relpath"], "reason": r["reason"], "duplicate_of": r["duplicate_of"],
                      "point_count": r["point_count"], "label": r["fine_label"]}
                     for r in roi_records if r["status"] == "excluded"]
    provider_points = Counter(f"{r['fine_label'] or 'unknown'}" for r in roi_records
                              if r["status"] == "used" for _ in range(r["point_count"] or 0))
    counts = _nested_counts(samples)
    totals = Counter(row["fine_label"] for row in samples)
    rare = {name: totals.get(name, 0) for name in FINE_LABELS if totals.get(name, 0) < 300}
    for record in images:
        if (source / record["relpath"]).stat().st_size != record["bytes"]:
            raise RuntimeError(f"source changed during build: {record['relpath']}")
    manifest = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "task": "center_point_patch_classification_class_and_subtype",
        "created_at": datetime.now(UTC).isoformat(), "started_at": started,
        "code_revision": code_revision(), "python": sys.version.split()[0],
        "source_root": str(source), "zip_archives_read": False,
        "labels_source": "provider", "label_basis": "ImageJ_point_ROI_filename",
        "expert_verified": False, "research_only": True,
        "classes": list(CLASSES), "subtypes": {k: list(v) for k, v in SUBTYPES.items()},
        "fine_labels": list(FINE_LABELS),
        "patch": {"size": PATCH_SIZE, "dtype": "uint8", "channels": "RGB", "format": "TIFF uncompressed",
                  "source_pixels": "lossless_tiff_raw_values", "resized": False, "normalized": False,
                  "padding": "none_boundary_patches_excluded",
                  "center_rule": "center = floor(x), floor(y) in TIFF pixel-corner coordinates",
                  "crop_rule": "left = center_x - 64, top = center_y - 64; centre 64 px = [32:96, 32:96]"},
        "coordinates": {"space": "tiff_pixel_grid_continuous_pixel_corner_origin",
                        "pixel_center": "i + 0.5", "x_axis": "columns, left to right",
                        "y_axis": "rows, top to bottom",
                        "transforms": {"identity": "x = roi_x", "hflip": "x = width - roi_x"},
                        "roi_columns": "roi_x/roi_y keep the provider coordinates unchanged"},
        "split_basis": "Area (wafer) groups; Before and After of the same Area share a split",
        "split_map": split_map, "orientation_overrides": overrides,
        "orientation_probe": {"method": "median 15x15 local std at points / at 8 shifted controls",
                              "min_points": MIN_PROBE_POINTS, "min_signal_ratio": MIN_SIGNAL_RATIO,
                              "min_preference": MIN_PREFERENCE},
        "exclusion_rules": {"center_box_px": CENTER_BOX,
                            "other_label_within_center": "excluded (ambiguous centre)",
                            "boundary": "excluded, no padding",
                            "same_coordinate_conflicting_labels": "excluded"},
        "counts": {"patches": len(samples), "by_split_class_subtype": counts,
                   "by_fine_label": dict(sorted(totals.items())),
                   "provider_points_in_used_roi_files": dict(sorted(provider_points.items())),
                   "unique_points": len(point_rows), "duplicate_point_refs": stats["duplicate_point_refs"],
                   "excluded_points_by_reason": dict(sorted(excluded_points.items())),
                   "images": len(images)},
        "rare_fine_labels_under_300": rare,
        "excluded_roi_files": excluded_rois,
        "sources": [{"relpath": r["relpath"], "sha256": r["sha256"], "bytes": r["bytes"],
                     "jpeg_sha256": r["jpeg_pair"].get("sha256"), "pixel_source": r["pixel_source"],
                     "pixel_check": r["pixel_check"]} for r in images],
        "roi_file_sha256": {r["relpath"]: r["sha256"] for r in roi_records},
        "files": {"samples": "samples.csv", "points": "points.csv", "roi_files": "roi_files.csv",
                  "images": "images.json", "contact_sheet": sheet, "contact_sheet_note": sheet_note},
        "limitations": [
            "provider labels from ROI file names; not expert verified",
            "TED a-f / TSD a-c physical definitions are not confirmed in this dataset",
            "unannotated regions are not negatives; there is no background class",
            "centre points only; not boxes or masks",
            "Areas without provider ROIs are not included",
        ],
    }
    manifest["samples_sha256"] = sha256_file(output / "samples.csv")
    write_json(output / "manifest.json", manifest)
    hashes = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*"))
              if p.is_file() and p.name != "output_hashes.json"}
    write_json(output / "output_hashes.json", hashes)
    return manifest


def validate(dataset: Path) -> dict:
    """Structural check: hashes, shapes, dtype, one split per Area."""
    dataset = Path(dataset).resolve()
    if (dataset / "BUILD_FAILED.json").exists():
        raise ValueError("dataset build failed; do not use")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema")
    if sha256_file(dataset / "samples.csv") != manifest["samples_sha256"]:
        raise ValueError("samples.csv hash mismatch")
    hashes = json.loads((dataset / "output_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        if sha256_file(dataset / relative) != expected:
            raise ValueError(f"hash mismatch: {relative}")
    rows = read_csv(dataset / "samples.csv")
    split_by_area: dict[str, set] = defaultdict(set)
    for row in rows:
        patch = tifffile.imread(dataset / row["path"])
        if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3) or patch.dtype != np.uint8:
            raise ValueError(f"bad patch {row['path']}")
        if row["fine_label"] not in FINE_LABELS:
            raise ValueError(f"unknown label {row['fine_label']}")
        split_by_area[row["area"]].add(row["split"])
    leaking = {area: sorted(s) for area, s in split_by_area.items() if len(s) > 1}
    if leaking:
        raise ValueError(f"areas in more than one split: {leaking}")
    if len(rows) != manifest["counts"]["patches"]:
        raise ValueError("patch count mismatch")
    return {"status": "passed", "patches": len(rows), "files_hashed": len(hashes),
            "scope": "structure and bytes only; labels are provider labels, not expert verified"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="build a new dataset folder")
    make.add_argument("--source", required=True, type=Path, help="the '2D XRT' folder")
    make.add_argument("--output", required=True, type=Path, help="new folder outside any git repo")
    make.add_argument("--splits", type=Path, help='JSON {"<area>": "train|val|test"}; default 1,9/2/8')
    make.add_argument("--orientation-overrides", type=Path,
                      help='JSON {"<roi relpath>": "identity|hflip"} kept outside git')
    make.add_argument("--preview-per-label", type=int, default=8)
    check = commands.add_parser("validate", help="verify an existing dataset folder")
    check.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate(args.dataset), ensure_ascii=False, indent=2))
        return 0
    splits = json.loads(args.splits.read_text(encoding="utf-8-sig")) if args.splits else None
    overrides = (json.loads(args.orientation_overrides.read_text(encoding="utf-8-sig"))
                 if args.orientation_overrides else None)
    manifest = build(args.source, args.output, splits, overrides, args.preview_per_label)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
