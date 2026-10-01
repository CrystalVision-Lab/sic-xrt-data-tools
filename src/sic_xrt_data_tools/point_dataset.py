"""Read-only ImageJ point to centered classification patch candidates, schema v1."""

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from roifile import ROI_TYPE, ImagejRoi

CLASSES = ("BPD", "TED", "TSD")
COLORS = {"BPD": "#ef4444", "TED": "#22c55e", "TSD": "#38bdf8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_from_name(name: str) -> str | None:
    match = re.search(r"(?:^|[^A-Za-z])(BPD|TED|TSD)(?:[^A-Za-z]|$)", name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def stage(name: str) -> str | None:
    lowered = name.lower()
    if re.search(r"(?:^|[^a-z])before(?:[^a-z]|$)", lowered):
        return "before"
    if re.search(r"(?:^|[^a-z])after(?:[^a-z]|$)", lowered):
        return "after"
    return None


def decode_roi(data: bytes) -> np.ndarray:
    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("empty_or_invalid_roi")
    roi = ImagejRoi.frombytes(data)
    if roi.roitype != ROI_TYPE.POINT:
        raise ValueError("non_point_roi")
    if roi.position > 1 or roi.z_position > 1 or roi.t_position > 1:
        raise ValueError("roi_refers_to_another_page")
    points = np.asarray(roi.coordinates(), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("invalid_point_coordinates")
    return points


def choose_image(roi_name: str, candidates: list[str], parent: str) -> tuple[str | None, str]:
    local = [name for name in candidates if str(PurePosixPath(name).parent) == parent]
    roi_stage = stage(PurePosixPath(roi_name).name)
    if len(local) > 1 and roi_stage:
        local = [name for name in local if stage(PurePosixPath(name).name) == roi_stage]
    if len(local) == 1:
        image_stage = stage(PurePosixPath(local[0]).name)
        if roi_stage and image_stage and roi_stage != image_stage:
            return None, "conflicting_before_after"
        return local[0], "same_directory_unique_stage_candidate_pending_visual_review"
    return None, "no_unique_tiff_in_roi_directory"


def discover(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    mappings, exclusions, archives = [], [], []
    for archive_path in sorted(root.glob("*.zip")):
        match = re.fullmatch(r"(\d+)\s+Area", archive_path.stem, re.IGNORECASE)
        if not match:
            exclusions.append({"archive": archive_path.name, "reason": "unrecognized_wafer_group"})
            continue
        wafer = match.group(1)
        before = archive_path.stat()
        archive_hash = sha256(archive_path)
        archives.append({"archive": archive_path.name, "wafer": wafer,
                         "sha256": archive_hash, "bytes": before.st_size,
                         "mtime_ns": before.st_mtime_ns})
        with zipfile.ZipFile(archive_path) as archive:
            images = [info.filename for info in archive.infolist()
                      if info.filename.lower().endswith((".tif", ".tiff"))]
            roi_entries = []
            for info in archive.infolist():
                if info.filename.lower().endswith(".roi"):
                    roi_entries.append((info.filename, archive.read(info),
                                        str(PurePosixPath(info.filename).parent)))
                elif info.filename.lower().endswith("/roiset.zip"):
                    if info.file_size > 10 * 1024 * 1024:
                        exclusions.append({"archive": archive_path.name,
                                           "roi": info.filename, "reason": "nested_roi_zip_too_large"})
                        continue
                    with zipfile.ZipFile(io.BytesIO(archive.read(info))) as nested:
                        for member in nested.infolist():
                            if member.filename.lower().endswith(".roi"):
                                roi_entries.append((info.filename + "::" + member.filename,
                                                    nested.read(member),
                                                    str(PurePosixPath(info.filename).parent)))
            for roi_name, content, parent in roi_entries:
                label = label_from_name(roi_name.rsplit("::", 1)[-1])
                item = {"archive": archive_path.name, "wafer": wafer,
                        "roi": roi_name, "label": label,
                        "roi_sha256": hashlib.sha256(content).hexdigest()}
                if not label:
                    exclusions.append({**item, "reason": "unknown_label"})
                    continue
                try:
                    points = decode_roi(content)
                except (ValueError, IndexError, KeyError) as error:
                    exclusions.append({**item, "reason": str(error)})
                    continue
                image, evidence = choose_image(roi_name.rsplit("::", 1)[-1], images, parent)
                if image is None:
                    exclusions.append({**item, "reason": evidence, "point_count": len(points)})
                    continue
                mappings.append({**item, "image": image, "points": points,
                                 "mapping_evidence": evidence, "archive_sha256": archive_hash})
        after = archive_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Original archive changed during inspection: {archive_path}")
    return mappings, exclusions, archives


def unique_points(mappings: list[dict]) -> tuple[list[dict], list[dict]]:
    points, duplicate_refs = [], []
    seen = {}
    for mapping in mappings:
        for number, (x, y) in enumerate(mapping["points"]):
            key = (mapping["archive"], mapping["image"], mapping["label"],
                   round(float(x), 4), round(float(y), 4))
            ref = {"roi": mapping["roi"], "point_index": number}
            if key in seen:
                points[seen[key]]["refs"].append(ref)
                duplicate_refs.append({"roi": mapping["roi"], "point_index": number,
                                       "canonical_point": seen[key]})
                continue
            seen[key] = len(points)
            points.append({k: v for k, v in mapping.items() if k != "points"} | {
                "x": float(x), "y": float(y), "refs": [ref]})
    return points, duplicate_refs


def patch_bounds(x: float, y: float, size: int, width: int, height: int):
    center_x, center_y = math.floor(x + 0.5), math.floor(y + 0.5)
    left, top = center_x - size // 2, center_y - size // 2
    if left < 0 or top < 0 or left + size > width or top + size > height:
        return None
    return left, top, left + size, top + size


def display_rgb(array: np.ndarray, limits=None) -> tuple[np.ndarray, tuple]:
    if limits is None:
        limits = tuple(float(v) for v in np.percentile(array, [1, 99]))
    low, high = limits
    if array.dtype == np.uint8:
        view = array
    else:
        view = np.clip((array.astype(np.float32) - low) * (255 / max(high - low, 1)),
                       0, 255).astype(np.uint8)
    if view.ndim == 2:
        view = np.repeat(view[..., None], 3, axis=2)
    return view, limits


def make_review(image, points: list[dict], output: Path, source_id: str, sizes: list[int]):
    stride = max(1, math.ceil(max(image.shape[:2]) / 1600))
    preview, limits = display_rgb(np.asarray(image[::stride, ::stride]))
    overview = Image.fromarray(preview)
    draw = ImageDraw.Draw(overview)
    for point in points:
        x, y = point["x"] / stride, point["y"] / stride
        color = COLORS[point["label"]]
        draw.ellipse((x-2, y-2, x+2, y+2), outline=color)
    overview_name = f"{source_id}_overview.png"
    overview.save(output / overview_name)
    sheets = []
    for label in CLASSES:
        candidates = [p for p in points if p["label"] == label and
                      patch_bounds(p["x"], p["y"], max(sizes), image.shape[1], image.shape[0])]
        if not candidates:
            continue
        indices = np.linspace(0, len(candidates) - 1, min(16, len(candidates)), dtype=int)
        sheet = Image.new("RGB", (4 * 280, math.ceil(len(indices) / 4) * 170), "#111827")
        canvas = ImageDraw.Draw(sheet)
        for cell, index in enumerate(indices):
            point = candidates[index]
            left, top, right, bottom = patch_bounds(point["x"], point["y"], max(sizes),
                                                   image.shape[1], image.shape[0])
            rgb, _ = display_rgb(np.asarray(image[top:bottom, left:right]), limits)
            raw = Image.fromarray(rgb).resize((128, 128))
            marked = raw.copy()
            overlay = ImageDraw.Draw(marked)
            overlay.ellipse((59, 59, 69, 69), outline=COLORS[label], width=1)
            px, py = (cell % 4) * 280, (cell // 4) * 170
            sheet.paste(raw, (px, py))
            sheet.paste(marked, (px + 134, py))
            canvas.text((px, py+132), f"{label} x={point['x']:.1f} y={point['y']:.1f}", fill="white")
            canvas.text((px, py+148), "original / marked (review only)", fill="white")
        name = f"{source_id}_{label}_samples.png"
        sheet.save(output / name)
        sheets.append(name)
    return overview_name, sheets


def build(root: Path, output: Path, split_map: dict[str, str], sizes=(128, 256)) -> dict:
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError("Output must be outside original data")
    if output.exists():
        raise ValueError("Output already exists; choose a new version folder")
    if any(v not in {"train", "val", "test", "unassigned"} for v in split_map.values()):
        raise ValueError("Invalid split name")
    sizes = sorted(set(sizes))
    if not sizes or any(s < 16 or s > 1024 or s % 2 for s in sizes):
        raise ValueError("Patch sizes must be even integers from 16 to 1024")
    if not (root / "2D XRT").is_dir():
        raise ValueError("Missing 2D XRT source folder")
    mappings, exclusions, archive_records = discover(root / "2D XRT")
    points, duplicates = unique_points(mappings)
    if not points:
        raise ValueError("No mapped, labeled point annotations found")
    output.mkdir(parents=True)
    review = output / "검수"
    review.mkdir()
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["archive"], point["image"])].append(point)
    rows, point_rows, source_records, reviews = [], [], [], []
    counts, rejected = Counter(), Counter()
    manifests = output / "기록"
    manifests.mkdir()
    try:
        for (archive_name, image_name), source_points in sorted(grouped.items()):
            source_id = hashlib.sha256((archive_name + "::" + image_name).encode()).hexdigest()[:12]
            wafer = source_points[0]["wafer"]
            split = split_map.get(wafer, "unassigned")
            print(f"Processing wafer {wafer}: {image_name}", flush=True)
            with tempfile.TemporaryDirectory(prefix="xrt-dataset-") as temporary:
                # Fixed scratch filename avoids trusting ZIP member paths during extraction.
                scratch = Path(temporary) / "source.tif"
                with (zipfile.ZipFile(root / "2D XRT" / archive_name) as archive,
                      archive.open(image_name) as source, scratch.open("wb") as destination):
                    shutil.copyfileobj(source, destination, 4 * 1024 * 1024)
                source_hash = sha256(scratch)
                with tifffile.TiffFile(scratch) as tiff:
                    if len(tiff.pages) != 1:
                        raise ValueError("2D point pipeline requires a single-page TIFF")
                    axes = tiff.series[0].axes
                    shape = tiff.series[0].shape
                    if axes not in {"YX", "YXS"} or len(shape) not in {2, 3}:
                        raise ValueError(f"Unsupported 2D axes: {axes}, shape: {shape}")
                    if len(shape) == 3 and shape[2] != 3:
                        raise ValueError("Only grayscale or RGB TIFF is supported")
                try:
                    image = tifffile.memmap(scratch, mode="r")
                except ValueError:
                    image = tifffile.imread(scratch)
                height, width = image.shape[:2]
                valid_points = []
                for point in source_points:
                    point_id = hashlib.sha256(f"{source_id}:{point['label']}:{point['x']:.4f}:{point['y']:.4f}".encode()).hexdigest()[:20]
                    point["point_id"] = point_id
                    in_bounds = 0 <= point["x"] < width and 0 <= point["y"] < height
                    point_rows.append({"point_id": point_id, "source_id": source_id,
                                       "wafer": wafer, "label": point["label"], "x": point["x"],
                                       "y": point["y"], "in_bounds": in_bounds,
                                       "refs_json": json.dumps(point["refs"], ensure_ascii=False)})
                    if in_bounds:
                        valid_points.append(point)
                    else:
                        rejected["point_out_of_image"] += 1
                sample = np.asarray(image[::max(1, height//1000), ::max(1, width//1000)])
                color_fraction = (float(np.mean(np.any(sample != sample[..., :1], axis=-1)))
                                  if sample.ndim == 3 else 0.0)
                source_records.append({"source_id": source_id, "archive": archive_name,
                                       "image": image_name, "sha256": source_hash,
                                       "shape": list(image.shape), "axes": axes,
                                       "dtype": str(image.dtype), "wafer": wafer, "split": split,
                                       "color_pixel_fraction_sampled": color_fraction,
                                       "visual_review": "pending"})
                overview, sheets = make_review(image, valid_points, review, source_id, sizes)
                reviews.append({"source_id": source_id, "wafer": wafer, "image": image_name,
                                "overview": overview, "sheets": sheets})
                coords = np.array([[p["x"], p["y"]] for p in valid_points])
                labels = np.array([p["label"] for p in valid_points])
                for point in valid_points:
                    for size in sizes:
                        bounds = patch_bounds(point["x"], point["y"], size, width, height)
                        if bounds is None:
                            rejected[f"edge_{size}"] += 1
                            continue
                        left, top, right, bottom = bounds
                        other = ((coords[:, 0] >= left) & (coords[:, 0] < right)
                                 & (coords[:, 1] >= top) & (coords[:, 1] < bottom)
                                 & (labels != point["label"]))
                        if np.any(other):
                            rejected[f"mixed_class_{size}"] += 1
                            continue
                        patch = np.array(image[top:bottom, left:right], copy=True)
                        patch_id = f"{point['point_id']}_{size}"
                        relative = Path(f"patches_{size}") / split / point["label"] / f"{patch_id}.tif"
                        target = output / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        tifffile.imwrite(target, patch, photometric="rgb" if patch.ndim == 3 else "minisblack",
                                         metadata=None)
                        rows.append({"patch_id": patch_id, "path": relative.as_posix(),
                                     "sha256": sha256(target), "point_id": point["point_id"],
                                     "source_id": source_id, "wafer": wafer, "split": split,
                                     "label": point["label"], "size": size, "x": point["x"],
                                     "y": point["y"], "left": left, "top": top,
                                     "dtype": str(patch.dtype), "review_status": "pending",
                                     "label_basis": "ImageJ_point_ROI_filename"})
                        counts[f"{size}/{split}/{point['label']}"] += 1
                if isinstance(image, np.memmap):
                    image._mmap.close()
                del image
                print(f"  patches so far: {len(rows)}", flush=True)
        write_csv(manifests / "samples.csv", rows, list(rows[0]) if rows else ["patch_id", "path"])
        write_csv(manifests / "points.csv", point_rows,
                  ["point_id", "source_id", "wafer", "label", "x", "y", "in_bounds", "refs_json"])
        mapping_report = [{k: (len(v) if k == "points" else v) for k, v in m.items()} for m in mappings]
        write_json(manifests / "mappings.json", mapping_report)
        write_json(manifests / "excluded_rois.json", exclusions)
        write_json(manifests / "duplicate_points.json", duplicates)
        write_json(manifests / "sources.json", source_records)
        write_json(manifests / "archives.json", archive_records)
        unlabeled = []
        mapped_images = set(grouped)
        for record in archive_records:
            with zipfile.ZipFile(root / "2D XRT" / record["archive"]) as archive:
                for entry in archive.infolist():
                    if entry.filename.lower().endswith((".jpg", ".tif", ".tiff")) and (
                            record["archive"], entry.filename) not in mapped_images:
                        unlabeled.append({"archive": record["archive"], "image": entry.filename,
                                          "bytes": entry.file_size, "status": "not_used_as_normal"})
        write_json(manifests / "unlabeled_2d.json", unlabeled)
        stacks = []
        for path in sorted((root / "3D XRT").glob("*.tif")):
            with tifffile.TiffFile(path) as stack:
                stacks.append({"path": str(path), "shape": list(stack.series[0].shape),
                               "axes": stack.series[0].axes, "dtype": str(stack.pages[0].dtype),
                               "status": "unlabeled_not_used_for_training"})
        write_json(manifests / "unlabeled_3d.json", stacks)
        for record in archive_records:
            now = (root / "2D XRT" / record["archive"]).stat()
            if (now.st_size, now.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise RuntimeError("Original changed during build")
        summary = {"schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
                   "task": "center_point_patch_classification_candidates",
                   "classes": list(CLASSES), "patch_sizes": sizes, "patch_count": len(rows),
                   "unique_points": len(points), "duplicate_point_refs": len(duplicates),
                   "excluded_rois": len(exclusions), "source_count": len(source_records),
                   "counts": dict(sorted(counts.items())), "rejected_patches": dict(rejected),
                   "split_map": split_map, "split_basis": "user_confirmed_distinct_Area_wafers",
                   "review_status": "pending", "normal_samples": 0,
                   "originals_modified": False,
                   "limitations": ["ROI filename labels and image matching require visual review",
                                   "not bounding boxes or segmentation masks",
                                   "no normal class; no unannotated region is assumed negative",
                                   "multiple sizes of the same point are not independent samples"]}
        write_json(output / "summary.json", summary)
        render_index(output, summary, reviews)
        (output / "읽어주세요.txt").write_text(
            "SiC XRT 점 주석 기반 분류 데이터셋 v1\n\n"
            "검수.html을 브라우저로 열어 원본과 주석의 정합을 확인하세요.\n"
            "patches_128 / patches_256에는 표시를 덧그리지 않은 원본 dtype TIFF 패치가 있습니다.\n"
            "train/val/test는 서로 다른 웨이퍼로 분리했습니다. 두 크기는 별도 실험으로 사용하세요.\n"
            "정답은 ROI 파일명의 BPD/TED/TSD에서 가져왔으며 검수 상태는 pending입니다.\n"
            "점은 박스/윤곽이 아닙니다. 정상 자료는 생성하지 않았습니다.\n"
            "색이 덧그려진 원본인지 검수한 뒤 학습하세요.\n"
            "다른 클래스의 점이 포함된 패치와 경계 밖 패치는 제외했습니다.\n"
            "기록/samples.csv에 원본 ID, 웨이퍼, 좌표, 클래스, 분할, SHA-256을 기록했습니다.\n"
            "기록/excluded_rois.json에 빈 파일/unknown/대응 불가 주석을 기록했습니다.\n"
            "기록/unlabeled_2d.json와 unlabeled_3d.json의 자료는 학습에 사용하지 않았습니다.\n"
            f"생성 패치: {len(rows):,}개 / 원본 영상: {len(source_records)}개\n",
            encoding="utf-8-sig")
        return summary
    except Exception:
        write_json(output / "BUILD_FAILED.json", {"status": "incomplete_do_not_train"})
        raise


def render_index(output: Path, summary: dict, reviews: list[dict]):
    sections = []
    for item in reviews:
        pictures = "".join(f'<a href="검수/{html.escape(name)}"><img loading="lazy" src="검수/{html.escape(name)}"></a>'
                           for name in [item["overview"], *item["sheets"]])
        sections.append(f'<section><h2>웨이퍼 {item["wafer"]}</h2><p>{html.escape(item["image"])}</p>{pictures}</section>')
    rows = "".join(f"<tr><td>{html.escape(key)}</td><td>{count:,}</td></tr>" for key, count in summary["counts"].items())
    content = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>SiC XRT 데이터셋 검수</title>
<style>body{{font:16px system-ui;background:#f4f6fa;color:#172033;max-width:1200px;margin:40px auto;padding:0 24px}}section{{background:white;padding:24px;margin:24px 0;border-radius:16px}}img{{max-width:100%;margin:12px 0}}table{{border-collapse:collapse}}td,th{{padding:8px 24px;border-bottom:1px solid #ddd}}.notice{{background:#fff3cd;padding:20px;border-radius:12px}}a{{color:#1d4ed8}}</style>
<h1>SiC XRT 점 주석 데이터셋 v1</h1><p>원본 {summary['source_count']}개 · 패치 {summary['patch_count']:,}개 · 128/256 크기</p>
<div class="notice">검수 상태: <b>확인 필요</b>. 정답은 ROI 파일명에서 가져왔습니다. 원본 위의 점이 실제 결함에 맞는지, 색/번호가 원본에 이미 덧그려져 있지는 않은지 확인하세요. 정상 클래스는 없습니다.</div>
<p>빨강 BPD · 초록 TED · 파랑 TSD. 샘플은 왼쪽 원본 / 오른쪽 중심 표시입니다. 표시는 검수 화면에만 있습니다.</p>
<p>웨이퍼 분할: {html.escape(str(summary['split_map']))}. 같은 웨이퍼의 모든 영상과 두 크기의 패치를 같은 분할에 두었습니다.</p>
<table><tr><th>크기/분할/클래스</th><th>패치 수</th></tr>{rows}</table>
<p><a href="기록/samples.csv">패치 정답·좌표 목록</a> · <a href="기록/excluded_rois.json">보류 주석</a> · <a href="읽어주세요.txt">사용 안내</a></p>
{''.join(sections)}</html>'''
    (output / "검수.html").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path, help="JSON wafer id to train/val/test")
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8-sig"))
    summary = build(args.source, args.output, splits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
