"""Restore supplied point subtypes into a reviewable annotation workspace v2."""

import argparse
import csv
import hashlib
import html
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from roifile import ROI_TYPE, ImagejRoi

from .source_registry import digest_file, validate_registry

FINE_CLASSES = ("BPD",) + tuple("TED_" + c for c in "abcdef") + tuple("TSD_" + c for c in "abc")
COLORS = {"BPD": "#f43f5e", "TED": "#22c55e", "TSD": "#38bdf8", "unknown": "#f59e0b"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_hint(name):
    match = re.search(r"(?:^|[^a-z])(BPD|TED|TSD)(?=$|[^a-z])(?:\s*\(([a-f])\))?", name, re.IGNORECASE)
    if not match:
        return "unknown", None, "unknown"
    base, subtype = match.group(1).upper(), match.group(2)
    subtype = subtype.lower() if subtype else None
    if base == "BPD" or subtype is None:
        return base, subtype, base
    return base, subtype, base + "_" + subtype


def roi_context(asset):
    pieces = asset["locator"].split("::")
    # The primary ZIP's first member is either an ROI or a nested RoiSet ZIP.
    if len(pieces) > 1 and pieces[0].lower().endswith("area.zip"):
        return pieces[0], str(PurePosixPath(pieces[1]).parent)
    path = PurePosixPath(pieces[0])
    parent = path.parent
    if parent.name.lower() == "roiset":
        parent = parent.parent
    return "", str(parent)


def image_context(asset):
    pieces = asset["locator"].split("::")
    return (pieces[0], str(PurePosixPath(pieces[1]).parent)) if len(pieces) > 1 \
        else ("", str(PurePosixPath(pieces[0]).parent))


def associate_roi(roi, images):
    local = [a for a in images if image_context(a) == roi_context(roi)
             and a["area_id"] == roi["area_id"] and a["decode_status"] == "decoded"
             and a["phase_hint"] != "unknown" and a["phase_hint"] == roi["phase_hint"]]
    tiffs = [a for a in local if a["suffix"] in {".tif", ".tiff"}]
    if len(tiffs) == 1:
        return tiffs[0], "unique_same_phase_tiff_in_same_directory", local
    numeric = re.match(r"^(\d+)_", roi["locator"].split("::")[-1].rsplit("/", 1)[-1])
    if numeric:
        matches = [a for a in local if PurePosixPath(a["locator"].split("::")[-1]).stem == numeric.group(1)]
        if len(matches) == 1:
            return matches[0], "explicit_leading_image_number_in_same_directory", local
    if len(local) == 1:
        return local[0], "unique_same_phase_image_in_same_directory", local
    return None, "ambiguous_or_missing_image", local


class SourceReader:
    """Validate every original used; prefer a verified loose alias for large images."""
    def __init__(self, root, assets):
        self.root = Path(root).resolve(strict=True)
        self.by_id = {a["asset_id"]: a for a in assets}
        self.checked = {}
        self.aliases = defaultdict(list)
        for asset in assets:
            if asset["source_kind"] == "file" and asset.get("sha256"):
                self.aliases[asset["sha256"]].append(asset)

    def verify_file(self, asset):
        path = (self.root / asset["locator"]).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("original_outside_source_root")
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        key = asset["asset_id"]
        if key not in self.checked:
            if digest_file(path) != asset["sha256"]:
                raise RuntimeError("original_hash_changed")
            self.checked[key] = signature
        if self.checked[key] != signature:
            raise RuntimeError("original_changed_during_restore")
        return path

    @contextmanager
    def open_asset(self, asset, prefer_loose=False):
        aliases = self.aliases.get(asset.get("sha256"), []) if prefer_loose else []
        if aliases:
            selected = min(aliases, key=lambda a: a["locator"])
            with self.verify_file(selected).open("rb") as stream:
                yield stream
            return
        if asset["source_kind"] == "file":
            with self.verify_file(asset).open("rb") as stream:
                yield stream
            return
        parent = self.by_id[asset["parent_asset_id"]]
        with self.open_asset(parent) as stream, zipfile.ZipFile(stream) as archive:
            member = archive.infolist()[asset["ordinal_path"][-1]]
            if member.filename != asset["locator"].split("::")[-1]:
                raise ValueError("zip_ordinal_path_mismatch")
            data = archive.read(member)
            if hashlib.sha256(data).hexdigest() != asset["sha256"]:
                raise RuntimeError("member_hash_changed")
            with io.BytesIO(data) as content:
                yield content

    def points(self, asset):
        with self.open_asset(asset) as stream:
            roi = ImagejRoi.frombytes(stream.read())
        if roi.roitype != ROI_TYPE.POINT:
            raise ValueError("not_point_roi")
        if max(roi.position, roi.z_position, roi.t_position) > 1:
            raise ValueError("roi_page_not_single_image")
        points = np.asarray(roi.coordinates(), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("invalid_point_coordinates")
        return points

    def image(self, asset):
        if asset["metadata"].get("page_count", 1) != 1:
            raise ValueError("stack_requires_page_selection")
        with self.open_asset(asset, prefer_loose=True) as stream:
            if asset["suffix"] in {".tif", ".tiff"}:
                with tifffile.TiffFile(stream) as tif:
                    return tif.pages[0].asarray()
            with Image.open(stream) as image:
                image.load()
                return np.asarray(image).copy()

    def verify_unchanged(self):
        for asset_id, signature in self.checked.items():
            path = self.root / self.by_id[asset_id]["locator"]
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != signature:
                raise RuntimeError("used_original_changed_after_restore")


def restore_points(roi_groups, images, reader, assets_by_id):
    points, mappings, exclusions = [], [], []
    seen = {}
    for group in roi_groups:
        refs = [assets_by_id[i] for i in group["asset_ids"]]
        primary = [a for a in refs if a["locator"].split("::")[0].lower().endswith("area.zip")]
        roi = min(primary or refs, key=lambda a: (len(a["ordinal_path"]), a["locator"]))
        base, subtype, fine = label_hint(PurePosixPath(roi["locator"].split("::")[-1]).name)
        mapping = {"roi_asset_id": roi["asset_id"], "roi_alias_ids": group["asset_ids"],
                   "all_byte_alias_ids": group.get("all_byte_alias_ids", group["asset_ids"]),
                   "roi_locator": roi["locator"], "base_label": base, "subtype": subtype,
                   "fine_label": fine, "annotation_origin": "supplied_roi_filename",
                   "review_status": "provider_imported_pending_review"}
        if roi["decode_status"] != "decoded":
            exclusions.append(mapping | {"reason": "invalid_or_empty_roi"})
            continue
        try:
            coordinates = reader.points(roi)
        except (ValueError, IndexError, KeyError) as error:
            exclusions.append(mapping | {"reason": str(error)})
            continue
        image, evidence, alternatives = associate_roi(roi, images)
        mapping.update(mapping_evidence=evidence, candidate_image_ids=[a["asset_id"] for a in alternatives],
                       coordinate_count=len(coordinates),
                       coordinate_set_sha256=hashlib.sha256(coordinates.tobytes()).hexdigest())
        if image is None:
            exclusions.append(mapping | {"reason": evidence})
            continue
        width, height = image["metadata"]["width"], image["metadata"]["height"]
        mapping.update(image_asset_id=image["asset_id"], area_id=image["area_id"],
                       phase=image["phase_hint"], mapping_status="candidate_pending_visual_review",
                       point_ids=[])
        for index, (x, y) in enumerate(coordinates):
            x, y = float(x), float(y)
            key = (image["asset_id"], fine, round(x, 4), round(y, 4))
            ref = {"roi_asset_id": roi["asset_id"], "point_index": index}
            if key in seen:
                point = points[seen[key]]
                point["roi_refs"].append(ref)
            else:
                point = {
                    "point_id": "pt_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24],
                    "image_asset_id": image["asset_id"], "area_id": image["area_id"],
                    "phase": image["phase_hint"], "x": x, "y": y, "base_label": base,
                    "subtype": subtype, "fine_label": fine, "roi_refs": [ref],
                    "coordinate_valid": 0 <= x < width and 0 <= y < height,
                    "mapping_status": "candidate_pending_visual_review",
                    "review_status": "provider_imported_pending_review",
                    "annotation_origin": "supplied_roi_filename", "split": "unassigned",
                    "physical_instance_id": None, "human_verified": False,
                    "eligible_for_verified_evaluation": False,
                }
                seen[key] = len(points)
                points.append(point)
            mapping["point_ids"].append(point["point_id"])
        mappings.append(mapping)
    locations = defaultdict(list)
    for point in points:
        locations[(point["image_asset_id"], round(point["x"], 4), round(point["y"], 4))].append(point)
    conflicts = []
    for group in locations.values():
        if len({p["fine_label"] for p in group}) > 1:
            conflicts.append({"point_ids": [p["point_id"] for p in group],
                              "reason": "same_coordinate_different_labels"})
            for point in group:
                point["review_status"] = "label_conflict_pending_review"
    return points, mappings, exclusions, conflicts


def semantic_roi_groups(groups, by_id):
    """Bytes alone cannot deduplicate label filenames or separate acquisition contexts."""
    result = []
    for group in groups:
        if group["kind"] != "roi":
            continue
        aliases = [by_id[i] for i in group["asset_ids"]]
        primary = [a for a in aliases if a["locator"].split("::")[0].lower().endswith("area.zip")]
        contexts = defaultdict(list)
        for asset in primary or aliases:
            fine = label_hint(PurePosixPath(asset["locator"].split("::")[-1]).name)[2]
            key = (roi_context(asset), asset["area_id"], asset["phase_hint"], fine)
            contexts[key].append(asset["asset_id"])
        for ids in contexts.values():
            result.append(group | {"asset_ids": ids, "all_byte_alias_ids": group["asset_ids"]})
    return result


def preview_rgb(array, limits=None):
    if array.dtype == np.uint8:
        view = array
    else:
        low, high = limits or np.percentile(array, [1, 99])
        view = np.clip((array.astype(np.float32) - low) * (255 / max(high - low, 1)), 0, 255).astype(np.uint8)
    if view.ndim == 2:
        view = np.repeat(view[..., None], 3, axis=2)
    return view[..., :3]


def make_review(image_asset, pixels, points, output):
    source_id = image_asset["asset_id"]
    stride = max(1, math.ceil(max(pixels.shape[:2]) / 1600))
    overview = Image.fromarray(preview_rgb(pixels[::stride, ::stride]))
    draw = ImageDraw.Draw(overview)
    for point in points:
        if not point["coordinate_valid"]:
            continue
        x, y = point["x"] / stride, point["y"] / stride
        color = COLORS.get(point["base_label"], "#f59e0b")
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color)
    name = "review/" + source_id + "_overview.png"
    overview.save(output / name)
    sheets = []
    for fine in sorted({p["fine_label"] for p in points}):
        candidates = [p for p in points if p["fine_label"] == fine and p["coordinate_valid"]]
        if not candidates:
            continue
        indices = np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int)
        sheet = Image.new("RGB", (560, math.ceil(len(indices) / 2) * 164), "#0f172a")
        painter = ImageDraw.Draw(sheet)
        for cell, index in enumerate(indices):
            point = candidates[index]
            x, y = math.floor(point["x"] + 0.5), math.floor(point["y"] + 0.5)
            left, top = max(0, x - 128), max(0, y - 128)
            right, bottom = min(pixels.shape[1], x + 128), min(pixels.shape[0], y + 128)
            raw = Image.fromarray(preview_rgb(pixels[top:bottom, left:right])).resize((128, 128))
            marked = raw.copy()
            mark = ImageDraw.Draw(marked)
            cx = (point["x"] - left) / (right - left) * 128
            cy = (point["y"] - top) / (bottom - top) * 128
            mark.line((cx - 5, cy, cx + 5, cy), fill="red")
            mark.line((cx, cy - 5, cx, cy + 5), fill="red")
            sx, sy = (cell % 2) * 280, (cell // 2) * 164
            sheet.paste(raw, (sx, sy))
            sheet.paste(marked, (sx + 132, sy))
            painter.text((sx, sy + 130), f"{fine} x={point['x']:.2f} y={point['y']:.2f}", fill="white")
            painter.text((sx, sy + 145), "raw / coordinate marker (not verified)", fill="white")
        sheet_name = f"review/{source_id}_{fine}.png"
        sheet.save(output / sheet_name)
        sheets.append({"fine_label": fine, "path": sheet_name, "sample_count": len(indices)})
    return {"image_asset_id": source_id, "area_id": image_asset["area_id"],
            "phase": image_asset["phase_hint"], "locator": image_asset["locator"],
            "overview": name, "sheets": sheets, "point_count": len(points),
            "orientation_applied": False, "review_status": "pending"}


def save_dashboard(output, summary, reviews, points):
    e = html.escape
    sections = []
    for review in reviews:
        cards = "".join(f"<details><summary>{e(sheet['fine_label'])} · {sheet['sample_count']}개 예시</summary>"
                        f"<img loading='lazy' src='{e(sheet['path'])}'></details>" for sheet in review["sheets"])
        sections.append(f"<section><h2>Area {e(review['area_id'])} · {e(review['phase'])}</h2>"
                        f"<p>{e(review['locator'])}<br>주석점 {review['point_count']}개 · 원시 좌표, 방향 변환 미적용</p>"
                        f"<img loading='lazy' class='overview' src='{e(review['overview'])}'>{cards}</section>")
    counts = "".join(f"<tr><td>{e(name)}</td><td>{count}</td></tr>" for name, count in summary["fine_counts"].items())
    point_index = [{k: p[k] for k in ("point_id", "image_asset_id", "area_id", "phase", "x", "y", "fine_label")}
                   for p in points]
    write_json(output / "review_point_index.json", point_index)
    body = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>전체 웨이퍼 주석 복원</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1e293b}}
.note{{background:#fff7ed;padding:20px;border:1px solid #fdba74}}section{{border-top:1px solid #cbd5e1;margin-top:32px}}
img{{max-width:100%}}.overview{{width:650px}}details{{margin:12px 0}}table{{border-collapse:collapse}}td,th{{border:1px solid #cbd5e1;padding:8px}}</style>
<h1>전체 웨이퍼 주석 복원 · 검수 자료</h1><p>큰 종류와 세부 문자, 원본 좌표와 출처를 복원했습니다.</p>
<div class='note'>현재 자료는 제공된 ROI를 복원한 검수용 데이터입니다. 영상 연결은 폴더·파일명·좌표 범위에 근거한 후보입니다.
사람이 확인한 정답으로 표시하지 않았습니다. 검수자 표시 이름: {e(summary['requested_reviewer_name'])} · 실제 생성 주체: Codex AI.
각 확대 화면은 원본과 좌표 표시 화면의 한 쌍이며, 일부 예시만 표시합니다. 모든 결함의 검수 완료를 뜻하지 않습니다.</div>
<p>복원한 영상별 고유 주석점 {summary['point_count']}개 · 연결 영상 {summary['mapped_images']}개 ·
ROI 없는 Area: {e(', '.join(summary['areas_without_roi']) or '없음')}</p>
<table><tr><th>제공된 세부 이름</th><th>영상별 점 수</th></tr>{counts}</table>
<p><a href='provider_points.csv'>전체 점 목록</a> · <a href='roi_mappings.json'>ROI와 영상 연결 근거</a> ·
<a href='annotation_exclusions.json'>연결 보류·빈 주석</a> · <a href='requirements_status.json'>요구사항별 진행 상태</a> ·
<a href='workspace.json'>전체 요약</a></p>{''.join(sections)}
<h2>검수 결과 기록</h2><p>review_template.csv에서 point_id와 검수 결과를 기록할 수 있습니다.
추가 결함·선 경로·전후 대응·3D 경로는 별도 작업 파일로 관리합니다. 확인되지 않은 영역을 정상으로 지정하지 않습니다.</p></html>"""
    (output / "전체웨이퍼_주석검수.html").write_text(body, encoding="utf-8")


def build_workspace(registry, output, reviewer_name="양희승", with_reviews=True):
    registry, output = Path(registry).resolve(), Path(output).resolve()
    validation = validate_registry(registry)
    if not validation["valid"] or validation["registry_status"] == "incomplete":
        raise ValueError("source_registry_not_ready")
    original_contract = json.loads((registry / "registry.json").read_text(encoding="utf-8"))
    original_root = Path(original_contract["source_root"]).resolve(strict=True)
    if output.exists():
        raise FileExistsError("workspace_output_already_exists")
    if output.is_relative_to(original_root) or original_root.is_relative_to(output) or output.is_relative_to(registry):
        raise ValueError("workspace_must_be_outside_original_and_registry")
    assets = [json.loads(line) for line in (registry / "assets.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {a["asset_id"]: a for a in assets}
    groups = json.loads((registry / "content_groups.json").read_text(encoding="utf-8"))
    images = []
    for group in groups:
        if group["kind"] == "image":
            primary = [by_id[i] for i in group["asset_ids"]
                       if by_id[i]["locator"].split("::")[0].lower().endswith("area.zip")]
            image = min(primary, key=lambda a: a["locator"]) if primary else by_id[group["preferred_reference_asset_id"]]
            if image["area_id"] or image.get("metadata", {}).get("page_count", 0) > 1:
                images.append(image | {"alias_asset_ids": group["asset_ids"]})
    reader = SourceReader(original_root, assets)
    points, mappings, exclusions, conflicts = restore_points(
        semantic_roi_groups(groups, by_id), images, reader, by_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / "review").mkdir()
    write_jsonl(output / "images.jsonl", images)
    write_jsonl(output / "provider_points.jsonl", points)
    write_json(output / "roi_mappings.json", mappings)
    write_json(output / "annotation_exclusions.json", exclusions)
    write_json(output / "label_conflicts.json", conflicts)
    grouped = defaultdict(list)
    for point in points:
        grouped[point["image_asset_id"]].append(point)
    reviews = []
    if with_reviews:
        for image_id, source_points in sorted(grouped.items()):
            print("Review: " + by_id[image_id]["locator"], flush=True)
            pixels = reader.image(by_id[image_id])
            reviews.append(make_review(by_id[image_id], pixels, source_points, output))
            del pixels
    reader.verify_unchanged()
    summary = {
        "point_count": len(points), "mapped_images": len(grouped),
        "fine_counts": dict(sorted(Counter(p["fine_label"] for p in points).items())),
        "area_fine_counts": dict(sorted(Counter(p["area_id"] + "/" + p["fine_label"] for p in points).items())),
        "coordinate_invalid_points": sum(not p["coordinate_valid"] for p in points),
        "areas_without_roi": original_contract["summary"]["areas_without_supplied_roi"],
        "roi_content_groups": sum(g["kind"] == "roi" for g in groups),
        "mapped_roi_groups": len(mappings), "excluded_roi_groups": len(exclusions),
        "label_conflicts": len(conflicts), "requested_reviewer_name": reviewer_name,
        "generated_by": "Codex_AI", "human_verified_points": 0,
        "source_originals_verified_for_restore": len(reader.checked),
    }
    contract = {"schema": "annotation_workspace", "schema_version": 2,
                "created_at": datetime.now(UTC).isoformat(), "registry_root": str(registry),
                "registry_manifest_sha256": digest_file(registry / "output_hashes.json"),
                "source_root": str(original_root), "summary": summary,
                "training_ready": False, "split_status": "unassigned",
                "subtype_definition_status": "provider_definition_pending",
                "source_originals_changed": False}
    write_json(output / "workspace.json", contract)
    requirements = {
        "2d_basic_and_subtype_labels": "provider_points_restored_pending_visual_and_semantic_review",
        "complete_detection_and_counts": "complete_annotation_regions_pending",
        "normal_dust_scratch_labels": "pending",
        "bpd_polyline_angle_length_curvature": "path_and_zero_direction_and_scale_pending",
        "before_after_registration_and_identity": "candidate_work_pending",
        "bpd_to_ted_conversion": "verified_identity_and_transition_labels_pending",
        "density_and_scalebar": "physical_xy_scale_and_valid_area_pending",
        "3d_paths_and_conversion_depth": "page_paths_and_depth_calibration_pending",
        "2d_3d_correlation": "specimen_mapping_pending",
        "wafer_holdout_evaluation": "all_wafer_verified_labels_pending",
    }
    write_json(output / "requirements_status.json", requirements)
    fields = ["point_id", "image_asset_id", "area_id", "phase", "x", "y", "base_label", "subtype",
              "fine_label", "coordinate_valid", "review_status", "human_verified", "split"]
    with (output / "provider_points.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    with (output / "review_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["point_id", "original_fine_label", "reviewer_display_name",
                                                    "actual_actor", "decision", "corrected_fine_label", "notes"])
        writer.writeheader()
        writer.writerows({"point_id": p["point_id"], "original_fine_label": p["fine_label"],
                          "reviewer_display_name": reviewer_name, "actual_actor": "",
                          "decision": "pending"} for p in points)
    write_json(output / "reviews.json", reviews)
    save_dashboard(output, summary, reviews, points)
    outputs = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(output / "output_hashes.json", {p.relative_to(output).as_posix(): digest_file(p) for p in outputs})
    return contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-name", default="양희승")
    parser.add_argument("--no-review-images", action="store_true")
    args = parser.parse_args(argv)
    result = build_workspace(args.registry, args.output, args.reviewer_name, not args.no_review_images)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
