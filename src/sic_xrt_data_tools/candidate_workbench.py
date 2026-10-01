"""Classical contrast/geometry candidates for annotation, never verified labels."""

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

from .annotation_workspace import SourceReader, make_review, write_json, write_jsonl
from .source_registry import digest_file, validate_registry


def validate_outputs(root):
    root = Path(root)
    expected = json.loads((root / "output_hashes.json").read_text(encoding="utf-8"))
    return all((root / name).is_file() and digest_file(root / name) == sha for name, sha in expected.items())


def gray_view(pixels, max_side):
    stride = max(1, math.ceil(max(pixels.shape[:2]) / max_side))
    sampled = pixels[::stride, ::stride]
    gray = sampled.mean(axis=2, dtype=np.float32) if sampled.ndim == 3 else sampled.astype(np.float32)
    if not np.isfinite(gray).all():
        raise ValueError("nonfinite_image_values")
    return gray, stride


def contrast_candidates(pixels, max_candidates=64, max_side=4096):
    """Bright/dark DoG extrema with spatial sampling; score is not a probability."""
    if max_candidates < 1 or max_side < 32:
        raise ValueError("invalid_candidate_limits")
    gray, stride = gray_view(pixels, max_side)
    response = ndimage.gaussian_filter(gray, 1.0) - ndimage.gaussian_filter(gray, 6.0)
    noise = float(np.median(np.abs(response - np.median(response))) * 1.4826)
    selected = []
    for sign, polarity in ((1, "bright"), (-1, "dark")):
        scored = sign * response
        threshold = max(noise * 4, float(np.percentile(scored, 99.5)), 1e-5)
        maxima = ndimage.maximum_filter(scored, size=7)
        valid = (scored == maxima) & (scored > threshold)
        valid[:3] = valid[-3:] = False
        valid[:, :3] = valid[:, -3:] = False
        ys, xs = np.nonzero(valid)
        order = np.argsort(scored[ys, xs])[::-1]
        cells = Counter()
        accepted = []
        for index in order:
            x, y = int(xs[index]), int(ys[index])
            cell = (min(3, x * 4 // gray.shape[1]), min(3, y * 4 // gray.shape[0]))
            if cells[cell] >= max(1, math.ceil(max_candidates / 32)):
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < 49 for px, py in accepted):
                continue
            accepted.append((x, y))
            cells[cell] += 1
            selected.append({"x": float(x * stride), "y": float(y * stride),
                             "polarity": polarity, "contrast_score": float(scored[y, x]),
                             "detector_stride": stride, "noise_mad": noise})
            if len(accepted) >= math.ceil(max_candidates / 2):
                break
    return sorted(selected, key=lambda p: -p["contrast_score"])[:max_candidates]


def line_geometry_candidate(pixels, point, radius=96):
    """Fit only a local straight segment to a contrast component near a supplied BPD point."""
    x, y = round(point["x"]), round(point["y"])
    left, top = max(0, x - radius), max(0, y - radius)
    patch = pixels[top:min(pixels.shape[0], y + radius + 1), left:min(pixels.shape[1], x + radius + 1)]
    result = {"point_id": point["point_id"], "image_asset_id": point["image_asset_id"],
              "status": "unresolved", "path_xy": None, "angle_raw_x_deg": None,
              "local_segment_length_px": None, "length_um": None, "curvature": None,
              "step_flow_zero_direction": None, "whole_line_length_confirmed": False,
              "human_verified": False, "review_status": "candidate_pending_review"}
    if patch.size == 0:
        return result | {"reason": "point_outside_image"}
    if patch.ndim == 3:
        chroma = np.ptp(patch.astype(np.float32), axis=2)
        if float(np.mean(chroma > 100)) > 0.25:
            return result | {"reason": "strong_color_contrast_requires_review"}
    gray = patch.mean(axis=2, dtype=np.float32) if patch.ndim == 3 else patch.astype(np.float32)
    response = ndimage.gaussian_filter(gray, 1) - ndimage.gaussian_filter(gray, 10)
    magnitude = np.abs(response)
    threshold = max(float(np.percentile(magnitude, 92)), 1e-5)
    components, _ = ndimage.label(magnitude > threshold)
    cy, cx = y - top, x - left
    nearby = components[max(0, cy - 6):cy + 7, max(0, cx - 6):cx + 7]
    ids = [int(i) for i in np.unique(nearby) if i]
    if not ids:
        return result | {"reason": "no_contrast_component_near_point"}
    candidates = []
    for component in ids:
        yy, xx = np.nonzero(components == component)
        if len(xx) < 8:
            continue
        coords = np.column_stack((xx, yy)).astype(np.float64)
        center = coords.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(coords.T))
        ratio = float(values[-1] / max(values[0], 1e-9))
        if ratio < 4:
            continue
        axis = vectors[:, -1]
        projection = (coords - center) @ axis
        ends = np.array([center + projection.min() * axis, center + projection.max() * axis])
        ends += [left, top]
        distance = float(np.min((xx - cx) ** 2 + (yy - cy) ** 2))
        candidates.append((distance, {
            "path_xy": ends.tolist(), "angle_raw_x_deg": float(np.degrees(np.arctan2(axis[1], axis[0])) % 180),
            "local_segment_length_px": float(projection.max() - projection.min()),
            "principal_axis_ratio": ratio, "component_pixels": len(xx),
            "context_clipped": bool(xx.min() == 0 or yy.min() == 0 or xx.max() == patch.shape[1] - 1
                                    or yy.max() == patch.shape[0] - 1),
        }))
    if not candidates:
        return result | {"reason": "not_a_clear_local_straight_segment"}
    result.update(min(candidates, key=lambda item: item[0])[1])
    return result | {"status": "local_straight_segment_candidate", "reason": "contrast_component_pca_fit"}


def neighbor_edges(previous, current, max_distance=24):
    """Mutual nearest neighbors between consecutive frames, no identity guarantee."""
    if not previous or not current:
        return []
    a = np.array([[p["x"], p["y"]] for p in previous])
    b = np.array([[p["x"], p["y"]] for p in current])
    distances = np.linalg.norm(a[:, None] - b[None, :], axis=2)
    for i, p in enumerate(previous):
        for j, q in enumerate(current):
            if p["polarity"] != q["polarity"]:
                distances[i, j] = np.inf
    edges = []
    for i, j in enumerate(np.argmin(distances, axis=1)):
        if int(np.argmin(distances[:, j])) == i and distances[i, j] <= max_distance:
            edges.append({"from_id": previous[i]["candidate_id"], "to_id": current[j]["candidate_id"],
                          "distance_px": float(distances[i, j]), "status": "geometric_neighbor_candidate",
                          "human_verified": False})
    return edges


def phase_shift_candidate(reference, moving):
    """Integer thumbnail translation only; report direct image correlation separately."""
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError("alignment_images_must_have_same_2d_shape")
    a = reference.astype(np.float32) - float(reference.mean())
    b = moving.astype(np.float32) - float(moving.mean())
    if min(float(a.std()), float(b.std())) < 1e-6:
        return {"status": "unresolved", "reason": "no_image_contrast", "shift_yx": None}
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.array(np.unravel_index(np.argmax(correlation), correlation.shape))
    shift = peak.astype(float)
    for axis, size in enumerate(correlation.shape):
        if shift[axis] > size // 2:
            shift[axis] -= size
    warped = ndimage.shift(b, shift, order=1, mode="constant", cval=np.nan)
    valid = np.isfinite(warped)
    if valid.sum() < 16 or min(float(a[valid].std()), float(warped[valid].std())) < 1e-6:
        quality = None
    else:
        quality = float(np.corrcoef(a[valid], warped[valid])[0, 1])
    return {"status": "thumbnail_translation_candidate", "shift_yx": shift.tolist(),
            "normalized_correlation": quality,
            "phase_peak_to_mean": float(correlation.max() / max(float(correlation.mean()), 1e-12)),
            "physical_field_of_view_confirmed": False, "human_verified": False}


def build_tracks(nodes, edges):
    by_id = {p["candidate_id"]: p for p in nodes}
    following = {e["from_id"]: e["to_id"] for e in edges}
    incoming = {e["to_id"] for e in edges}
    tracks = []
    for start in sorted(set(following) - incoming):
        ids = [start]
        while ids[-1] in following and following[ids[-1]] not in ids:
            ids.append(following[ids[-1]])
        if len(ids) < 3:
            continue
        points = [by_id[i] for i in ids]
        tracks.append({"track_id": "track_" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:24],
                       "stack_asset_id": points[0]["image_asset_id"], "candidate_ids": ids,
                       "frame_xy": [[p["frame_index"], p["x"], p["y"]] for p in points],
                       "status": "geometric_path_candidate", "human_verified": False,
                       "base_label": None, "length_um": None, "conversion_depth_um": None,
                       "depth_calibration_status": "pending"})
    return tracks


def tag_candidates(items, image, frame=None):
    result = []
    for item in items:
        key = [image["asset_id"], frame, item["x"], item["y"], item["polarity"]]
        result.append(item | {
            "candidate_id": "cand_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24],
            "image_asset_id": image["asset_id"], "area_id": image["area_id"],
            "phase": image["phase_hint"], "frame_index": frame,
            "base_label": None, "subtype": None, "probability": None,
            "annotation_origin": "classical_contrast_proposal", "review_status": "unreviewed",
            "human_verified": False, "split": "unassigned", "eligible_for_verified_evaluation": False,
        })
    return result


def review_candidates(items):
    return [p | {"coordinate_valid": True, "fine_label": "contrast_" + p["polarity"],
                 "base_label": "unknown"} for p in items]


def overview_pair_candidates(images):
    groups = defaultdict(lambda: defaultdict(list))
    for image in images:
        if image["area_id"] and image["nominal_magnification_hint"] == "100" \
                and image["phase_hint"] in {"before", "after"} and image["decode_status"] == "decoded":
            stem = PurePosixPath(image["locator"].split("::")[-1]).stem
            if not stem.isnumeric():
                groups[image["area_id"]][image["phase_hint"]].append(image)
    return [(area, phases["before"][0], phases["after"][0]) for area, phases in sorted(groups.items(), key=lambda i: int(i[0]))
            if len(phases["before"]) == len(phases["after"]) == 1]


def thumbnail_gray(pixels, side=512):
    gray, _ = gray_view(pixels, side * 2)
    low, high = np.percentile(gray, [1, 99])
    normalized = np.clip((gray - low) * (255 / max(high - low, 1)), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(normalized).resize((side, side))).astype(np.float32)


def build_workbench(workspace, output, max_2d=64, max_3d=24):
    if min(max_2d, max_3d) < 1:
        raise ValueError("invalid_candidate_limits")
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if not validate_outputs(workspace):
        raise ValueError("annotation_workspace_hash_invalid")
    contract = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
    if contract["schema"] != "annotation_workspace" or contract["schema_version"] != 2:
        raise ValueError("unsupported_workspace_schema")
    source = Path(contract["source_root"]).resolve(strict=True)
    if output.exists():
        raise FileExistsError("candidate_output_already_exists")
    if output.is_relative_to(source) or source.is_relative_to(output) or output.is_relative_to(workspace):
        raise ValueError("candidate_output_must_be_independent")
    registry = Path(contract["registry_root"])
    if digest_file(registry / "output_hashes.json") != contract["registry_manifest_sha256"] \
            or not validate_outputs(registry) or not validate_registry(registry)["valid"]:
        raise ValueError("source_registry_changed_since_restore")
    all_assets = [json.loads(line) for line in (registry / "assets.jsonl").read_text(encoding="utf-8").splitlines()]
    images = [json.loads(line) for line in (workspace / "images.jsonl").read_text(encoding="utf-8").splitlines()]
    provider_points = [json.loads(line) for line in (workspace / "provider_points.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped_points = defaultdict(list)
    for point in provider_points:
        grouped_points[point["image_asset_id"]].append(point)
    reader = SourceReader(source, all_assets)
    output.mkdir(parents=True, exist_ok=False)
    (output / "review").mkdir()
    nodes_2d, nodes_3d, edges, line_candidates, surveys, reviews = [], [], [], [], [], []
    align_views = {}
    pairs = overview_pair_candidates(images)
    pair_ids = {a["asset_id"] for _, before, after in pairs for a in (before, after)}
    for image in sorted(images, key=lambda a: (int(a["area_id"] or 999), a["locator"])):
        image_id = image["asset_id"]
        if image["decode_status"] != "decoded":
            surveys.append({"image_asset_id": image_id, "status": "source_decode_failed",
                            "area_id": image["area_id"], "candidate_count": 0})
            continue
        if image.get("metadata", {}).get("page_count", 1) == 1:
            print("2D candidates: " + image["locator"], flush=True)
            pixels = reader.image(image)
            proposals = tag_candidates(contrast_candidates(pixels, max_candidates=max_2d), image)
            known = grouped_points[image_id]
            if known:
                xy = np.array([[p["x"], p["y"]] for p in known])
                proposals = [p for p in proposals if np.min(np.sum((xy - [p["x"], p["y"]]) ** 2, axis=1)) > 16**2]
            nodes_2d.extend(proposals)
            line_candidates.extend(line_geometry_candidate(pixels, p) for p in known if p["base_label"] == "BPD")
            if proposals:
                reviews.append(make_review(image, pixels, review_candidates(proposals), output))
            if image_id in pair_ids:
                align_views[image_id] = thumbnail_gray(pixels)
            surveys.append({"image_asset_id": image_id, "area_id": image["area_id"], "phase": image["phase_hint"],
                            "status": "candidate_scan_completed", "candidate_count": len(proposals),
                            "annotation_coverage": "provider_subset" if known else "unknown",
                            "human_complete_annotation": False, "defect_absence_confirmed": False})
            del pixels
        else:
            print("3D candidates: " + image["locator"], flush=True)
            preview_pages = set(np.linspace(0, image["metadata"]["page_count"] - 1, 4, dtype=int))
            previous = []
            with reader.open_asset(image, prefer_loose=True) as stream, tifffile.TiffFile(stream) as tif:
                for index, page in enumerate(tif.pages):
                    pixels = page.asarray()
                    proposals = tag_candidates(contrast_candidates(pixels, max_candidates=max_3d, max_side=1536), image, index)
                    edges.extend(neighbor_edges(previous, proposals))
                    nodes_3d.extend(proposals)
                    if index in preview_pages and proposals:
                        display_asset = image | {"asset_id": image_id + f"_frame{index:04d}",
                                                 "area_id": "3D", "phase_hint": f"frame {index}"}
                        reviews.append(make_review(display_asset, pixels, review_candidates(proposals), output))
                    surveys.append({"image_asset_id": image_id, "frame_index": index,
                                    "status": "candidate_scan_completed", "candidate_count": len(proposals),
                                    "human_complete_annotation": False, "depth_um": None})
                    previous = proposals
                    del pixels
    alignments, pre_post_edges = [], []
    by_image = defaultdict(list)
    for candidate in nodes_2d:
        by_image[candidate["image_asset_id"]].append(candidate)
    for area, before, after in pairs:
        a, b = align_views[before["asset_id"]], align_views[after["asset_id"]]
        shift = phase_shift_candidate(a, b)
        record = {"area_id": area, "before_asset_id": before["asset_id"], "after_asset_id": after["asset_id"],
                  "pair_evidence": "unique_x100_overviews_in_same_area", "transform_coordinate_system": "512x512_thumbnail",
                  "before_raw_size": [before["metadata"]["width"], before["metadata"]["height"]],
                  "after_raw_size": [after["metadata"]["width"], after["metadata"]["height"]],
                  "exif_transform_applied": False, **shift}
        alignments.append(record)
        if shift["status"] == "thumbnail_translation_candidate" and (shift["normalized_correlation"] or 0) >= 0.3:
            before_nodes, after_nodes = [], []
            dy, dx = shift["shift_yx"]
            for point in by_image[before["asset_id"]]:
                before_nodes.append(point | {"x": point["x"] / before["metadata"]["width"] * 512,
                                             "y": point["y"] / before["metadata"]["height"] * 512})
            for point in by_image[after["asset_id"]]:
                after_nodes.append(point | {"x": point["x"] / after["metadata"]["width"] * 512 + dx,
                                            "y": point["y"] / after["metadata"]["height"] * 512 + dy})
            matches = neighbor_edges(before_nodes, after_nodes, max_distance=5)
            pre_post_edges.extend(match | {"area_id": area, "distance_units": "thumbnail_pixels",
                                           "transition_label": "unknown", "conversion_confirmed": False} for match in matches)
    reader.verify_unchanged()
    tracks = build_tracks(nodes_3d, edges)
    write_jsonl(output / "candidates_2d.jsonl", nodes_2d)
    write_jsonl(output / "candidates_3d.jsonl", nodes_3d)
    write_jsonl(output / "bpd_local_line_candidates.jsonl", line_candidates)
    write_jsonl(output / "survey_coverage.jsonl", surveys)
    write_json(output / "stack_neighbor_candidates.json", edges)
    write_json(output / "stack_path_candidates.json", tracks)
    write_json(output / "before_after_alignment_candidates.json", alignments)
    write_json(output / "before_after_identity_candidates.json", pre_post_edges)
    write_json(output / "reviews.json", reviews)
    with (output / "candidate_review_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate_id", "image_asset_id", "area_id", "frame_index", "x", "y",
                                                  "proposed_polarity", "reviewed_base_label", "reviewed_subtype", "review_status",
                                                  "reviewer_name", "actual_actor", "reviewed_at", "notes"])
        writer.writeheader()
        for p in nodes_2d + nodes_3d:
            writer.writerow({"candidate_id": p["candidate_id"], "image_asset_id": p["image_asset_id"], "area_id": p["area_id"],
                             "frame_index": p["frame_index"], "x": p["x"], "y": p["y"], "proposed_polarity": p["polarity"],
                             "review_status": "unreviewed", "reviewer_name": contract["summary"]["requested_reviewer_name"]})
    write_json(output / "future_wafer_folds.json", {
        "status": "planning_only_not_applied", "unit": "user_confirmed_distinct_wafer_Area",
        "three_dimensional_stacks": "exclude_until_physical_Area_correspondence_is_confirmed",
        "development_policy": "tune_only_inside_training_wafers_with_grouped_inner_validation",
        "outer_held_out_policy": "evaluate_once_per_fixed_configuration; never tune_on_outer_scores",
        "folds": [{"held_out_area": str(i), "training_areas": [str(j) for j in range(1, 10) if j != i],
                   "ready": False, "blockers": ["human_complete_annotations", "fine_class_definition",
                                                "deduplicate_cross_acquisition_identity", "annotated_evaluation_coverage"]}
                  for i in range(1, 10)],
    })
    summary = {
        "schema": "candidate_workbench", "schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "annotation_workspace": str(workspace), "workspace_manifest_sha256": digest_file(workspace / "output_hashes.json"),
        "candidate_counts_2d": dict(sorted(Counter(p["area_id"] for p in nodes_2d).items(), key=lambda i: int(i[0]))),
        "candidates_2d": len(nodes_2d), "candidates_3d": len(nodes_3d), "survey_records": len(surveys),
        "bpd_local_segments_attempted": len(line_candidates),
        "bpd_local_segments_proposed": sum(p["status"] == "local_straight_segment_candidate" for p in line_candidates),
        "stack_neighbor_edges": len(edges), "stack_path_candidates": len(tracks),
        "before_after_pair_candidates": len(alignments), "before_after_identity_candidates": len(pre_post_edges),
        "original_files_verified": len(reader.checked), "human_verified_candidates": 0, "training_ready": False,
        "split_status": "unassigned", "source_originals_changed": False,
        "candidate_parameters": {"max_2d_per_image": max_2d, "max_3d_per_frame": max_3d,
                                 "dog_sigma": [1, 6], "noise_mad_multiplier": 4, "tail_percentile": 99.5,
                                 "neighbor_distance_raw_pixels": 24, "no_defect_probability_claim": True},
    }
    write_json(output / "candidate_summary.json", summary)
    save_candidate_dashboard(output, summary, reviews, alignments)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    write_json(output / "output_hashes.json", {p.relative_to(output).as_posix(): digest_file(p) for p in files})
    return summary


def save_candidate_dashboard(output, summary, reviews, alignments):
    e = html.escape
    counts = "".join(f"<tr><td>Area {area}</td><td>{count}</td></tr>" for area, count in summary["candidate_counts_2d"].items())
    cards = []
    for review in reviews:
        sheets = "".join(f"<img loading='lazy' src='{e(sheet['path'])}'>" for sheet in review["sheets"])
        cards.append(f"<details><summary>{e(review['area_id'])} · {e(review['phase'])} · {review['point_count']}개 후보 · {e(review['locator'])}</summary>"
                     f"<img loading='lazy' class='overview' src='{e(review['overview'])}'>{sheets}</details>")
    pairs = "".join(f"<tr><td>Area {p['area_id']}</td><td>{e(str(p.get('shift_yx')))}</td>"
                    f"<td>{e(str(p.get('normalized_correlation')))}</td><td>후보·검수 전</td></tr>" for p in alignments)
    body = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>전체 웨이퍼 검출·경로 후보</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1e293b}}img{{max-width:100%}}.overview{{width:650px}}
.note{{background:#fff7ed;padding:20px}}details{{border:1px solid #cbd5e1;padding:12px;margin:12px 0}}td,th{{border:1px solid #cbd5e1;padding:8px}}table{{border-collapse:collapse}}</style>
<h1>전체 웨이퍼 검출·경로 후보</h1><div class='note'>밝고 어두운 국소 대비를 찾아 검수 대상을 제안한 자료입니다.
TED/TSD/BPD 정답이나 결함 존재 확률을 새로 지정하지 않았습니다. 검출 누락·오탐·먼지·스크래치가 있을 수 있으며,
주석이 없는 영역을 정상으로 처리하지 않습니다. 샘플 수는 설정한 상한 안의 후보 수입니다. 전체 결함 개수가 아닙니다.</div>
<p>2D 후보 {summary['candidates_2d']}개 · 3D 후보 {summary['candidates_3d']}개 ·
3D 경로 후보 {summary['stack_path_candidates']}개 · BPD 국소 직선 후보 {summary['bpd_local_segments_proposed']}개</p>
<table><tr><th>웨이퍼</th><th>2D 검수 후보 수</th></tr>{counts}</table>
<h2>열처리 전·후 정렬 후보</h2><p>512×512 보기 영상의 이동 추정입니다. 원본 좌표 변환에는 각 영상의 치수 비율이 필요하고,
실제 같은 시야인지·EXIF 반전·정렬 오차를 검수해야 합니다. 수가 줄었다는 이유로 소멸이나 전환을 확정하지 않습니다.</p>
<table><tr><th>Area</th><th>이동 [y,x]</th><th>영상 상관</th><th>확인 상태</th></tr>{pairs}</table>
<h2>각도·길이·깊이의 해석</h2><p>BPD 선은 기존 주석점 부근의 짧은 직선 성분 후보입니다. 각도는 원본 +X 축 기준이며
step-flow 0° 기준은 미확정입니다. 길이는 국소 구간의 픽셀 길이이고 전체 BPD 길이·곡률·µm 길이가 아닙니다.
3D 경로는 연속 프레임의 가까운 대비점 연결 후보이며, 동일 결함·종류·전환 깊이는 아직 검증되지 않았습니다.</p>
<p><a href='candidate_summary.json'>요약</a> · <a href='candidates_2d.jsonl'>2D 전체 후보</a> ·
<a href='candidates_3d.jsonl'>3D 전체 후보</a> · <a href='bpd_local_line_candidates.jsonl'>BPD 국소 경로 후보</a> ·
<a href='before_after_alignment_candidates.json'>전후 정렬 후보</a> · <a href='stack_path_candidates.json'>3D 경로 후보</a></p>
<h2>원본 대비와 좌표 예시</h2>{''.join(cards)}</html>"""
    (output / "전체웨이퍼_후보검수.html").write_text(body, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-2d-per-image", default=64, type=int)
    parser.add_argument("--max-3d-per-frame", default=24, type=int)
    args = parser.parse_args(argv)
    result = build_workbench(args.workspace, args.output, args.max_2d_per_image, args.max_3d_per_frame)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
