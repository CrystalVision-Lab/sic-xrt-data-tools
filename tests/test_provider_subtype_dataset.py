"""Synthetic-only tests for the provider subtype dataset (no real XRT data)."""

import io
import json
import math
import shutil

import numpy as np
import pytest
import tifffile
from PIL import Image
from roifile import ROI_TYPE, ImagejRoi

from sic_xrt_data_tools.annotations.provider_roi import (
    decode_points,
    discover,
    image_hint,
    parse_label,
)
from sic_xrt_data_tools.provider_subtype_dataset import (
    build,
    decide_transform,
    read_csv,
    validate,
)
from sic_xrt_data_tools.tiff.orientation import (
    apply_transform,
    contrast_ratio,
    exact_row_agreement,
    transform_points,
)

W, H = 640, 512


@pytest.mark.parametrize(("name", "expected"), [
    ("After an_1_TED(a)", ("TED", "a", None)),
    ("Before an_area 2_1-1_BPD ", ("BPD", None, None)),
    ("Before an_area 2_1_1 TSD(b)", ("TSD", "b", None)),
    ("BPD", ("BPD", None, None)),
    ("4_After annealing_BPD", ("BPD", None, None)),
    ("Before an_unknown", (None, None, "unknown_label")),
    ("After an_1_TED", ("TED", None, "missing_subtype")),
    ("After an_1_TSD(e)", ("TSD", "e", "invalid_subtype")),
    ("After an_1_BPD(a)", ("BPD", "a", "unexpected_subtype")),
])
def test_parse_label(name, expected):
    assert parse_label(name + ".roi") == expected


@pytest.mark.parametrize(("stem", "area", "expected"), [
    ("Before an_8_1_TSD(b)", "8", "1"),
    ("Before an_9_3_TED(a)", "9", "3"),
    ("Before an_area 2_1-1_TED(c)", "2", "1_1"),
    ("After an_1_BPD", "1", "1"),
    ("BPD", "1", None),
])
def test_image_hint(stem, area, expected):
    assert image_hint(stem, area) == expected


def test_decode_points_rejects_empty_and_garbage():
    with pytest.raises(ValueError, match="empty_file"):
        decode_points(b"")
    with pytest.raises(ValueError, match="not_an_imagej_roi"):
        decode_points(b"x" * 80)
    points, subpixel = decode_points(point_roi([[10.5, 20.25]]).tobytes())
    assert subpixel and points.tolist() == [[10.5, 20.25]]


def test_transforms_are_involutions_and_match_arrays():
    xy = np.array([[0.5, 0.5], [10.25, 3.75]])
    for name in ("identity", "hflip", "vflip", "rot180"):
        assert np.allclose(transform_points(transform_points(xy, name, W, H), name, W, H), xy)
    image = np.zeros((H, W), np.uint8)
    image[3, 10] = 255
    moved = apply_transform(image, "hflip")
    x, y = transform_points(np.array([[10.5, 3.5]]), "hflip", W, H)[0]
    assert moved[math.floor(y), math.floor(x)] == 255


def test_exact_row_agreement_counts_damaged_rows():
    a = np.random.default_rng(0).integers(0, 255, (100, 20, 3), dtype=np.uint8)
    b = a.copy()
    b[10:30] = 0
    assert exact_row_agreement(a, a) == 1.0
    assert exact_row_agreement(a, b) == pytest.approx(0.8)


def test_decide_transform_rules():
    strong, flat = {"ratio": 3.0}, {"ratio": 1.0}
    assert decide_transform({"identity": strong}, "identity", 50)[0] == "identity"
    assert decide_transform({"identity": strong, "hflip": flat}, "hflip", 50)[0] == "identity"
    assert decide_transform({"identity": flat, "hflip": strong}, "hflip", 50)[0] == "hflip"
    assert decide_transform({"identity": flat, "hflip": strong}, "hflip", 3) == (None, "contrast_probe_inconclusive")


# ---------------------------------------------------------------- synthetic tree

def grid_points(n=20, x0=90, y0=90, step=52):
    cols = 9
    return np.array([[x0 + step * (i % cols) + 0.5, y0 + step * (i // cols) + 0.5] for i in range(n)])


def render(spots, seed):
    """TIFF-frame image: noisy background with bright (TSD) / dark (TED) spots."""
    rng = np.random.default_rng(seed)
    image = rng.normal(120, 5, (H, W, 1)).repeat(3, axis=2) + np.array([8, 4, -10])
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    for (x, y), amplitude in spots:
        image += (amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.5 ** 2)))[..., None]
    return np.clip(image, 0, 255).astype(np.uint8)


def write_pair(folder, stem, tiff_frame, damage_rows=None):
    """Mimic the provider pair: JPEG raw pixels are the mirror of the TIFF."""
    folder.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(tiff_frame[:, ::-1])).save(buffer, "JPEG", quality=95)
    (folder / f"{stem}.jpg").write_bytes(buffer.getvalue())
    decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    tiff = np.ascontiguousarray(decoded[:, ::-1])
    if damage_rows is not None:
        tiff = tiff.copy()
        tiff[damage_rows] = tiff[damage_rows][:, ::-1, ::-1]
    tifffile.imwrite(folder / f"{stem}.tif", tiff, photometric="rgb", metadata=None)
    return tiff, decoded


def point_roi(points, name=None):
    roi = ImagejRoi.frompoints(np.asarray(points, np.float32))
    roi.roitype = ROI_TYPE.POINT
    if name:
        roi.name = name
    return roi


def write_roi(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    point_roi(points).tofile(path)


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "2D XRT"
    tsd = grid_points(20)
    ted = grid_points(20, x0=116, y0=116)
    # Area 1: everything on the TIFF grid, plus duplicates / broken / unknown files.
    a1 = root / "1 Area" / "before annealing_1"
    write_pair(a1, "before annealing_1", render([(p, 90) for p in tsd] + [(p, -70) for p in ted], 1))
    write_roi(a1 / "Before an_1_TSD(a).roi", tsd)
    write_roi(a1 / "Before an_1_TED(b).roi", ted)
    (a1 / "RoiSet").mkdir()
    shutil.copy(a1 / "Before an_1_TSD(a).roi", a1 / "RoiSet" / "Before an_1_TSD(a).roi")
    (a1 / "Before an_1_TED(e).roi").write_bytes(b"")
    write_roi(a1 / "Before an_unknown.roi", tsd[:3])
    # a TED(c) point 5 px from a TSD point: ambiguous centre; and one near the edge
    write_roi(a1 / "Before an_1_TED(c).roi", [tsd[0] + [5, 0], [20.5, 250.5]])
    x100 = root / "1 Area" / "Before annealing_1_x100"
    x100.mkdir(parents=True)
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(x100 / "4.jpg")
    write_roi(x100 / "4_Before annealing_BPD.roi", [[10.5, 10.5]])
    # Area 2: TSD drawn on the raw JPEG grid (mirrored), TED on the TIFF grid.
    a2 = root / "2 Area" / "before annealing_2"
    rare = np.array([[150.5, 420.5], [400.5, 420.5]])
    spots = [(p, 90) for p in [*tsd, *rare]] + [(p, -70) for p in ted]
    write_pair(a2, "before annealing_1_1", render(spots, 2))
    write_roi(a2 / "Before an_area 2_1_1_TSD(a).roi", transform_points(tsd, "hflip", W, H))
    write_roi(a2 / "Before an_area 2_1_1_TSD(c).roi", transform_points(rare, "hflip", W, H))
    write_roi(a2 / "Before an_area 2_1-1_TED(c).roi", ted)
    # Area 3: damaged TIFF copy; pixels must be rebuilt from the JPEG.
    a3 = root / "3 Area" / "After annealing_3"
    write_pair(a3, "After annealing_3", render([(p, -70) for p in ted], 3), damage_rows=slice(200, 400))
    write_roi(a3 / "After an_3_TED(d).roi", ted)
    return root


def build_tree(tree, tmp_path, **kwargs):
    return build(tree, tmp_path / "out", {"1": "train", "2": "val", "3": "test"}, **kwargs)


def test_build_end_to_end(tree, tmp_path):
    manifest = build_tree(tree, tmp_path)
    out = tmp_path / "out"
    assert manifest["schema"] == "provider_subtype_point_dataset" and manifest["schema_version"] == 1
    assert manifest["labels_source"] == "provider" and manifest["expert_verified"] is False
    reasons = {e["relpath"].split("/")[-1]: e["reason"] for e in manifest["excluded_roi_files"]}
    assert reasons["Before an_1_TED(e).roi"] == "empty_file"
    assert reasons["Before an_unknown.roi"] == "unknown_label"
    assert reasons["4_Before annealing_BPD.roi"] == "x100_magnified_image_different_scale"
    assert any(e["reason"] == "duplicate_bytes" for e in manifest["excluded_roi_files"])

    rois = {r["relpath"].split("/")[-1]: r for r in read_csv(out / "roi_files.csv")}
    assert rois["Before an_area 2_1_1_TSD(a).roi"]["coordinate_transform"] == "hflip"
    assert rois["Before an_area 2_1_1_TSD(c).roi"]["transform_basis"] == "inherited_from_same_image_same_class_files"
    assert rois["Before an_area 2_1-1_TED(c).roi"]["coordinate_transform"] == "identity"

    images = {i["relpath"].split("/")[-1]: i for i in json.loads((out / "images.json").read_text("utf-8"))}
    assert images["before annealing_1.tif"]["pixel_check"] == "tiff_bit_identical_to_jpeg_hflip"
    assert images["After annealing_3.tif"]["pixel_source"] == "jpeg_hflip"

    points = read_csv(out / "points.csv")
    excluded = {p["reason"] for p in points if p["status"] == "excluded"}
    assert {"other_label_within_center_32px", "patch_crosses_image_boundary"} <= excluded

    samples = read_csv(out / "samples.csv")
    counts = manifest["counts"]["by_split_class_subtype"]
    assert counts["val"]["TSD"] == {"a": 20, "c": 2}
    assert counts["test"]["TED"] == {"d": 20}
    assert {s["split"] for s in samples if s["area"] == "1"} == {"train"}
    for row in samples:
        patch = tifffile.imread(out / row["path"])
        assert patch.shape == (128, 128, 3) and patch.dtype == np.uint8
        centre = patch[60:68, 60:68].astype(float).mean() - patch.astype(float).mean()
        assert (centre > 20) if row["label"] == "TSD" else (centre < -15)
    # pixels equal the verified TIFF crop for intact images
    row = next(s for s in samples if s["image"].endswith("before annealing_1.tif"))
    source = tifffile.imread(tree / row["image"])
    left, top = int(row["left"]), int(row["top"])
    assert np.array_equal(tifffile.imread(out / row["path"]), source[top:top + 128, left:left + 128])
    assert left == math.floor(float(row["x"])) - 64
    assert validate(out)["status"] == "passed"
    assert (out / "preview" / "contact_sheet.png").exists()


def test_refuses_unsafe_outputs(tree, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        build_tree(tree, tmp_path)
    with pytest.raises(ValueError, match="outside the source"):
        build(tree, tree / "inside", {})
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="git work tree"):
        build(tree, repo / "dataset", {})


def test_validate_detects_tampering(tree, tmp_path):
    build_tree(tree, tmp_path)
    out = tmp_path / "out"
    patch = next((out / "patches_128").rglob("*.tif"))
    tifffile.imwrite(patch, np.zeros((128, 128, 3), np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(out)


def test_discover_marks_coordinate_duplicates(tmp_path):
    folder = tmp_path / "9 Area" / "before annealing_9"
    folder.mkdir(parents=True)
    tifffile.imwrite(folder / "before annealing_1.tif", np.zeros((300, 300, 3), np.uint8), photometric="rgb")
    points = grid_points(3)
    write_roi(folder / "Before an_9_3_TED(a).roi", points)
    (folder / "RoiSet").mkdir()
    point_roi(points, name="renamed").tofile(folder / "RoiSet" / "Before an_9_1_TED(a).roi")
    found = {r.relpath.split("/")[-1]: r for r in discover(tmp_path)}
    kept, dup = found["Before an_9_1_TED(a).roi"], found["Before an_9_3_TED(a).roi"]
    assert kept.status == "mapped" and dup.reason == "duplicate_coordinates"
    assert dup.hint_matches_image is False
    assert contrast_ratio(np.zeros((300, 300, 3), np.uint8), points)["points_used"] == 3
