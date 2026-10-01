import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import tifffile
from roifile import ROI_TYPE, ImagejRoi

from sic_xrt_data_tools.balanced_subset import create_subsets
from sic_xrt_data_tools.point_dataset import (
    build,
    choose_image,
    decode_roi,
    label_from_name,
    patch_bounds,
    unique_points,
)
from sic_xrt_data_tools.validate_point_dataset import validate


def roi_bytes(points):
    roi = ImagejRoi.frompoints(np.asarray(points, dtype=np.float32))
    roi.roitype = ROI_TYPE.POINT
    return roi.tobytes()


def test_subpixel_point_coordinates_preserved():
    np.testing.assert_allclose(decode_roi(roi_bytes([[60.25, 40.75]])), [[60.25, 40.75]])


def test_blank_and_non_point_rois_rejected():
    with pytest.raises(ValueError, match="invalid_roi"):
        decode_roi(b"")
    roi = ImagejRoi.frompoints(np.array([[1, 1], [3, 3], [5, 1]]))
    with pytest.raises(ValueError, match="non_point"):
        decode_roi(roi.tobytes())


def test_mapping_never_chooses_an_ambiguous_or_opposite_stage_image():
    images = ["a/after_1.tif", "a/before_1.tif"]
    assert choose_image("after_1_BPD.roi", images, "a")[0] == "a/after_1.tif"
    assert choose_image("BPD.roi", images, "a")[0] is None
    assert choose_image("after_1_BPD.roi", ["a/before_1.tif"], "a")[0] is None
    assert choose_image("BPD.roi", images, "b")[0] is None


def test_unknown_class_is_not_made_normal():
    assert label_from_name("unknown.roi") is None
    assert label_from_name("Before_1_TED(a).roi") == "TED"


def test_crop_boundary_and_rounding():
    assert patch_bounds(20.5, 30.5, 16, 100, 100) == (13, 23, 29, 39)
    assert patch_bounds(0, 0, 16, 100, 100) is None


def test_same_point_in_nested_set_is_not_an_additional_sample():
    base = {"archive": "1 Area.zip", "image": "x.tif", "label": "BPD", "roi": "a.roi",
            "points": np.array([[20.25, 30.5]])}
    points, duplicates = unique_points([base, {**base, "roi": "RoiSet.zip::a.roi"}])
    assert len(points) == 1
    assert len(duplicates) == 1
    assert len(points[0]["refs"]) == 2


def test_end_to_end_preserves_dtype_pixels_and_wafer_split(tmp_path):
    source = tmp_path / "source"
    folder = source / "2D XRT"
    folder.mkdir(parents=True)
    originals = {}
    array = np.arange(128*128, dtype=np.uint16).reshape(128, 128)
    tiff = io.BytesIO()
    tifffile.imwrite(tiff, array, metadata=None)
    for wafer in (1, 2, 8):
        path = folder / f"{wafer} Area.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("before/image.tif", tiff.getvalue())
            archive.writestr("before/BPD.roi", roi_bytes([[32, 32], [1, 1]]))
            archive.writestr("before/TED.roi", roi_bytes([[36, 32], [96, 96]]))
            archive.writestr("before/unknown.roi", roi_bytes([[64, 64]]))
        originals[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    output = tmp_path / "dataset"
    summary = build(source, output, {"1": "train", "2": "val", "8": "test"}, sizes=(16, 32))
    assert summary["normal_samples"] == 0
    assert summary["rejected_patches"]["mixed_class_16"] > 0
    assert summary["excluded_rois"] == 3
    assert summary["patch_count"] > 0
    with (output / "기록/samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    for sample in samples:
        patch = tifffile.imread(output / sample["path"])
        left, top, size = int(sample["left"]), int(sample["top"]), int(sample["size"])
        np.testing.assert_array_equal(patch, array[top:top+size, left:left+size])
        assert patch.dtype == np.uint16
        assert hashlib.sha256((output / sample["path"]).read_bytes()).hexdigest() == sample["sha256"]
    groups = {}
    for sample in samples:
        groups.setdefault(sample["wafer"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert json.loads((output / "summary.json").read_text())["review_status"] == "pending"
    assert validate(output)["status"] == "passed"
    with pytest.raises(ValueError, match="Every class"):
        create_subsets(output)
    for path, (data, mtime) in originals.items():
        assert path.read_bytes() == data
        assert path.stat().st_mtime_ns == mtime
    with pytest.raises(ValueError, match="already exists"):
        build(source, output, {}, sizes=(16,))
    (output / samples[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(output)


def test_rgb_crop_and_nested_annotation_deduplication(tmp_path):
    folder = tmp_path / "original" / "2D XRT"
    folder.mkdir(parents=True)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 23, 45, 67
    tiff = io.BytesIO()
    tifffile.imwrite(tiff, image, photometric="rgb", metadata=None)
    roi = roi_bytes([[40.25, 40.75]])
    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w") as nested:
        nested.writestr("BPD.roi", roi)
    with zipfile.ZipFile(folder / "1 Area.zip", "w") as archive:
        archive.writestr("a/image.tif", tiff.getvalue())
        archive.writestr("a/BPD.roi", roi)
        archive.writestr("a/RoiSet.zip", nested_bytes.getvalue())
    output = tmp_path / "dataset"
    summary = build(folder.parent, output, {"1": "train"}, sizes=(16,))
    assert summary["patch_count"] == 1
    assert summary["duplicate_point_refs"] == 1
    path = next(output.glob("patches_16/train/BPD/*.tif"))
    np.testing.assert_array_equal(tifffile.imread(path), image[33:49, 32:48])
    assert validate(output)["status"] == "passed"


def test_output_inside_originals_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        build(tmp_path, tmp_path / "processed", {}, sizes=(16,))


def test_balanced_view_is_deterministic_and_train_only(tmp_path):
    (tmp_path / "기록").mkdir()
    (tmp_path / "summary.json").write_text(json.dumps({"patch_sizes": [128],
                                                       "classes": ["BPD", "TED", "TSD"]}))
    rows = []
    for label, count in (("BPD", 2), ("TED", 5), ("TSD", 3)):
        for number in range(count):
            rows.append({"patch_id": f"{label}_{number}", "split": "train", "label": label, "size": 128})
        rows.append({"patch_id": f"{label}_val", "split": "val", "label": label, "size": 128})
    with (tmp_path / "기록/samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = create_subsets(tmp_path)
    assert result["sizes"]["128"]["total"] == 6
    path = tmp_path / "기록/train_balanced_128.csv"
    first = path.read_bytes()
    create_subsets(tmp_path)
    assert path.read_bytes() == first
    with path.open(encoding="utf-8-sig", newline="") as stream:
        chosen = list(csv.DictReader(stream))
    assert all(row["split"] == "train" for row in chosen)


def test_missing_source_does_not_create_false_success(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        build(tmp_path / "missing", tmp_path / "dataset", {}, sizes=(16,))
    assert not (tmp_path / "dataset").exists()
