import io
import json
import zipfile

import numpy as np
import pytest
import tifffile
from roifile import ROI_TYPE, ImagejRoi

from sic_xrt_data_tools.annotation_workspace import (
    associate_roi,
    build_workspace,
    label_hint,
)
from sic_xrt_data_tools.source_registry import Registrar, digest_file


def point_roi(points, name=None):
    roi = ImagejRoi.frompoints(np.array(points, np.float32))
    roi.roitype = ROI_TYPE.POINT
    if name:
        roi.name = name
    return roi.tobytes()


def test_restore_subtypes_duplicates_conflicts_and_unknown_points(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = io.BytesIO()
    tifffile.imwrite(image, np.zeros((64, 64), np.uint8))
    roi = point_roi([[20.25, 30.75], [1, 1]])
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as z:
        z.writestr("Before_TED(a).roi", roi)
    with zipfile.ZipFile(raw / "1 Area.zip", "w") as z:
        z.writestr("1 Area/Before/before_1.tif", image.getvalue())
        z.writestr("1 Area/Before/Before_TED(a).roi", roi)
        z.writestr("1 Area/Before/RoiSet.zip", nested.getvalue())
        # Identical ROI bytes may carry a different label filename. Preserve this conflict.
        z.writestr("1 Area/Before/Before_TSD(b).roi", point_roi([[20.25, 30.75]], "Before_TSD(b)"))
        z.writestr("1 Area/Before/Before_unknown.roi", point_roi([[5, 5]], "unknown"))
        z.writestr("1 Area/Before/Before_TED(e).roi", b"")
    before = digest_file(raw / "1 Area.zip")
    registry = tmp_path / "registered"
    Registrar(raw, registry).run()
    output = tmp_path / "workspace"
    result = build_workspace(registry, output, with_reviews=True)
    assert result["schema_version"] == 2
    assert result["summary"]["point_count"] == 4
    assert result["summary"]["fine_counts"] == {"TED_a": 2, "TSD_b": 1, "unknown": 1}
    assert result["summary"]["label_conflicts"] == 1
    assert result["summary"]["excluded_roi_groups"] == 1
    assert result["training_ready"] is False
    assert digest_file(raw / "1 Area.zip") == before
    points = [json.loads(s) for s in (output / "provider_points.jsonl").read_text().splitlines()]
    assert any(p["x"] == 20.25 and p["y"] == 30.75 for p in points)
    assert all(p["human_verified"] is False and p["split"] == "unassigned" for p in points)
    hashes = json.loads((output / "output_hashes.json").read_text(encoding="utf-8"))
    assert all(digest_file(output / name) == sha for name, sha in hashes.items())


def test_numeric_image_association_remains_candidate_and_phase_safe():
    roi = {"locator": "1 Area.zip::1 Area/After_x100/4_After_BPD.roi", "area_id": "1", "phase_hint": "after"}
    images = [{"locator": f"1 Area.zip::1 Area/After_x100/{i}.jpg", "area_id": "1",
               "phase_hint": "after", "decode_status": "decoded", "suffix": ".jpg"}
              for i in (3, 4, 5)]
    image, evidence, _ = associate_roi(roi, images)
    assert image["locator"].endswith("/4.jpg")
    assert evidence == "explicit_leading_image_number_in_same_directory"
    images[1]["phase_hint"] = "before"
    assert associate_roi(roi, images)[0] is None


def test_label_names_are_not_guessed_or_read_from_compound_words():
    assert label_hint("Before_TED(f).roi") == ("TED", "f", "TED_f")
    assert label_hint("TSD(c).roi") == ("TSD", "c", "TSD_c")
    assert label_hint("unknown.roi") == ("unknown", None, "unknown")
    assert label_hint("TEDDY.roi") == ("unknown", None, "unknown")


def test_tampered_registry_and_changed_original_are_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    roi = point_roi([[3, 4]])
    with zipfile.ZipFile(raw / "1 Area.zip", "w") as z:
        z.writestr("1 Area/Before/TED(a).roi", roi)
    registry = tmp_path / "registered"
    Registrar(raw, registry).run()
    (raw / "1 Area.zip").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash_changed"):
        build_workspace(registry, tmp_path / "workspace", with_reviews=False)


def test_same_roi_bytes_with_different_label_names_are_not_collapsed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = io.BytesIO()
    tifffile.imwrite(image, np.zeros((32, 32), np.uint8))
    roi = point_roi([[12, 14]])
    with zipfile.ZipFile(raw / "1 Area.zip", "w") as z:
        z.writestr("1 Area/Before/before.tif", image.getvalue())
        z.writestr("1 Area/Before/TED(a).roi", roi)
        z.writestr("1 Area/Before/TSD(b).roi", roi)
    registry = tmp_path / "registered"
    Registrar(raw, registry).run()
    result = build_workspace(registry, tmp_path / "workspace", with_reviews=False)
    assert result["summary"]["point_count"] == 2
    assert result["summary"]["label_conflicts"] == 1
    (registry / "assets.jsonl").write_text("")
    with pytest.raises(ValueError, match="not_ready"):
        build_workspace(registry, tmp_path / "workspace", with_reviews=False)
