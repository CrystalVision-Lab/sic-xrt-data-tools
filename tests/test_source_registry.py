import io
import json
import struct
import zipfile

import numpy as np
import pytest
import tifffile
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from roifile import ImagejRoi

from sic_xrt_data_tools.source_registry import (
    Registrar,
    digest_file,
    inspect_image,
    metadata_json_value,
    name_hints,
    refresh_registry_reports,
    unsafe_member,
    validate_registry,
)


def image_bytes():
    stream = io.BytesIO()
    image = Image.new("RGB", (21, 13), (12, 30, 80))
    exif = Image.Exif()
    exif[274] = 2
    image.save(stream, format="JPEG", exif=exif)
    return stream.getvalue()


def test_tiff_stream_with_numeric_linux_temporary_file_name_is_decoded():
    stream = io.BytesIO()
    tifffile.imwrite(stream, np.arange(64, dtype=np.uint16).reshape(8, 8))
    stream.name = 7  # Linux TemporaryFile exposes its file descriptor as its name.
    result = inspect_image(stream, ".tif")
    assert result["pages_decoded"] == 1 and result["width"] == result["height"] == 8
    assert result["dtype"] == "uint16"


def read_assets(output):
    return [json.loads(line) for line in (output / "assets.jsonl").read_text().splitlines()]


def test_nested_files_preserve_hints_and_failures_without_ground_truth(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    output = tmp_path / "registered"
    roi = ImagejRoi.frompoints(np.array([[2.25, 3.75], [5, 6]], dtype=np.float32)).tobytes()
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as z:
        z.writestr("before_TED(a).roi", roi)
    with zipfile.ZipFile(source / "1 Area.zip", "w") as z:
        z.writestr("1 Area/Before_x100/1.jpg", image_bytes())
        z.writestr("1 Area/After/before.jpg", image_bytes())
        z.writestr("1 Area/Before/broken.jpg", b"not an image")
        z.writestr("1 Area/Before/before_TED(a).roi", roi)
        z.writestr("1 Area/Before/RoiSet.zip", nested.getvalue())
        z.writestr("1 Area/Before/empty_TED(e).roi", b"")
    before = digest_file(source / "1 Area.zip")
    contract = Registrar(source, output, "user_confirmed_distinct_wafers").run()
    assert digest_file(source / "1 Area.zip") == before
    assert contract["status"] == "completed_with_issues"
    assert contract["summary"]["source_tree_unchanged"]
    assert contract["checks"]["new_labels_created"] is False
    assert contract["checks"]["splits_created"] is False
    assets = read_assets(output)
    jpg = next(a for a in assets if a["locator"].endswith("x100/1.jpg"))
    assert jpg["zip_crc_verified"]
    assert jpg["metadata"]["exif_orientation"] == 2
    assert jpg["metadata"]["orientation_applied"] is False
    assert jpg["metadata"]["pages_decoded"] == 1
    conflict = next(a for a in assets if a["locator"].endswith("After/before.jpg"))
    assert conflict["phase_hint"] == "unknown"
    rois = [a for a in assets if a["kind"] == "roi" and a["decode_status"] == "decoded"]
    assert len(rois) == 2
    assert all(a["metadata"]["subtype_name_hint"] == "a" for a in rois)
    assert all(a["metadata"]["semantic_review_status"] == "not_reviewed" for a in rois)
    relationships = json.loads((output / "relationships.json").read_text())
    assert any(r["type"] == "exact_byte_duplicate" for r in relationships)
    assert all(not r.get("physical_defect_identity_confirmed", False) for r in relationships)
    assert validate_registry(output)["valid"]
    assert refresh_registry_reports(output)["valid"]
    new_contract = json.loads((output / "registry.json").read_text())
    assert new_contract["summary"]["image_byte_duplicate_groups"] == 1
    assert new_contract["summary"]["unique_byte_content_counts"]["image"] == 2
    (output / "issues.json").write_text("[]")
    assert not validate_registry(output)["valid"]


def test_stack_decodes_every_page_and_keeps_spacing_unconfirmed(tmp_path):
    source = tmp_path / "3D XRT"
    source.mkdir()
    pixels = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)
    tifffile.imwrite(source / "No12-before.tif", pixels, imagej=True,
                     metadata={"axes": "ZYX", "unit": "micron"}, resolution=(2, 4))
    output = tmp_path / "registered"
    contract = Registrar(source, output).run()
    asset = read_assets(output)[0]
    assert contract["summary"]["image_pages_decoded"] == 3
    assert asset["metadata"]["series"][0]["shape"] == [3, 5, 7]
    assert asset["metadata"]["physical_xy_scale_candidate"] == {"x_um_per_pixel": 0.5,
                                                                "y_um_per_pixel": 0.25}
    assert asset["metadata"]["physical_xy_scale_status"] == "unconfirmed"
    assert asset["metadata"]["depth_spacing_status"] == "missing"


def test_corrupt_zip_crc_is_incomplete_and_other_members_continue(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "2 Area.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("corrupt.txt", b"content")
        z.writestr("good.txt", b"good")
    data = bytearray(path.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", data, 26)
    data[30 + name_length + extra_length] ^= 1
    path.write_bytes(data)
    output = tmp_path / "registered"
    contract = Registrar(source, output).run()
    assert contract["status"] == "incomplete"
    assets = read_assets(output)
    assert next(a for a in assets if a["locator"].endswith("good.txt"))["integrity_status"] == "verified"
    assert next(a for a in assets if a["locator"].endswith("corrupt.txt"))["zip_crc_verified"] is False


def test_registry_refuses_overwrite_and_source_nested_output(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "file.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="outside_source"):
        Registrar(source, source / "output")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        Registrar(source, output)
    assert (source / "file.txt").read_text() == "preserve me"


def test_archive_paths_are_never_used_for_extraction(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    with zipfile.ZipFile(source / "a.zip", "w") as z:
        z.writestr("../escape.txt", b"retain only as registered bytes")
        z.writestr("good.jpg", image_bytes())
    output = tmp_path / "registered"
    result = Registrar(source, output).run()
    assert not (tmp_path / "escape.txt").exists()
    assert result["summary"]["issues_by_code"]["unsafe_archive_path"] == 1
    assert unsafe_member(r"C:\bad.txt")
    assert unsafe_member(r"a\..\bad.txt")


def test_large_image_limit_and_phase_conflict_are_explicit():
    with pytest.warns(Image.DecompressionBombWarning), pytest.raises(ValueError, match="decode_limit"):
        inspect_image(io.BytesIO(image_bytes()), ".jpg", pixel_limit=200)
    assert name_hints("1 Area.zip::1 Area/After/before.tif")["phase_hint"] == "unknown"
    assert name_hints("3 Area.zip::3 Area/After_x100/1.jpg")["nominal_magnification_hint"] == "100"
    assert name_hints("3D XRT/No12-afterAnnealing.tif")["phase_hint"] == "after"
    assert name_hints("5 Area/After annealing_5x100/1.jpg")["nominal_magnification_hint"] == "100"
    assert name_hints("3D XRT/N12_flipped.tif")["filename_flip_axis"] == "unknown"


def test_duplicate_member_names_receive_distinct_ids(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    with zipfile.ZipFile(source / "a.zip", "w") as z:
        z.writestr("same.txt", b"first")
        with pytest.warns(UserWarning):
            z.writestr("same.txt", b"second")
    output = tmp_path / "registered"
    Registrar(source, output).run()
    assets = read_assets(output)
    assert len({a["asset_id"] for a in assets}) == 3
    assert validate_registry(output)["valid"]


def test_invalid_archive_never_reports_completed_registration(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "bad.zip").write_bytes(b"not a ZIP")
    contract = Registrar(source, tmp_path / "registered").run()
    assert contract["status"] == "incomplete"


def test_change_during_scan_is_detected(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    source.mkdir()
    path = source / "change.txt"
    path.write_bytes(b"before")
    original_digest = digest_file
    calls = 0

    def change_after_first_hash(target):
        nonlocal calls
        result = original_digest(target)
        if target == path:
            calls += 1
            if calls == 1:
                path.write_bytes(b"changed")
        return result

    monkeypatch.setattr("sic_xrt_data_tools.source_registry.digest_file", change_after_first_hash)
    output = tmp_path / "registered"
    result = Registrar(source, output).run()
    assert result["status"] == "incomplete"
    assert result["summary"]["issues_by_code"]["source_changed_during_scan"] == 1


def test_rational_and_numpy_metadata_serialization_preserves_values():
    values = {"dpi": [IFDRational(144, 2), IFDRational(720, 10)], "count": np.int64(3)}
    normalized = json.loads(json.dumps(values, default=metadata_json_value))
    assert normalized == {"dpi": [{"numerator": 144, "denominator": 2},
                                  {"numerator": 720, "denominator": 10}], "count": 3}


def test_tree_comparison_does_not_depend_on_windows_case_sorting(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "Before.txt").write_text("one")
    (source / "after.txt").write_text("two")
    (source / "BEFORE2.txt").write_text("three")
    result = Registrar(source, tmp_path / "registered").run()
    assert result["summary"]["source_tree_unchanged"]
    assert result["status"] == "completed"
