"""Read-only original registration, source_registry v1 (not ground truth)."""

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import re
import tempfile
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
from PIL import Image
from roifile import ImagejRoi

CHUNK = 4 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def metadata_json_value(value):
    """Preserve library scalar/rational metadata without lossy string conversion."""
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return {"numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"unsupported_metadata_type: {type(value).__name__}")


def digest_stream(stream, sink=None):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK):
        digest.update(chunk)
        size += len(chunk)
        if sink is not None:
            sink.write(chunk)
    return digest.hexdigest(), size


def digest_file(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)[0]


def file_signature(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def name_hints(locator):
    components = locator.replace("\\", "/").replace("::", "/").split("/")
    evidence = []
    for component in components:
        phases = re.findall(r"(?<![a-z])(before|after)(?=annealing|[^a-z]|$)", component.lower())
        for phase in sorted(set(phases)):
            evidence.append({"component": component, "phase": phase})
    phases = {item["phase"] for item in evidence}
    area = re.search(r"(?:^|[/ :])([1-9][0-9]*)\s*area(?:[/. :]|$)", locator, re.IGNORECASE)
    magnification = re.search(r"(?<![a-z])x(\d+)(?:[_. /]|$)", locator, re.IGNORECASE)
    return {
        "area_id": area.group(1) if area else None,
        "phase_hint": next(iter(phases)) if len(phases) == 1 else "unknown",
        "phase_name_conflict": len(phases) > 1,
        "phase_evidence": evidence,
        "nominal_magnification_hint": magnification.group(1) if magnification else None,
        "magnification_calibrated": False,
        "filename_flip_hint": "flipped" in locator.lower(),
        "filename_flip_axis": "unknown",
    }


def unsafe_member(name):
    normalized = name.replace("\\", "/")
    return (
        PurePosixPath(normalized).is_absolute()
        or ".." in PurePosixPath(normalized).parts
        or bool(re.match(r"^[a-z]:", normalized, re.IGNORECASE))
        or "\x00" in name
    )


def inspect_image(stream, suffix, pixel_limit=500_000_000, metadata=None):
    """Decode raw pixels; never apply EXIF, rescale, stack all frames, or infer labels."""
    stream.seek(0)
    meta = {} if metadata is None else metadata
    meta.update(coordinate_system="raw_pixel_xy", orientation_applied=False)
    pixel_digest = hashlib.sha256()
    if suffix in {".tif", ".tiff"}:
        # Linux TemporaryFile.name is an integer descriptor. Older tifffile versions
        # interpret that as a path unless an explicit display name is supplied.
        with tifffile.TiffFile(stream, name="registered_source" + suffix) as tif:
            first = tif.pages[0]
            meta.update(
                format="TIFF", width=first.imagewidth, height=first.imagelength,
                page_count=len(tif.pages), dtype=str(first.dtype),
                series=[{"shape": list(s.shape), "axes": s.axes, "dtype": str(s.dtype)}
                        for s in tif.series],
                exif_orientation=int(first.tags["Orientation"].value)
                if "Orientation" in first.tags else 1,
            )
            tags = first.tags
            meta["resolution_tags"] = {
                name: list(tags[name].value) if isinstance(tags[name].value, tuple)
                else int(tags[name].value)
                for name in ("XResolution", "YResolution", "ResolutionUnit") if name in tags
            }
            ij = tif.imagej_metadata or {}
            meta["imagej"] = {k: ij[k] for k in ("unit", "spacing", "slices", "frames", "images")
                              if k in ij}
            meta["depth_spacing_status"] = "candidate" if ij.get("spacing") else "missing"
            meta["physical_xy_scale_status"] = "unconfirmed"
            unit = str(ij.get("unit", "")).lower()
            if unit in {"micron", "um", "µm", "micrometer"}:
                candidate = {}
                for axis in ("X", "Y"):
                    tag = tags.get(axis + "Resolution")
                    if tag and isinstance(tag.value, tuple) and tag.value[0] > 0:
                        candidate[axis.lower() + "_um_per_pixel"] = tag.value[1] / tag.value[0]
                if candidate:
                    meta["physical_xy_scale_candidate"] = candidate
                    meta["scale_evidence"] = "ImageJ unit + TIFF resolution; geometry unconfirmed"
            meta["pages_decoded"] = 0
            for page in tif.pages:
                if int(np.prod(page.shape, dtype=np.int64)) > pixel_limit * 4:
                    raise ValueError("page_exceeds_decode_limit")
                pixels = page.asarray()
                pixel_digest.update(json.dumps([list(pixels.shape), str(pixels.dtype)]).encode())
                pixel_digest.update(memoryview(np.ascontiguousarray(pixels)).cast("B"))
                meta["pages_decoded"] += 1
                del pixels
    else:
        # Large microscopy images are expected. An explicit bound replaces Pillow's default.
        previous_limit = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = pixel_limit
            with Image.open(stream) as image:
                if image.width * image.height > pixel_limit:
                    raise ValueError("image_exceeds_decode_limit")
                meta.update(format=image.format, width=image.width, height=image.height,
                            mode=image.mode, page_count=getattr(image, "n_frames", 1),
                            exif_orientation=int(image.getexif().get(274, 1)),
                            display_dpi=image.info.get("dpi"),
                            physical_xy_scale_status="unconfirmed", pages_decoded=0)
                for index in range(meta["page_count"]):
                    image.seek(index)
                    image.load()
                    pixel_digest.update(json.dumps([image.size, image.mode]).encode())
                    # Avoid another full-sized image/bytes copy for very large JPEGs.
                    for top in range(0, image.height, 128):
                        pixel_digest.update(image.crop((0, top, image.width,
                                                      min(top + 128, image.height))).tobytes())
                    meta["pages_decoded"] += 1
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit
    meta["raw_decoded_pixels_sha256"] = pixel_digest.hexdigest()
    return meta


def inspect_roi(stream, name):
    stream.seek(0)
    data = stream.read()
    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("empty_or_invalid_imagej_roi")
    roi = ImagejRoi.frombytes(data)
    coordinates = np.asarray(roi.coordinates())
    if coordinates.size and not np.isfinite(coordinates).all():
        raise ValueError("nonfinite_roi_coordinates")
    match = re.search(r"(?:^|[^a-z])(TED|TSD|BPD)(?:\s*\(([a-f])\))?", name, re.IGNORECASE)
    return {
        "roi_type": roi.roitype.name, "coordinate_count": len(coordinates),
        "bounds_xy": [coordinates.min(axis=0).tolist(), coordinates.max(axis=0).tolist()]
        if coordinates.size else None,
        "internal_name": roi.name, "position": roi.position,
        "z_position": roi.z_position, "t_position": roi.t_position,
        "class_name_hint": match.group(1).upper() if match else None,
        "subtype_name_hint": match.group(2).lower() if match and match.group(2) else None,
        "semantic_review_status": "not_reviewed", "coordinate_system": "raw_pixel_xy",
        "annotation_origin": "supplied_file", "image_association_status": "not_assigned",
    }


def content_summary(assets):
    """Count exact byte identities while preserving every location alias."""
    groups = defaultdict(list)
    for asset in assets:
        if asset.get("sha256") and asset["integrity_status"] == "verified":
            groups[(asset["kind"], asset["sha256"])].append(asset)
    content_groups = []
    for (kind, sha), members in sorted(groups.items()):
        preferred = min(members, key=lambda a: (a["source_kind"] != "zip_member",
                                               len(a["ordinal_path"]), a["locator"]))
        areas = sorted({a["area_id"] for a in members if a["area_id"]}, key=int)
        phases = sorted({a["phase_hint"] for a in members if a["phase_hint"] != "unknown"})
        content_groups.append({"kind": kind, "sha256": sha,
                               "preferred_reference_asset_id": preferred["asset_id"],
                               "asset_ids": [a["asset_id"] for a in members],
                               "area_name_hints": areas, "phase_name_hints": phases,
                               "physical_defect_identity_confirmed": False})
    images = [a for a in assets if a["kind"] == "image"]
    archive_images = [a for a in images if a["area_id"] and a["source_kind"] == "zip_member"]
    summary = {
        "unique_byte_content_counts": dict(Counter(g["kind"] for g in content_groups)),
        "image_byte_duplicate_groups": sum(g["kind"] == "image" and len(g["asset_ids"]) > 1
                                            for g in content_groups),
        "area_archive_image_counts": dict(Counter(a["area_id"] for a in archive_images)),
        "area_loose_image_counts": dict(Counter(a["area_id"] for a in images
                                                 if a["area_id"] and a["source_kind"] == "file")),
        "archive_area_images": len(archive_images),
        "areas_without_supplied_roi": sorted(
            {a["area_id"] for a in images if a["area_id"]}
            - {a["area_id"] for a in assets if a["kind"] == "roi" and a["area_id"]}, key=int),
        "image_identical_bytes_across_areas": sum(g["kind"] == "image" and len(g["area_name_hints"]) > 1
                                                   for g in content_groups),
    }
    return summary, content_groups


class Registrar:
    def __init__(self, source, output, area_basis="unconfirmed", progress=None):
        self.source = Path(source).resolve(strict=True)
        self.output = Path(output).resolve()
        if not self.source.is_dir():
            raise ValueError("source_must_be_directory")
        if self.output == self.source or self.output.is_relative_to(self.source):
            raise ValueError("output_must_be_outside_source")
        if self.source.is_relative_to(self.output):
            raise ValueError("output_must_not_contain_source")
        if self.output.exists():
            raise FileExistsError("registry_output_already_exists")
        self.area_basis = area_basis
        self.progress = progress or (lambda message: None)
        self.assets = []
        self.issues = []
        self.relationships = []

    def issue(self, asset, code, detail="", severity="warning"):
        self.issues.append({"asset_id": asset["asset_id"], "locator": asset["locator"],
                            "code": code, "severity": severity, "detail": detail})

    def asset(self, locator, source_kind, ordinal_path, parent=None):
        suffix = Path(locator.split("::")[-1]).suffix.lower()
        asset = {
            "asset_id": "src_" + hashlib.sha256(
                json.dumps([locator, ordinal_path], ensure_ascii=False).encode()).hexdigest()[:24],
            "locator": locator, "source_kind": source_kind, "parent_asset_id": parent,
            "ordinal_path": ordinal_path, "suffix": suffix,
            "kind": "image" if suffix in IMAGE_SUFFIXES else "roi" if suffix == ".roi"
            else "archive" if suffix == ".zip" else "auxiliary",
            "integrity_status": "pending", "decode_status": "not_applicable",
            **name_hints(locator),
        }
        self.assets.append(asset)
        if asset["phase_name_conflict"]:
            self.issue(asset, "phase_name_conflict", "Before/after names disagree; no phase assigned.")
        return asset

    def inspect_payload(self, stream, asset, depth=0):
        if asset["kind"] == "archive":
            if depth > 3:
                asset["decode_status"] = "not_inspected"
                self.issue(asset, "nested_archive_depth_limit", severity="error")
                return
            self.inspect_archive(stream, asset, depth)
            return
        if asset["kind"] not in {"image", "roi"}:
            return
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                if asset["kind"] == "image":
                    asset["metadata"] = {}
                    inspect_image(stream, asset["suffix"], metadata=asset["metadata"])
                else:
                    asset["metadata"] = inspect_roi(stream, asset["locator"])
                # Normalize before the global save, so third-party scalars cannot lose a full scan.
                asset["metadata"] = json.loads(json.dumps(asset["metadata"], default=metadata_json_value))
            asset["decode_status"] = "decoded"
            for warning in captured:
                self.issue(asset, "decoder_warning", str(warning.message)[:500])
            orientation = asset["metadata"].get("exif_orientation", 1)
            if orientation != 1:
                self.issue(asset, "orientation_transform_required",
                           f"EXIF orientation {orientation}; original pixels/coordinates preserved.")
        except Exception as error:  # noqa: BLE001 - isolate third-party decoder faults per asset
            asset["decode_status"] = "failed"
            self.issue(asset, "decode_failed", type(error).__name__ + ": " + str(error)[:500],
                       severity="error")

    def inspect_archive(self, stream, parent, depth):
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream) as archive:
                entries = archive.infolist()
                parent["metadata"] = {"entry_count": len(entries), "member_crc_scope": "all_files"}
                names = Counter(entry.filename for entry in entries)
                for index, entry in enumerate(entries):
                    if entry.is_dir():
                        continue
                    asset = self.asset(parent["locator"] + "::" + entry.filename, "zip_member",
                                       parent["ordinal_path"] + [index], parent["asset_id"])
                    asset.update(declared_bytes=entry.file_size, zip_crc32=f"{entry.CRC:08x}")
                    if unsafe_member(entry.filename):
                        self.issue(asset, "unsafe_archive_path", "Never extracted to named path.", "error")
                    if names[entry.filename] > 1:
                        self.issue(asset, "duplicate_archive_name", "Distinct ZIP ordinals retained.")
                    if entry.file_size > 8 * 1024**3:
                        asset["integrity_status"] = "not_checked"
                        self.issue(asset, "archive_member_size_limit", severity="error")
                        continue
                    needs_payload = asset["kind"] in {"image", "roi", "archive"}
                    with tempfile.TemporaryFile(dir=self.output / "scratch") as temporary:
                        try:
                            with archive.open(entry) as member:
                                sha, size = digest_stream(member, temporary if needs_payload else None)
                            asset.update(sha256=sha, bytes=size, zip_crc_verified=True,
                                         integrity_status="verified" if size == entry.file_size else "failed")
                            if size != entry.file_size:
                                self.issue(asset, "member_size_mismatch", severity="error")
                            if needs_payload:
                                temporary.seek(0)
                                self.inspect_payload(temporary, asset, depth + 1)
                        except Exception as error:  # noqa: BLE001 - continue other archive members
                            asset.update(integrity_status="failed", zip_crc_verified=False)
                            self.issue(asset, "member_read_failed", type(error).__name__, "error")
                    if asset["kind"] == "image":
                        self.progress(f"image {sum(a['kind'] == 'image' for a in self.assets)}: "
                                      f"{asset['decode_status']}")
                parent["decode_status"] = "enumerated"
        except Exception as error:  # noqa: BLE001 - preserve failed archive in registry
            parent["decode_status"] = "failed"
            self.issue(parent, "archive_read_failed", type(error).__name__, "error")

    def derive_relationships(self):
        hashes = defaultdict(list)
        groups = defaultdict(list)
        for asset in self.assets:
            if asset.get("sha256") and asset["integrity_status"] == "verified":
                hashes[asset["sha256"]].append(asset["asset_id"])
            if asset["kind"] == "image" and asset["area_id"]:
                groups[asset["area_id"]].append(asset)
        for sha, ids in sorted(hashes.items()):
            if len(ids) > 1:
                self.relationships.append({"type": "exact_byte_duplicate", "status": "confirmed",
                                           "sha256": sha, "asset_ids": ids,
                                           "physical_defect_identity_confirmed": False})
        for area, assets in sorted(groups.items(), key=lambda item: int(item[0])):
            self.relationships.append({
                "type": "area_group", "status": self.area_basis, "area_id": area,
                "asset_ids": [a["asset_id"] for a in assets],
                "before_candidates": [a["asset_id"] for a in assets if a["phase_hint"] == "before"],
                "after_candidates": [a["asset_id"] for a in assets if a["phase_hint"] == "after"],
                "unassigned_phase": [a["asset_id"] for a in assets if a["phase_hint"] == "unknown"],
                "field_of_view_registration_status": "not_done",
                "physical_defect_identity_confirmed": False,
            })
            magnifications = defaultdict(list)
            for asset in assets:
                key = (asset["phase_hint"], asset["nominal_magnification_hint"])
                magnifications[key].append(asset["asset_id"])
            for (phase, magnification), ids in magnifications.items():
                self.relationships.append({"type": "capture_family_candidate", "status": "candidate",
                                           "area_id": area, "phase_hint": phase,
                                           "nominal_magnification_hint": magnification,
                                           "asset_ids": ids, "scale_confirmed": False,
                                           "same_field_of_view_confirmed": False})
        # Paired stack filenames identify a candidate acquisition family only.
        stacks = defaultdict(list)
        for asset in self.assets:
            if asset["kind"] == "image" and "3d" in asset["locator"].lower():
                match = re.search(r"(?:no|n)(\d+)", Path(asset["locator"]).name, re.IGNORECASE)
                if match:
                    stacks[match.group(1)].append(asset["asset_id"])
        for number, ids in sorted(stacks.items()):
            self.relationships.append({"type": "stack_family_candidate", "status": "candidate",
                                       "name_number_hint": number, "asset_ids": ids,
                                       "area_mapping_status": "unknown",
                                       "physical_defect_identity_confirmed": False})

    def refine_name_metadata(self):
        """Derive only filename/record checks; never modify bytes or semantic labels."""
        derived_codes = {"phase_name_conflict", "unassigned_2d_area", "missing_stack_spacing",
                         "image_bytes_shared_across_areas", "image_bytes_shared_across_phases"}
        self.issues = [i for i in self.issues if i["code"] not in derived_codes]
        for asset in self.assets:
            asset.update(name_hints(asset["locator"]))
            if asset["phase_name_conflict"]:
                self.issue(asset, "phase_name_conflict", "Before/after names disagree; no phase assigned.")
            if asset["kind"] != "image":
                continue
            if "2d xrt" in asset["locator"].lower() and not asset["area_id"]:
                self.issue(asset, "unassigned_2d_area", "Area is absent from the original path.")
            meta = asset.get("metadata", {})
            if meta.get("page_count", 0) > 1 and meta.get("depth_spacing_status") == "missing":
                self.issue(asset, "missing_stack_spacing", "Layer pitch not provided; depth uncalibrated.")
        _, groups = content_summary(self.assets)
        by_id = {a["asset_id"]: a for a in self.assets}
        for group in groups:
            if group["kind"] != "image":
                continue
            asset = by_id[group["preferred_reference_asset_id"]]
            if len(group["area_name_hints"]) > 1:
                self.issue(asset, "image_bytes_shared_across_areas",
                           "Identical bytes in Areas " + ", ".join(group["area_name_hints"]))
            if len(group["phase_name_hints"]) > 1:
                self.issue(asset, "image_bytes_shared_across_phases",
                           "Identical bytes carry both before and after names.")

    def run(self):
        files = sorted(p for p in self.source.rglob("*") if p.is_file())
        if not files:
            raise ValueError("no_source_files")
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "scratch").mkdir()
        started = datetime.now(UTC).isoformat()
        snapshots = {}
        for path in files:
            relative = path.relative_to(self.source).as_posix()
            asset = self.asset(relative, "file", [])
            if path.is_symlink() or not path.resolve().is_relative_to(self.source):
                asset["integrity_status"] = "not_checked"
                self.issue(asset, "source_link_not_followed", severity="error")
                continue
            self.progress("source: " + relative)
            try:
                before = file_signature(path)
                asset["signature_before"] = before
                asset["sha256"] = digest_file(path)
                asset["bytes"] = before["bytes"]
                asset["integrity_status"] = "verified"
                with path.open("rb") as stream:
                    self.inspect_payload(stream, asset)
                after = file_signature(path)
                verification_sha = digest_file(path)
                asset.update(signature_after=after, verification_sha256=verification_sha)
                asset["source_unchanged_during_scan"] = before == after and asset["sha256"] == verification_sha
                if not asset["source_unchanged_during_scan"]:
                    asset["integrity_status"] = "changed"
                    self.issue(asset, "source_changed_during_scan", severity="error")
                snapshots[relative] = after
            except Exception as error:  # noqa: BLE001 - preserve failures and scan remaining originals
                asset["integrity_status"] = "failed"
                self.issue(asset, "source_read_failed", type(error).__name__, "error")
        # Check previously processed originals again after the complete run.
        final_files = sorted(p.relative_to(self.source).as_posix()
                             for p in self.source.rglob("*") if p.is_file())
        final_tree_unchanged = set(final_files) == {p.relative_to(self.source).as_posix() for p in files}
        try:
            final_stats_unchanged = all(file_signature(self.source / relative) == signature
                                        for relative, signature in snapshots.items())
        except OSError:
            final_stats_unchanged = False
        self.refine_name_metadata()
        self.derive_relationships()
        (self.output / "scratch").rmdir()
        summary = {
            "source_files": len(files), "source_bytes": sum(a.get("bytes", 0) for a in self.assets
                                                            if a["source_kind"] == "file"),
            "registered_assets": len(self.assets), "kind_counts": dict(Counter(a["kind"] for a in self.assets)),
            "integrity_counts": dict(Counter(a["integrity_status"] for a in self.assets)),
            "decode_counts": dict(Counter(a["decode_status"] for a in self.assets)),
            "image_pages_decoded": sum(a.get("metadata", {}).get("pages_decoded", 0)
                                       for a in self.assets if a["kind"] == "image"),
            "issues_by_code": dict(Counter(i["code"] for i in self.issues)),
            "source_tree_unchanged": final_tree_unchanged,
            "source_final_stats_unchanged": final_stats_unchanged,
            "area_image_counts": dict(Counter(a["area_id"] for a in self.assets
                                              if a["kind"] == "image" and a["area_id"])),
        }
        blocking = any(a["integrity_status"] != "verified"
                       or a["kind"] == "archive" and a["decode_status"] != "enumerated"
                       for a in self.assets)
        contract = {
            "schema": "source_registry", "schema_version": 1, "source_root": str(self.source),
            "tool": "sic_xrt_data_tools.source_registry",
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("numpy", "tifffile", "Pillow", "roifile", "imagecodecs")},
            "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
            "status": "incomplete" if blocking or not final_tree_unchanged or not final_stats_unchanged
            else "completed_with_issues" if self.issues else "completed",
            "area_group_basis": self.area_basis, "summary": summary,
            "checks": {"source_sha256": "two_pass", "zip_crc": "all_members",
                       "image_decode": "all_pages_raw_pixels", "visual_quality_review": "not_done",
                       "annotation_semantic_review": "not_done", "before_after_registration": "not_done",
                       "new_labels_created": False, "splits_created": False, "training_run": False},
        }
        self.save(contract)
        return contract

    def save(self, contract):
        def write_json(name, data):
            (self.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                       default=metadata_json_value)
                                           + "\n", encoding="utf-8")

        additional, groups = content_summary(self.assets)
        contract["summary"].update(additional)
        write_json("registry.json", contract)
        write_json("content_groups.json", groups)
        with (self.output / "assets.jsonl").open("w", encoding="utf-8") as stream:
            for asset in self.assets:
                stream.write(json.dumps(asset, ensure_ascii=False, default=metadata_json_value) + "\n")
        write_json("relationships.json", self.relationships)
        write_json("issues.json", self.issues)
        fields = ["asset_id", "locator", "kind", "source_kind", "area_id", "phase_hint",
                  "phase_name_conflict", "nominal_magnification_hint", "bytes", "sha256",
                  "integrity_status", "decode_status", "metadata_json"]
        with (self.output / "전체_원본_목록.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for asset in self.assets:
                writer.writerow(asset | {"metadata_json": json.dumps(asset.get("metadata", {}),
                                                                    ensure_ascii=False,
                                                                    default=metadata_json_value)})
        with (self.output / "확인_필요_목록.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["asset_id", "locator", "code", "severity", "detail"])
            writer.writeheader()
            writer.writerows(self.issues)
        self.write_report(contract)
        files = sorted(p for p in self.output.iterdir() if p.is_file() and p.name != "output_hashes.json")
        write_json("output_hashes.json", {p.name: digest_file(p) for p in files})

    def write_report(self, contract):
        e = html.escape
        summary = contract["summary"]
        rows = []
        for area, count in sorted(summary["area_archive_image_counts"].items(), key=lambda item: int(item[0])):
            members = [a for a in self.assets if a["kind"] == "image" and a["area_id"] == area
                       and a["source_kind"] == "zip_member"]
            rois = [a for a in self.assets if a["kind"] == "roi" and a["area_id"] == area]
            phases = Counter(a["phase_hint"] for a in members)
            failed = sum(a["decode_status"] == "failed" for a in members)
            roi_unique = len({a.get('sha256') for a in rois if a.get('sha256')})
            rows.append(f"<tr><td>Area {e(area)}</td><td>{count}</td><td>{phases['before']}</td>"
                        f"<td>{phases['after']}</td><td>{phases['unknown']}</td><td>{len(rois)}</td>"
                        f"<td>{roi_unique}</td><td>{failed}</td></tr>")
        explanations = {
            "decode_failed": ("영상·주석 읽기 실패", "원본 재확보 또는 별도 검수가 필요합니다."),
            "phase_name_conflict": ("열처리 전·후 이름 충돌", "폴더와 파일의 전·후 표기가 달라 단계 판정을 보류했습니다."),
            "orientation_transform_required": ("영상 방향 정보 확인", "자동으로 뒤집어 표시할 때 ROI 좌표도 같은 변환이 필요합니다."),
            "unassigned_2d_area": ("Area 미지정 영상", "원본 경로에 Area 정보가 없어 웨이퍼 판정을 보류했습니다."),
            "missing_stack_spacing": ("3D 깊이 간격 없음", "층 간 실제 거리가 없어 깊이를 µm 단위로 확정할 수 없습니다."),
            "image_bytes_shared_across_areas": ("Area 간 동일 영상", "다른 Area에 동일 바이트 영상이 있어 대응 관계 확인이 필요합니다."),
            "image_bytes_shared_across_phases": ("전·후 동일 영상", "같은 바이트 영상에 전·후 이름이 함께 있어 출처 확인이 필요합니다."),
        }
        issues = "".join(f"<tr><td>{e(explanations.get(i['code'], (i['code'], ''))[0])}</td>"
                         f"<td>{e(i['locator'])}</td><td>"
                         f"{e(explanations.get(i['code'], ('', ''))[1])}"
                         f"<br><small>{e(i['detail'])}</small></td></tr>" for i in self.issues)
        stacks = [a for a in self.assets if a["kind"] == "image" and a["source_kind"] == "file"
                  and a.get("metadata", {}).get("page_count", 0) > 1]
        stack_rows = "".join(f"<tr><td>{e(a['locator'])}</td><td>{a.get('metadata', {}).get('page_count', '?')}</td>"
                             f"<td>{e(a['decode_status'])}</td><td>미확정</td></tr>" for a in stacks)
        report = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>전체 원본 등록 결과</title><style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:18px 0}}td,th{{border:1px solid #cbd5e1;padding:8px;text-align:left;overflow-wrap:anywhere}}
th{{background:#e2e8f0}}.note{{padding:18px;background:#fff7ed;border:1px solid #fdba74}}a{{color:#075985}}
</style><h1>전체 원본 등록 결과</h1><p>등록 완료 시간: {e(contract['finished_at'])}</p>
<p>원본 위치: {e(contract['source_root'])}</p><p>원본 파일 {summary['source_files']}개 ·
등록 항목 {summary['registered_assets']}개 · 디코딩한 영상 페이지 {summary['image_pages_decoded']}개</p>
<p>Area ZIP 안의 2D 영상 {summary['archive_area_images']}개. 동일 바이트 영상 복사본 묶음
{summary['image_byte_duplicate_groups']}개. 전체 영상의 서로 다른 바이트 내용은
{summary['unique_byte_content_counts'].get('image', 0)}개이며, 독립 촬영/결함 수는 미확정입니다.</p>
<p>상태: {e({'completed': '등록 완료', 'completed_with_issues': '등록 완료 · 확인 필요 항목 있음', 'incomplete': '등록 미완료'}.get(contract['status'], contract['status']))}.
SHA-256 두 번 대조, ZIP 파일 CRC와 영상 전체 페이지 읽기를 검사했습니다.</p>
<div class="note">이 결과는 원본 등록입니다. 파일이 읽힌다는 사실은 결함 라벨의 정답이나 영상의 시각적 품질을 보증하지 않습니다.
기존 ROI 이름은 출처 속성으로 보존했습니다. 신규 정답·학습 분할·전후 동일 결함 대응은 만들지 않았습니다.
Before/After와 x100 표기는 이름에서 읽은 후보이며, EXIF 반전은 적용하지 않았습니다.</div>
<h2>2D Area별 영상과 주석 파일</h2><p>영상 수는 Area ZIP 구성원 기준입니다. 풀어 둔 복사본도 전체 목록에 함께 등록했습니다.
주석 수에는 풀어 둔 파일·직접 ROI·중첩 ZIP의 중복 내보내기가 포함됩니다.
영상 수도 전체·확대·분할·다른 형식 내보내기를 포함하며 독립 결함/시료 수가 아닙니다.</p>
<table><tr><th>그룹</th><th>ZIP 내 영상</th><th>Before 후보</th><th>After 후보</th><th>단계 미확정</th><th>ROI 등록 항목</th><th>ROI 바이트 종류</th><th>ZIP 영상 읽기 실패</th></tr>{''.join(rows)}</table>
<p>제공된 ROI 파일이 없는 Area: {e(', '.join(summary['areas_without_supplied_roi']) or '없음')}.
해당 Area에 결함이 없다는 뜻은 아닙니다.</p>
<h2>3D TIFF</h2><table><tr><th>원본</th><th>페이지</th><th>읽기 상태</th><th>깊이 간격·2D 대응</th></tr>{stack_rows}</table>
<h2>확인할 항목</h2><table><tr><th>종류</th><th>원본 참조</th><th>설명</th></tr>{issues}</table>
<h2>기록 파일</h2><p><a href="전체_원본_목록.csv">전체 원본 목록 CSV</a> ·
<a href="확인_필요_목록.csv">확인 필요 목록 CSV</a> · <a href="registry.json">등록 요약</a> ·
<a href="assets.jsonl">원본별 전체 속성</a> · <a href="relationships.json">확정 중복과 관계 후보</a> ·
<a href="content_groups.json">동일 바이트 묶음과 대표 참조</a> ·
<a href="output_hashes.json">기록 파일 SHA-256</a></p>
<h2>다음 단계</h2><p>제공자 ROI의 TED a–f / TSD a–c 이름과 좌표를 원본 영상에 연결하고,
라벨이 없는 Area 및 전후 추적·선 경로·3D 경로에 필요한 정답을 별도로 검수합니다.</p></html>"""
        (self.output / "전체원본_등록결과.html").write_text(report, encoding="utf-8")


def refresh_registry_reports(output):
    """Rebuild validated reports without rereading or changing original files."""
    if not validate_registry(output)["valid"]:
        raise ValueError("existing_registry_validation_failed")
    output = Path(output).resolve()
    contract = json.loads((output / "registry.json").read_text(encoding="utf-8"))
    registrar = object.__new__(Registrar)
    registrar.output = output
    registrar.assets = [json.loads(line) for line in (output / "assets.jsonl").read_text(
        encoding="utf-8").splitlines()]
    registrar.issues = json.loads((output / "issues.json").read_text(encoding="utf-8"))
    registrar.area_basis = contract["area_group_basis"]
    registrar.relationships = []
    registrar.refine_name_metadata()
    registrar.derive_relationships()
    contract["summary"]["issues_by_code"] = dict(Counter(i["code"] for i in registrar.issues))
    if contract["status"] == "completed" and registrar.issues:
        contract["status"] = "completed_with_issues"
    contract["report_refreshed_at"] = datetime.now(UTC).isoformat()
    registrar.save(contract)
    return validate_registry(output)


def validate_registry(output):
    output = Path(output)
    expected = json.loads((output / "output_hashes.json").read_text(encoding="utf-8"))
    failures = [name for name, sha in expected.items() if digest_file(output / name) != sha]
    contract = json.loads((output / "registry.json").read_text(encoding="utf-8"))
    assets = [json.loads(line) for line in (output / "assets.jsonl").read_text(encoding="utf-8").splitlines()]
    if contract["schema"] != "source_registry" or contract["schema_version"] != 1:
        failures.append("schema")
    ids = [a["asset_id"] for a in assets]
    if len(ids) != len(set(ids)) or len(assets) != contract["summary"]["registered_assets"]:
        failures.append("asset_identity_or_count")
    relationships = json.loads((output / "relationships.json").read_text(encoding="utf-8"))
    known = set(ids)
    if any(not set(r["asset_ids"]).issubset(known) for r in relationships):
        failures.append("relationship_reference")
    return {"valid": not failures, "failures": failures, "files_hashed": len(expected),
            "assets": len(assets), "registry_status": contract["status"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--area-group-basis", choices=["unconfirmed", "user_confirmed_distinct_wafers"],
                        default="unconfirmed")
    args = parser.parse_args(argv)
    contract = Registrar(args.source, args.output, args.area_group_basis,
                         progress=lambda message: print(message, flush=True)).run()
    validation = validate_registry(args.output)
    print(json.dumps({"summary": contract["summary"], "validation": validation}, ensure_ascii=False), flush=True)
    return 0 if validation["valid"] and contract["status"] != "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
