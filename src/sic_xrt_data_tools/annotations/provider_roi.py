"""Discover provider ImageJ point ROI files and map them to their source TIFF.

Labels come only from the ROI file name (``BPD``, ``TED(a)`` ... ``TSD(c)``).
They are provider labels, not expert-verified ground truth.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from roifile import ROI_TYPE, ImagejRoi

SUBTYPES = {"BPD": (), "TED": tuple("abcdef"), "TSD": tuple("abc")}
CLASSES = tuple(SUBTYPES)
FINE_LABELS = tuple(
    name for cls, subs in SUBTYPES.items() for name in ([cls] if not subs else [f"{cls}_{s}" for s in subs])
)

_LABEL = re.compile(r"(?<![A-Za-z])(BPD|TED|TSD)(?![A-Za-z])\s*(?:\(\s*([A-Za-z])\s*\))?", re.IGNORECASE)
_AREA = re.compile(r"^(\d+)\s*Area$", re.IGNORECASE)


def fine_label(cls: str, subtype: str | None) -> str:
    return cls if subtype is None else f"{cls}_{subtype}"


def parse_label(filename: str) -> tuple[str | None, str | None, str | None]:
    """Return ``(class, subtype, problem)`` from an ROI file name.

    ``problem`` is ``None`` for a usable label; otherwise a short reason code.
    """
    stem = Path(filename).stem
    matches = list(_LABEL.finditer(stem))
    if not matches:
        return None, None, "unknown_label"
    if len({m.group(1).upper() for m in matches}) > 1:
        return None, None, "multiple_class_names"
    match = matches[-1]
    cls = match.group(1).upper()
    subtype = match.group(2).lower() if match.group(2) else None
    allowed = SUBTYPES[cls]
    if not allowed:
        return (cls, None, None) if subtype is None else (cls, subtype, "unexpected_subtype")
    if subtype is None:
        return cls, None, "missing_subtype"
    if subtype not in allowed:
        return cls, subtype, "invalid_subtype"
    return cls, subtype, None


def stage_of(text: str) -> str | None:
    lowered = text.lower()
    has_before = re.search(r"(?<![a-z])before(?![a-z])", lowered) is not None
    has_after = re.search(r"(?<![a-z])after(?![a-z])", lowered) is not None
    if has_before == has_after:
        return None
    return "before" if has_before else "after"


def image_hint(roi_stem: str, area: str | None) -> str | None:
    """Digits before the class name, e.g. ``Before an_8_1_TSD(b)`` -> ``1``.

    A leading area number is dropped. Used only as a consistency warning.
    """
    match = _LABEL.search(roi_stem)
    prefix = roi_stem[: match.start()] if match else roi_stem
    prefix = re.sub(r"area\s*\d+", "", prefix, flags=re.IGNORECASE) if area else prefix
    tokens = re.findall(r"\d+", prefix)
    if area and len(tokens) > 1 and tokens[0] == area:
        tokens = tokens[1:]
    return "_".join(tokens) or None


def image_digits(image_stem: str) -> str | None:
    tokens = re.findall(r"\d+", image_stem)
    return "_".join(tokens) or None


def hint_matches(hint: str | None, image_stem: str) -> bool | None:
    if hint is None:
        return None
    digits = image_digits(image_stem)
    if digits is None:
        return False
    return hint in {digits, digits.split("_")[0]}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_points(data: bytes) -> tuple[np.ndarray, bool]:
    """Decode a single-image POINT ROI into continuous pixel coordinates.

    ImageJ sub-pixel coordinates use the pixel-corner origin, so pixel ``i``
    spans ``[i, i + 1)`` and its centre is ``i + 0.5``. Integer-only ROIs store
    pixel indices, which are shifted by ``+0.5`` to the same convention.
    """
    if len(data) == 0:
        raise ValueError("empty_file")
    if len(data) < 64 or data[:4] != b"Iout":
        raise ValueError("not_an_imagej_roi")
    roi = ImagejRoi.frombytes(data)
    if roi.roitype != ROI_TYPE.POINT:
        raise ValueError("not_a_point_roi")
    if max(roi.position, roi.z_position, roi.t_position) > 1:
        raise ValueError("roi_refers_to_another_page")
    points = np.asarray(roi.coordinates(), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("invalid_point_coordinates")
    subpixel = bool(roi.subpixelresolution and roi.subpixel_coordinates is not None)
    if not subpixel:
        points = points + 0.5
    return points, subpixel


@dataclass
class RoiFile:
    path: Path
    relpath: str
    sha256: str
    size: int
    area: str | None
    stage: str | None
    cls: str | None = None
    subtype: str | None = None
    status: str = "pending"
    reason: str | None = None
    points: np.ndarray | None = None
    subpixel: bool | None = None
    image: Path | None = None
    image_relpath: str | None = None
    mapping_basis: str | None = None
    image_hint: str | None = None
    hint_matches_image: bool | None = None
    duplicate_of: str | None = None
    warnings: list[str] = field(default_factory=list)
    transform: str | None = None
    transform_basis: str | None = None
    probes: dict | None = None

    @property
    def fine_label(self) -> str | None:
        return fine_label(self.cls, self.subtype) if self.cls else None


def area_of(relpath: Path) -> str | None:
    for part in relpath.parts:
        match = _AREA.match(part.strip())
        if match:
            return match.group(1)
    return None


def _image_directory(roi_path: Path) -> Path:
    parent = roi_path.parent
    return parent.parent if parent.name.lower() == "roiset" else parent


def _choose_tiff(roi: RoiFile) -> tuple[Path | None, str]:
    directory = _image_directory(roi.path)
    if re.search(r"x\s*100", directory.name, re.IGNORECASE):
        return None, "x100_magnified_image_different_scale"
    tiffs = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if not tiffs:
        has_jpg = any(p.suffix.lower() in {".jpg", ".jpeg"} for p in directory.iterdir())
        return None, "no_tiff_in_directory" + ("_jpg_only" if has_jpg else "")
    if len(tiffs) > 1:
        if roi.stage is None:
            return None, "ambiguous_tiff_no_stage_in_name"
        tiffs = [p for p in tiffs if stage_of(p.stem) == roi.stage]
        if len(tiffs) != 1:
            return None, "ambiguous_tiff_for_stage"
        return tiffs[0], "same_directory_tiff_matching_stage"
    image_stage = stage_of(tiffs[0].stem)
    if roi.stage and image_stage and roi.stage != image_stage:
        return None, "roi_stage_conflicts_with_tiff"
    return tiffs[0], "same_directory_unique_tiff"


def discover(root: Path) -> list[RoiFile]:
    """Find every ``.roi`` below ``root`` (ZIP archives are not read)."""
    root = Path(root).resolve()
    files: list[RoiFile] = []
    by_hash: dict[str, RoiFile] = {}
    paths = sorted(root.rglob("*.roi"), key=lambda p: (p.parent.name.lower() == "roiset", str(p).lower()))
    for path in paths:
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        data = path.read_bytes()
        area = area_of(rel)
        stem = path.stem.strip()
        stage = stage_of(stem) or stage_of(str(rel.parent))
        roi = RoiFile(path=path, relpath=rel.as_posix(), sha256=sha256_bytes(data), size=len(data),
                      area=area, stage=stage)
        files.append(roi)
        roi.cls, roi.subtype, problem = parse_label(stem)
        if area is None:
            roi.status, roi.reason = "excluded", "no_area_folder"
            continue
        if problem:
            roi.status, roi.reason = "excluded", problem
            continue
        try:
            roi.points, roi.subpixel = decode_points(data)
        except (ValueError, IndexError, KeyError, TypeError) as error:
            roi.status, roi.reason = "excluded", str(error) or "decode_error"
            continue
        if roi.sha256 in by_hash:
            roi.status, roi.reason = "excluded", "duplicate_bytes"
            roi.duplicate_of = by_hash[roi.sha256].relpath
            continue
        by_hash[roi.sha256] = roi
        image, basis = _choose_tiff(roi)
        if image is None:
            roi.status, roi.reason = "excluded", basis
            continue
        roi.image, roi.image_relpath, roi.mapping_basis = image, image.relative_to(root).as_posix(), basis
        roi.image_hint = image_hint(stem, area)
        roi.hint_matches_image = hint_matches(roi.image_hint, image.stem)
        if roi.hint_matches_image is False:
            roi.warnings.append("image_number_in_roi_name_differs_from_tiff_name")
        roi.status = "mapped"
    _mark_coordinate_duplicates(files)
    return files


def _mark_coordinate_duplicates(files: list[RoiFile]) -> None:
    """Different bytes, same image + label + coordinates (e.g. renamed copies)."""
    def preference(roi: RoiFile):
        in_roiset = Path(roi.relpath).parent.name.lower() == "roiset"
        return (roi.hint_matches_image is False, in_roiset, roi.relpath.lower())

    canonical: dict[tuple, RoiFile] = {}
    for roi in sorted((r for r in files if r.status == "mapped"), key=preference):
        key = (roi.image_relpath, roi.fine_label, np.round(roi.points, 4).tobytes())
        if key in canonical:
            roi.status, roi.reason = "excluded", "duplicate_coordinates"
            roi.duplicate_of = canonical[key].relpath
        else:
            canonical[key] = roi
