"""Evidence for which pixel grid a set of point annotations was drawn on.

Two independent checks, neither of which uses labels or model scores:

* ``image_relation`` compares a same-named JPEG and TIFF at low resolution
  (identity / horizontal / vertical mirror / 180 degree rotation).
* ``contrast_ratio`` measures local contrast at annotated points relative to
  the same points shifted by fixed offsets. Annotations drawn on the image
  grid land on defects and score clearly above 1; annotations from a mirrored
  grid land on background and score close to 1.
"""

from __future__ import annotations

import numpy as np

TRANSFORMS = ("identity", "hflip", "vflip", "rot180")
CONTROL_SHIFTS = ((41, 0), (0, 41), (-41, 0), (0, -41), (29, 29), (-29, -29), (29, -29), (-29, 29))
WINDOW_RADIUS = 7


def apply_transform(array: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return array
    if name == "hflip":
        return array[:, ::-1]
    if name == "vflip":
        return array[::-1]
    if name == "rot180":
        return array[::-1, ::-1]
    raise ValueError(f"unknown transform {name}")


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else 0.0


def image_relation(tiff_small: np.ndarray, jpeg_small: np.ndarray) -> dict:
    """Correlate two low-resolution grayscale views under each transform."""
    height = min(tiff_small.shape[0], jpeg_small.shape[0])
    width = min(tiff_small.shape[1], jpeg_small.shape[1])
    scores = {}
    for name in TRANSFORMS:
        candidate = apply_transform(jpeg_small, name)
        scores[name] = round(_correlation(tiff_small[:height, :width], candidate[:height, :width]), 4)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, runner_up = ordered[0], ordered[1]
    confident = best[1] >= 0.3 and best[1] - runner_up[1] >= 0.2
    return {"correlation": scores, "best": best[0] if confident else "undetermined",
            "method": "pearson_grayscale_1_8_scale_raw_pixels_exif_ignored"}


def exact_row_agreement(reference: np.ndarray, candidate: np.ndarray, block: int = 256) -> float:
    """Fraction of rows whose every pixel is bit-identical (bounded memory)."""
    if reference.shape != candidate.shape:
        return 0.0
    equal = 0
    for start in range(0, reference.shape[0], block):
        a, b = reference[start:start + block], candidate[start:start + block]
        equal += int(np.all((a == b).reshape(len(a), -1), axis=1).sum())
    return equal / reference.shape[0]


def mirror_x(x: np.ndarray, width: int) -> np.ndarray:
    """Horizontal mirror in continuous pixel-corner coordinates."""
    return width - np.asarray(x, dtype=np.float64)


def transform_points(xy: np.ndarray, name: str, width: int, height: int) -> np.ndarray:
    """Map continuous coordinates through the same transform as ``apply_transform``."""
    if name not in TRANSFORMS:
        raise ValueError(f"unknown transform {name}")
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
    if name in {"hflip", "rot180"}:
        xy[:, 0] = width - xy[:, 0]
    if name in {"vflip", "rot180"}:
        xy[:, 1] = height - xy[:, 1]
    return xy


def _window_std(image: np.ndarray, px: np.ndarray, py: np.ndarray, radius: int) -> np.ndarray:
    offsets = np.arange(-radius, radius + 1)
    ys = py[:, None, None] + offsets[None, :, None]
    xs = px[:, None, None] + offsets[None, None, :]
    windows = image[ys, xs].astype(np.float32)
    if windows.ndim == 4:
        windows = windows.mean(axis=-1)
    return windows.reshape(len(px), -1).std(axis=1)


def contrast_ratio(image: np.ndarray, xy: np.ndarray, radius: int = WINDOW_RADIUS) -> dict:
    """Median local std at points divided by median local std at shifted controls."""
    height, width = image.shape[:2]
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    px = np.floor(xy[:, 0]).astype(np.int64)
    py = np.floor(xy[:, 1]).astype(np.int64)
    margin = radius + max(max(abs(dx), abs(dy)) for dx, dy in CONTROL_SHIFTS)
    keep = (px >= margin) & (py >= margin) & (px < width - margin) & (py < height - margin)
    px, py = px[keep], py[keep]
    if len(px) == 0:
        return {"points_used": 0, "ratio": None}
    at_points = float(np.median(_window_std(image, px, py, radius)))
    controls = np.concatenate([_window_std(image, px + dx, py + dy, radius) for dx, dy in CONTROL_SHIFTS])
    at_controls = float(np.median(controls))
    ratio = at_points / at_controls if at_controls > 0 else None
    return {"points_used": len(px), "median_std_points": round(at_points, 3),
            "median_std_controls": round(at_controls, 3),
            "ratio": None if ratio is None else round(ratio, 4)}
