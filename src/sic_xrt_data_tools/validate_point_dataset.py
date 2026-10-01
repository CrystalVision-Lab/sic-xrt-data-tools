"""Validate schema-v1 patch hashes, image metadata and group separation."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import tifffile

from .point_dataset import sha256, write_json


def validate(root: Path) -> dict:
    root = root.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1 or (root / "BUILD_FAILED.json").exists():
        raise ValueError("Unsupported or incomplete dataset")
    with (root / "기록/samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != summary["patch_count"]:
        raise ValueError("Sample count does not match summary or dataset is empty")
    sources = {item["source_id"]: item for item in json.loads(
        (root / "기록/sources.json").read_text(encoding="utf-8"))}
    counts, wafer_splits, point_splits = Counter(), defaultdict(set), defaultdict(set)
    ids, paths = set(), set()
    for row in rows:
        if row["patch_id"] in ids or row["path"] in paths:
            raise ValueError("Duplicate patch ID or path")
        ids.add(row["patch_id"])
        paths.add(row["path"])
        path = (root / row["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Patch missing or path outside dataset")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Patch hash mismatch: {row['patch_id']}")
        source = sources[row["source_id"]]
        if source["wafer"] != row["wafer"] or source["split"] != row["split"]:
            raise ValueError("Patch group/split differs from original")
        size, left, top = int(row["size"]), int(row["left"]), int(row["top"])
        if left < 0 or top < 0 or left + size > source["shape"][1] or top + size > source["shape"][0]:
            raise ValueError("Patch crop exceeds original bounds")
        with tifffile.TiffFile(path) as image:
            expected = [size, size, *source["shape"][2:]]
            if list(image.series[0].shape) != expected or str(image.pages[0].dtype) != row["dtype"]:
                raise ValueError("Patch shape/dtype mismatch")
        if row["dtype"] != source["dtype"] or row["label"] not in summary["classes"]:
            raise ValueError("Invalid class or original dtype not preserved")
        wafer_splits[row["wafer"]].add(row["split"])
        point_splits[row["point_id"]].add(row["split"])
        counts[f"{size}/{row['split']}/{row['label']}"] += 1
    if any(len(splits) != 1 for splits in [*wafer_splits.values(), *point_splits.values()]):
        raise ValueError("Wafer/point leaked across splits")
    if dict(counts) != summary["counts"]:
        raise ValueError("Class counts differ from summary")
    missing = {split: [label for label in summary["classes"]
                       if not any(r["split"] == split and r["label"] == label for r in rows)]
               for split in ("train", "val", "test")}
    return {"status": "passed", "files_checked": len(rows), "all_patch_hashes": "passed",
            "shape_dtype": "passed", "wafer_point_split_separation": "passed",
            "missing_classes_by_split": missing, "semantic_visual_review": "pending"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = validate(args.root)
    write_json(args.root / "validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
