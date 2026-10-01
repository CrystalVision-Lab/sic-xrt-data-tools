"""Generate deterministic, class-balanced training CSV views without copying pixels."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .point_dataset import write_csv, write_json


def create_subsets(root: Path, seed: int = 42) -> dict:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    with (root / "기록/samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {"seed": seed, "purpose": "optional_training_only_views", "sizes": {}}
    for size in summary["patch_sizes"]:
        groups = defaultdict(list)
        for row in rows:
            if row["split"] == "train" and int(row["size"]) == size:
                groups[row["label"]].append(row)
        minimum = min((len(groups[label]) for label in summary["classes"]), default=0)
        if minimum == 0:
            raise ValueError("Every class needs training examples for a balanced view")
        selected = []
        generator = random.Random(seed)
        for label in summary["classes"]:
            selected.extend(generator.sample(sorted(groups[label], key=lambda r: r["patch_id"]), minimum))
        selected.sort(key=lambda row: row["patch_id"])
        name = f"train_balanced_{size}.csv"
        write_csv(root / "기록" / name, selected, list(rows[0]))
        result["sizes"][str(size)] = {"csv": f"기록/{name}", "per_class": minimum,
                                     "total": len(selected), "split": "train"}
    write_json(root / "balanced_subsets.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_subsets(args.root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
