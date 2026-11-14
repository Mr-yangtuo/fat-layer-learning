#!/usr/bin/env python3
"""Generate a fixed 90/10 split while keeping every images/annotations directory intact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from LineDet.dataset.split_manifest import load_manifest_pairs
from LineDet.dataset.utils import find_image_annotation_pairs


Pair = Tuple[str, str]


def choose_validation_groups(group_sizes: Dict[str, int], target: int, seed: int) -> set[str]:
    """Choose a deterministic subset whose number of samples is closest to target."""
    keys = sorted(group_sizes)
    random.Random(seed).shuffle(keys)
    max_group = max(group_sizes.values())
    cap = min(sum(group_sizes.values()) - 1, target + max_group)
    parents: Dict[int, Tuple[int, str] | None] = {0: None}
    for key in keys:
        size = group_sizes[key]
        for current in sorted(list(parents), reverse=True):
            new_sum = current + size
            if new_sum <= cap and new_sum not in parents:
                parents[new_sum] = (current, key)
    candidates = [value for value in parents if 0 < value < sum(group_sizes.values())]
    if not candidates:
        raise ValueError("At least two non-empty directory groups are required")
    best = min(candidates, key=lambda value: (abs(value - target), value < target, value))
    selected: set[str] = set()
    cursor = best
    while cursor:
        parent = parents[cursor]
        if parent is None:
            break
        cursor, key = parent
        selected.add(key)
    return selected


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def manifest_digest(records: Sequence[dict]) -> str:
    payload = "\n".join(
        f"{record['split']}\t{record['group']}\t{record['image']}\t{record['annotation']}"
        for record in sorted(records, key=lambda item: (item["split"], item["image"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite fixed manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)

    pairs = sorted(find_image_annotation_pairs(str(root)), key=lambda pair: pair[0])
    if not pairs:
        raise ValueError(f"No valid image/annotation pairs found under {root}")

    grouped: Dict[str, List[Pair]] = defaultdict(list)
    for image_path, annotation_path in pairs:
        image = Path(image_path).resolve()
        annotation = Path(annotation_path).resolve()
        group_path = image.parent.parent
        group = os.path.relpath(group_path, root)
        grouped[group].append((str(image), str(annotation)))

    target = max(1, round(len(pairs) * args.val_ratio))
    val_groups = choose_validation_groups(
        {group: len(items) for group, items in grouped.items()}, target, args.seed
    )

    train_records: List[dict] = []
    val_records: List[dict] = []
    for group in sorted(grouped):
        split = "val" if group in val_groups else "train"
        destination = val_records if split == "val" else train_records
        for image_path, annotation_path in sorted(grouped[group]):
            destination.append({
                "split": split,
                "group": group,
                "image": os.path.relpath(image_path, root),
                "annotation": os.path.relpath(annotation_path, root),
            })

    all_records = train_records + val_records
    write_jsonl(output / "train.jsonl", train_records)
    write_jsonl(output / "val.jsonl", val_records)
    meta = {
        "schema_version": 1,
        "name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": args.data_root,
        "data_root_real": str(root),
        "seed": args.seed,
        "requested_val_ratio": args.val_ratio,
        "total_pairs": len(pairs),
        "train_pairs": len(train_records),
        "val_pairs": len(val_records),
        "actual_val_ratio": len(val_records) / len(pairs),
        "total_groups": len(grouped),
        "train_groups": len(grouped) - len(val_groups),
        "val_groups": len(val_groups),
        "group_definition": "parent directory containing images/ and annotations/",
        "manifest_sha256": manifest_digest(all_records),
    }
    with (output / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)

    loaded_train = load_manifest_pairs(str(root), str(output), "train")
    loaded_val = load_manifest_pairs(str(root), str(output), "val")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Validated fixed split: train={len(loaded_train)}, val={len(loaded_val)}, overlap=0")


if __name__ == "__main__":
    main()
