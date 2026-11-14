"""Read and validate fixed, directory-grouped train/validation manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("image", "annotation", "group"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"{path}:{line_number}: missing non-empty '{key}'")
            records.append(record)
    return records


def _absolute_path(data_root: Path, relative_path: str) -> str:
    if os.path.isabs(relative_path):
        raise ValueError(f"Manifest paths must be relative: {relative_path}")
    path = (data_root / relative_path).resolve(strict=False)
    lexical = os.path.abspath(os.path.join(str(data_root), relative_path))
    if os.path.commonpath([str(data_root), lexical]) != str(data_root):
        raise ValueError(f"Manifest path escapes data root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file is missing: {path}")
    return str(path)


def load_manifest_pairs(data_root: str, manifest_dir: str, mode: str) -> List[Tuple[str, str]]:
    """Load one fixed split and assert train/val are path- and group-disjoint."""
    root = Path(data_root).resolve()
    manifest = Path(manifest_dir).resolve()
    meta_path = manifest / "meta.json"
    train_path = manifest / "train.jsonl"
    val_path = manifest / "val.jsonl"
    for required in (meta_path, train_path, val_path):
        if not required.is_file():
            raise FileNotFoundError(f"Incomplete split manifest, missing: {required}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    expected_root = meta.get("data_root_real")
    if expected_root and os.path.realpath(expected_root) != os.path.realpath(root):
        raise ValueError(
            f"Manifest data root mismatch: manifest={expected_root}, requested={root}"
        )

    train_records = _read_jsonl(train_path)
    val_records = _read_jsonl(val_path)
    train_groups = {record["group"] for record in train_records}
    val_groups = {record["group"] for record in val_records}
    group_overlap = train_groups & val_groups
    if group_overlap:
        example = sorted(group_overlap)[0]
        raise ValueError(f"Train/val group overlap detected: {example}")

    def identity(record: Dict[str, str]) -> Tuple[str, str]:
        return (
            os.path.realpath(root / record["image"]),
            os.path.realpath(root / record["annotation"]),
        )

    train_ids = {identity(record) for record in train_records}
    val_ids = {identity(record) for record in val_records}
    pair_overlap = train_ids & val_ids
    if pair_overlap:
        raise ValueError(f"Train/val file overlap detected: {next(iter(pair_overlap))[0]}")

    if mode == "train":
        selected = train_records
    elif mode == "val":
        selected = val_records
    elif mode == "all":
        selected = train_records + val_records
    else:
        raise ValueError(f"Unsupported dataset mode for manifest: {mode}")

    pairs = [
        (_absolute_path(root, record["image"]), _absolute_path(root, record["annotation"]))
        for record in selected
    ]
    print(
        f"[split-manifest] mode={mode}, pairs={len(pairs)}, "
        f"train_groups={len(train_groups)}, val_groups={len(val_groups)}, manifest={manifest}"
    )
    return pairs
