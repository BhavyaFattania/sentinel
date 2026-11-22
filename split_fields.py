"""Validate the on-disk cohort before training or comparing models."""

import csv
from pathlib import Path

SPLITS = ("train", "val", "test")
AFF_FILES = ("s2.npy", "weather.npy", "dem.npy", "soil.npy", "coord.npy", "yield.npy", "mask.npy")


def require_fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Split output directory is not empty: {path}. Choose a fresh directory.")


def write_manifest(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("country", "split", "field"))
        writer.writeheader()
        writer.writerows(rows)


def load_split(root, required):
    root = Path(root)
    seen = set()
    resolved = set()
    actual = set()
    sets = []
    for split in SPLITS:
        paths = sorted(p for p in (root / split).glob("*/*") if p.is_dir())
        if not paths:
            raise ValueError(f"Empty {split} split: {root}")
        for path in paths:
            key = (path.parent.name, path.name)
            source = path.resolve()
            if key in seen or source in resolved:
                raise ValueError(f"Field occurs more than once across splits: {'/'.join(key)}")
            seen.add(key)
            resolved.add(source)
            actual.add((key[0], split, key[1]))
            missing = [name for name in required if not (path / name).is_file()]
            if missing:
                raise ValueError(f"{path} is missing required arrays: {missing}")
        sets.append([str(path) for path in paths])
    manifest = root / "split_manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as f:
            rows = [(r["country"], r["split"], r["field"]) for r in csv.DictReader(f)]
        if len(rows) != len(set(rows)) or set(rows) != actual:
            raise ValueError(f"{manifest} does not match the fields on disk")
    return tuple(sets)


def field_ids(paths):
    return sorted(f"{Path(path).parent.name}/{Path(path).name}" for path in paths)
