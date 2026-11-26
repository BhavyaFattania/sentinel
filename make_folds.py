"""
make_folds.py - per-country, field-level train/val/test folders using ALL fields that have sar.npy.

folds/<Country>/fold_<k>/{train,val,test}/<Country>/<field>/   (symlinks; --copy to copy)

--scheme repeated (default): K independent random 70/15/15 splits per country (seed+k).
--scheme cv                : true K-fold CV, test=fold k, val=fold k+1, train=rest (80/10/10 for K=10).
A field is in exactly one split. --group-by farm keeps a farm together.

python make_folds.py --npy npy --out folds --k 10
"""

import os, glob, shutil, random, argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from split_fields import AFF_FILES, require_fresh, write_manifest


def link(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    shutil.copytree(src, dst) if copy else os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="npy")
    ap.add_argument("--out", default="folds")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--scheme", choices=["repeated", "cv"], default="repeated")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument(
        "--s2-only", action="store_true", help="build optical/AFF pilot folds without radar"
    )
    ap.add_argument("--sar-file", default="sar.npy")
    ap.add_argument("--group-by", choices=["field", "farm"], default="field")
    a = ap.parse_args()
    if a.k < 1 or (a.scheme == "cv" and a.k < 3):
        ap.error("Use at least three folds for CV, or one for repeated holdout")
    if min(a.val_frac, a.test_frac) <= 0 or a.val_frac + a.test_frac >= 1:
        ap.error("Validation and test fractions must be positive and sum to less than one")
    require_fresh(a.out)

    by = {}
    for d in sorted(glob.glob(os.path.join(a.npy, "*", "*"))):
        names = AFF_FILES + (() if a.s2_only else (a.sar_file,))
        if all((Path(d) / name).is_file() for name in names):
            by.setdefault(Path(d).parent.name, []).append(d)
    if not by:
        ap.error("No complete fields found")

    rows = []
    folds = []
    for c, ds in by.items():
        if a.group_by == "farm":
            groups = {}
            for d in ds:
                meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
                key = (meta.get("data_provider"), meta["farm_identifier"])
                groups.setdefault(key, []).append(d)
            ds = list(groups.values())
        else:
            ds = [[d] for d in ds]
        n = len(ds)
        if n < 3 or (a.scheme == "cv" and n < a.k):
            ap.error(f"{c} has {n} groups, too few for the requested split")
        base = random.Random(a.seed)
        order = ds[:]
        base.shuffle(order)
        cv_folds = np.array_split(np.arange(n), a.k) if a.scheme == "cv" else None
        for k in range(a.k):
            if a.scheme == "repeated":
                o = ds[:]
                random.Random(a.seed + k).shuffle(o)
                nv, nt = max(1, round(a.val_frac * n)), max(1, round(a.test_frac * n))
                parts = dict(val=o[:nv], test=o[nv : nv + nt], train=o[nv + nt :])
            else:
                te = set(cv_folds[k])
                va = set(cv_folds[(k + 1) % a.k])
                parts = dict(
                    test=[order[i] for i in sorted(te)],
                    val=[order[i] for i in sorted(va)],
                    train=[order[i] for i in range(n) if i not in te and i not in va],
                )
            if any(not groups for groups in parts.values()):
                ap.error(f"{c} fold {k} has an empty split")
            folds.append((c, k, parts))
    for c, k, parts in folds:
        root = Path(a.out) / c / f"fold_{k}"
        local_rows = []
        for split, groups in parts.items():
            for group in groups:
                for d in group:
                    field = Path(d).name
                    link(d, root / split / c / field, a.copy)
                    local_rows.append(dict(country=c, split=split, field=field))
        write_manifest(root / "split_manifest.csv", local_rows)
        rows.extend(dict(row, fold=k) for row in local_rows)
        counts = {
            split: sum(row["split"] == split for row in local_rows)
            for split in ("train", "val", "test")
        }
        print(f"{c} fold {k}: {counts}")
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fold_manifest.csv"), index=False)


if __name__ == "__main__":
    main()
