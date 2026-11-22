"""
make_folds.py - per-country, field-level train/val/test folders using ALL fields that have sar.npy.

folds/<Country>/fold_<k>/{train,val,test}/<Country>/<field>/   (symlinks; --copy to copy)

--scheme repeated (default): K independent random 70/15/15 splits per country (seed+k).
--scheme cv                : true K-fold CV, test=fold k, val=fold k+1, train=rest (80/10/10 for K=10).
Splits are by FIELD only (no year/region/farm grouping): a field is in exactly one split.

python make_folds.py --npy npy --out folds --k 10
"""
import os, glob, shutil, random, argparse
import numpy as np, pandas as pd

def link(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst): return
    shutil.copytree(src, dst) if copy else os.symlink(os.path.abspath(src), dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="npy"); ap.add_argument("--out", default="folds")
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--scheme", choices=["repeated", "cv"], default="repeated")
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--copy", action="store_true")
    a = ap.parse_args()

    by = {}
    for d in sorted(glob.glob(os.path.join(a.npy, "*", "*"))):
        if os.path.exists(os.path.join(d, "sar.npy")):
            by.setdefault(d.split(os.sep)[-2], []).append(d)

    rows = []
    for c, ds in by.items():
        n = len(ds)
        base = random.Random(a.seed); order = ds[:]; base.shuffle(order)
        cv_folds = np.array_split(np.arange(n), a.k) if a.scheme == "cv" else None
        for k in range(a.k):
            if a.scheme == "repeated":
                o = ds[:]; random.Random(a.seed + k).shuffle(o)
                nv, nt = max(1, round(a.val_frac * n)), max(1, round(a.test_frac * n))
                parts = dict(val=o[:nv], test=o[nv:nv + nt], train=o[nv + nt:])
            else:
                te = set(cv_folds[k]); va = set(cv_folds[(k + 1) % a.k])
                parts = dict(test=[order[i] for i in sorted(te)], val=[order[i] for i in sorted(va)],
                             train=[order[i] for i in range(n) if i not in te and i not in va])
            for sp, lst in parts.items():
                for d in lst:
                    link(d, os.path.join(a.out, c, f"fold_{k}", sp, c, os.path.basename(d)), a.copy)
                    rows.append(dict(country=c, fold=k, split=sp, field=os.path.basename(d)))
        p0 = {sp: sum(1 for r in rows if r["country"] == c and r["fold"] == 0 and r["split"] == sp) for sp in ("train", "val", "test")}
        print(f"{c}: {n} fields -> per fold train/val/test = {p0['train']}/{p0['val']}/{p0['test']}  ({a.k} x {a.scheme})")
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "fold_manifest.csv"), index=False)

if __name__ == "__main__":
    main()