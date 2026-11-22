"""
make_splits.py - build splits/{train,val,test}/<Country>/<field>/ from npy/<Country>/<field>/

Rules
  train : EVERY Germany field
  val/test : Argentina + Brazil. Within each (country, year) group the fields are shuffled and
             split BY FIELD: val = n//2, test = n - n//2  (4 fields -> 2 val + 2 test).
             A field is never divided: all its data goes to one split only.
             A year with a single field goes to test.
Only fields that have sar.npy are used. Symlinks by default (no extra disk); --copy to copy.

python make_splits.py --npy npy --out splits
"""
import os, json, glob, shutil, random, argparse
import pandas as pd
from split_fields import require_fresh

def link(src, dst, copy):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst): return
    if copy: shutil.copytree(src, dst)
    else: os.symlink(os.path.abspath(src), dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="npy"); ap.add_argument("--out", default="splits")
    ap.add_argument("--train-countries", nargs="+", default=["Germany"])
    ap.add_argument("--eval-countries", nargs="+", default=["Argentina", "Brazil"])
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--copy", action="store_true")
    a = ap.parse_args()
    require_fresh(a.out)
    if set(a.train_countries) & set(a.eval_countries):
        ap.error("Training and evaluation countries must be disjoint")
    rng = random.Random(a.seed); rows = []

    for d in sorted(glob.glob(os.path.join(a.npy, "*", "*"))):
        mp = os.path.join(d, "meta.json")
        if not (os.path.exists(mp) and os.path.exists(os.path.join(d, "sar.npy"))): continue
        c, f = d.split(os.sep)[-2], d.split(os.sep)[-1]
        rows.append(dict(country=c, field=f, year=json.load(open(mp)).get("year"), path=d))
    if not rows:
        ap.error("No fields have metadata and sar.npy")
    df = pd.DataFrame(rows); df["split"] = ""

    df.loc[df.country.isin(a.train_countries), "split"] = "train"
    for (c, y), g in df[df.country.isin(a.eval_countries)].groupby(["country", "year"]):
        idx = list(g.index); rng.shuffle(idx)
        nv = len(idx) // 2
        df.loc[idx[:nv], "split"] = "val"; df.loc[idx[nv:], "split"] = "test"

    df = df[df.split != ""]
    if set(df["split"]) != {"train", "val", "test"}:
        ap.error("No complete train/validation/test split can be made")
    for _, r in df.iterrows():
        link(r.path, os.path.join(a.out, r.split, r.country, r.field), a.copy)
    df.drop(columns="path").to_csv(os.path.join(a.out, "split_manifest.csv"), index=False)

    print(df.groupby(["split", "country"]).size().unstack(fill_value=0))
    print("\nval/test fields per country-year:")
    print(df[df.split != "train"].groupby(["country", "year", "split"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()