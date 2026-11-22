"""
build_sar_npy.py - for each field in <npy>/<country>/<field>/, build sar.npy (24,2,H,W)
= Gaussian-weighted (sigma days) composite of ALL that field's co-registered S1 images,
centred on each of the 24 slot dates in dates.npy. NaN where no SAR contributes.
Aligned to the npy grid via the field's raw S2 reference tif (coreg_pairs.csv), cropped/padded
top-left to (H,W).

python build_sar_npy.py --npy npy --pairs YieldSAT_Project/data/coreg_pairs.csv --log YieldSAT_Project/data/local_export_log.csv --map "F:\\YieldSAT_Project=$PWD/YieldSAT_Project" --map "F:\\Raw=$PWD/Raw" --max-fields 2
"""
import os, re, glob, argparse, datetime as dt
import numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

RE = re.compile(r"(\d{8})T\d{6}")

def remap(p, win, lin):
    """F:\\YieldSAT_Project\\x -> /linux/root/x (Windows paths stored in the CSVs)"""
    p = str(p).replace("\\", "/")
    w = win.replace("\\", "/").rstrip("/")
    return lin.rstrip("/") + p[len(w):] if p.lower().startswith(w.lower()) else p

def s1_date(i):
    m = RE.search(str(i))
    return np.datetime64(dt.datetime.strptime(m.group(1), "%Y%m%d").date()) if m else None

def load_s1(path, bounds, rh, rw, H, W):
    with rasterio.open(path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        a = src.read([1, 2], window=win, out_shape=(2, rh, rw), boundless=True,
                     fill_value=np.nan, resampling=Resampling.nearest).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    out = np.full((2, H, W), np.nan, np.float32)
    h, w = min(H, rh), min(W, rw)
    out[:, :h, :w] = a[:, :h, :w]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--log", required=True); ap.add_argument("--sigma", type=float, default=10)
    ap.add_argument("--max-fields", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--name", default="sar.npy", help="output file name, e.g. sar_s3.npy for sigma=3")
    ap.add_argument("--map", action="append", default=[],
                    help='Windows=Linux prefix remap, repeatable, e.g. --map "F:\\YieldSAT_Project=/home/u/YieldSAT_Project" --map "F:\\Raw=/home/u/Raw"')
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs).set_index("field_id")["s2_matched_path"].to_dict()
    log = pd.read_csv(a.log)
    maps = [m.split("=", 1) for m in a.map]
    def rm(p):
        for w, l in maps: p = remap(p, w, l)
        return p
    pairs = {k: rm(v) for k, v in pairs.items()}
    log["local_path"] = log["local_path"].map(rm)
    log["d"] = log["image_id"].map(s1_date)
    by_field = {f: g.dropna(subset=["d"]) for f, g in log.groupby("field_id")}

    done = 0
    for d in sorted(glob.glob(os.path.join(a.npy, "*", "*"))):
        f = os.path.basename(d)
        out = os.path.join(d, a.name)
        if (os.path.exists(out) and not a.overwrite) or not os.path.exists(os.path.join(d, "dates.npy")):
            continue
        if f not in by_field or f not in pairs:
            print("skip (no SAR/ref):", f); continue
        H, W = np.load(os.path.join(d, "mask.npy")).shape
        slots = np.load(os.path.join(d, "dates.npy")).astype("datetime64[D]")
        with rasterio.open(pairs[f]) as r:
            bounds, rh, rw = r.bounds, r.height, r.width
        if abs(rh - H) > 3 or abs(rw - W) > 3:
            print(f"WARN {f}: ref grid {rh}x{rw} vs npy {H}x{W}")
        g = by_field[f]
        stack = np.stack([load_s1(p, bounds, rh, rw, H, W) for p in g["local_path"]])  # (N,2,H,W)
        dd = g["d"].values.astype("datetime64[D]")
        days = (slots[:, None] - dd[None, :]).astype(np.float32)          # (24,N)
        Wt = np.exp(-0.5 * (days / a.sigma) ** 2).astype(np.float32)
        valid = np.isfinite(stack).astype(np.float32)
        num = np.einsum("tn,nchw->tchw", Wt, np.nan_to_num(stack))
        den = np.einsum("tn,nchw->tchw", Wt, valid)
        comp = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
        np.save(out, comp)
        done += 1
        print(f"[{done}] {f} sar {comp.shape} from {len(g)} images, nan={np.isnan(comp).mean():.3f}")
        if a.max_fields and done >= a.max_fields:
            break

if __name__ == "__main__":
    main()