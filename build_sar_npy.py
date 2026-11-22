"""
build_sar_npy.py - for each field in <npy>/<country>/<field>/, build sar.npy (24,2,H,W)
= Gaussian-weighted (sigma days) composite of ALL that field's co-registered S1 images,
centred on each of the 24 slot dates in dates.npy. NaN where no SAR contributes.
Warped directly to the field's matching raw S2 reference grid from coreg_pairs.csv.

python build_sar_npy.py --npy npy --pairs YieldSAT_Project/data/coreg_pairs.csv --log YieldSAT_Project/data/local_export_log.csv --map "F:\\YieldSAT_Project=$PWD/YieldSAT_Project" --map "F:\\Raw=$PWD/Raw" --max-fields 2
"""
import os, re, glob, argparse, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.vrt import WarpedVRT
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

def load_s1(path, grid):
    with rasterio.open(path) as src, WarpedVRT(
        src, **grid, dtype="float32", nodata=np.nan, resampling=Resampling.nearest
    ) as aligned:
        return aligned.read([1, 2], masked=True).filled(np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--log", required=True); ap.add_argument("--sigma", type=float, default=10)
    ap.add_argument("--max-fields", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--name", default="sar.npy", help="output file name, e.g. sar_s3.npy for sigma=3")
    ap.add_argument("--map", action="append", default=[],
                    help='Windows=Linux prefix remap, repeatable, e.g. --map "F:\\YieldSAT_Project=/home/u/YieldSAT_Project" --map "F:\\Raw=/home/u/Raw"')
    ap.add_argument("--path-column", default="local_path", help="column containing the TIFFs to use")
    ap.add_argument("--input-db", action="store_true", help="combine GEE dB inputs in power before returning dB")
    a = ap.parse_args()
    if not np.isfinite(a.sigma) or a.sigma <= 0:
        ap.error("--sigma must be finite and positive")

    pairs = pd.read_csv(a.pairs).set_index("field_id")["s2_matched_path"].to_dict()
    log = pd.read_csv(a.log)
    maps = [m.split("=", 1) for m in a.map]
    def rm(p):
        for w, l in maps: p = remap(p, w, l)
        return p
    pairs = {k: rm(v) for k, v in pairs.items()}
    log = log.dropna(subset=[a.path_column]).copy()
    log[a.path_column] = log[a.path_column].map(rm)
    log["d"] = log["image_id"].map(s1_date)
    by_field = dict(tuple(log.dropna(subset=["d"]).groupby("field_id")))

    done = 0
    for d in sorted(glob.glob(os.path.join(a.npy, "*", "*"))):
        f = os.path.basename(d)
        out = os.path.join(d, a.name)
        if (os.path.exists(out) and not a.overwrite) or not os.path.exists(os.path.join(d, "dates.npy")):
            continue
        if f not in by_field or f not in pairs:
            print("skip (no SAR/ref):", f); continue
        field = Path(d)
        shape = np.load(field / "mask.npy").shape
        slots = np.load(field / "dates.npy").astype("datetime64[D]")
        if (field / "slot_dates.npy").is_file():
            slot_dates = np.load(field / "slot_dates.npy").astype("datetime64[D]")
            slots = np.where(np.isnat(slots), slot_dates, slots)
        if (field / "season.npy").is_file():
            slots = np.where(np.load(field / "season.npy"), slots, np.datetime64("NaT", "D"))
        with rasterio.open(pairs[f]) as ref:
            if ref.shape != shape:
                raise ValueError(f"{f}: reference shape {ref.shape} differs from exported mask {shape}")
            grid = dict(crs=ref.crs, transform=ref.transform, height=ref.height, width=ref.width)
        g = by_field[f]
        stack = np.stack([load_s1(p, grid) for p in g[a.path_column]])
        if a.input_db:
            stack = np.power(10, stack / 10)
        dates = g["d"].values.astype("datetime64[D]")
        days = (slots[:, None] - dates[None, :]).astype(np.float32)
        weights = np.exp(-0.5 * (days / a.sigma) ** 2).astype(np.float32)
        weights[np.isnat(slots)] = 0
        valid = np.isfinite(stack)
        num = np.einsum("tn,nchw->tchw", weights, np.where(valid, stack, 0))
        den = np.einsum("tn,nchw->tchw", weights, valid.astype(np.float32))
        comp = np.divide(num, den, out=np.full(num.shape, np.nan, np.float32), where=den > 1e-6)
        if a.input_db:
            comp = 10 * np.log10(comp, where=comp > 0, out=np.full_like(comp, np.nan))
        temp = field / (Path(out).name + ".partial.npy")
        np.save(temp, comp)
        temp.replace(out)
        done += 1
        print(f"[{done}] {f} sar {comp.shape} from {len(g)} images, nan={np.isnan(comp).mean():.3f}")
        if a.max_fields and done >= a.max_fields:
            break

if __name__ == "__main__":
    main()