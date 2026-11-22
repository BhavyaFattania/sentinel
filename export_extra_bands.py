"""
export_extra_bands.py - add the non-S2 YieldSAT bands to each existing field folder in npy/<Country>/<field>/
    weather.npy (24,4)  temp_max, temp_mean, temp_min, total_prec   (same for all pixels of a field)
    dem.npy     (5,H,W) aspect, curvature, dem, slope, twi
    soil.npy    (S,H,W) soil props x depths (default 8 props x 3 depths 0-5,5-15,15-30 = 24, value only)
    coord.npy   (3,H,W) coord_x, coord_y, coord_z
    extra_bands.json    band names used (so the choice is documented)
Grid = same row/col as s2.npy. Soil list is an ASSUMPTION that gives 24 channels - confirm with the BTech.

python export_extra_bands.py "Preprocessed/Argentina/merge_s2-soil-dem-weather-coords.nc" --npy npy
"""
import os, json, argparse
import numpy as np, xarray as xr

WEATHER = ["temp_max", "temp_mean", "temp_min", "total_prec"]
DEM = ["aspect", "curvature", "dem", "slope", "twi"]
COORD = ["coord_x", "coord_y", "coord_z"]
PROPS = ["cec", "cfvo", "clay", "nitrogen", "phh2o", "sand", "silt", "soc"]

def decode(da):
    return da.values, {int(k): v for k, v in da.attrs.items() if str(k).lstrip("-").isdigit()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("nc"); ap.add_argument("--npy", default="npy")
    ap.add_argument("--soil-depths", nargs="+", default=["0-5", "5-15", "15-30"])
    ap.add_argument("--soil-uncertainty", action="store_true"); ap.add_argument("--max-fields", type=int, default=None)
    a = ap.parse_args()

    soil = [f"{p}_{d}" for p in PROPS for d in a.soil_depths]
    if a.soil_uncertainty: soil += [f"{p}_{d}_uncertainty" for p in PROPS for d in a.soil_depths]
    ds = xr.open_dataset(a.nc); bands = [str(b) for b in ds["band"].values]
    wanted = WEATHER + DEM + soil + COORD
    miss = [b for b in wanted if b not in bands]
    assert not miss, f"bands not found in .nc: {miss}"
    ib = [bands.index(b) for b in wanted]
    nw, nd, ns = len(WEATHER), len(DEM), len(soil)

    country = os.path.basename(os.path.dirname(os.path.abspath(a.nc)))
    fc, fmap = decode(ds["field_shared_name"]); code_of = {n: c for c, n in fmap.items()}
    row, col = ds["row"].values.astype(np.int64), ds["col"].values.astype(np.int64)
    order = np.argsort(fc, kind="stable"); groups = {int(fc[g[0]]): g for g in np.split(order, np.flatnonzero(np.diff(fc[order])) + 1)}

    base = os.path.join(a.npy, country); done = 0
    for f in sorted(os.listdir(base)):
        d = os.path.join(base, f)
        if not os.path.exists(os.path.join(d, "mask.npy")) or f not in code_of: continue
        if os.path.exists(os.path.join(d, "weather.npy")): continue
        idx = np.sort(groups[code_of[f]])
        sel = slice(int(idx[0]), int(idx[-1]) + 1) if idx[-1] - idx[0] + 1 == len(idx) else idx
        x = ds["sample"].isel(index=sel, band=ib).values.astype(np.float32)        # (N,24,B)
        H, W = np.load(os.path.join(d, "mask.npy")).shape
        r, c = row[idx], col[idx]
        weather = np.nanmean(x[:, :, :nw], axis=0)                                  # (24,4)
        stat = x[:, 0, nw:]                                                         # (N, nd+ns+3)
        grid = np.full((stat.shape[1], H, W), np.nan, np.float32); grid[:, r, c] = stat.T
        np.save(os.path.join(d, "weather.npy"), weather)
        np.save(os.path.join(d, "dem.npy"), grid[:nd]); np.save(os.path.join(d, "soil.npy"), grid[nd:nd + ns])
        np.save(os.path.join(d, "coord.npy"), grid[nd + ns:])
        done += 1
        if done == 1:
            json.dump(dict(weather=WEATHER, dem=DEM, soil=soil, coord=COORD), open(os.path.join(a.npy, "extra_bands.json"), "w"), indent=1)
            print("first field", f, "weather range", np.nanmin(weather), np.nanmax(weather), "| dem range", np.nanmin(grid[:nd]), np.nanmax(grid[:nd]))
        if done % 20 == 0: print(done, "fields", flush=True)
        if a.max_fields and done >= a.max_fields: break
    print("done", done, "fields in", country)

if __name__ == "__main__":
    main()