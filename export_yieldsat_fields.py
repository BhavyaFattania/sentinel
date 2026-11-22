"""Export field arrays from one YieldSAT NetCDF using its original raster grid."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import netCDF4
import numpy as np
import rasterio

from export_extra_bands import WEATHER, DEM, COORD, PROPS

S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]
SOIL = [f"{prop}_{depth}" for prop in PROPS for depth in ("0-5", "5-15", "15-30")]


def as_float(array):
    return np.ma.asarray(array, dtype=np.float32).filled(np.nan)


def field_dates(ds, idx):
    times = ds["times"]
    values = np.ma.asarray(times[idx], dtype=np.float64).filled(np.nan)
    dates = np.full(values.shape[1], np.datetime64("NaT", "D"))
    for slot, column in enumerate(values.T):
        unique = np.unique(column[np.isfinite(column)])
        if len(unique) > 1:
            raise ValueError(f"Multiple optical dates in slot {slot}")
        if len(unique):
            date = netCDF4.num2date(unique[0], times.units, times.calendar)
            dates[slot] = np.datetime64(date.strftime("%Y-%m-%d"))
    return dates


def export_field(ds, idx, band_indices, raw, destination, meta):
    reference = next(iter(sorted((raw / "s2_images").glob("*.tif"))))
    yield_ref = next(iter(sorted((raw / "yield_masks").glob("mean*.tif"))))
    with rasterio.open(reference) as src, rasterio.open(yield_ref) as labels:
        shape = src.shape
        if (shape, src.crs, src.transform) != (labels.shape, labels.crs, labels.transform):
            raise ValueError(f"Optical and yield grids differ for {raw.name}")
        target = labels.read(1, masked=True).filled(np.nan)
    rows = ds["row"][idx].astype(np.intp)
    cols = ds["col"][idx].astype(np.intp)
    y = as_float(ds["target"][idx])
    if len(np.unique(rows * shape[1] + cols)) != len(rows):
        raise ValueError(f"Duplicate pixel indices for {raw.name}")
    if not np.allclose(target[rows, cols], y, equal_nan=True):
        raise ValueError(f"NetCDF targets disagree with the raw yield raster for {raw.name}")

    x = as_float(ds["sample"][idx, :, band_indices])
    dates = field_dates(ds, idx)
    t = x.shape[1]
    n_s2, n_weather = len(S2), len(WEATHER)
    static = x[:, 0, n_s2 + n_weather:]
    # Static data are repeated in the source. Allow the first slot to be missing.
    for slot in range(1, t):
        missing = ~np.isfinite(static)
        if not missing.any():
            break
        static[missing] = x[:, slot, n_s2 + n_weather:][missing]
    weather_values = x[:, :, n_s2:n_s2 + n_weather]
    count = np.isfinite(weather_values).sum(axis=0)
    weather = np.divide(np.nansum(weather_values, axis=0), count,
                        out=np.full(count.shape, np.nan, np.float32), where=count > 0)
    s2 = np.full((t, n_s2, *shape), np.nan, np.float32)
    s2[:, :, rows, cols] = x[:, :, :n_s2].transpose(1, 2, 0)
    grids = np.full((static.shape[1], *shape), np.nan, np.float32)
    grids[:, rows, cols] = static.T
    dem, soil, coord = np.split(grids, [len(DEM), len(DEM) + len(SOIL)])
    ymap = np.full(shape, np.nan, np.float32)
    ymap[rows, cols] = y
    mask = np.zeros(shape, bool)
    mask[rows, cols] = True
    # A slot centre is distinct from the selected optical acquisition date.
    harvest_year = int(meta["year"])
    start = np.datetime64(f"{harvest_year - 1}-01", "M")
    months = start + np.arange(t).astype("timedelta64[M]")
    month_start = months.astype("datetime64[D]")
    month_end = (months + np.timedelta64(1, "M")).astype("datetime64[D]")
    sow = np.datetime64(datetime.strptime(meta["seeding_date"], "%d.%m.%Y").date())
    harvest = np.datetime64(datetime.strptime(meta["harvesting_date"], "%d.%m.%Y").date())
    season = (month_end > sow) & (month_start <= harvest)
    centers = month_start + (month_end - month_start) // 2

    destination.mkdir(parents=True)
    for name, array in dict(s2=s2, weather=weather, dem=dem, soil=soil, coord=coord,
                            dates=dates, slot_dates=centers, season=season, mask=mask, yield_=ymap).items():
        np.save(destination / ("yield.npy" if name == "yield_" else f"{name}.npy"), array)
    meta["s2_reference"] = str(reference.resolve())
    meta["bands"] = dict(s2=S2, weather=WEATHER, dem=DEM, soil=SOIL, coord=COORD)
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{raw.name}: {shape}, labeled={np.isfinite(y).sum()}, optical dates={(~np.isnat(dates)).sum()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nc", type=Path)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop")
    parser.add_argument("--year", type=int)
    parser.add_argument("--max-fields", type=int)
    args = parser.parse_args()
    country = args.nc.parent.name
    with netCDF4.Dataset(args.nc) as ds:
        bands = list(ds["band"][:])
        wanted = S2 + WEATHER + DEM + SOIL + COORD
        missing = set(wanted) - set(bands)
        if missing:
            parser.error(f"Missing NetCDF bands: {sorted(missing)}")
        band_indices = [bands.index(b) for b in wanted]
        field = ds["field_shared_name"]
        codes = np.asarray(field[:])
        order = np.argsort(codes, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
        done = 0
        for start, end in zip(bounds[:-1], bounds[1:]):
            idx = order[start:end]
            name = field.getncattr(str(int(codes[idx[0]])))
            raw = args.raw / country / name
            if args.crop and f"_{args.crop}_" not in name:
                continue
            if args.year and not name.endswith(f"_{args.year}"):
                continue
            metadata = raw / f"metadata-{name}.json"
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            dest = args.out / country / name
            if dest.exists():
                raise FileExistsError(f"Use a fresh output directory: {dest}")
            selection = slice(int(idx[0]), int(idx[-1]) + 1) if idx[-1] - idx[0] + 1 == len(idx) else idx
            export_field(ds, selection, band_indices, raw, dest, meta)
            done += 1
            if args.max_fields is not None and done >= args.max_fields:
                break
    if not done:
        parser.error("No matching fields")


if __name__ == "__main__":
    main()
