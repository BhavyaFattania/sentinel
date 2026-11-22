"""Build pilot VV/VH arrays on the YieldSAT grid from public Sentinel-1 RTC COGs."""

import argparse
from collections import defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

API = "https://planetarycomputer.microsoft.com/api"
COLLECTION = "sentinel-1-rtc"
TOKENS = {}
TOKEN_LOCK = Lock()


def get_json(url):
    for delay in (2, 5, 10, None):
        try:
            with urlopen(url, timeout=25) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or delay is None:
                endpoint = "SAS" if "/sas/" in url else "STAC"
                raise RuntimeError(f"Planetary Computer {endpoint} HTTP {exc.code}") from None
            retry_after = exc.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 60))
        except (TimeoutError, URLError) as exc:
            if delay is None:
                raise RuntimeError(f"Planetary Computer connection failed: {exc.reason if isinstance(exc, URLError) else exc}") from None
        time.sleep(delay)


def sign_asset(href):
    parts = urlsplit(href)
    account = parts.hostname.removesuffix(".blob.core.windows.net")
    container = parts.path.split("/")[1]
    key = (account, container)
    with TOKEN_LOCK:
        entry = TOKENS.get(key)
        if entry is None or entry[0] <= datetime.now(timezone.utc) + timedelta(minutes=1):
            data = get_json(f"{API}/sas/v1/token/{account}/{container}")
            entry = datetime.fromisoformat(data["msft:expiry"]), data["token"]
            TOKENS[key] = entry
    return f"{href}?{entry[1]}"


def search(bounds, sow, harvest):
    params = {"bbox": ",".join(map(str, bounds)), "limit": 100,
              "datetime": f"{sow}T00:00:00Z/{harvest}T23:59:59Z"}
    url = f"{API}/stac/v1/collections/{COLLECTION}/items?{urlencode(params)}"
    while url:
        page = get_json(url)
        yield from page["features"]
        url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)


def read_backscatter(item, grid):
    channels = []
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="20", GDAL_HTTP_MAX_RETRY="2"):
        for pol in ("vv", "vh"):
            signed = sign_asset(item["assets"][pol]["href"])
            try:
                with rasterio.open(signed) as src, WarpedVRT(src, **grid, nodata=np.nan) as aligned:
                    channels.append(aligned.read(1,masked=True).filled(np.nan))
            except rasterio.errors.RasterioError:
                raise RuntimeError(f"{item['id']} {pol} raster could not be read") from None
    return np.stack(channels).astype(np.float32)


def build(field):
    radar_file = field / "sar.npy"
    provenance_file = field / "sar_sources.json"
    if radar_file.is_file() and provenance_file.is_file():
        return field.name, "already built"
    meta = json.loads((field / "meta.json").read_text(encoding="utf-8"))
    sow, harvest = [np.datetime64(datetime.strptime(meta[name], "%d.%m.%Y").date())
                    for name in ("seeding_date", "harvesting_date")]
    with rasterio.open(meta["s2_reference"]) as ref:
        bounds = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
        grid = dict(crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height)
    shape = (grid["height"], grid["width"])
    support = np.load(field / "mask.npy")
    optical_dates = np.load(field / "dates.npy")
    slot_dates = np.load(field / "slot_dates.npy")
    season = np.load(field / "season.npy")
    data = np.full((len(slot_dates), 2, *shape), np.nan, np.float32)
    groups = defaultdict(list)
    for item in search(bounds, sow, harvest):
        p = item["properties"]
        orbit = (p.get("sat:orbit_state"), p.get("sat:relative_orbit"))
        if None not in orbit and {"vv", "vh"} <= item["assets"].keys():
            date = np.datetime64(p["datetime"][:10])
            groups[orbit].append((date, item))
    if not groups:
        raise ValueError(f"No dual-polarization RTC items for {field.name}")
    orbit, scenes = max(sorted(groups.items()), key=lambda entry: len({d for d, _ in entry[1]}))
    sources = []
    cache = {}
    for slot in np.flatnonzero(season):
        center = optical_dates[slot] if not np.isnat(optical_dates[slot]) else slot_dates[slot]
        center = max(sow, min(harvest, center))
        nearby = sorted(scenes, key=lambda pair: (abs(pair[0] - center), pair[1]["id"]))
        for date, scene in nearby:
            days = int((date - center) / np.timedelta64(1, "D"))
            if abs(days) > 12:
                break
            if scene["id"] not in cache:
                power = read_backscatter(scene, grid)
                power[~np.isfinite(power) | (power <= 0)] = np.nan
                cache[scene["id"]] = power
            power = cache[scene["id"]]
            coverage = np.isfinite(power[:,support]).all(axis=0).mean()
            if coverage < 0.95:
                continue
            data[slot] = power
            sources.append(dict(slot=int(slot), target_date=str(center), date=str(date),
                                item=scene["id"], labeled_coverage=float(coverage)))
            break
    if not sources:
        raise ValueError(f"No RTC slots with adequate coverage for {field.name}")
    temp = field / "sar.partial.npy"
    np.save(temp, data)
    temp.replace(radar_file)
    provenance = dict(collection=COLLECTION, units="linear backscatter power", orbit=orbit, slots=sources)
    provenance_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return field.name, f"{len(sources)} radar slots on orbit {orbit}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    fields = sorted(p.parent for p in args.npy.glob("*/*/meta.json"))
    if not fields:
        parser.error("No exported field inputs")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(build, field): field.name for field in fields}
        for job in as_completed(jobs):
            try:
                field, status = job.result()
                print(f"{field}: {status}", flush=True)
            except (ValueError, RuntimeError) as exc:
                failures += 1
                print(f"{jobs[job]}: {exc}", flush=True)
    if failures:
        parser.exit(1, f"{failures}/{len(fields)} fields failed.\n")


if __name__ == "__main__":
    main()
