"""Build pilot VV/VH arrays on the YieldSAT grid from public Sentinel-1 RTC COGs."""

import argparse
import csv
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


class CoverageError(ValueError):
    pass


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
                raise RuntimeError(
                    f"Planetary Computer connection failed: {exc.reason if isinstance(exc, URLError) else exc}"
                ) from None
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
    params = {
        "collections": COLLECTION,
        "bbox": ",".join(map(str, bounds)),
        "limit": 100,
        "datetime": f"{sow}T00:00:00Z/{harvest}T23:59:59Z",
    }
    url = f"{API}/stac/v1/search?{urlencode(params)}"
    while url:
        page = get_json(url)
        for item in page["features"]:
            date = np.datetime64(item["properties"]["datetime"][:10])
            if sow <= date <= harvest:
                yield item
        url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)


def read_backscatter(item, grid):
    channels = []
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="20", GDAL_HTTP_MAX_RETRY="2"
    ):
        for pol in ("vv", "vh"):
            signed = sign_asset(item["assets"][pol]["href"])
            try:
                with rasterio.open(signed) as src, WarpedVRT(src, **grid, nodata=np.nan) as aligned:
                    channels.append(aligned.read(1, masked=True).filled(np.nan))
            except rasterio.errors.RasterioError:
                raise RuntimeError(f"{item['id']} {pol} raster could not be read") from None
    return np.stack(channels).astype(np.float32)


# Rebuild the recorded slots from their exact STAC item IDs.
def replay_slots(grid, data, sources):
    cache = {}
    for source in sources:
        item_id = source["item"]
        if item_id not in cache:
            item = get_json(f"{API}/stac/v1/collections/{COLLECTION}/items/{item_id}")
            power = read_backscatter(item, grid)
            power[~np.isfinite(power) | (power <= 0)] = np.nan
            cache[item_id] = power
        data[source["slot"]] = cache[item_id]


def build(field, min_slots=1, recorded=None):
    radar_file = field / "sar.npy"
    provenance_file = field / "sar_sources.json"
    if radar_file.is_file() and provenance_file.is_file():
        existing = json.loads(provenance_file.read_text(encoding="utf-8"))
        if len(existing["slots"]) >= min_slots and (
            (recorded is None and existing.get("catalog") == "stac-search")
            or (
                recorded is not None
                and existing.get("catalog") == "recorded-items"
                and existing["slots"] == recorded["slots"]
            )
        ):
            return len(existing["slots"]), "already built"
    meta = json.loads((field / "meta.json").read_text(encoding="utf-8"))
    sow, harvest = [
        np.datetime64(datetime.strptime(meta[name], "%d.%m.%Y").date())
        for name in ("seeding_date", "harvesting_date")
    ]
    with rasterio.open(meta["s2_reference"]) as ref:
        bounds = transform_bounds(ref.crs, "EPSG:4326", *ref.bounds)
        grid = dict(crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height)
    shape = (grid["height"], grid["width"])
    support = np.load(field / "mask.npy")
    optical_dates = np.load(field / "dates.npy")
    slot_dates = np.load(field / "slot_dates.npy")
    season = np.load(field / "season.npy")
    data = np.full((len(slot_dates), 2, *shape), np.nan, np.float32)
    if recorded is not None:
        orbit, sources = recorded["orbit"], recorded["slots"]
        if len(sources) < min_slots:
            raise CoverageError(f"Only {len(sources)} recorded radar slots; need {min_slots}")
        replay_slots(grid, data, sources)
    else:
        groups = defaultdict(list)
        for item in search(bounds, sow, harvest):
            p = item["properties"]
            orbit = (p.get("sat:orbit_state"), p.get("sat:relative_orbit"))
            if None not in orbit and {"vv", "vh"} <= item["assets"].keys():
                date = np.datetime64(p["datetime"][:10])
                groups[orbit].append((date, item))
        if not groups:
            raise CoverageError(f"No dual-polarization RTC items for {field.name}")
        centers = {}
        for slot in np.flatnonzero(season):
            center = optical_dates[slot] if not np.isnat(optical_dates[slot]) else slot_dates[slot]
            centers[int(slot)] = max(sow, min(harvest, center))

        def candidate_count(scenes):
            limit = np.timedelta64(12, "D")
            return sum(
                any(abs(date - center) <= limit for date, _ in scenes)
                for center in centers.values()
            )

        best = 0
        for orbit, scenes in sorted(
            groups.items(), key=lambda item: (-candidate_count(item[1]), item[0])
        ):
            if candidate_count(scenes) < min_slots:
                continue
            sources = []
            cache = {}
            data.fill(np.nan)
            for slot, center in centers.items():
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
                    coverage = np.isfinite(power[:, support]).all(axis=0).mean()
                    if coverage < 0.95:
                        continue
                    data[slot] = power
                    sources.append(
                        dict(
                            slot=slot,
                            target_date=str(center),
                            date=str(date),
                            item=scene["id"],
                            labeled_coverage=float(coverage),
                        )
                    )
                    break
            if len(sources) >= min_slots:
                break
            best = max(best, len(sources))
        else:
            raise CoverageError(f"Only {best} usable radar slots found; need {min_slots}")
    temp = field / "sar.partial.npy"
    np.save(temp, data)
    temp.replace(radar_file)
    provenance = dict(
        collection=COLLECTION,
        catalog="recorded-items" if recorded is not None else "stac-search",
        units="linear backscatter power",
        orbit=orbit,
        slots=sources,
    )
    provenance_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return len(sources), f"orbit {orbit}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--country", help="limit coverage checks to one country")
    parser.add_argument("--field", action="append", help="process only these exact field IDs")
    parser.add_argument("--min-slots", type=int, default=1)
    parser.add_argument(
        "--sources", type=Path, help="Replay exact item IDs and slots from a JSONL file"
    )
    parser.add_argument("--status-csv", type=Path)
    args = parser.parse_args()
    if not 1 <= args.min_slots <= 24:
        parser.error("--min-slots must be between 1 and 24")
    if args.workers < 1:
        parser.error("--workers must be positive")
    fields = sorted(
        p.parent
        for p in args.npy.glob("*/*/meta.json")
        if (args.country is None or p.parent.parent.name == args.country)
        and (args.field is None or p.parent.name in args.field)
    )
    records = {}
    if args.sources:
        rows = (json.loads(line) for line in args.sources.read_text(encoding="utf-8").splitlines())
        records = {(row["country"], row["field"]): row for row in rows}
        expected = {
            key
            for key in records
            if (args.country is None or key[0] == args.country)
            and (args.field is None or key[1] in args.field)
        }
        fields = [p for p in fields if (p.parent.name, p.name) in expected]
        if {(p.parent.name, p.name) for p in fields} != expected:
            parser.error("Some recorded fields are missing. Export the field inputs first.")
    if not fields:
        parser.error("No exported field inputs")
    if args.field and set(args.field) != {p.name for p in fields}:
        parser.error("Some requested fields were not found")

    def run(writer=None, report=None):
        counts = {"ready": 0, "coverage": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                pool.submit(
                    build, field, args.min_slots, records.get((field.parent.name, field.name))
                ): field
                for field in fields
            }
            for job in as_completed(jobs):
                field = jobs[job]
                slots = 0
                try:
                    slots, detail = job.result()
                    status = "ready"
                except CoverageError as exc:
                    status, detail = "coverage", str(exc)
                except Exception as exc:
                    status, detail = "failed", str(exc)
                counts[status] += 1
                print(f"{field.name}: {status} ({slots} slots) {detail}", flush=True)
                if writer:
                    writer.writerow(
                        dict(
                            country=field.parent.name,
                            field=field.name,
                            status=status,
                            radar_slots=slots,
                            detail=detail,
                        )
                    )
                    report.flush()
        print(counts, flush=True)
        return counts["failed"] + (counts["coverage"] if args.sources else 0)

    if args.status_csv:
        with args.status_csv.open("w", newline="", encoding="utf-8") as report:
            writer = csv.DictWriter(
                report, fieldnames=("country", "field", "status", "radar_slots", "detail")
            )
            writer.writeheader()
            failed = run(writer, report)
    else:
        failed = run()
    if failed:
        parser.exit(1)


if __name__ == "__main__":
    main()
