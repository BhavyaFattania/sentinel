"""
02_speckle_filter_and_export.py

Phase 2 â€” Multi-temporal speckle filtering (in GEE, before export).
Phase 3 â€” Reproject to the Sentinel-2 grid and export GeoTIFFs.

For each field, builds an ee.ImageCollection from the images logged in
gee_scene_log.csv, grouped by consistent geometry (relative orbit + pass
direction), applies a multi-temporal boxcar/mean-ratio speckle filter
(Quegan & Yu, 2001 style), reprojects each filtered image onto the target
Sentinel-2 CRS/transform, and exports to Google Drive.

You must supply the target S2 CRS + transform per field (read from one of
your existing Sentinel-2 GeoTIFFs) â€” see `get_s2_grid()` below, which uses
rasterio to read that from a local file path per field.

Usage:
    python 02_speckle_filter_and_export.py --ee-project PROJECT --mt-window 5
"""

import os
import time
import math
import csv
import argparse
import datetime as dt

import ee
import pandas as pd
import rasterio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUBMITTED_LOG = os.path.join(DATA_DIR, "submitted_exports.csv")


# ============================================================
# Multi-temporal speckle filter
# ============================================================
def multitemporal_filter(collection, band, window=5):
    """
    Simple multi-temporal speckle filter (Quegan & Yu, 2001 style):
    divide each image by its local spatial mean, average those ratios across
    acquisitions, and multiply by the local spatial mean of the target image.
    Work in linear power, then return filtered backscatter in dB.

    This assumes `collection` already contains only images sharing
    consistent geometry (same relative orbit + pass direction) so the
    temporal mean is not corrupted by geometry-driven backscatter shifts.
    """
    kernel = ee.Kernel.square(radius=window // 2, units="pixels")

    def _power_and_local_mean(img):
        db = img.select(band)
        power = db.multiply(math.log(10) / 10).exp()
        mean = power.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel)
        return img.addBands(mean.rename("_local_mean")).addBands(
            power.divide(mean).rename("_ratio")
        )

    normalized = collection.map(_power_and_local_mean)
    temporal_ratio = normalized.select("_ratio").mean()

    def _filter_one(img):
        db = img.select("_local_mean").multiply(temporal_ratio).log10().multiply(10)
        return img.addBands(db.rename(band + "_mtfilt"), overwrite=True).select(
            img.bandNames().remove("_local_mean").remove("_ratio").add(band + "_mtfilt")
        )

    return normalized.map(_filter_one)


# ============================================================
# Resumability â€” track what's already been submitted so a rerun
# (after a crash, a Ctrl+C, or hitting the queue cap) doesn't
# resubmit duplicates.
# ============================================================
def load_submitted_set(path=SUBMITTED_LOG):
    """Return a set of (field_id, image_id) pairs already submitted in a past run."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(zip(df["field_id"], df["image_id"]))


def log_submitted(field_id, image_id, task_id, path=SUBMITTED_LOG):
    """Append one successfully-submitted export to the resumability log immediately
    (not batched), so progress isn't lost even if the script stops abruptly."""
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["field_id", "image_id", "task_id", "submitted_at"])
        writer.writerow([field_id, image_id, task_id, dt.datetime.now().isoformat()])


# ============================================================
# Target grid (read from an existing Sentinel-2 GeoTIFF)
# ============================================================
from rasterio.warp import transform_bounds


# ============================================================
# Target grid (read from an existing Sentinel-2 GeoTIFF)
# ============================================================
def get_s2_grid(s2_tif_path):
    """Return (crs_wkt, transform_list, width, height, bounds_wgs84) from a S2 GeoTIFF.
    bounds_wgs84 is used to CLIP the export to this field's extent â€” without it,
    exporting a reprojected full Sentinel-1 scene (~250km swath) instead of just
    the field produces multi-GB files instead of the expected ~100-200KB."""
    with rasterio.open(s2_tif_path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return src.crs.to_wkt(), list(src.transform)[:6], src.width, src.height, bounds_wgs84


def count_active_tasks():
    """
    Query Earth Engine directly for how many tasks are currently
    READY or RUNNING (i.e. counted against the 3000-task queue cap).
    """
    tasks = ee.batch.Task.list()
    active_states = {"READY", "RUNNING", "UNSUBMITTED"}
    return sum(1 for t in tasks if t.status().get("state") in active_states)


def wait_for_queue_capacity(threshold=2800, poll_interval_seconds=180):
    """
    Block until the number of active (READY/RUNNING) EE tasks drops below
    `threshold`. Prints the ACTUAL queue size each check, rather than
    guessing blindly â€” this tells you directly whether the queue is
    genuinely draining or stuck, instead of leaving you to wonder.

    No timeout here on purpose: the correct response to a full queue is
    always to keep waiting, not to give up â€” GEE will eventually work
    through it (it may just take a while for a large backlog).
    """
    while True:
        try:
            active = count_active_tasks()
        except Exception as e:
            print(f"  (Couldn't check queue size: {e} â€” proceeding cautiously.)")
            return
        if active < threshold:
            return
        print(
            f"  Queue at {active} active tasks (need < {threshold}) â€” "
            f"waiting {poll_interval_seconds}s and checking again... "
            f"[see https://code.earthengine.google.com/tasks]"
        )
        time.sleep(poll_interval_seconds)


def export_image(
    img,
    bands,
    field_id,
    image_id,
    country,
    crs_wkt,
    transform,
    bounds_wgs84,
    folder,
    retry_wait_seconds=120,
    submitted_log=SUBMITTED_LOG,
):
    """
    Export the given bands together as one multi-band GeoTIFF, reprojected
    onto the S2 grid AND CLIPPED to the field's extent (bounds_wgs84) â€”
    without the clip, this exports the full ~250km Sentinel-1 scene instead
    of just the field, producing multi-gigabyte files instead of ~100-200KB.

    If Earth Engine's task queue is full ("Too many tasks already in the
    queue"), this WAITS and RETRIES indefinitely instead of crashing â€” the
    queue drains over time as GEE processes earlier tasks, so pausing and
    trying again is always the correct response here, never a fatal error.
    Any other kind of error (bad CRS, missing asset, etc.) is NOT retried â€”
    it's raised immediately since retrying won't fix it.
    """
    region = ee.Geometry.Rectangle(list(bounds_wgs84))
    projected = img.select(bands).clip(region).reproject(crs=crs_wkt, crsTransform=transform)
    # EE task 'description' has a hard 100-character limit (alnum, . , : ; _ - only).
    # The actual output filename (fileNamePrefix) has no such limit, so all the
    # full identifying detail (country, field_id, image_id) stays there â€” the
    # description just needs to be short enough for the Tasks list.
    description = f"{field_id}_{image_id}"[:100]
    task = ee.batch.Export.image.toDrive(
        image=projected,
        description=description,
        folder=folder,
        fileNamePrefix=f"{country}_{field_id}_{image_id}",
        region=region,
        crs=crs_wkt,
        crsTransform=transform,
        maxPixels=1e9,
    )

    attempt = 0
    while True:
        attempt += 1
        try:
            task.start()
            log_submitted(field_id, image_id, task.id, submitted_log)
            return task
        except ee.ee_exception.EEException as e:
            if "Too many tasks" in str(e):
                print(f"  Queue full (attempt {attempt}) â€” checking real queue size...")
                wait_for_queue_capacity(poll_interval_seconds=retry_wait_seconds)
                continue
            raise  # a real error (bad CRS, missing asset, etc.) â€” don't retry, surface it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ee-project", required=True, help="Earth Engine-enabled Google Cloud project"
    )
    parser.add_argument(
        "--data-dir", default=DATA_DIR, help="directory containing the scene inventory CSVs"
    )
    parser.add_argument(
        "--mt-window", type=int, default=5, help="Speckle filter kernel size (pixels)"
    )
    parser.add_argument("--drive-folder", type=str, default="YieldSAT_S1_exports")
    parser.add_argument(
        "--max-new-tasks",
        type=int,
        default=2500,
        help="Stop submitting new tasks after this many in one run, to stay under GEE's "
        "3000-task queue cap (leaves headroom for tasks still draining). "
        "Rerun the script later to continue where it left off.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SPECKLE FILTER + EXPORT TO S2 GRID")
    print("=" * 60)

    ee.Initialize(project=args.ee_project)
    scene_log = pd.read_csv(os.path.join(args.data_dir, "gee_scene_log.csv"))
    field_scene_map = pd.read_csv(os.path.join(args.data_dir, "field_scene_map.csv"))
    submitted_log = os.path.join(args.data_dir, "submitted_exports.csv")

    already_submitted = load_submitted_set(submitted_log)
    if already_submitted:
        print(
            f"Resuming: {len(already_submitted)} image(s) already submitted in a previous run â€” will skip those."
        )
        print()

    print("Checking current Earth Engine queue size before starting...")
    wait_for_queue_capacity(threshold=2800, poll_interval_seconds=60)
    print("Queue has capacity â€” starting submissions.")
    print()

    tasks_started = 0
    tasks_skipped_resume = 0

    field_ids = list(scene_log.groupby("field_id").groups.keys())
    total_fields = len(field_ids)
    stop_early = False

    for field_num, (field_id, group) in enumerate(scene_log.groupby("field_id"), start=1):
        if stop_early:
            break
        print(f"[{field_num}/{total_fields}] {field_id}")
        # Group further by consistent geometry so the temporal mean isn't
        # corrupted by mixed relative orbits / pass directions.
        for (orbit_dir, rel_orbit), geom_group in group.groupby(
            ["orbit_direction", "relative_orbit"]
        ):
            if len(geom_group) < 3:
                # Multi-temporal filtering needs a minimum stack size to be useful.
                print(
                    f"  orbit {rel_orbit} ({orbit_dir}): "
                    f"only {len(geom_group)} scenes, skipping (need >= 3)"
                )
                continue

            image_ids = geom_group["gee_system_index"].tolist()
            collection = ee.ImageCollection(
                [ee.Image(f"COPERNICUS/S1_GRD/{iid}") for iid in image_ids]
            )

            filtered = multitemporal_filter(collection, band="VV", window=args.mt_window)
            filtered = multitemporal_filter(filtered, band="VH", window=args.mt_window)

            ref_matches = field_scene_map.loc[field_scene_map["field_id"] == field_id]
            if ref_matches.empty or "s2_grid_ref_path" not in ref_matches.columns:
                print(f"  no s2_grid_ref_path found in field_scene_map.csv, skipping")
                continue
            ref_row = ref_matches.iloc[0]
            s2_path = ref_row["s2_grid_ref_path"]
            if pd.isna(s2_path) or not os.path.exists(s2_path):
                print(f"  s2_grid_ref_path missing or file not found ({s2_path}), skipping")
                continue
            crs_wkt, transform, _, _, bounds_wgs84 = get_s2_grid(s2_path)

            country = geom_group["country"].iloc[0]
            filtered_list = filtered.toList(filtered.size())
            for i, image_id in enumerate(image_ids):
                if (field_id, image_id) in already_submitted:
                    tasks_skipped_resume += 1
                    continue
                img = ee.Image(filtered_list.get(i))
                export_image(
                    img,
                    bands=["VV_mtfilt", "VH_mtfilt"],
                    field_id=field_id,
                    image_id=image_id,
                    country=country,
                    crs_wkt=crs_wkt,
                    transform=transform,
                    bounds_wgs84=bounds_wgs84,
                    folder=args.drive_folder,
                    submitted_log=submitted_log,
                )
                tasks_started += 1

                if tasks_started >= args.max_new_tasks:
                    print(f"\nReached --max-new-tasks limit ({args.max_new_tasks}) for this run.")
                    print("Rerun the script to continue with the remaining fields.")
                    stop_early = True
                    break
            if stop_early:
                break

    print()
    print("=" * 60)
    print(f"Export tasks started this run : {tasks_started}")
    print(f"Skipped (already submitted)   : {tasks_skipped_resume}")
    print(f"Total ever submitted          : {tasks_started + len(already_submitted)}")
    print("Check progress at https://code.earthengine.google.com/tasks")
    print("=" * 60)


if __name__ == "__main__":
    main()
