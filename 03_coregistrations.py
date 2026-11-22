"""
03_coregistration.py

Phase 4 — Co-registration verification + sub-pixel shift correction.

For every S1/S2 matched pair (from field_scene_map.csv, nearest-date
pairing), estimates the pixel-level (dx, dy) shift between the two
rasters using phase cross-correlation, logs the result for every field
(not just a spot sample), and writes a shift-corrected copy of the S1
raster for pairs above a confidence threshold.

Requires: rasterio, scikit-image, numpy
    pip install rasterio scikit-image numpy --break-system-packages

Usage:
    python 03_coregistration.py --confidence-threshold 0.3
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from skimage.registration import phase_cross_correlation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COREG_PAIRS = os.path.join(DATA_DIR, "coreg_pairs.csv")  # from 08_prepare_coregistration_pairs.py
COREG_LOG = os.path.join(DATA_DIR, "coregistration_log.csv")


def read_overlap(s1_path, s2_path, band=1):
    """
    Read the OVERLAPPING real-world area from both rasters, not the full
    extent of each. This handles the common case where S1 and S2 share
    the exact same pixel grid (same transform/pixel size/origin) but
    differ by a few rows/cols of edge padding — e.g. from GEE's region
    clipping rounding outward inconsistently per field. Rather than
    requiring identical array shapes upfront (which fails on that
    harmless padding), this finds the shared bounding box and reads
    only that window from each file, so the arrays that come out are
    already the same shape whenever the two rasters genuinely overlap.
    """
    with rasterio.open(s1_path) as s1_src, rasterio.open(s2_path) as s2_src:
        if s1_src.crs != s2_src.crs:
            raise ValueError(f"CRS mismatch: {s1_src.crs} vs {s2_src.crs} — not comparable directly.")

        # Intersection of the two rasters' real-world bounds.
        left = max(s1_src.bounds.left, s2_src.bounds.left)
        bottom = max(s1_src.bounds.bottom, s2_src.bounds.bottom)
        right = min(s1_src.bounds.right, s2_src.bounds.right)
        top = min(s1_src.bounds.top, s2_src.bounds.top)

        if left >= right or bottom >= top:
            raise ValueError(
                f"No spatial overlap between rasters: S1 bounds {s1_src.bounds}, "
                f"S2 bounds {s2_src.bounds}"
            )

        s1_window = from_bounds(left, bottom, right, top, transform=s1_src.transform)
        s2_window = from_bounds(left, bottom, right, top, transform=s2_src.transform)

        s1_arr = s1_src.read(band, window=s1_window, boundless=False).astype(np.float32)
        s2_arr = s2_src.read(band, window=s2_window, boundless=False).astype(np.float32)

        # from_bounds can still round to a 1-pixel-different shape between the
        # two windows in rare edge cases — trim both to the smaller common
        # shape as a final safety net so phase_cross_correlation never sees
        # mismatched arrays.
        min_h = min(s1_arr.shape[0], s2_arr.shape[0])
        min_w = min(s1_arr.shape[1], s2_arr.shape[1])
        s1_arr = s1_arr[:min_h, :min_w]
        s2_arr = s2_arr[:min_h, :min_w]

        s1_transform = s1_src.window_transform(s1_window)

    return s1_arr, s2_arr, s1_transform


def normalise(arr):
    """Min-max normalise for a fair cross-correlation comparison across modalities."""
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
    lo, hi = np.percentile(arr, [2, 98])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def estimate_shift(s1_path, s2_path, debug=False):
    """
    Estimate (row_shift, col_shift) in pixels between S1 and S2 rasters,
    reading only their overlapping real-world area (see read_overlap()).

    Returns (shift_y, shift_x, error) where `error` is phase_cross_correlation's
    normalised RMS error (lower = more confident match).
    """
    s1_arr, s2_arr, s1_transform = read_overlap(s1_path, s2_path)

    if s1_arr.size == 0:
        raise ValueError("Overlapping window is empty after cropping — check raster bounds/CRS.")

    if debug:
        def stats(name, arr):
            print(f"    {name}: shape={arr.shape} dtype={arr.dtype} "
                  f"min={np.nanmin(arr):.4g} max={np.nanmax(arr):.4g} "
                  f"mean={np.nanmean(arr):.4g} std={np.nanstd(arr):.4g} "
                  f"nan_count={np.isnan(arr).sum()}")
        stats("S1 raw", s1_arr)
        stats("S2 raw", s2_arr)

    s1_norm = normalise(s1_arr)
    s2_norm = normalise(s2_arr)

    if debug:
        stats("S1 normalised", s1_norm)
        stats("S2 normalised", s2_norm)

    shift, error, _ = phase_cross_correlation(s2_norm, s1_norm, upsample_factor=10)
    return shift[0], shift[1], error, s1_transform


def apply_shift_and_write(s1_path, out_path, shift_y, shift_x):
    """Write a shift-corrected copy of the S1 raster (transform-only correction)."""
    with rasterio.open(s1_path) as src:
        profile = src.profile.copy()
        data = src.read()
        transform = src.transform

    # Shift is in pixels; convert to a translated affine transform.
    corrected_transform = transform * Affine.translation(shift_x, shift_y)
    profile.update(transform=corrected_transform)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Max acceptable phase_cross_correlation error to auto-apply a shift",
    )
    parser.add_argument(
        "--pairs-file",
        type=str,
        default=COREG_PAIRS,
        help="CSV with field_id, s1_export_path, s2_matched_path columns "
             "(output of 08_prepare_coregistration_pairs.py by default)",
    )
    parser.add_argument(
        "--debug-stats",
        type=int,
        default=0,
        help="Print raw/normalised array statistics for the first N fields, "
             "then stop — use this to diagnose degenerate arrays (e.g. all-NaN, "
             "constant) before running the full batch.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CO-REGISTRATION CHECK + CORRECTION")
    print("=" * 60)

    pairs = pd.read_csv(args.pairs_file)  # must include s1_export_path, s2_matched_path

    if args.debug_stats:
        print(f"DEBUG MODE — printing array stats for the first {args.debug_stats} fields, then stopping.")
        print()
        for idx, row in pairs.head(args.debug_stats).iterrows():
            print(f"[{row['field_id']}]")
            try:
                estimate_shift(row["s1_export_path"], row["s2_matched_path"], debug=True)
            except Exception as e:
                print(f"    FAILED: {e}")
            print()
        return

    results = []

    for idx, row in pairs.iterrows():
        field_id = row["field_id"]
        s1_path = row["s1_export_path"]
        s2_path = row["s2_matched_path"]

        print(f"[{idx + 1}/{len(pairs)}] {field_id}", end=" ... ")

        try:
            shift_y, shift_x, error, transform = estimate_shift(s1_path, s2_path)
        except Exception as e:
            print(f"FAILED ({e})")
            results.append(
                {
                    "field_id": field_id,
                    "shift_y_px": None,
                    "shift_x_px": None,
                    "error": None,
                    "correction_applied": False,
                    "flag": "failed",
                }
            )
            continue

        pixel_size = abs(transform.a)
        shift_mag_px = float(np.hypot(shift_y, shift_x))
        confident = error <= args.confidence_threshold
        large_shift = shift_mag_px > 3  # more than 3 pixels is worth a manual look

        applied = False
        if confident:
            corrected_path = s1_path.replace(".tif", "_coreg.tif")
            apply_shift_and_write(s1_path, corrected_path, shift_y, shift_x)
            applied = True

        flag = "ok"
        if not confident:
            flag = "low_confidence"
        elif large_shift:
            flag = "large_shift_review"

        print(
            f"shift=({shift_y:.2f},{shift_x:.2f})px "
            f"(~{shift_mag_px * pixel_size:.1f} m), error={error:.3f}, flag={flag}"
        )

        results.append(
            {
                "field_id": field_id,
                "shift_y_px": shift_y,
                "shift_x_px": shift_x,
                "shift_magnitude_px": shift_mag_px,
                "shift_magnitude_m": shift_mag_px * pixel_size,
                "error": error,
                "correction_applied": applied,
                "flag": flag,
            }
        )

    log_df = pd.DataFrame(results)
    log_df.to_csv(COREG_LOG, index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total pairs               : {len(log_df)}")
    print(f"Corrections applied        : {int(log_df['correction_applied'].sum())}")
    print(f"Low confidence (unfixed)   : {(log_df['flag'] == 'low_confidence').sum()}")
    print(f"Large shift (needs review) : {(log_df['flag'] == 'large_shift_review').sum()}")
    print(f"Failed                     : {(log_df['flag'] == 'failed').sum()}")
    print(f"Written to                 : {COREG_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()