"""
03b_coregistration_edge_verify.py

Phase 4b — Co-registration verification using EDGE/GRADIENT similarity
instead of raw pixel intensity.

Why this exists: 03_coregistrations.py compares raw S1/S2 pixel values
with phase_cross_correlation. That fails for SAR-optical pairs (error≈1.0,
essentially random shift output) because a field that is bright in
optical can be dark in SAR and vice versa — raw brightness just isn't a
shared signal between the two modalities.

What IS shared between SAR and optical: edges. Field boundaries, roads,
waterways, and other physical discontinuities show up as gradient
transitions in both modalities, even though the sign/magnitude of the
transition differs. This is the same principle behind the published
SAR-optical registration literature (OS-SIFT, HOPC, ROS-PC, OS-Flow,
etc.) — they all build descriptors from gradient/phase-congruency maps
rather than raw intensity, specifically because edge structure is
resistant to the radiometric differences between the two sensors.
This script is a lightweight instance of that same idea: replace raw
intensity with a Sobel gradient-magnitude map, then run the same
phase_cross_correlation on that.

This is a VERIFICATION check, not a registration pipeline. If your S1
exports were already reprojected onto the S2 grid (Script 02/03's
reproject/crsTransform step), this should show near-zero, high-confidence
shifts for fields with enough physical edge structure. It's not a
guarantee for fields with no such structure (e.g. a single homogeneous
crop with no boundary/road/waterway inside the clipped tile) — those will
still show a low-confidence result, and that's expected, not a bug.

Requires: rasterio, scikit-image, numpy, scipy
    pip install rasterio scikit-image numpy scipy --break-system-packages

Usage:
    python 03b_coregistration_edge_verify.py --ncc-threshold 0.15
    python 03b_coregistration_edge_verify.py --diagnose-n 5   # inspect a few fields first
    python 03b_coregistration_edge_verify.py --pairs-file data/coreg_pairs.csv --compare-log data/coregistration_log.csv
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import rasterio
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COREG_PAIRS = os.path.join(DATA_DIR, "coreg_pairs.csv")
EDGE_COREG_LOG = os.path.join(DATA_DIR, "coregistration_edge_log.csv")


# ============================================================
# Same overlap-reading logic as 03_coregistrations.py, kept
# self-contained here so this script can run standalone.
# ============================================================
def read_overlap(s1_path, s2_path, band=1):
    with rasterio.open(s1_path) as s1_src:
        s1_bounds = s1_src.bounds
        crs = s1_src.crs

        with rasterio.open(s2_path) as s2_src:
            s2_bounds = s2_src.bounds

            left = max(s1_bounds.left, s2_bounds.left)
            bottom = max(s1_bounds.bottom, s2_bounds.bottom)
            right = min(s1_bounds.right, s2_bounds.right)
            top = min(s1_bounds.top, s2_bounds.top)

            if left >= right or bottom >= top:
                raise ValueError(
                    f"Rasters do not overlap: S1 bounds={s1_bounds}, S2 bounds={s2_bounds}"
                )

            s1_window = s1_src.window(left, bottom, right, top)
            s2_window = s2_src.window(left, bottom, right, top)

            s1_arr = s1_src.read(band, window=s1_window).astype(np.float32)
            s2_arr = s2_src.read(band, window=s2_window).astype(np.float32)

            min_rows = min(s1_arr.shape[0], s2_arr.shape[0])
            min_cols = min(s1_arr.shape[1], s2_arr.shape[1])

            if min_rows < 8 or min_cols < 8:
                # Edge detection needs a bit more context than raw correlation did;
                # a 4x4 overlap has essentially no room for gradient structure.
                raise ValueError(
                    f"Overlap region too small for edge analysis ({min_rows}x{min_cols} px)."
                )

            s1_arr = s1_arr[:min_rows, :min_cols]
            s2_arr = s2_arr[:min_rows, :min_cols]

            overlap_transform = s1_src.window_transform(s1_window)

    return s1_arr, s2_arr, overlap_transform, crs


def normalise(arr):
    """Min-max normalise (percentile-clipped) for a fair comparison."""
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
    lo, hi = np.percentile(arr, [2, 98])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def edge_map(arr):
    """
    Sobel gradient-magnitude map: the shared signal between SAR and
    optical. Denoise lightly first (SAR speckle otherwise dominates
    the gradient with noise rather than real structure) then take the
    Sobel magnitude, then normalise so both modalities' edge maps are
    on a comparable 0-1 scale before correlation.
    """
    from scipy.ndimage import uniform_filter

    smoothed = uniform_filter(arr, size=3)
    edges = sobel(smoothed)
    return normalise(edges)


def manual_ncc_at_shift(ref, moving, shift_y, shift_x):
    """
    Plain normalized cross-correlation coefficient between `ref` and
    `moving` shifted by (shift_y, shift_x), computed directly (no FFT
    whitening) so we have a transparent, non-black-box sanity check on
    skimage's error metric. 1.0 = perfect match, 0 = uncorrelated.
    Uses scipy.ndimage.shift for sub-pixel shifting, then compares only
    the region both arrays still cover after the shift.
    """
    from scipy.ndimage import shift as nd_shift

    shifted = nd_shift(moving, shift=(shift_y, shift_x), order=1, mode="constant", cval=np.nan)
    valid = ~np.isnan(shifted)
    if valid.sum() < 16:
        return np.nan
    a = ref[valid].ravel()
    b = shifted[valid].ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < 1e-9:
        return np.nan
    return float((a * b).sum() / denom)


def estimate_shift_edge(s1_path, s2_path):
    """
    Same overlap-reading as the intensity-based script, run on Sobel edge
    maps instead of raw pixel values — but with normalization=None so
    skimage does plain (not phase-whitened) cross-correlation. Phase-only
    correlation (skimage's default) whitens every frequency to unit
    amplitude, which is fine for broadband imagery but degenerates for
    sparse edge maps (mostly background, a few strong edges): the mostly-
    empty background gets amplified as much as the real edges and buries
    the correlation peak in noise, which is why the raw default gave
    error=1.000 for essentially every field regardless of content.

    Also returns a manually-computed NCC at the found shift, as an
    independent, transparent cross-check on skimage's error value.
    """
    s1_arr, s2_arr, overlap_transform, _ = read_overlap(s1_path, s2_path)

    s1_edges = edge_map(s1_arr)
    s2_edges = edge_map(s2_arr)

    shift, error, _ = phase_cross_correlation(
        s2_edges, s1_edges, upsample_factor=10, normalization=None
    )
    ncc = manual_ncc_at_shift(s2_edges, s1_edges, shift[0], shift[1])
    return shift[0], shift[1], error, ncc, overlap_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ncc-threshold", type=float, default=0.15,
        help="Min acceptable manual NCC (normalized cross-correlation, -1..1) "
             "to flag a field as 'confident'. NCC is more interpretable than "
             "skimage's phase-correlation error for sparse edge maps — 0.15-0.2 "
             "is a reasonable starting bar; tune after inspecting a few fields.",
    )
    parser.add_argument(
        "--pairs-file", type=str, default=COREG_PAIRS,
        help="CSV with field_id, s1_export_path, s2_matched_path columns "
             "(output of 08_prepare_coregistration_pairs.py by default)",
    )
    parser.add_argument(
        "--compare-log", type=str, default=None,
        help="Optional: path to the intensity-based coregistration_log.csv, "
             "to print a side-by-side comparison at the end.",
    )
    parser.add_argument(
        "--diagnose-n", type=int, default=0,
        help="If >0, print raw array/edge-map statistics (min/max/mean/std/"
             "nonzero-fraction) for the first N pairs before scoring anything. "
             "Use this if results still look degenerate (e.g. all errors "
             "identical) — it tells you whether the edge maps themselves are "
             "sane or whether something upstream (units, all-zero tiles, "
             "band selection) is degenerate.",
    )
    args = parser.parse_args()

    if args.diagnose_n > 0:
        pairs_preview = pd.read_csv(args.pairs_file).head(args.diagnose_n)
        print("=" * 60)
        print(f"DIAGNOSTIC: raw stats for first {args.diagnose_n} pair(s)")
        print("=" * 60)
        for _, row in pairs_preview.iterrows():
            try:
                s1_arr, s2_arr, _, _ = read_overlap(row["s1_export_path"], row["s2_matched_path"])
                s1_e = edge_map(s1_arr)
                s2_e = edge_map(s2_arr)
                for label, a in [("S1 raw", s1_arr), ("S2 raw", s2_arr),
                                  ("S1 edges", s1_e), ("S2 edges", s2_e)]:
                    print(f"  {row['field_id']:45s} {label:9s} "
                          f"shape={a.shape} min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} "
                          f"mean={np.nanmean(a):.4g} std={np.nanstd(a):.4g} "
                          f"nonzero_frac={np.mean(a != 0):.3f}")
            except Exception as e:
                print(f"  {row['field_id']}: DIAGNOSTIC FAILED ({e})")
        print("=" * 60)
        print()

    print("=" * 60)
    print("CO-REGISTRATION VERIFICATION — EDGE/GRADIENT BASED")
    print("=" * 60)

    pairs = pd.read_csv(args.pairs_file)
    results = []

    for idx, row in pairs.iterrows():
        field_id = row["field_id"]
        s1_path = row["s1_export_path"]
        s2_path = row["s2_matched_path"]

        print(f"[{idx + 1}/{len(pairs)}] {field_id}", end=" ... ")

        try:
            shift_y, shift_x, error, ncc, transform = estimate_shift_edge(s1_path, s2_path)
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({
                "field_id": field_id, "shift_y_px": None, "shift_x_px": None,
                "error": None, "ncc": None, "flag": "failed",
            })
            continue

        pixel_size = abs(transform.a)
        shift_mag_px = float(np.hypot(shift_y, shift_x))
        confident = (not np.isnan(ncc)) and ncc >= args.ncc_threshold

        flag = "ok" if confident else "low_confidence"
        print(
            f"shift=({shift_y:.2f},{shift_x:.2f})px "
            f"(~{shift_mag_px * pixel_size:.1f} m), error={error:.3f}, ncc={ncc:.3f}, flag={flag}"
        )

        results.append({
            "field_id": field_id,
            "shift_y_px": shift_y,
            "shift_x_px": shift_x,
            "shift_magnitude_px": shift_mag_px,
            "shift_magnitude_m": shift_mag_px * pixel_size,
            "error": error,
            "ncc": ncc,
            "flag": flag,
        })

    log_df = pd.DataFrame(results)
    log_df.to_csv(EDGE_COREG_LOG, index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n = len(log_df)
    n_ok = (log_df["flag"] == "ok").sum()
    n_low = (log_df["flag"] == "low_confidence").sum()
    n_fail = (log_df["flag"] == "failed").sum()
    print(f"Total pairs        : {n}")
    print(f"Confident match     : {n_ok}  ({100*n_ok/n:.1f}%)")
    print(f"Low confidence      : {n_low}  ({100*n_low/n:.1f}%)")
    print(f"Failed (no overlap) : {n_fail}")
    ok_df = log_df[log_df["flag"] == "ok"]
    if not ok_df.empty:
        print()
        print("Shift magnitude (m) among confident matches:")
        print(ok_df["shift_magnitude_m"].describe())
    print()
    print(f"Written to: {EDGE_COREG_LOG}")

    if args.compare_log and os.path.exists(args.compare_log):
        print()
        print("=" * 60)
        print("COMPARISON WITH INTENSITY-BASED LOG")
        print("=" * 60)
        intensity_df = pd.read_csv(args.compare_log)
        merged = log_df.merge(
            intensity_df[["field_id", "error", "flag"]],
            on="field_id", suffixes=("_edge", "_intensity"),
        )
        print(f"Fields improved from low_confidence -> ok using edges: "
              f"{((merged['flag_intensity'] == 'low_confidence') & (merged['flag_edge'] == 'ok')).sum()}")
        print("Mean error, intensity-based :", intensity_df["error"].mean())
        print("Mean error, edge-based      :", log_df["error"].mean())
    print("=" * 60)


if __name__ == "__main__":
    main()