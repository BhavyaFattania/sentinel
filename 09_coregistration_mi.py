"""
09_coregistration_mi.py

Cross-modal (SAR <-> optical) co-registration using Mutual Information (MI),
following the standard methodology established for exactly this problem:

    Suri, S. & Reinartz, P. (2010). "Mutual-Information-Based Registration
    of TerraSAR-X and Ikonos Imagery in Urban Areas." IEEE Transactions on
    Geoscience and Remote Sensing, 48(2), 939-949.

WHY NOT PHASE CROSS-CORRELATION (what the earlier script used):
Phase correlation assumes the two images' pixel intensities are directly
comparable/correlated. That's a reasonable assumption for two optical
images or two SAR images, but it does NOT hold between SAR and optical:
a field's radar backscatter brightness has no direct relationship to its
visible-light reflectance. Suri & Reinartz (2010) is the paper that
established MI as the right similarity metric for this exact cross-modal
case: instead of asking "do the raw values correlate?", MI asks "how much
does knowing image A's value at a pixel tell you about image B's value
there?" - which holds even when the two modalities have completely
different, nonlinear brightness relationships, as long as the same
underlying land-cover structure is producing both.

METHOD:
For a small range of candidate pixel shifts (dx, dy), shift the S1 image,
take the overlapping region with S2, and compute the mutual information
between the two. The shift that MAXIMISES MI is the estimated alignment
correction - this is a direct, literal implementation of the Suri &
Reinartz grid-search MI registration approach, at the pixel/sub-field
scale appropriate for this dataset (a full SIFT/feature-matching pipeline,
as used in later papers e.g. Ye et al. 2017, is unnecessary at this
field-crop scale where an exhaustive small-window search is fast and
exact).

Usage:
    python 09_coregistration_mi.py --max-shift 15
    python 09_coregistration_mi.py --max-shift 15 --use-gradient
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy.ndimage import sobel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COREG_PAIRS = os.path.join(DATA_DIR, "coreg_pairs.csv")
COREG_LOG = os.path.join(DATA_DIR, "coregistration_mi_log.csv")


# ============================================================
# Reading the overlapping real-world window from both rasters
# (same approach as 03_coregistration.py's fix — handles the
# few-pixel extent padding from GEE's region clipping)
# ============================================================
def read_overlap(s1_path, s2_path, band=1, margin_px=0):
    with rasterio.open(s1_path) as s1_src, rasterio.open(s2_path) as s2_src:
        if s1_src.crs != s2_src.crs:
            raise ValueError(f"CRS mismatch: {s1_src.crs} vs {s2_src.crs}")

        left = max(s1_src.bounds.left, s2_src.bounds.left)
        bottom = max(s1_src.bounds.bottom, s2_src.bounds.bottom)
        right = min(s1_src.bounds.right, s2_src.bounds.right)
        top = min(s1_src.bounds.top, s2_src.bounds.top)
        if left >= right or bottom >= top:
            raise ValueError("No spatial overlap between rasters.")

        s1_window = from_bounds(left, bottom, right, top, transform=s1_src.transform)
        s2_window = from_bounds(left, bottom, right, top, transform=s2_src.transform)

        s1_arr = s1_src.read(band, window=s1_window, boundless=False).astype(np.float32)
        s2_arr = s2_src.read(band, window=s2_window, boundless=False).astype(np.float32)

        min_h = min(s1_arr.shape[0], s2_arr.shape[0])
        min_w = min(s1_arr.shape[1], s2_arr.shape[1])
        s1_arr = s1_arr[:min_h, :min_w]
        s2_arr = s2_arr[:min_h, :min_w]
        s1_transform = s1_src.window_transform(s1_window)
        pixel_size = abs(s1_src.transform.a)

    return s1_arr, s2_arr, s1_transform, pixel_size


# ============================================================
# Mutual Information
# ============================================================
def mutual_information(a, b, bins=32):
    """
    Standard histogram-based MI, per Suri & Reinartz (2010):
    MI(A,B) = sum p(a,b) * log( p(a,b) / (p(a)*p(b)) )
    Also returns Normalised MI (NMI = (H(A)+H(B)) / H(A,B)), which is
    bounded and comparable ACROSS different field sizes/content — raw MI
    isn't, since it scales with image entropy/size.
    """
    a = a[np.isfinite(a)].ravel() if a.ndim == 1 else a
    joint_hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    joint_prob = joint_hist / joint_hist.sum()

    p_a = joint_prob.sum(axis=1)
    p_b = joint_prob.sum(axis=0)

    nonzero = joint_prob > 0
    h_joint = -np.sum(joint_prob[nonzero] * np.log2(joint_prob[nonzero]))
    h_a = -np.sum(p_a[p_a > 0] * np.log2(p_a[p_a > 0]))
    h_b = -np.sum(p_b[p_b > 0] * np.log2(p_b[p_b > 0]))

    mi = h_a + h_b - h_joint
    nmi = (h_a + h_b) / h_joint if h_joint > 0 else 0.0
    return mi, nmi


def preprocess(arr, use_gradient=False):
    """Optionally use gradient magnitude instead of raw intensity — structural
    edges (field boundaries, roads) are more directly comparable across SAR
    and optical than raw brightness is, per Ye et al. (2017)'s structural-
    similarity line of work building on the MI approach."""
    arr = np.nan_to_num(arr, nan=np.nanmedian(arr))
    if use_gradient:
        gx = sobel(arr, axis=0)
        gy = sobel(arr, axis=1)
        arr = np.hypot(gx, gy)
    return arr


# ============================================================
# Grid-search MI shift estimation (the core Suri & Reinartz method)
# ============================================================
def estimate_shift_mi(s1_arr, s2_arr, max_shift=15, bins=32, peak_exclusion_radius=2):
    """
    Exhaustive search over integer pixel shifts (dy, dx) in
    [-max_shift, max_shift]^2, picking the shift that maximises MI
    between the overlapping region of shifted-S1 and S2.

    Also computes PEAK SHARPNESS: best_mi / second_best_mi, where
    "second best" excludes shifts within `peak_exclusion_radius` pixels
    of the winner (so we're comparing the true peak against a genuinely
    DIFFERENT candidate location, not just a neighbouring pixel of the
    same peak). This is the standard confidence measure from stereo/
    template matching: a real, correct match produces one sharp, distinct
    peak (ratio well above 1.0); a false or unreliable match produces a
    flat, noisy MI surface where many unrelated shifts score almost as
    well as the "winner" (ratio close to 1.0) — meaning the reported
    shift is essentially arbitrary, even if its MI Gain over zero-shift
    looked fine in isolation.

    Returns (best_dy, best_dx, best_mi, best_nmi, mi_at_zero_shift,
    peak_sharpness).
    """
    h, w = s2_arr.shape
    surface = {}  # (dy, dx) -> mi
    mi_zero = None

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0_s1, y1_s1 = max(0, dy), min(h, h + dy)
            x0_s1, x1_s1 = max(0, dx), min(w, w + dx)
            y0_s2, y1_s2 = max(0, -dy), min(h, h - dy)
            x0_s2, x1_s2 = max(0, -dx), min(w, w - dx)

            if y1_s1 <= y0_s1 or x1_s1 <= x0_s1:
                continue

            patch_s1 = s1_arr[y0_s1:y1_s1, x0_s1:x1_s1]
            patch_s2 = s2_arr[y0_s2:y1_s2, x0_s2:x1_s2]
            if patch_s1.size < bins * 2:
                continue

            mi, nmi = mutual_information(patch_s1, patch_s2, bins=bins)
            surface[(dy, dx)] = (mi, nmi)
            if dy == 0 and dx == 0:
                mi_zero = mi

    if not surface:
        raise ValueError("No valid overlap found across the search range — field may be too small.")

    best_dy, best_dx = max(surface, key=lambda k: surface[k][0])
    best_mi, best_nmi = surface[(best_dy, best_dx)]

    # Second-best MI among candidates far enough from the winner to count
    # as a genuinely different location, not just its immediate neighbour.
    far_candidates = [
        mi
        for (dy, dx), (mi, _) in surface.items()
        if max(abs(dy - best_dy), abs(dx - best_dx)) > peak_exclusion_radius
    ]
    second_best_mi = max(far_candidates) if far_candidates else best_mi
    peak_sharpness = best_mi / second_best_mi if second_best_mi > 0 else float("inf")

    return best_dy, best_dx, best_mi, best_nmi, mi_zero, peak_sharpness


def apply_shift_and_write(s1_path, out_path, shift_y, shift_x):
    with rasterio.open(s1_path) as src:
        profile = src.profile.copy()
        data = src.read()
        transform = src.transform
    corrected_transform = transform * Affine.translation(-shift_x, -shift_y)
    profile.update(transform=corrected_transform)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=str, default=COREG_PAIRS)
    parser.add_argument(
        "--max-shift",
        type=int,
        default=15,
        help="Search range in pixels (+/-) — keep small, this is O(max_shift^2)",
    )
    parser.add_argument("--bins", type=int, default=32, help="Histogram bins for MI estimation")
    parser.add_argument(
        "--use-gradient",
        action="store_true",
        help="Use gradient magnitude (Sobel) instead of raw intensity — "
        "often more robust cross-modally (Ye et al., 2017)",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.05,
        help="Minimum fractional MI improvement over zero-shift to trust "
        "and auto-apply the correction",
    )
    parser.add_argument(
        "--sharpness-threshold",
        type=float,
        default=1.05,
        help="Minimum peak_sharpness (best_MI / second_best_MI at a genuinely "
        "different candidate shift) to trust the match. Catches two false-"
        "positive modes MI Gain alone misses: (1) a flat/noisy MI surface "
        "where many unrelated shifts score nearly as well as the 'winner' "
        "— the reported shift is then essentially arbitrary even if its "
        "gain over zero-shift looks fine; (2) a small field with too little "
        "content for a reliable histogram, where MI can look inflated by "
        "chance. Default 1.05 requires the peak to be at least 5%% better "
        "than any distinct alternative.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CO-REGISTRATION — MUTUAL INFORMATION METHOD")
    print("(Suri & Reinartz, 2010, IEEE TGRS 48(2):939-949)")
    print("=" * 60)

    pairs = pd.read_csv(args.pairs_file)
    results = []

    for idx, row in pairs.iterrows():
        field_id = row["field_id"]
        print(f"[{idx + 1}/{len(pairs)}] {field_id}", end=" ... ")
        try:
            s1_arr, s2_arr, s1_transform, pixel_size = read_overlap(
                row["s1_export_path"], row["s2_matched_path"]
            )
            s1_p = preprocess(s1_arr, args.use_gradient)
            s2_p = preprocess(s2_arr, args.use_gradient)

            dy, dx, mi, nmi, mi_zero, sharpness = estimate_shift_mi(
                s1_p, s2_p, max_shift=args.max_shift, bins=args.bins
            )

            improvement = (
                (mi - mi_zero) / abs(mi_zero) if mi_zero and mi_zero != 0 else float("inf")
            )
            # A match is only trustworthy if BOTH hold: (1) meaningfully better than doing
            # nothing (MI gain), AND (2) a genuinely distinct peak, not one of many similarly
            # -scoring candidates (sharpness). Either alone can be misleading — see the two
            # false-positive modes documented on --sharpness-threshold below.
            confident = (
                improvement >= args.improvement_threshold and sharpness >= args.sharpness_threshold
            )
            shift_mag_m = np.hypot(dy, dx) * pixel_size
            flag = (
                "ok"
                if confident
                else (
                    "low_sharpness" if sharpness < args.sharpness_threshold else "low_improvement"
                )
            )
            if abs(dy) >= args.max_shift or abs(dx) >= args.max_shift:
                flag = "hit_search_boundary_review"

            applied = False
            if flag == "ok" and (dy != 0 or dx != 0):
                corrected_path = row["s1_export_path"].replace(".tif", "_mi_coreg.tif")
                apply_shift_and_write(row["s1_export_path"], corrected_path, dy, dx)
                applied = True

            print(
                f"shift=({dy},{dx})px (~{shift_mag_m:.1f}m) NMI={nmi:.4f} "
                f"MI_gain={improvement:.2%} sharpness={sharpness:.2f}x flag={flag}"
            )

            results.append(
                {
                    "field_id": field_id,
                    "shift_y_px": dy,
                    "shift_x_px": dx,
                    "shift_magnitude_m": shift_mag_m,
                    "mi": mi,
                    "nmi": nmi,
                    "mi_improvement_pct": improvement,
                    "peak_sharpness": sharpness,
                    "correction_applied": applied,
                    "flag": flag,
                }
            )
        except Exception as e:
            print(f"FAILED ({e})")
            results.append(
                {
                    "field_id": field_id,
                    "shift_y_px": None,
                    "shift_x_px": None,
                    "shift_magnitude_m": None,
                    "mi": None,
                    "nmi": None,
                    "mi_improvement_pct": None,
                    "peak_sharpness": None,
                    "correction_applied": False,
                    "flag": "failed",
                }
            )

    log_df = pd.DataFrame(results)
    log_df.to_csv(COREG_LOG, index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total fields                : {len(log_df)}")
    print(f"Corrections applied           : {int(log_df['correction_applied'].sum())}")
    print(f"Low improvement (unfixed)     : {(log_df['flag'] == 'low_improvement').sum()}")
    print(f"Low sharpness / ambiguous peak: {(log_df['flag'] == 'low_sharpness').sum()}")
    print(
        f"Hit search boundary (review)  : {(log_df['flag'] == 'hit_search_boundary_review').sum()}"
    )
    print(f"Failed                        : {(log_df['flag'] == 'failed').sum()}")
    print(f"Written to                    : {COREG_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
