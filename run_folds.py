"""
run_folds.py - train s2 (optical-only) and fused on every fold of every country, then summarise.
Resumable: a (country, fold, mode) with runs/<country>/fold_<k>/results_<mode>.json is skipped.
Summary: mean +/- std of TEST rmse / r2 per country and mode, plus paired Wilcoxon (s2 vs fused) on per-fold test RMSE.

python run_folds.py --npy npy --folds folds --runs runs --modes s2 fused
python run_folds.py --countries Germany --fold-ids 0 1 2     # subset / quick test
python run_folds.py --summary-only
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def plot_pred(runs, countries, modes):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for c in countries:
        fig, axs = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 5), squeeze=False)
        for ax, m in zip(axs[0], modes):
            fs = glob.glob(os.path.join(runs, c, "fold_*", f"preds_{m}.csv"))
            if not fs: ax.set_title(f"{m}: no predictions"); continue
            d = pd.concat([pd.read_csv(f) for f in fs])
            err = d.pred - d.actual
            rmse = float(np.sqrt((err ** 2).mean())); r2 = 1 - (err ** 2).sum() / ((d.actual - d.actual.mean()) ** 2).sum()
            ax.scatter(d.actual, d.pred, s=14, alpha=0.6)
            lo, hi = min(d.actual.min(), d.pred.min()), max(d.actual.max(), d.pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", label="1:1")
            ax.set_xlabel("actual field yield (t/ha)"); ax.set_ylabel("predicted yield (t/ha)")
            ax.set_title(f"{c} - {m}\nRMSE {rmse:.2f} t/ha, R2 {r2:.2f}, n={len(d)}"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(runs, f"pred_vs_actual_{c}.png"), dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="npy"); ap.add_argument("--folds", default="folds"); ap.add_argument("--runs", default="runs")
    ap.add_argument("--modes", nargs="+", default=["s2", "fused"]); ap.add_argument("--countries", nargs="+", default=None)
    ap.add_argument("--fold-ids", nargs="+", type=int, default=None); ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--extra", default="", help='extra args for the trainer, e.g. --extra "--gate-rule off --tdrop 0"')
    ap.add_argument("--trainer", default="train_fusion_v2.py", help="training script in the same folder")
    a = ap.parse_args()
    trainer = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.trainer)

    countries = a.countries or sorted(os.listdir(a.folds))
    countries = [c for c in countries if os.path.isdir(os.path.join(a.folds, c))]
    for c in countries:
        ids = a.fold_ids if a.fold_ids is not None else sorted(int(x.split("_")[1]) for x in os.listdir(os.path.join(a.folds, c)))
        for k in ids:
            for m in a.modes:
                out = os.path.join(a.runs, c, f"fold_{k}")
                if a.summary_only or os.path.exists(os.path.join(out, f"results_{m}.json")): continue
                print(f"=== {c} fold {k} mode {m} ===", flush=True)
                subprocess.run([sys.executable, trainer, "--npy", a.npy, "--split-dir", os.path.join(a.folds, c, f"fold_{k}"),
                                "--mode", m, "--accum", str(a.accum), "--out", out] + a.extra.split(), check=False)

    rows = []
    for f in glob.glob(os.path.join(a.runs, "*", "fold_*", "results_*.json")):
        r = json.load(open(f)); parts = f.split(os.sep)
        rows.append(dict(country=parts[-3], fold=int(parts[-2].split("_")[1]), mode=r["mode"], epochs=r["epochs_run"],
                         rmse=r["test"]["rmse"], r2=r["test"]["r2"], n_test=r["test"]["n"],
                         baseline=r.get("baseline_rmse"),
                         train_rmse=r.get("train", {}).get("rmse"), train_r2=r.get("train", {}).get("r2"),
                         val_rmse=r.get("val", {}).get("rmse"), val_r2=r.get("val", {}).get("r2")))
    if not rows: print("no results yet"); return
    df = pd.DataFrame(rows).sort_values(["country", "mode", "fold"]); df.to_csv(os.path.join(a.runs, "all_folds.csv"), index=False)
    print("\n" + df.groupby(["country", "mode"])[["rmse", "r2"]].agg(["mean", "std", "count"]).round(3).to_string())

    if df["baseline"].notna().any():
        b = df.dropna(subset=["baseline"]).drop_duplicates(["country", "fold"]).groupby("country")["baseline"].agg(["mean", "std", "count"]).round(3)
        print("\nMean-baseline test RMSE (predict train-mean yield) - the model must beat this:"); print(b.to_string())
    if df["train_rmse"].notna().any():
        t = df.groupby(["country", "mode"])[["train_rmse", "val_rmse", "rmse", "train_r2", "val_r2", "r2"]].mean().round(3)
        print("\nTrain / Val / Test, field level, mean over folds (best checkpoint):")
        print(t.rename(columns={"rmse": "test_rmse", "r2": "test_r2"}).to_string())
    plot_pred(a.runs, sorted(df.country.unique()), a.modes)

    from scipy.stats import wilcoxon
    print("\nPaired test on per-fold TEST RMSE (s2 vs fused, same folds):")
    for c, g in df.groupby("country"):
        p = g.pivot(index="fold", columns="mode", values="rmse").dropna()
        if {"s2", "fused"} <= set(p.columns) and len(p) >= 3:
            d = p["s2"] - p["fused"]                       # > 0 means fused better
            try: pv = wilcoxon(p["s2"], p["fused"]).pvalue
            except ValueError: pv = float("nan")
            print(f"{c}: folds={len(p)} s2={p['s2'].mean():.3f} fused={p['fused'].mean():.3f} "
                  f"gain={d.mean() / p['s2'].mean():+.1%} fused_wins={(d > 0).sum()}/{len(p)} p={pv:.4f}")

if __name__ == "__main__":
    main()