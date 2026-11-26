"""Train selected Sentinel yield models and compare matching held-out fields."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_predictions(root, predictions):
    for country in sorted({key[0] for key in predictions}):
        modes = sorted({key[2] for key in predictions if key[0] == country})
        figure, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 5), squeeze=False)
        for axis, mode in zip(axes[0], modes):
            data = pd.concat(
                [data for (c, _, m), data in predictions.items() if (c, m) == (country, mode)]
            )
            lower = min(data.actual.min(), data.pred.min())
            upper = max(data.actual.max(), data.pred.max())
            axis.scatter(data.actual, data.pred, s=14, alpha=0.6)
            axis.plot([lower, upper], [lower, upper], "r--")
            axis.set(
                xlabel="Actual yield (t/ha)",
                ylabel="Predicted yield (t/ha)",
                title=f"{country} | {mode} | {len(data)} field predictions",
            )
        figure.tight_layout()
        figure.savefig(root / f"pred_vs_actual_{country}.png", dpi=130)
        plt.close(figure)


def collect_fold(args, trainer, fold, out, extra):
    config = dict(
        trainer=trainer.name,
        trainer_hash=hashlib.sha256(trainer.read_bytes()).hexdigest(),
        split=str(fold.resolve()),
        npy=str(args.npy.resolve()),
        extra=extra,
        accum=args.accum,
    )
    manifest = fold / "split_manifest.csv"
    if manifest.is_file():
        config["split_hash"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config_path = out / "run_config.json"
    if not args.summary_only:
        if config_path.exists():
            old = json.loads(config_path.read_text(encoding="utf-8"))
            if config != old:
                raise ValueError(
                    f"{out} contains a different configuration. Use a new --runs directory."
                )
        else:
            if out.exists() and any(out.iterdir()):
                raise ValueError(f"{out} already contains artifacts. Use a new --runs directory.")
            out.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    runs, predictions = {}, {}
    for mode in args.modes:
        result = out / f"results_{mode}.json"
        pred_file = out / f"preds_{mode}.csv"
        if not args.summary_only and not (result.is_file() and pred_file.is_file()):
            print(f"=== {fold.parent.name} {fold.name} mode {mode} ===", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(trainer),
                    "--npy",
                    str(args.npy),
                    "--split-dir",
                    str(fold),
                    "--mode",
                    mode,
                    "--accum",
                    str(args.accum),
                    "--out",
                    str(out),
                    *extra,
                ],
                check=True,
            )
        runs[mode] = json.loads(result.read_text(encoding="utf-8"))
        if runs[mode]["mode"] != mode:
            raise ValueError(f"Mode mismatch in {result}")
        data = pd.read_csv(pred_file).sort_values("field").reset_index(drop=True)
        if data.field.duplicated().any() or len(data) != runs[mode]["test"]["n"]:
            raise ValueError(f"Invalid field predictions in {pred_file}")
        if not np.isfinite(data[["actual", "pred"]].to_numpy()).all():
            raise ValueError(f"Non-finite predictions in {pred_file}")
        predictions[mode] = data
    if "s2" in runs and "fused" in runs:
        if runs["s2"].get("field_ids") != runs["fused"].get("field_ids"):
            raise ValueError(f"The two modes used different split fields in {fold}")
        columns = ["field", "actual"]
        if not predictions["s2"][columns].equals(predictions["fused"][columns]):
            raise ValueError(f"The two modes used different test labels in {fold}")
    return runs, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", type=Path, default=Path("npy"))
    parser.add_argument("--folds", type=Path, default=Path("folds"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--modes", nargs="+", default=["s2", "fused"])
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--fold-ids", nargs="+", type=int)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--extra", default="", help="extra quoted arguments for the trainer")
    parser.add_argument("--trainer", default="train_aff.py")
    args = parser.parse_args()
    trainer = Path(__file__).resolve().with_name(args.trainer)
    if not trainer.is_file():
        parser.error(f"Trainer does not exist: {trainer}")
    if len(set(args.modes)) != len(args.modes):
        parser.error("Duplicate model modes")
    if not args.folds.is_dir():
        parser.error(f"Fold directory does not exist: {args.folds}")
    extra = shlex.split(args.extra)

    countries = args.countries or sorted(d.name for d in args.folds.iterdir() if d.is_dir())
    rows, all_predictions = [], {}
    selected = []
    for country in countries:
        country_path = args.folds / country
        if not country_path.is_dir():
            parser.error(f"No folds for {country}")
        ids = (
            args.fold_ids
            if args.fold_ids is not None
            else sorted(
                int(d.name.removeprefix("fold_"))
                for d in country_path.iterdir()
                if d.is_dir() and re.fullmatch(r"fold_\d+", d.name)
            )
        )
        for k in ids:
            fold = country_path / f"fold_{k}"
            if not fold.is_dir():
                parser.error(f"Fold does not exist: {fold}")
            selected.append((country, k, fold))
    if not selected:
        parser.error("No matching folds")
    for country, k, fold in selected:
        results, predictions = collect_fold(
            args, trainer, fold, args.runs / country / fold.name, extra
        )
        for mode, result in results.items():
            rows.append(
                dict(
                    country=country,
                    fold=k,
                    mode=mode,
                    epochs=result["epochs_run"],
                    rmse=result["test"]["rmse"],
                    r2=result["test"]["r2"],
                    n_test=result["test"]["n"],
                    baseline=result.get("baseline_rmse"),
                    train_rmse=result.get("train", {}).get("rmse"),
                    train_r2=result.get("train", {}).get("r2"),
                    val_rmse=result.get("val", {}).get("rmse"),
                    val_r2=result.get("val", {}).get("r2"),
                )
            )
            all_predictions[country, k, mode] = predictions[mode]
    summary = pd.DataFrame(rows).sort_values(["country", "mode", "fold"])
    summary.to_csv(args.runs / "all_folds.csv", index=False)
    print(
        summary.groupby(["country", "mode"])[["rmse", "r2"]]
        .agg(["mean", "std", "count"])
        .round(3)
        .to_string()
    )
    for country, grouped in summary.groupby("country"):
        paired = grouped.pivot(index="fold", columns="mode", values="rmse")
        if {"s2", "fused"} <= set(paired.columns):
            differences = paired["s2"] - paired["fused"]
            print(
                f"{country}: mean paired reduction {differences.mean():+.3f} t/ha, "
                f"fused wins {(differences > 0).sum()}/{len(differences)} folds"
            )
    plot_predictions(args.runs, all_predictions)


if __name__ == "__main__":
    main()
