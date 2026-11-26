# Sentinel wheat yield prediction

This repository reproduces the YieldSAT wheat input ablation for 263 fields across 27 farms in Argentina and Brazil. The six trained variants share the same farm splits. The frozen reference run supplies the all-input AFF and country training mean.

| Path | Purpose |
| --- | --- |
| `export_yieldsat_fields.py` | Export model-ready arrays from YieldSAT Raw and Preprocessed files. |
| `fetch_sar_rtc.py` | Fetch and align Sentinel-1 VV/VH. |
| `experiments/wheat_ablation.py` | Verify, train, predict, and report the input ablation. |
| `experiments/run_loro.py` | Rebuild the earlier depth comparison and the two local reference rows. |
| `reproduction/reference/` | Original manifest, folds, reference predictions, validation metrics, prepared-array SHA-256 checksums, and radar acquisition records. |
| `reproduction/saved/` | Saved predictions and validation metrics for regenerating the original table. |

## Set up and regenerate the original table

Use a full clone of the reproduction branch so the reference commit is available. Commands below use PowerShell from the repository root and Python 3.12. CUDA is required for training. The original environment used PyTorch 2.11.0+cu128.

```powershell
git clone --branch codex/yieldsat-pilot https://github.com/BhavyaFattania/sentinel.git
Set-Location sentinel
py -3.12 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-wheat.txt

$report = New-Item -ItemType Directory -Force runs/saved-report
Copy-Item reproduction/saved/* $report.FullName -Recurse -Force
& $py experiments/wheat_ablation.py --report-only --out $report.FullName
```

The saved inputs reproduce the original score table without training. Compare the resulting CSV with `reproduction/saved/input_ablation_scores.csv`.

## Prepare data and retrain

Keep the original [YieldSAT](https://yieldsat.github.io/) Raw and Preprocessed directories available locally. The raster data are external inputs. Use the commands below to rebuild the prepared arrays, or obtain a copy of the original prepared arrays for exact input parity.

```powershell
$yieldSat = "D:\path\to\YieldSAT"
New-Item -ItemType Directory -Force data | Out-Null
Copy-Item reproduction/reference/wheat-field-manifest.csv data/
foreach ($country in @("Argentina", "Brazil")) {
    $nc = Join-Path $yieldSat "Preprocessed/$country/merge_s2-soil-dem-weather-coords.nc"
    & $py export_yieldsat_fields.py $nc --raw (Join-Path $yieldSat Raw) --out data/fields --crop wheat --resume
    if ($LASTEXITCODE -ne 0) { throw "Export failed for $country" }
}
& $py fetch_sar_rtc.py --npy data/fields --min-slots 4 --sources reproduction/reference/radar_sources.jsonl --status-csv data/radar-status.csv
if ($LASTEXITCODE -ne 0) { throw "Radar replay failed" }
& $py experiments/wheat_ablation.py --check
& $py experiments/wheat_ablation.py
```

Three Argentine wheat fields lacked qualifying radar coverage in the original run. The frozen manifest selects 123 Argentine and 140 Brazilian fields. Check the status CSV for any newly missing eligible fields. The radar command replays the original acquisition IDs and slots. Verify the arrays against the recorded checksums, since catalog assets or raster processing libraries can change. The `--check` mode verifies the manifest and fold membership. Use this separate check for the actual arrays:

```powershell
Import-Csv reproduction/reference/field_sha256.csv | ForEach-Object {
    $file = Join-Path data $_.path
    if (!(Test-Path -LiteralPath $file) -or
        ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $_.sha256)) {
        throw "Prepared array differs or is missing: $file"
    }
}
```

Training writes to `runs/wheat-ablation`, including fold checkpoints, validation history, held-out predictions, CSV reports, and a plot. Override locations with `--repo`, `--data`, `--previous`, and `--out`. To regenerate the local reference rows from training, first run `experiments/run_loro.py --repo . --data data --out runs/reference`, then pass `--previous runs/reference --out runs/wheat-from-new-reference` to the ablation. Different GPU and library configurations can yield different floating-point results.

## How preprocessing and loading work

The exporter reads monthly Sentinel-2 surface reflectance, weather, static maps, and yield from each country's YieldSAT NetCDF and uses field metadata and raster information from Raw. It writes per-field NumPy arrays with a common raster grid and valid yield mask. The radar fetcher searches Microsoft Planetary Computer Sentinel-1 RTC, pairs observations within 12 days of target slots on a consistent orbit, requires at least 95% labeled-area coverage for each accepted slot, and warps VV and VH linear backscatter to the field grid. The experiment requires at least four accepted slots.

The loader caches arrays and valid pixel indices on the GPU. For each fold it estimates feature and yield scaling from training fields, replaces missing normalized input values with -1, and gathers 5×5 patches centered on labeled pixels. A field has 24 seasonal slots. Weather is shared across the pixels of its field.

## Architecture

| Input | Tensor per sample | Encoder |
| --- | --- | --- |
| Sentinel-2 optical | 12 bands × 24 slots × 5 × 5 | Conv3D with a 1 × 5 × 5 kernel, then a two-layer LSTM |
| Sentinel-1 radar | 2 polarizations × 24 slots × 5 × 5 | Separate Conv3D and two-layer LSTM |
| Weather | 24 slots × 4 variables | Two-layer LSTM |
| Terrain | 5 channels × 5 × 5 | Conv2D and a linear projection |
| Soil | 24 channels × 5 × 5 | Conv2D and a linear projection |
| Coordinates | 3 channels × 5 × 5 | Conv2D and a linear projection |

Each enabled branch produces a 64-value vector. A learned query scores the projected modality keys. Softmax weights combine the projected values into a 64-value vector. A linear output head predicts the normalized yield of the center pixel. The ablation retrains the network with the branches enabled for each input combination.

## Training and scoring

- **Splits.** Leave one entire farm out for testing. Reserve two Argentine farms and one Brazilian farm for validation. Use the remaining farms for training.
- **Epoch.** Draw 24,576 labeled pixels approximately equally across training farms and shuffle them into batches of 256. Train with pixel-level MSE, Adam at 0.001, and mixed precision. Fold seeds start at 42.
- **Checkpoint.** Train for at most 10 epochs. Starting at epoch five, stop after three epochs without a qualifying validation improvement of more than 0.015 t/ha. Restore the lowest observed validation farm-macro RMSE checkpoint. Validation samples at most 1,024 pixels per field.
- **Test.** Predict all labeled pixels on the held-out farm. Average their predictions within each field, calculate RMSE across fields in each farm, then average the farm RMSE values. The country training mean assigns each test field the mean yield of training fields from that country.
- **Published reference.** The [YieldSAT paper, Table 17](https://arxiv.org/html/2604.00940v1) reports field-level AFF RMSE of 0.78 t/ha in Argentina and 0.55 t/ha in Brazil. That benchmark used Sentinel-2 and auxiliary inputs, with its own cohort and training protocol. This repository's paired comparison uses the frozen local folds and pooled Argentina/Brazil training.
