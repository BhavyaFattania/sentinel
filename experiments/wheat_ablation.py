"""Train and score the paired YieldSAT wheat input ablation."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COUNTRIES = ("Argentina", "Brazil")


CHANNELS = {"s2": 12, "sar": 2, "weather": 4, "dem": 5, "soil": 24, "coord": 3}


RASTER = ("s2", "sar", "dem", "soil", "coord")


IMAGE = ("s2", "sar")


EPOCHS = 10


SAMPLES = 24576


BATCH = 256


VAL_PIXELS = 1024


SEED = 42


ARMS = {
    "s2": ("s2",),
    "s1_s2": ("s2", "sar"),
    "s1_s2_weather": ("s2", "sar", "weather"),
    "s1_s2_terrain": ("s2", "sar", "dem"),
    "s1_s2_soil": ("s2", "sar", "soil"),
    "s1_s2_coord": ("s2", "sar", "coord"),
}


AUX = ("weather", "dem", "soil", "coord")


LABELS = {
    "country_mean": "Country training mean",
    "s2": "S2 alone",
    "s1_s2": "S1 + S2",
    "s1_s2_weather": "S1 + S2 + weather",
    "s1_s2_terrain": "S1 + S2 + terrain",
    "s1_s2_soil": "S1 + S2 + soil",
    "s1_s2_coord": "S1 + S2 + coordinates",
    "all_inputs": "S1 + S2 + all four",
}


PAPER = {
    "Argentina": {
        "S2 optical-only 3D-LSTM": (0.92, 0.16),
        "S2 + auxiliary AFF": (0.78, 0.13),
    },
    "Brazil": {
        "S2 optical-only 3D-LSTM": (0.57, 0.23),
        "S2 + auxiliary AFF": (0.55, 0.20),
    },
}


@dataclass(frozen=True)
class Settings:
    repo: Path
    data: Path
    previous: Path
    out: Path
    check: bool
    report_only: bool


@dataclass(frozen=True)
class Fold:
    region: str
    train: list[int]
    validation: list[int]
    test: list[int]


@dataclass(frozen=True)
class ExperimentData:
    fields: list[dict[str, str]]
    groups: list[str]
    paths: list[Path]
    folds: list[Fold]
    roster: list[dict[str, str]]
    reused_predictions: list[dict[str, str]]
    protocol: dict


# Read CLI overrides while defaulting to the existing workspace layout.
def parse_settings():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--data", type=Path, default=repo / "data")
    parser.add_argument("--previous", type=Path, default=repo / "reproduction" / "reference")
    parser.add_argument("--out", type=Path, default=repo / "runs" / "wheat-ablation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify the manifests and folds without training",
    )
    mode.add_argument(
        "--report-only", action="store_true", help="Score existing predictions in --out"
    )
    return Settings(**vars(parser.parse_args()))


# Experiment files and field data


# Fingerprint a reference file used to reproduce the paired experiment.
def hash_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# Read saved experiment rows with their original CSV column names.
def read_csv(p):
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Write rows to a CSV using an atomic replacement.
def write_csv(path, rows):
    temp = path.with_suffix(".partial")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


# Write experiment metadata as JSON using an atomic replacement.
def save_json(path, data):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


# Collect finite-value sums, squared sums, and counts for normalization.
def moments(a, axis, channels):
    a = np.moveaxis(a, axis, 0).reshape(channels, -1)
    finite = np.isfinite(a)
    clean = np.where(finite, a, 0).astype(np.float64)
    return np.stack((finite.sum(1), clean.sum(1), np.square(clean).sum(1)))


class FieldBank:
    """Cache field inputs without mixing training, validation or test moments."""

    # Load field arrays, pixel indices, and normalization statistics into the cache.
    def __init__(self, paths, device):
        self.device = device
        shapes = [np.load(p / "mask.npy", mmap_mode="r").shape for p in paths]
        widths = [w + 4 for h, w in shapes]
        self.widths = torch.tensor(widths, device=device)
        sizes = [(h + 4) * (w + 4) for h, w in shapes]
        offsets = np.r_[0, np.cumsum(sizes)]
        self.offsets = torch.tensor(offsets[:-1], device=device)
        self.axis = torch.arange(5, device=device)
        self.bank = {
            k: torch.empty(
                (int(offsets[-1]), c * (24 if k in IMAGE else 1)),
                device=device,
                dtype=torch.float16 if k in IMAGE else torch.float32,
            )
            for k, c in CHANNELS.items()
            if k in RASTER
        }
        statistics, weather, items, values, actual = defaultdict(list), [], [], [], []
        self.limits = [0]
        for i, path in enumerate(paths):
            for k, c in CHANNELS.items():
                a = np.load(path / f"{k}.npy")
                statistics[k].append(moments(a, 1 if k in IMAGE or k == "weather" else 0, c))
                if k == "weather":
                    weather.append(a)
                    continue
                a = np.pad(a, ((0, 0),) * (a.ndim - 2) + ((2, 2), (2, 2)), mode="edge")
                dtype = np.float16 if k in IMAGE else np.float32
                # The optical cache uses half-scaled DNs to retain the finite float32 range.
                if k == "s2":
                    a = a * 0.5
                if k in IMAGE and np.any(np.isfinite(a) & (abs(a) > np.finfo(np.float16).max)):
                    raise ValueError(f"Input overflow: {path.name}/{k}")
                buf = np.ascontiguousarray(
                    a.reshape(self.bank[k].shape[1], sizes[i]).T, dtype=dtype
                )
                self.bank[k][int(offsets[i]) : int(offsets[i + 1])].copy_(torch.from_numpy(buf))
            y = np.load(path / "yield.npy")
            valid = np.isfinite(y) & np.load(path / "mask.npy")
            r, c = np.nonzero(valid)
            data = y[r, c]
            if len(data) == 0:
                raise ValueError(f"No labeled pixels in {path}")
            values.append(data)
            statistics["yield"].append(moments(data[None], 0, 1))
            actual.append(float(data.mean()))
            items.append(np.column_stack((np.full(len(data), i), r, c)))
            self.limits.append(self.limits[-1] + len(data))
            if (i + 1) % 50 == 0:
                print(f"Cached {i+1}/{len(paths)} fields", flush=True)
        self.weather = torch.as_tensor(np.stack(weather), device=device)
        self.items = torch.as_tensor(np.concatenate(items), device=device)
        self.targets = torch.as_tensor(np.concatenate(values), device=device)
        self.stats = {k: np.stack(v) for k, v in statistics.items()}
        self.actual = np.array(actual)

    # Derive input and yield scaling from the current fold's training fields.
    def fold_stats(self, train):
        norm = {}
        for k, stats in self.stats.items():
            count, total, squared = stats[train].sum(axis=0)
            n = np.maximum(count, 1)
            mean = total / n
            sd = np.sqrt(np.maximum(squared / n - mean**2, 0))
            sd[sd <= 1e-6 * np.maximum(np.abs(mean), 1)] = 1
            if k == "yield":
                ystat = float(mean[0]), float(sd[0])
            else:
                if k in IMAGE:
                    mean, sd = np.tile(mean, 24), np.tile(sd, 24)
                norm[k] = (
                    torch.tensor(mean, device=self.device, dtype=torch.float32),
                    torch.tensor(sd, device=self.device, dtype=torch.float32),
                )
        return norm, ystat

    # Extract and normalize model inputs and targets for selected pixel indices.
    def batch(self, ids, norm, ystat, masked, *, active=None):
        fields, r, c = self.items[ids].unbind(-1)
        widths = self.widths[fields]
        centers = self.offsets[fields] + r * widths + c
        patch = (
            centers[:, None, None] + widths[:, None, None] * self.axis[None, :, None] + self.axis
        )

        # Scale finite inputs and fill missing values with the model's sentinel value.
        def scale(a, k):
            mean, sd = norm[k]
            x = a.float()
            if k == "s2":
                x = 2 * x
            return torch.where(torch.isfinite(x), (x - mean) / sd, -1).half()

        # Arrange a satellite patch as channels by time by height by width.
        def image(k):
            raw = self.bank[k][patch]
            bands = CHANNELS[k]
            x = scale(raw, k).reshape(-1, 5, 5, 24, bands).permute(0, 4, 3, 1, 2).contiguous()
            if masked:
                valid = torch.isfinite(raw[:, 2, 2]).reshape(-1, 24, bands).all(-1)
                return x, valid
            return x

        # Arrange a static patch as channels by height by width.
        def static(k):
            return scale(self.bank[k][patch], k).permute(0, 3, 1, 2).contiguous()

        return (
            image("s2"),
            scale(self.weather[fields], "weather")
            if active is None or "weather" in active
            else None,
            static("dem") if active is None or "dem" in active else None,
            static("soil") if active is None or "soil" in active else None,
            static("coord") if active is None or "coord" in active else None,
            image("sar") if active is None or "sar" in active else None,
            (self.targets[ids] - ystat[0]) / ystat[1],
        )


# Generate a held-out test farm and separate validation farms for each fold.
def folds(groups):
    unique = sorted(set(groups))
    for j, region in enumerate(unique):
        rng = np.random.default_rng(SEED + 1000 + j)
        valid = set()
        for country, n in (("Argentina", 2), ("Brazil", 1)):
            candidates = [g for g in unique if g != region and g.startswith(country + "/")]
            valid.update(rng.choice(candidates, n, replace=False))
        tr = [i for i, g in enumerate(groups) if g != region and g not in valid]
        va = [i for i, g in enumerate(groups) if g in valid]
        te = [i for i, g in enumerate(groups) if g == region]
        assert set(groups[i] for i in tr).isdisjoint(valid | {region}) and va and te
        yield Fold(region, tr, va, te)


# Validate the model provenance, field manifest, and frozen fold assignments.
def load_experiment(settings: Settings) -> ExperimentData:
    reference = json.loads((settings.previous / "protocol.json").read_text(encoding="utf-8"))
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(settings.repo),
            "merge-base",
            "--is-ancestor",
            reference["commit"],
            "HEAD",
        ],
        check=False,
    )
    manifest = settings.data / "wheat-field-manifest.csv"
    if ancestry.returncode or hash_file(manifest) != reference["manifest_sha256"]:
        raise ValueError(
            "Reference commit is missing from checkout history or the bundled field manifest changed"
        )
    if reference["model_variants"]["aff_lstm2"] != [2, False]:
        raise ValueError("The full model must be the two-layer AFF with final-state pooling")
    if (SAMPLES, BATCH, EPOCHS) != tuple(
        reference[k] for k in ("samples_per_epoch", "batch", "max_epochs")
    ):
        raise ValueError("Training constants differ from the full model")
    config = dict(
        reference=reference,
        runner_sha256=hash_file(Path(__file__)),
        fold_manifest_sha256=hash_file(settings.previous / "fold_manifest.csv"),
        full_predictions_sha256=hash_file(settings.previous / "loro_predictions.csv"),
        arms={k: list(v) for k, v in ARMS.items()},
        full_model="reuse paired aff_lstm2 predictions",
        recurrence="two LSTM layers, 64 hidden units per active temporal branch",
    )
    fields = sorted(
        [r for r in read_csv(manifest) if r["country"] in COUNTRIES and r["status"] == "ready"],
        key=lambda r: (r["country"], r["field"]),
    )
    groups = [f"{r['country']}/{r['provider']}/{r['farm']}" for r in fields]
    paths = [settings.data / "fields" / r["country"] / r["field"] for r in fields]
    stages = list(folds(groups))
    assert len(fields) == 263 and len(stages) == 27
    roster = []
    for j, fold in enumerate(stages):
        for role, ii in (
            ("train", fold.train),
            ("val", fold.validation),
            ("test", fold.test),
        ):
            roster.extend(
                dict(
                    fold=str(j + 1),
                    heldout=fold.region,
                    role=role,
                    country=fields[i]["country"],
                    farm=groups[i],
                    field=fields[i]["field"],
                )
                for i in ii
            )
    if roster != read_csv(settings.previous / "fold_manifest.csv"):
        raise ValueError("Generated fold assignments do not match the full-model run")

    existing = read_csv(settings.previous / "loro_predictions.csv")
    reused = []
    for row in existing:
        if row["arm"] in ("aff_lstm2", "country_mean"):
            reused.append(
                {
                    **row,
                    "arm": "all_inputs" if row["arm"] == "aff_lstm2" else row["arm"],
                }
            )
    assert len(reused) == 526
    return ExperimentData(fields, groups, paths, stages, roster, reused, config)


# Neural network and fold training


# ================= BTech AFF code (cleaned; ImgBranch64 = AFFS2Branch64 with cin channels) =================
class ImgBranch64(nn.Module):
    # Build the convolution and LSTM for one satellite time series.
    def __init__(self, cin, p=0.2):
        super().__init__()
        self.conv3d = nn.Conv3d(cin, 64, kernel_size=(1, 5, 5))
        self.bn = nn.BatchNorm3d(64)
        self.leaky = nn.LeakyReLU(inplace=True)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 64)
        self.fc_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    # Encode a sequence of satellite patches into a 64-feature vector.
    def forward(self, x):  # [B,C,T,5,5]
        x = self.leaky(self.bn(self.conv3d(x)))
        if x.shape[-2:] != (1, 1):
            raise RuntimeError(f"Expected 1x1 after conv, got {x.shape[-2:]}")
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B,T,64]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.dropout(self.relu(self.fc_bn(self.fc(x))))


class TemporalBranch64(nn.Module):
    # Build the LSTM for a non-image time series such as weather.
    def __init__(self, input_dim, p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)

    # Summarize a temporal input using the LSTM's final output.
    def forward(self, x):  # [B,T,C]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.dropout(self.relu(self.bn(self.fc(x))))


class SpatialBranch64(nn.Module):
    # Build a spatial encoder for a static input such as soil or terrain.
    def __init__(self, input_dim, p=0.2):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, 64, kernel_size=5)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 64)
        self.fc_bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(p)

    # Convert a static patch into a 64-feature vector.
    def forward(self, x):  # [B,C,5,5]
        x = self.relu(self.bn(self.conv(x))).squeeze(-1).squeeze(-1)
        return self.dropout(self.relu(self.fc_bn(self.fc(x))))


class LearnableQueryAttention(nn.Module):
    # Initialize the learned query and the attention projections.
    def __init__(self, dim=64, dropout=0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

    # Use the learned query to compute a weighted sum of modality features.
    def forward(self, reps):
        x = torch.stack(reps, dim=1)
        B = x.shape[0]
        q = self.q_proj(self.query).expand(B, -1, -1)
        k, v = self.k_proj(x), self.v_proj(x)
        w = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1))
        return torch.matmul(w, v).squeeze(1)


class MirandaFeatureFusion(nn.Module):
    # Construct the input encoders, attention fusion, and yield head.
    def __init__(self, soil_ch=24, use_sar=False, drop=(), p=0.2):
        super().__init__()
        d = set(drop)  # drop = branches to remove (ablation)
        self.s2 = None if "s2" in d else ImgBranch64(12, p)
        self.weather = None if "weather" in d else TemporalBranch64(4, p)
        self.dem = None if "dem" in d else SpatialBranch64(5, p)
        self.soil = None if "soil" in d else SpatialBranch64(soil_ch, p)
        self.coord = None if "coord" in d else SpatialBranch64(3, p)
        self.attn = LearnableQueryAttention(64, p)
        self.out = nn.Linear(64, 1)
        self.sar = ImgBranch64(2, p) if use_sar else None

    # Encode the enabled modalities and predict yield for the center pixel.
    def forward(self, s2_x, weather_x, topo_x, soil_x, coord_x, sar_x=None):
        reps = []
        if self.s2 is not None:
            reps.append(self.s2(s2_x))
        if self.sar is not None:
            reps.append(self.sar(sar_x))
        if self.weather is not None:
            reps.append(self.weather(weather_x))
        if self.dem is not None:
            reps.append(self.dem(topo_x))
        if self.soil is not None:
            reps.append(self.soil(soil_x))
        if self.coord is not None:
            reps.append(self.coord(coord_x))
        return self.out(self.attn(reps)).squeeze(-1)


# Pass the modality tensors from a batch through the model.
def fwd(model, b):
    s2, wx, dem, soil, co, sar, _ = b
    return model(s2, wx, dem, soil, co, sar)


# Initialize the AFF network and disable the branches outside this input combination.
def build_model(arm, seed, device):
    torch.manual_seed(seed)
    net = MirandaFeatureFusion(use_sar=True)
    for channel in ("sar", *AUX):
        if channel not in ARMS[arm]:
            setattr(net, channel, None)
    torch.manual_seed(seed + 1)
    return net.to(device)


# Predict field yields by averaging pixel estimates in physical units.
@torch.inference_mode()
def predict(model, bank, fields, norm, ystat, cap=None, *, active=None):
    model.eval()
    pred = []
    for i in fields:
        lo, hi = bank.limits[i : i + 2]
        if cap is not None and hi - lo > cap:
            batches = (
                torch.linspace(lo, hi - 1, cap, device=bank.device, dtype=torch.float64)
                .long()
                .split(1024)
            )
            n = cap
        else:
            batches = (slice(j, min(j + 1024, hi)) for j in range(lo, hi, 1024))
            n = hi - lo
        total = torch.zeros((), device=bank.device)
        for idx in batches:
            batch = bank.batch(idx, norm, ystat, False, active=active)
            with torch.autocast("cuda", dtype=torch.float16):
                total += fwd(model, batch).float().sum()
        pred.append(total * (ystat[1] / n) + ystat[0])
    result = torch.stack(pred).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite field predictions")
    return result


# Sample an approximately equal number of training pixels per farm.
def region_batches(bank, fields, groups, rng):
    grouped = defaultdict(list)
    for i in fields:
        grouped[groups[i]].append(i)
    out = []
    for j, g in enumerate(sorted(grouped)):
        count = SAMPLES // len(grouped) + (j < SAMPLES % len(grouped))
        idx = np.concatenate([np.arange(bank.limits[i], bank.limits[i + 1]) for i in grouped[g]])
        out.append(rng.choice(idx, count, replace=len(idx) < count))
    return rng.permutation(np.concatenate(out))


# Calculate the mean of the individual validation farm RMSE values.
def val_score(actual, pred, groups):
    sq = (actual - pred) ** 2
    return float(
        np.mean(
            [np.sqrt(np.mean(sq[np.array([x == g for x in groups])])) for g in sorted(set(groups))]
        )
    )


# Train one fold and restore the checkpoint with the lowest validation RMSE.
def fit(bank, train, val, groups, norm, ystat, arm, seed, *, active):
    start = time.monotonic()
    model = build_model(arm, seed, bank.device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    best, reference, bad = float("inf"), float("inf"), 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ids = torch.as_tensor(region_batches(bank, train, groups, rng), device=bank.device)
        losses = []
        for idx in ids.split(BATCH):
            batch = bank.batch(idx, norm, ystat, False, active=active)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = fwd(model, batch)
            loss = nn.functional.mse_loss(pred.float(), batch[-1])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.detach())
        val_pred = predict(model, bank, val, norm, ystat, VAL_PIXELS, active=active)
        error = val_score(bank.actual[val], val_pred, [groups[i] for i in val])
        loss = float(torch.stack(losses).mean())
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        history.append(dict(epoch=epoch, train_mse=loss, val_macro_rmse=error))
        if error < best:
            best = error
            selected = epoch
            state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
        if error < reference - 0.015:
            reference, bad = error, 0
        else:
            bad += 1
        if epoch >= 5 and bad >= 3:
            break
    model.load_state_dict(state)
    return (
        model,
        history,
        dict(
            best_epoch=selected,
            val_macro_rmse=best,
            epochs=len(history),
            seconds=round(time.monotonic() - start, 2),
        ),
    )


# Train missing fold and input combinations, then write the combined predictions.
def run_training(data: ExperimentData, out: Path):
    source_actual = {row["field"]: float(row["actual"]) for row in data.reused_predictions}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the full ablation")
    device = torch.device("cuda")
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != data.protocol:
        raise ValueError("Experiment configuration changed. Use a new output directory")
    save_json(protocol, data.protocol)

    write_csv(out / "fold_manifest.csv", data.roster)

    print(
        f"{len(data.fields)} wheat fields, {len(data.folds)} paired folds | {torch.cuda.get_device_name()}",
        flush=True,
    )
    bank = FieldBank(data.paths, device)
    if not np.array_equal(
        bank.actual, np.asarray([source_actual[f["field"]] for f in data.fields])
    ):
        raise ValueError("Measured field yields differ from full-model predictions")
    # Compare every enabled input against the prior loader on one mini-batch.
    norm, ystat = bank.fold_stats(data.folds[0].train)
    idx = torch.tensor([0, bank.limits[20], bank.limits[120]], device=device)
    inputs = bank.batch(idx, norm, ystat, False)
    for active in ARMS.values():
        for k, val, ref in zip(
            ("s2", "weather", "dem", "soil", "coord", "sar"),
            bank.batch(idx, norm, ystat, False, active=active),
            inputs,
        ):
            if k in active:
                torch.testing.assert_close(val, ref, atol=0, rtol=0)
            else:
                assert val is None
    print(
        f"Ablation cache verified against the full loader; {torch.cuda.memory_allocated()/2**30:.2f} GiB.",
        flush=True,
    )

    predictions = list(data.reused_predictions)
    for j, fold in enumerate(data.folds):
        t = time.monotonic()
        target = out / f"fold-{j+1:02d}"
        target.mkdir(exist_ok=True)
        norm, ystat = bank.fold_stats(fold.train)
        for arm, active in ARMS.items():
            result = target / f"{arm}-predictions.csv"
            if result.exists():
                predictions.extend(read_csv(result))
                continue
            net, history, info = fit(
                bank,
                fold.train,
                fold.validation,
                data.groups,
                norm,
                ystat,
                arm,
                SEED + j,
                active=active,
            )
            write_csv(target / f"{arm}-history.csv", history)
            save_json(target / f"{arm}-validation.json", info)
            torch.save(net.state_dict(), target / f"{arm}-weights.pt")
            estimates = predict(net, bank, fold.test, norm, ystat, active=active)
            entries = [
                dict(
                    fold=j + 1,
                    region=fold.region,
                    country=data.fields[i]["country"],
                    field=data.fields[i]["field"],
                    arm=arm,
                    actual=float(bank.actual[i]),
                    predicted=float(p),
                    selected="",
                )
                for i, p in zip(fold.test, estimates)
            ]
            write_csv(result, entries)
            predictions.extend(entries)
            del net
            print(
                f"Fold {j+1:02d}/27 {arm} | checkpoint {info['best_epoch']} | {info['seconds']:.1f} s",
                flush=True,
            )
        print(
            f"Completed fold {j+1}/27 {fold.region} in {time.monotonic()-t:.1f} s",
            flush=True,
        )
    assert len(predictions) == 263 * (len(ARMS) + 2)
    write_csv(out / "loro_input_predictions.csv", predictions)
    print(
        f"Finished {len(data.folds)} folds and {len(predictions)} predictions",
        flush=True,
    )


# Scoring and entry point


# Calculate farm-averaged and pooled error metrics for a group of predictions.
def score(block):
    err = block.predicted.to_numpy() - block.actual.to_numpy()
    squared = err**2
    per_farm = (
        block.assign(squared=squared, abs_error=abs(err))
        .groupby("region")
        .agg(mse=("squared", "mean"), mae=("abs_error", "mean"))
    )
    dev = block.actual - block.actual.mean()
    return dict(
        fields=len(block),
        farms=len(per_farm),
        loro_rmse_t_ha=float(np.sqrt(per_farm.mse).mean()),
        farm_macro_mae_t_ha=float(per_farm.mae.mean()),
        pooled_rmse_t_ha=float(np.sqrt(squared.mean())),
        pooled_mae_t_ha=float(abs(err).mean()),
        pooled_r2=float(1 - squared.sum() / (dev**2).sum()),
    )


# Check prediction coverage and write the ablation scores, comparisons, and plot.
def write_report(path: Path, previous: Path):
    data = pd.read_csv(path / "loro_input_predictions.csv")
    roster = pd.read_csv(path / "fold_manifest.csv")
    test = roster[roster.role == "test"]
    expected = set(zip(test.fold, test.heldout, test.field))
    assert len(expected) == 263 and set(data.arm) == set(LABELS) and data.region.nunique() == 27
    assert set(data.country) == set(COUNTRIES)
    for _, a in data.groupby("arm"):
        assert len(a) == len(expected) and set(zip(a.fold, a.region, a.field)) == expected
    assert data.groupby("field").actual.nunique().eq(1).all()
    assert np.isfinite(data[["predicted", "actual"]].to_numpy()).all()
    assert all(f.groupby("farm").role.nunique().eq(1).all() for _, f in roster.groupby("fold"))
    scores = []
    for country in ("All", *COUNTRIES):
        area = data if country == "All" else data[data.country == country]
        for arm, label in LABELS.items():
            scores.append(
                dict(
                    country=country,
                    arm=arm,
                    inputs=label,
                    **score(area[area.arm == arm]),
                )
            )
    scores = pd.DataFrame(scores)
    scores.to_csv(path / "input_ablation_scores.csv", index=False)

    by_farm = (
        data.assign(sq=(data.predicted - data.actual) ** 2)
        .groupby(["country", "region", "arm"], as_index=False)
        .agg(fields=("field", "size"), mse=("sq", "mean"))
    )
    by_farm["rmse_t_ha"] = np.sqrt(by_farm.pop("mse"))
    by_farm.to_csv(path / "input_ablation_by_farm.csv", index=False)
    RMSE = by_farm.pivot(index=["country", "region", "fields"], columns="arm", values="rmse_t_ha")
    comparisons = {
        "s2": "country_mean",
        "s1_s2": "s2",
        **{name: "s1_s2" for name in list(LABELS)[3:]},
    }
    result = []
    boot = np.random.default_rng(11123)
    for country in ("All", *COUNTRIES):
        region = RMSE if country == "All" else RMSE.xs(country, level="country")
        draws = boot.integers(0, len(region), size=(20000, len(region)))
        for arm, base in comparisons.items():
            deltas = (region[base] - region[arm]).to_numpy()
            q = np.quantile(deltas[draws].mean(axis=1), [0.025, 0.975])
            result.append(
                dict(
                    country=country,
                    arm=arm,
                    reference=base,
                    rmse_reduction_t_ha=float(deltas.mean()),
                    farms_better=int(sum(deltas > 0)),
                    farms=len(deltas),
                    descriptive_bootstrap_low=float(q[0]),
                    descriptive_bootstrap_high=float(q[1]),
                )
            )
    gains = pd.DataFrame(result)
    gains.to_csv(path / "input_ablation_paired_gains.csv", index=False)
    nested_parts = []
    nested_choices = []
    candidates = list(LABELS)[2:]
    for fold, part in data.groupby("fold"):
        folder = f"fold-{fold:02d}"
        scores_by_validation = {}
        for arm in candidates:
            if arm == "all_inputs":
                v = json.loads((previous / folder / "validation.json").read_text(encoding="utf-8"))
                scores_by_validation[arm] = v["arms"]["aff_lstm2"]["val_macro_rmse"]
            else:
                v = json.loads(
                    (path / folder / f"{arm}-validation.json").read_text(encoding="utf-8")
                )
                scores_by_validation[arm] = v["val_macro_rmse"]
        picked = min(candidates, key=scores_by_validation.__getitem__)
        nested_parts.append(part[part.arm == picked])
        nested_choices.append(
            dict(
                fold=int(fold),
                region=part.region.iloc[0],
                arm=picked,
                validation_rmse=scores_by_validation[picked],
            )
        )
    picked = pd.concat(nested_parts)
    nested = []
    for country in ("All", *COUNTRIES):
        block = picked if country == "All" else picked[picked.country == country]
        nested.append(dict(country=country, **score(block)))
    pd.DataFrame(nested).to_csv(path / "nested_validation_selected_scores.csv", index=False)
    pd.DataFrame(nested_choices).to_csv(path / "nested_validation_choices.csv", index=False)
    print("Models selected with validation farms only:")
    print(pd.DataFrame(nested).round(4).to_string(index=False))
    print(pd.DataFrame(nested_choices).arm.value_counts().to_string())

    paper_rows = [
        dict(
            country=c,
            model=m,
            mean_rmse_t_ha=v[0],
            std_t_ha=v[1],
            source="YieldSAT appendix Table 17, field LORO",
        )
        for c, ref in PAPER.items()
        for m, v in ref.items()
    ]
    pd.DataFrame(paper_rows).to_csv(path / "yieldsat_reference.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3), sharey=True)
    for ax, country in zip(axes, ("All", *COUNTRIES)):
        d = scores[scores.country == country]
        y = np.arange(len(d))
        vals = d.loro_rmse_t_ha.to_numpy()
        colors = ["#7d8792", "#5d748a", "#416f90", *(["#248477"] * 4), "#865f34"]
        ax.barh(y, vals, 0.66, color=colors)
        ax.set_yticks(y, d.inputs)
        if country in PAPER:
            ref = PAPER[country]["S2 + auxiliary AFF"][0]
            ax.axvline(ref, ls="--", lw=1.2, color="#984d5d", label=f"Published AFF {ref:.2f}")
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        for i, v in enumerate(vals):
            ax.text(v + 0.015, i, f"{v:.3f}", ha="left", va="center", fontsize=9)
        ax.set_xlim(0, 1.25 * max(vals))
        ax.set_xlabel("Mean farm RMSE (t/ha)")
        ax.set_title(f"{country}, {int(d.farms.iloc[0])} farms", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Wheat input ablations on unseen farms", fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Dashed lines are published reference scores from a different training protocol.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path / "loro_input_ablation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        scores[["country", "arm", "loro_rmse_t_ha", "pooled_rmse_t_ha", "pooled_r2"]]
        .round(4)
        .to_string(index=False)
    )
    print(gains.round(4).to_string(index=False))
    print(
        f"Verified {len(data)} predictions, one per field and input set, on 27 matched outer folds."
    )


# Dispatch validation, training, and reporting from one CLI.
def main():
    settings = parse_settings()
    if not settings.report_only:
        data = load_experiment(settings)
        if settings.check:
            print(
                f"Verified {len(data.fields)} fields, {len(data.folds)} folds, and "
                f"{len(data.reused_predictions)} reference predictions."
            )
            return
        run_training(data, settings.out)
    write_report(settings.out, settings.previous)


if __name__ == "__main__":
    main()
