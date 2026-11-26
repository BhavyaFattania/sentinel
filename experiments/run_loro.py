"""Farm-held-out wheat experiments using the published AFF encoders."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

COUNTRIES = ("Argentina", "Brazil")
ARMS = {
    "aff_lstm2": (2, False),
    "aff_lstm1": (1, False),
    "aff_lstm1_valid_pool": (1, True),
}
CHANNELS = {"s2": 12, "sar": 2, "weather": 4, "dem": 5, "soil": 24, "coord": 3}
RASTER = ("s2", "sar", "dem", "soil", "coord")
IMAGE = ("s2", "sar")
EPOCHS = 10
SAMPLES = 24576
BATCH = 256
VAL_PIXELS = 1024
SEED = 42


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
                buf = np.ascontiguousarray(a.reshape(self.bank[k].shape[1], sizes[i]).T, dtype=dtype)
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
        patch = centers[:, None, None] + widths[:, None, None] * self.axis[None, :, None] + self.axis

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
            scale(self.weather[fields], "weather") if active is None or "weather" in active else None,
            static("dem") if active is None or "dem" in active else None,
            static("soil") if active is None or "soil" in active else None,
            static("coord") if active is None or "coord" in active else None,
            image("sar") if active is None or "sar" in active else None,
            (self.targets[ids] - ystat[0]) / ystat[1],
        )


class ValidSlotImage(nn.Module):
    # Reuse an image encoder's convolution and recurrent layers.
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    # Average recurrent outputs over valid satellite observations.
    def forward(self, inputs):
        x, validity = inputs
        b = self.encoder
        x = b.leaky(b.bn(b.conv3d(x))).squeeze(-1).squeeze(-1).permute(0, 2, 1)
        x, _ = b.lstm(x)
        mask = validity.unsqueeze(-1).to(x.dtype)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return b.dropout(b.relu(b.fc_bn(b.fc(x))))


# Build the depth and pooling variant for the original LORO experiment.
def build_model(aff, arm, seed, device):
    depth, masked = ARMS[arm]
    torch.manual_seed(seed)
    model = aff.MirandaFeatureFusion(use_sar=True)
    if depth == 1:
        for branch in (model.s2, model.sar, model.weather):
            old = branch.lstm
            new = nn.LSTM(old.input_size, old.hidden_size, num_layers=1, batch_first=True)
            new.load_state_dict({k: old.state_dict()[k] for k in new.state_dict()})
            branch.lstm = new
    if masked:
        model.s2 = ValidSlotImage(model.s2)
        model.sar = ValidSlotImage(model.sar)
    torch.manual_seed(seed + 1)
    return model.to(device)


# Predict field yields by averaging pixel estimates in physical units.
@torch.inference_mode()
def predict(aff, model, bank, fields, norm, ystat, arm, cap=None, *, active=None):
    masked = ARMS[arm][1] if active is None else False
    model.eval()
    pred = []
    for i in fields:
        lo, hi = bank.limits[i : i + 2]
        if cap is not None and hi - lo > cap:
            batches = (
                torch.linspace(lo, hi - 1, cap, device=bank.device, dtype=torch.float64).long().split(1024)
            )
            n = cap
        else:
            batches = (slice(j, min(j + 1024, hi)) for j in range(lo, hi, 1024))
            n = hi - lo
        total = torch.zeros((), device=bank.device)
        for idx in batches:
            batch = bank.batch(idx, norm, ystat, masked, active=active)
            with torch.autocast("cuda", dtype=torch.float16):
                total += aff.fwd(model, batch).float().sum()
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
        np.mean([np.sqrt(np.mean(sq[np.array([x == g for x in groups])])) for g in sorted(set(groups))])
    )


# Train one fold and restore the checkpoint with the lowest validation RMSE.
def fit(aff, bank, train, val, groups, norm, ystat, arm, seed, *, model_factory=build_model, active=None):
    masked = ARMS[arm][1] if active is None else False
    start = time.monotonic()
    model = model_factory(aff, arm, seed, bank.device)
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
            batch = bank.batch(idx, norm, ystat, masked, active=active)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = aff.fwd(model, batch)
            loss = nn.functional.mse_loss(pred.float(), batch[-1])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.detach())
        val_pred = predict(aff, model, bank, val, norm, ystat, arm, VAL_PIXELS, active=active)
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
        yield region, tr, va, te


# Run the original depth comparison and save the fold predictions.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    import train_aff as aff

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = a.data / "wheat-field-manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["country"] in COUNTRIES and r["status"] == "ready"]
    rows.sort(key=lambda r: (r["country"], r["field"]))
    paths = [a.data / "fields" / r["country"] / r["field"] for r in rows]
    groups = [f"{r['country']}/{r['provider']}/{r['farm']}" for r in rows]
    stages = list(folds(groups))
    assert len(rows) == 263 and len(stages) == 27
    assert sorted(i for _, _, _, t in stages for i in t) == list(range(len(rows)))
    config = dict(
        commit=subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip(),
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        countries=list(COUNTRIES),
        region="country/provider/farm",
        outer_folds=len(stages),
        fields=len(rows),
        model_variants={k: list(v) for k, v in ARMS.items()},
        samples_per_epoch=SAMPLES,
        batch=BATCH,
        max_epochs=EPOCHS,
        lr=0.001,
        training_sampler="equal pixel draws per training farm",
        validation="two separate Argentine farms and one separate Brazilian farm; at most 1024 pixels per field",
        pick="lowest validation mean-farm RMSE; min 5 epochs, patience 3, min improvement 0.015",
        test="all labeled pixels in held-out farms",
        seeds="42 plus outer fold index",
    )
    a.out.mkdir(parents=True, exist_ok=True)
    p = a.out / "protocol.json"
    if p.exists() and json.loads(p.read_text(encoding="utf-8")) != config:
        raise ValueError("Protocol mismatch. Choose a fresh run directory.")
    save_json(p, config)
    split = []
    for j, (region, tr, va, te) in enumerate(stages):
        for role, indices in (("train", tr), ("val", va), ("test", te)):
            split.extend(
                dict(
                    fold=j + 1,
                    heldout=region,
                    role=role,
                    country=rows[i]["country"],
                    farm=groups[i],
                    field=rows[i]["field"],
                )
                for i in indices
            )
    write_csv(a.out / "fold_manifest.csv", split)
    print(
        f"{len(rows)} fields across {len(stages)} farms | {torch.cuda.get_device_name()}",
        flush=True,
    )
    bank = FieldBank(paths, device)
    print(f"GPU cache: {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    complete = []
    for j, (region, tr, va, te) in enumerate(stages):
        target = a.out / f"fold-{j+1:02d}"
        target.mkdir(exist_ok=True)
        completed = target / "predictions.csv"
        if completed.exists():
            complete.append(j + 1)
            continue
        t = time.monotonic()
        norm, ystat = bank.fold_stats(tr)
        seed = SEED + j
        print(
            f"Fold {j+1}/{len(stages)} {region} | train {len(tr)}, val {len(va)}, test {len(te)}",
            flush=True,
        )
        models, summaries = {}, {}
        for arm in ARMS:
            net, history, info = fit(aff, bank, tr, va, groups, norm, ystat, arm, seed)
            models[arm] = net
            summaries[arm] = info
            write_csv(target / f"history-{arm}.csv", history)
            torch.save(net.state_dict(), target / f"weights-{arm}.pt")
            print(
                f"  {arm}: val farm RMSE {info['val_macro_rmse']:.3f}, best epoch {info['best_epoch']}, "
                f"{info['seconds']:.1f} s",
                flush=True,
            )
        picked = min(ARMS, key=lambda k: summaries[k]["val_macro_rmse"])
        save_json(
            target / "validation.json",
            dict(
                regions=sorted(set(groups[i] for i in va)),
                selected=picked,
                arms=summaries,
            ),
        )
        pred = {arm: predict(aff, model, bank, te, norm, ystat, arm) for arm, model in models.items()}
        country = rows[te[0]]["country"]
        base = float(np.mean([bank.actual[i] for i in tr if rows[i]["country"] == country]))
        pred["country_mean"] = np.full(len(te), base)
        pred["validation_selected"] = pred[picked]
        results = []
        for arm, estimates in pred.items():
            for i, estimate in zip(te, estimates):
                results.append(
                    dict(
                        fold=j + 1,
                        region=region,
                        country=country,
                        field=rows[i]["field"],
                        arm=arm,
                        actual=float(bank.actual[i]),
                        predicted=float(estimate),
                        selected=picked if arm == "validation_selected" else "",
                    )
                )
        write_csv(completed, results)
        print(f"  complete in {time.monotonic()-t:.1f} s", flush=True)
        del models, net
        complete.append(j + 1)
    result = []
    for j in complete:
        with (a.out / f"fold-{j:02d}" / "predictions.csv").open(newline="", encoding="utf-8") as f:
            result.extend(csv.DictReader(f))
    write_csv(a.out / "loro_predictions.csv", result)
    print(
        f"Completed {len(complete)} folds and {len(result)} prediction rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
