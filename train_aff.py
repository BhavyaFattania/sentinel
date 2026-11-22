"""
train_aff.py - BTech/Miranda AFF reproduction (pixel-level, 5x5 patches, T=24) + optional SAR branch.

--mode s2    : AFF exactly as in the BTech code: S2 | weather | DEM | soil | coord  -> learnable-query attention
--mode fused : the same + a SAR branch (same Conv3d(1,5,5)+LSTM branch as S2, 2 channels) = 6 branches
Samples are PIXELS (5x5 neighbourhood, all 24 slots, missing -> --nan-fill). Splits are by FIELD
(folds/<Country>/fold_k from make_folds.py). Field yield = mean of predicted valid pixels; only FIELD-level
RMSE/R2 are reported (as in YieldSAT Table 13). Output files are run_folds.py compatible.

python train_aff.py --npy npy --split-dir folds/Argentina/fold_0 --mode s2 --out runs_aff/Argentina/fold_0
python run_folds.py --npy npy --folds folds --runs runs_aff --trainer train_aff.py --countries Argentina
"""
import os, re, csv, json, math, argparse, random
import numpy as np, torch, torch.nn as nn
from split_fields import AFF_FILES, SPLITS, field_ids, load_split


# ================= BTech AFF code (cleaned; ImgBranch64 = AFFS2Branch64 with cin channels) =================
class ImgBranch64(nn.Module):
    def __init__(self, cin, p=0.2):
        super().__init__()
        self.conv3d = nn.Conv3d(cin, 64, kernel_size=(1, 5, 5))
        self.bn = nn.BatchNorm3d(64); self.leaky = nn.LeakyReLU(inplace=True)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 64); self.fc_bn = nn.BatchNorm1d(64); self.relu = nn.ReLU(); self.dropout = nn.Dropout(p)
    def forward(self, x):                                   # [B,C,T,5,5]
        x = self.leaky(self.bn(self.conv3d(x)))
        if x.shape[-2:] != (1, 1): raise RuntimeError(f"Expected 1x1 after conv, got {x.shape[-2:]}")
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)       # [B,T,64]
        x, _ = self.lstm(x); x = x[:, -1, :]
        return self.dropout(self.relu(self.fc_bn(self.fc(x))))

class TemporalBranch64(nn.Module):
    def __init__(self, input_dim, p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 64); self.bn = nn.BatchNorm1d(64); self.relu = nn.ReLU(); self.dropout = nn.Dropout(p)
    def forward(self, x):                                   # [B,T,C]
        x, _ = self.lstm(x); x = x[:, -1, :]
        return self.dropout(self.relu(self.bn(self.fc(x))))

class SpatialBranch64(nn.Module):
    def __init__(self, input_dim, p=0.2):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, 64, kernel_size=5); self.bn = nn.BatchNorm2d(64); self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 64); self.fc_bn = nn.BatchNorm1d(64); self.dropout = nn.Dropout(p)
    def forward(self, x):                                   # [B,C,5,5]
        x = self.relu(self.bn(self.conv(x))).squeeze(-1).squeeze(-1)
        return self.dropout(self.relu(self.fc_bn(self.fc(x))))

class LearnableQueryAttention(nn.Module):
    def __init__(self, dim=64, dropout=0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False); self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False); self.dropout = nn.Dropout(dropout); self.scale = math.sqrt(dim)
    def forward(self, reps):
        x = torch.stack(reps, dim=1); B = x.shape[0]
        q = self.q_proj(self.query).expand(B, -1, -1); k, v = self.k_proj(x), self.v_proj(x)
        w = self.dropout(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.scale, dim=-1))
        return torch.matmul(w, v).squeeze(1)

class MirandaFeatureFusion(nn.Module):
    def __init__(self, soil_ch=24, use_sar=False, drop=(), p=0.2):
        super().__init__(); d = set(drop)                      # drop = branches to remove (ablation)
        self.s2 = None if "s2" in d else ImgBranch64(12, p)
        self.weather = None if "weather" in d else TemporalBranch64(4, p)
        self.dem = None if "dem" in d else SpatialBranch64(5, p)
        self.soil = None if "soil" in d else SpatialBranch64(soil_ch, p)
        self.coord = None if "coord" in d else SpatialBranch64(3, p)
        self.attn = LearnableQueryAttention(64, p); self.out = nn.Linear(64, 1)
        self.sar = ImgBranch64(2, p) if use_sar else None
    def forward(self, s2_x, weather_x, topo_x, soil_x, coord_x, sar_x=None):
        reps = []
        if self.s2 is not None: reps.append(self.s2(s2_x))
        if self.sar is not None: reps.append(self.sar(sar_x))
        if self.weather is not None: reps.append(self.weather(weather_x))
        if self.dem is not None: reps.append(self.dem(topo_x))
        if self.soil is not None: reps.append(self.soil(soil_x))
        if self.coord is not None: reps.append(self.coord(coord_x))
        return self.out(self.attn(reps)).squeeze(-1)


# ================= data =================
CH_AX = {"s2": 1, "sar": 1, "weather": 1, "dem": 0, "soil": 0, "coord": 0}

def raw_field(d, use_sar, sar_file):
    L = lambda n: np.load(os.path.join(d, n))
    f = dict(s2=L("s2.npy"), weather=L("weather.npy"), dem=L("dem.npy"), soil=L("soil.npy"), coord=L("coord.npy"),
             y=L("yield.npy"), mask=L("mask.npy"))
    if use_sar: f["sar"] = L(sar_file)
    return f

def compute_norm(dirs, use_sar, sar_file, n_fields=120):
    acc, ys = {}, []
    for d in random.sample(dirs, min(n_fields, len(dirs))):
        f = raw_field(d, use_sar, sar_file); ok = np.isfinite(f["y"]) & f["mask"]; ys.append(f["y"][ok])
        for g in (k for k in CH_AX if k in f):
            a = f[g].astype(np.float64); a = np.moveaxis(a, CH_AX[g], 0).reshape(a.shape[CH_AX[g]], -1)
            fin = np.isfinite(a); z = np.where(fin, a, 0.0)
            s = acc.setdefault(g, [0, 0, 0]); s[0] = s[0] + z.sum(1); s[1] = s[1] + (z ** 2).sum(1); s[2] = s[2] + fin.sum(1)
    norm = {}
    for g, (s, ss, n) in acc.items():
        m = s / np.maximum(n, 1)
        sd = np.sqrt(np.maximum(ss / np.maximum(n, 1) - m ** 2, 0))
        # Constant channels have no learned scale. Prevent division by an epsilon
        # from amplifying small held-out departures into float16 overflow.
        sd[sd <= 1e-6 * np.maximum(np.abs(m), 1)] = 1
        norm[g] = m, sd
    y = np.concatenate(ys); return norm, (float(y.mean()), float(y.std()) + 1e-6)

class FieldData:
    def __init__(self, d, norm, ystat, use_sar, sar_file, fill):
        f = raw_field(d, use_sar, sar_file); self.name = os.path.basename(d)
        for g in (k for k in CH_AX if k in f):
            a = f[g].astype(np.float32); bc = [1] * a.ndim; bc[CH_AX[g]] = -1
            a = (a - norm[g][0].astype(np.float32).reshape(bc)) / norm[g][1].astype(np.float32).reshape(bc)
            a = np.where(np.isfinite(a), a, fill).astype(np.float32)
            if g in ("s2", "sar", "dem", "soil", "coord"):
                a = np.pad(a, ((0, 0),) * (a.ndim - 2) + ((2, 2), (2, 2)), mode="edge")
                if np.abs(a).max() <= np.finfo(np.float16).max:
                    a = a.astype(np.float16)
            setattr(self, g, a)
        ok = np.isfinite(f["y"]) & f["mask"]
        if not ok.any():
            raise ValueError(f"No finite yield targets in {d}")
        self.rc = np.argwhere(ok).astype(np.int64)
        self.ymap = np.where(ok, (np.nan_to_num(f["y"]) - ystat[0]) / ystat[1], 0.0).astype(np.float32)
        self.true_mean = float(f["y"][ok].mean())

def make_batch(fields, items, use_sar, dev):
    B = len(items); T = fields[0].s2.shape[0]
    s2 = np.empty((B, 12, T, 5, 5), np.float32); wx = np.empty((B, T, 4), np.float32)
    dem = np.empty((B, 5, 5, 5), np.float32); soil = np.empty((B, fields[0].soil.shape[0], 5, 5), np.float32)
    co = np.empty((B, 3, 5, 5), np.float32); sar = np.empty((B, 2, T, 5, 5), np.float32) if use_sar else None
    y = np.empty(B, np.float32)
    for i, (fi, r, c) in enumerate(items):
        fd = fields[fi]
        s2[i] = fd.s2[:, :, r:r + 5, c:c + 5].transpose(1, 0, 2, 3)
        if use_sar: sar[i] = fd.sar[:, :, r:r + 5, c:c + 5].transpose(1, 0, 2, 3)
        wx[i] = fd.weather; dem[i] = fd.dem[:, r:r + 5, c:c + 5]; soil[i] = fd.soil[:, r:r + 5, c:c + 5]
        co[i] = fd.coord[:, r:r + 5, c:c + 5]; y[i] = fd.ymap[r, c]
    t = lambda a: torch.from_numpy(a).to(dev)
    return t(s2), t(wx), t(dem), t(soil), t(co), (t(sar) if use_sar else None), t(y)

def fwd(model, b):
    s2, wx, dem, soil, co, sar, _ = b
    return model(s2, wx, dem, soil, co, sar)

def evaluate(model, fields, ystat, use_sar, dev, chunk=2048, want_pix=False):
    model.eval(); P, Y, N = [], [], []; se = n = 0.0
    with torch.no_grad():
        for fd in fields:
            preds = []
            for i in range(0, len(fd.rc), chunk):
                rc = fd.rc[i:i + chunk]
                b = make_batch([fd], [(0, r, c) for r, c in rc], use_sar, dev)
                p = fwd(model, b).cpu().numpy()
                preds.append(p)
            p = np.concatenate(preds) * ystat[1] + ystat[0]
            if not np.isfinite(p).all():
                raise FloatingPointError(f"Non-finite predictions for {fd.name}")
            ytrue = fd.ymap[fd.rc[:, 0], fd.rc[:, 1]] * ystat[1] + ystat[0]
            se += ((p - ytrue) ** 2).sum(); n += len(p)
            P.append(float(p.mean())); Y.append(fd.true_mean); N.append(fd.name)
    P, Y = np.array(P), np.array(Y)
    out = dict(rmse=float(np.sqrt(np.mean((P - Y) ** 2))), r2=float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()),
               n=len(Y))
    if want_pix: out["pix_rmse"] = float(np.sqrt(se / max(n, 1)))     # INTERNAL: only for --stop-metric pixel, never reported
    return out, N, P, Y

def plot_history(hist, path, mode):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    e = [h["epoch"] for h in hist]; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(e, [h["train_loss"] for h in hist]); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train pixel MSE (z)"); ax[0].set_title(f"Train loss - {mode}")
    ax[1].plot(e, [h["val_rmse"] for h in hist], "r-"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val field RMSE (t/ha)")
    b = ax[1].twinx(); b.plot(e, [h["val_r2"] for h in hist], "g--"); b.set_ylabel("val field R2")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ================= main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default="npy"); ap.add_argument("--split-dir", required=True)
    ap.add_argument("--mode", default="s2", choices=["s2", "fused"]); ap.add_argument("--out", default="runs_aff")
    ap.add_argument("--sar-file", default="sar.npy", help="e.g. sar_s3.npy built with build_sar_npy.py --sigma 3 --name sar_s3.npy")
    ap.add_argument("--nan-fill", type=float, default=-1.0); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--pixels-per-epoch", type=int, default=200000)
    ap.add_argument("--max-epochs", type=int, default=100); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=0.01); ap.add_argument("--min-epochs", type=int, default=5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--stop-metric", choices=["field", "pixel"], default="field",
                    help="metric for early stopping / best checkpoint / LR schedule. 'pixel' = validation pixel RMSE (as in the BTech), far less "
                         "noisy than field RMSE on ~20 val fields; used internally only, reported results stay field-level")
    ap.add_argument("--scheduler", choices=["none", "plateau"], default="none", help="plateau = ReduceLROnPlateau(factor 0.5, patience 3) on val field RMSE")
    ap.add_argument("--btech-config", action="store_true", help="BTech feature_aff settings: Adam lr 0.006, batch 2048, max 50 epochs, patience 10, "
                    "ReduceLROnPlateau(0.5, 3), wd 0, full pass over train pixels each epoch, seed+fold")
    ap.add_argument("--dropout", type=float, default=0.2, help="dropout in all branches and attention (BTech value 0.2)")
    ap.add_argument("--drop-branches", nargs="*", default=[], choices=["s2", "weather", "dem", "soil", "coord"],
                    help="ablation: remove these branches, e.g. --drop-branches weather  |  --drop-branches s2 (SAR+static only)"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accum", type=int, default=1, help="ignored (run_folds compatibility)"); ap.add_argument("--max-fields", type=int, default=None)
    a = ap.parse_args()
    if a.btech_config:
        a.lr, a.batch, a.max_epochs, a.patience, a.min_epochs, a.min_delta = 0.006, 2048, 50, 10, 1, 0.0
        a.wd, a.scheduler, a.pixels_per_epoch = 0.0, "plateau", 10 ** 9
    fm = re.search(r"fold_(\d+)", a.split_dir); a.seed = a.seed + (int(fm.group(1)) if fm else 0)      # seed + fold
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); use_sar = a.mode == "fused"

    required = AFF_FILES + ((a.sar_file,) if use_sar else ())
    tr_d, va_d, te_d = load_split(a.split_dir, required)
    if a.max_fields: tr_d, va_d, te_d = tr_d[:a.max_fields], va_d[:a.max_fields], te_d[:a.max_fields]
    used_fields = {name: field_ids(paths) for name, paths in zip(SPLITS, (tr_d, va_d, te_d))}
    print(f"fields train/val/test: {len(tr_d)}/{len(va_d)}/{len(te_d)} | {dev} | mode={a.mode} | sar_file={a.sar_file if use_sar else '-'}")

    norm, ystat = compute_norm(tr_d, use_sar, a.sar_file)
    load = lambda ds: [FieldData(d, norm, ystat, use_sar, a.sar_file, a.nan_fill) for d in ds]
    tr, va, te = load(tr_d), load(va_d), load(te_d)
    soil_ch = tr[0].soil.shape[0]; print(f"soil channels: {soil_ch}")
    model = MirandaFeatureFusion(soil_ch, use_sar, a.drop_branches, a.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3) if a.scheduler == "plateau" else None

    items = np.concatenate([np.column_stack([np.full(len(f.rc), i), f.rc]) for i, f in enumerate(tr)])
    if min(len(items), a.batch, a.pixels_per_epoch) < 2:
        ap.error("At least two training pixels are required per batch")
    os.makedirs(a.out, exist_ok=True); ck = os.path.join(a.out, f"best_{a.mode}.pt")
    csv_p = os.path.join(a.out, f"history_{a.mode}.csv"); png_p = os.path.join(a.out, f"curve_{a.mode}.png")
    if os.path.exists(csv_p): os.remove(csv_p)
    best, bad, hist = 1e9, 0, []
    for ep in range(1, a.max_epochs + 1):
        model.train(); sel = np.random.choice(len(items), min(a.pixels_per_epoch, len(items)), replace=False); tot = nb = 0
        for i in range(0, len(sel), a.batch):
            idx = sel[i:i + a.batch]
            if len(idx) < 2: continue
            b = make_batch(tr, [tuple(x) for x in items[idx]], use_sar, dev)
            loss = ((fwd(model, b) - b[-1]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        v, _, _, _ = evaluate(model, va, ystat, use_sar, dev, want_pix=a.stop_metric == "pixel")
        mval = v["pix_rmse"] if a.stop_metric == "pixel" else v["rmse"]       # early-stopping metric
        if sched: sched.step(mval)
        row = dict(epoch=ep, train_loss=tot / max(nb, 1), val_rmse=v["rmse"], val_r2=v["r2"]); hist.append(row)
        new = not os.path.exists(csv_p)
        with open(csv_p, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
        print(f"ep {ep} train={row['train_loss']:.4f} val_field_rmse={v['rmse']:.3f} r2={v['r2']:.3f} lr={opt.param_groups[0]['lr']:.4f}", flush=True)
        bad = 0 if mval < best - a.min_delta else bad + 1
        if mval < best: best = mval; torch.save(model.state_dict(), ck)
        if ep % 5 == 0: plot_history(hist, png_p, a.mode)
        if ep >= a.min_epochs and bad >= a.patience: print(f"early stop at epoch {ep}"); break
    plot_history(hist, png_p, a.mode)

    model.load_state_dict(torch.load(ck, map_location=dev))
    tm, names, P, Y = evaluate(model, te, ystat, use_sar, dev)
    vam = evaluate(model, va, ystat, use_sar, dev)[0]
    trm = evaluate(model, tr, ystat, use_sar, dev)[0]          # best checkpoint on all train fields (field-level)
    print(f"FINAL field-level  TRAIN rmse={trm['rmse']:.3f} r2={trm['r2']:.3f} | VAL rmse={vam['rmse']:.3f} r2={vam['r2']:.3f} | "
          f"TEST rmse={tm['rmse']:.3f} r2={tm['r2']:.3f}")
    csv.writer(open(os.path.join(a.out, f"preds_{a.mode}.csv"), "w", newline="")).writerows(
        [["field", "actual", "pred"]] + [[n, round(float(y), 4), round(float(p), 4)] for n, y, p in zip(names, Y, P)])
    train_mean = float(np.mean([f.true_mean for f in tr]))
    res = dict(mode=a.mode, epochs_run=len(hist), val_best_rmse=best, train=trm, val=vam, test=tm, field_ids=used_fields, device=str(dev),
               baseline_rmse=float(np.sqrt(np.mean((Y - train_mean) ** 2))), model="AFF-pixel", stop_metric=a.stop_metric, drop_branches=a.drop_branches, dropout=a.dropout, wd=a.wd, seed=a.seed, lr=a.lr, batch=a.batch, btech_config=a.btech_config, sar_file=a.sar_file if use_sar else None)
    print("TEST:", res["test"]); json.dump(res, open(os.path.join(a.out, f"results_{a.mode}.json"), "w"), indent=2)

if __name__ == "__main__":
    main()