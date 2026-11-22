"""
train_fusion.py - field-level yield regression, S2 + SAR feature fusion.

Per field (batch=1, no crop/pad/aug):
  S2  (24,12,H,W) -> ConvLSTM(12->32) -> (24,32,H,W)
  SAR (24, 2,H,W) -> ConvLSTM( 2->32) -> (24,32,H,W)
  AFF per t: g = sigmoid(Conv1x1([f_s2; f_sar; v_s2; v_sar]))  (66->32)
             fused = g*f_s2 + (1-g)*f_sar            (24,32,H,W)
  masked spatial mean -> (24,32) -> flatten 768 -> Linear 128 -> ReLU -> Linear 1
Target: field-mean yield, z-scored per country (train fields only).
Validity (NaN = cloud/missing) goes only to AFF, not to the encoders.

python train_fusion.py --npy npy --split-dir splits --mode fused --accum 16   # epochs are dynamic (early stopping)
--split-dir splits : use folders from make_splits.py (Germany train; Argentina/Brazil val/test)
--mode: fused | s2 (gate=1) | sar (gate=0)
"""
import os, csv, glob, json, argparse, random
import numpy as np, torch, torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(s, cin, ch, k=3):
        super().__init__(); s.ch = ch
        s.conv = nn.Conv2d(cin + ch, 4 * ch, k, padding=k // 2)
    def forward(s, x, h, c):
        i, f, o, g = s.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c

class ConvLSTM(nn.Module):
    def __init__(s, cin, ch):
        super().__init__(); s.cell = ConvLSTMCell(cin, ch); s.ch = ch
    def forward(s, x):                      # (T,C,H,W) -> (T,ch,H,W)
        T, _, H, W = x.shape
        h = x.new_zeros(1, s.ch, H, W); c = h.clone(); out = []
        for t in range(T):
            h, c = s.cell(x[t:t + 1], h, c); out.append(h)
        return torch.cat(out, 0)

class FusionNet(nn.Module):
    def __init__(s, mode="fused", ch=32, T=24, cloud_thr=0.7, sar_min=0.7, rule=True):
        super().__init__(); s.mode = mode
        s.cloud_thr, s.sar_min, s.rule = cloud_thr, sar_min, rule
        s.enc_s2 = ConvLSTM(12, ch); s.enc_sar = ConvLSTM(2, ch)
        s.gate = nn.Conv2d(2 * ch + 2, ch, 1)
        s.head = nn.Sequential(nn.Linear(T * ch, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def fuse(s, s2, sar, v2, vs, mask=None):           # -> fused features (T,ch,H,W), gate (T,ch,H,W) or None
        f2, fs = s.enc_s2(s2), s.enc_sar(sar); g = None
        if s.mode == "s2": f = f2
        elif s.mode == "sar": f = fs
        else:
            g = torch.sigmoid(s.gate(torch.cat([f2, fs, v2, vs], 1)))
            if s.rule and mask is not None:
                mf = mask.float()
                q = (v2[:, 0] * mf).sum((1, 2)) / mf.sum().clamp(min=1)        # (T,) share of field pixels with usable S2
                cloudy = ((1 - q) >= s.cloud_thr)[:, None, None, None]         # slot >= cloud_thr missing/cloudy
                g = torch.where(cloudy, g.clamp(max=1 - s.sar_min), g)         # SAR weight >= sar_min
                g = torch.where((q <= 0)[:, None, None, None], torch.zeros_like(g), g)   # no S2 at all -> SAR only
                g = g * v2                                                     # unusable optical pixel -> SAR only
                g = torch.where((vs == 0) & (v2 > 0), torch.ones_like(g), g)   # no SAR at pixel -> optical only
            f = g * f2 + (1 - g) * fs
        act = ((v2 + vs).flatten(1).amax(1) > 0).float()   # slot has any valid data (out-of-season -> 0)
        return f * act[:, None, None, None], g
    def forward(s, s2, sar, v2, vs, mask):  # s2 (T,12,H,W) sar (T,2,H,W) v* (T,1,H,W) mask (1,H,W)
        f, _ = s.fuse(s2, sar, v2, vs, mask)
        m = mask.float()                                   # (1,H,W)
        pooled = (f * m).sum((2, 3)) / m.sum().clamp(min=1)  # (T,ch)
        return s.head(pooled.flatten()[None])[0, 0]

def load_field(d):
    s2 = np.load(os.path.join(d, "s2.npy")); sar = np.load(os.path.join(d, "sar.npy"))
    y = np.load(os.path.join(d, "yield.npy"))
    return s2, sar, y

class Stats:
    def __init__(s, dirs, n=40):
        a2, a1, ys = [], [], {}
        for d in random.sample(dirs, min(n, len(dirs))):
            s2, sar, _ = load_field(d)
            a2.append(s2.reshape(24, 12, -1).transpose(1, 0, 2).reshape(12, -1)[:, ::50])
            a1.append(sar.reshape(24, 2, -1).transpose(1, 0, 2).reshape(2, -1)[:, ::50])
        a2, a1 = np.concatenate(a2, 1), np.concatenate(a1, 1)
        s.m2, s.s2 = np.nanmean(a2, 1), np.nanstd(a2, 1) + 1e-6
        s.m1, s.s1 = np.nanmean(a1, 1), np.nanstd(a1, 1) + 1e-6
        by = {}
        for d in dirs:
            c = d.split(os.sep)[-2]
            by.setdefault(c, []).append(float(np.nanmean(np.load(os.path.join(d, "yield.npy")))))
        s.y = {c: (np.nanmean(v), np.nanstd(v) + 1e-6) for c, v in by.items()}
        allv = [x for v in by.values() for x in v]
        s.gy = (float(np.mean(allv)), float(np.std(allv)) + 1e-6)   # global, train fields only
        s.global_y = False

def prep(d, st, dev):
    s2, sar, y = load_field(d)
    mask = np.isfinite(y) & np.load(os.path.join(d, "mask.npy"))
    sp = os.path.join(d, "season.npy")      # (24,) bool: slot inside seeding..harvest
    season = (np.load(sp) if os.path.exists(sp) else np.ones(24, bool)).astype(np.float32)[:, None, None, None]
    v2 = np.isfinite(s2).all(1, keepdims=True).astype(np.float32) * season   # (T,1,H,W)
    vs = np.isfinite(sar).all(1, keepdims=True).astype(np.float32) * season
    s2 = np.nan_to_num((s2 - st.m2[None, :, None, None]) / st.s2[None, :, None, None])
    sar = np.nan_to_num((sar - st.m1[None, :, None, None]) / st.s1[None, :, None, None])
    s2, sar = s2 * season, sar * season     # out-of-season slots -> 0 input
    c = d.split(os.sep)[-2]; mu, sd = st.gy if st.global_y else st.y[c]
    tgt = (float(np.nanmean(y[mask])) - mu) / sd
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
    return t(s2), t(sar), t(v2), t(vs), torch.from_numpy(mask[None]).to(dev), tgt, (mu, sd)

def evaluate(model, dirs, st, dev, return_preds=False):
    model.eval(); P, Y, Z, N = [], [], [], []
    with torch.no_grad():
        for d in dirs:
            s2, sar, v2, vs, m, tgt, (mu, sd) = prep(d, st, dev)
            p = model(s2, sar, v2, vs, m).item()
            P.append(p * sd + mu); Y.append(tgt * sd + mu); Z.append((p - tgt) ** 2); N.append(os.path.basename(d))
    P, Y = np.array(P), np.array(Y)
    out = dict(loss=float(np.mean(Z)), rmse=float(np.sqrt(np.mean((P - Y) ** 2))), r2=float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()), n=len(Y))
    return (out, N, P, Y) if return_preds else out

def plot_history(hist, path, mode):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    e = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(e, [h["train_loss"] for h in hist], label="train"); ax[0].plot(e, [h["val_loss"] for h in hist], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (z-scored MSE)"); ax[0].legend(); ax[0].set_title(f"Loss - {mode}")
    ax[1].plot(e, [h["val_rmse"] for h in hist], "r-", label="val RMSE (t/ha)"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("RMSE (t/ha)")
    b = ax[1].twinx(); b.plot(e, [h["val_r2"] for h in hist], "g--", label="val R2"); b.set_ylabel("R2")
    ax[1].set_title("Validation metrics"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True); ap.add_argument("--mode", default="fused", choices=["fused", "s2", "sar"])
    ap.add_argument("--max-epochs", type=int, default=300, help="upper limit only; early stopping ends training sooner")
    ap.add_argument("--patience", type=int, default=10, help="stop after this many epochs without val-RMSE improvement > min-delta")
    ap.add_argument("--min-delta", type=float, default=0.01, help="smallest val-RMSE improvement (t/ha) that counts")
    ap.add_argument("--min-epochs", type=int, default=10); ap.add_argument("--plot-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="continue from runs/last_<mode>.pt after a crash")
    ap.add_argument("--cloud-thr", type=float, default=0.7, help="slot with >= this share of field pixels missing/cloudy in S2 -> SAR favoured")
    ap.add_argument("--sar-min-weight", type=float, default=0.7, help="minimum SAR weight on such slots")
    ap.add_argument("--gate-rule", choices=["on", "off"], default="on", help="off = purely learned gate (old behaviour)")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-dir", default=None, help="folder from make_splits.py (train/val/test); overrides the internal random split")
    ap.add_argument("--out", default="runs"); ap.add_argument("--max-fields", type=int, default=None)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ok = lambda d: os.path.exists(os.path.join(d, "sar.npy"))
    tr, va, te = [], [], []
    dirs = [d for d in glob.glob(os.path.join(a.npy, "*", "*")) if ok(d)] if not a.split_dir else []
    if a.split_dir:
        g = lambda s: sorted(d for d in glob.glob(os.path.join(a.split_dir, s, "*", "*")) if ok(d))
        tr, va, te = g("train"), g("val"), g("test")
        if a.max_fields: tr, va, te = tr[:a.max_fields], va[:a.max_fields], te[:a.max_fields]
    elif a.max_fields: dirs = dirs[:a.max_fields]
    for c in ([] if a.split_dir else sorted({d.split(os.sep)[-2] for d in dirs})):          # per-country random split by field
        ds = sorted(d for d in dirs if d.split(os.sep)[-2] == c); random.shuffle(ds)
        n = len(ds); nv = max(1, round(.15 * n)); nt = max(1, round(.15 * n))
        va += ds[:nv]; te += ds[nv:nv + nt]; tr += ds[nv + nt:]
    print(f"fields train/val/test: {len(tr)}/{len(va)}/{len(te)} | {dev} | mode={a.mode}")

    st = Stats(tr)
    st.global_y = bool(a.split_dir)   # val/test countries are unseen: use ONE yield scale from train fields (no test leakage)
    model = FusionNet(a.mode, cloud_thr=a.cloud_thr, sar_min=a.sar_min_weight, rule=a.gate_rule == 'on').to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(m2=st.m2.tolist(), s2=st.s2.tolist(), m1=st.m1.tolist(), s1=st.s1.tolist(),
                   y={k: [float(x) for x in v] for k, v in st.y.items()}, gy=list(st.gy),
                   global_y=st.global_y, cloud_thr=a.cloud_thr, sar_min=a.sar_min_weight, rule=a.gate_rule == 'on'), open(os.path.join(a.out, f"stats_{a.mode}.json"), "w"))
    ck = os.path.join(a.out, f"best_{a.mode}.pt"); last = os.path.join(a.out, f"last_{a.mode}.pt")
    csv_p = os.path.join(a.out, f"history_{a.mode}.csv"); png_p = os.path.join(a.out, f"curve_{a.mode}.png")
    best, bad, hist, start = 1e9, 0, [], 1
    if a.resume and os.path.exists(last):
        c = torch.load(last, map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        best, bad, hist, start = c["best"], c["bad"], c["hist"], c["epoch"] + 1
        print(f"resumed from epoch {c['epoch']}")
    elif os.path.exists(csv_p):
        os.remove(csv_p)

    for ep in range(start, a.max_epochs + 1):
        model.train(); random.shuffle(tr); tot = 0; opt.zero_grad()
        for i, d in enumerate(tr, 1):
            s2, sar, v2, vs, m, tgt, _ = prep(d, st, dev)
            loss = (model(s2, sar, v2, vs, m) - tgt) ** 2
            (loss / a.accum).backward(); tot += loss.item()
            if i % a.accum == 0 or i == len(tr):
                opt.step(); opt.zero_grad()
        v = evaluate(model, va, st, dev)
        row = dict(epoch=ep, train_loss=tot / len(tr), val_loss=v["loss"], val_rmse=v["rmse"], val_r2=v["r2"])
        hist.append(row)
        new = not os.path.exists(csv_p)
        with open(csv_p, "a", newline="") as f:           # written EVERY epoch -> nothing lost on a crash
            w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
        print(f"ep {ep} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} val_rmse={v['rmse']:.3f} r2={v['r2']:.3f}")
        bad = 0 if v["rmse"] < best - a.min_delta else bad + 1
        if v["rmse"] < best: best = v["rmse"]; torch.save(model.state_dict(), ck)
        stop = ep >= a.min_epochs and bad >= a.patience
        if ep % a.plot_every == 0 or stop:
            plot_history(hist, png_p, a.mode)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, best=best, bad=bad, hist=hist), last)
        if stop:
            print(f"early stop at epoch {ep}: no val-RMSE gain > {a.min_delta} for {a.patience} epochs"); break
    plot_history(hist, png_p, a.mode)

    model.load_state_dict(torch.load(ck, map_location=dev))
    tm, names, P, Y = evaluate(model, te, st, dev, return_preds=True)
    csv.writer(open(os.path.join(a.out, f"preds_{a.mode}.csv"), "w", newline="")).writerows(
        [["field", "actual", "pred"]] + [[n, round(float(y), 4), round(float(p), 4)] for n, y, p in zip(names, Y, P)])
    base = float(np.sqrt(np.mean((Y - st.gy[0]) ** 2)))     # RMSE of predicting the train-mean yield
    res = dict(mode=a.mode, epochs_run=len(hist), val_best_rmse=best, test=tm, baseline_rmse=base)
    print("TEST:", res["test"]); json.dump(res, open(os.path.join(a.out, f"results_{a.mode}.json"), "w"), indent=2)

if __name__ == "__main__":
    main()