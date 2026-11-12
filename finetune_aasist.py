"""finetune_aasist.py - fine-tune the pretrained AASIST so it stops reading silence / loudness / channel.

What it trains on
  * ASVspoof 2019 LA train clips (ids LA_T_*), both classes, balanced 50/50 per batch
  * optional: real phone speech from CallHome (callhome_out/callNNN_mix.wav, made by callhome_extract.py)
    as extra BONAFIDE examples, so natural pauses + phone audio are not 'fake'.  The LAST --ch-holdout
    fraction of the calls is never trained on: it is the held-out false-alarm test.
How the shortcuts are attacked (augment.py): every sample, whatever its label, gets random edge
trimming + random noise padding + sometimes pause removal (silence stops being a label), a random
gain of -25..+10 dB, and with p=0.6 a telephone chain (8 kHz, noise, band-pass, mu-law, gating).

What it prints / saves (in --out):
  metrics.csv   per epoch: train loss, EER under asis / no_pauses / pad_edges / phone8k / gain-12 dB
                on validation clips, false-alarm rate on the held-out CallHome calls, selection score
  baseline row  epoch 0 = the pretrained weights, so you can compare before / after
  best.pth      lowest selection score = mean(EER of the 5 conditions) + 0.5 * CallHome false-alarm rate
  last.pth      the final epoch
Load a result anywhere with aasist_io.load("finetune_out/best.pth").

Usage (GPU strongly recommended: on Colab T4 an epoch of 150 steps takes minutes, on a 2-thread CPU hours):
    python finetune_aasist.py --la-root "C:\\...\\project\\data" ^
        --protocol "C:\\...\\ASVspoof_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt" ^
        --callhome-dir callhome_out --epochs 3 --steps-per-epoch 150
    python finetune_aasist.py --la-root ... --epochs 1 --steps-per-epoch 3 --batch-size 4 --val-per-class 10   # smoke test
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aasist_io  # noqa: E402
import augment  # noqa: E402
from silence_experiment import compute_eer  # noqa: E402

SR = 16000
N = augment.N


# ----------------------------------------------------------------------------- data
def discover_la(root, protocols):
    """-> {utt: (path, label)} with label 1 = bonafide, 0 = spoof, from every flac under root that has a label."""
    labels = {}
    files = sorted(glob.glob(os.path.join(root, "**", "ASVspoof2019.LA.cm.*.txt"), recursive=True))
    for p in list(protocols) + files:
        for line in open(p):
            f = line.split()
            if len(f) >= 5 and f[4] in ("bonafide", "spoof"):
                labels[f[1]] = 1 if f[4] == "bonafide" else 0
    items = {}
    for path in glob.glob(os.path.join(root, "**", "*.flac"), recursive=True):
        utt = os.path.splitext(os.path.basename(path))[0]
        if utt in labels:
            items[utt] = (path, labels[utt])
    return items


def split_la(items, val_per_class, seed):
    """train = LA_T_*; validation = LA_D_* (dev) if present, else a hold-out of LA_T_ (same speakers: optimistic)."""
    rng = np.random.default_rng(seed)
    train = {u: v for u, v in items.items() if u.startswith("LA_T_")}
    val = {u: v for u, v in items.items() if u.startswith("LA_D_")}
    note = "dev clips (LA_D_)"
    if sum(1 for v in val.values() if v[1] == 1) < 10 or sum(1 for v in val.values() if v[1] == 0) < 10:
        utts = sorted(train)
        rng.shuffle(utts)
        k = max(20, len(utts) // 10)
        val = {u: train[u] for u in utts[:k]}
        train = {u: train[u] for u in utts[k:]}
        note = "10% hold-out of LA_T_ (no dev clips found; same speakers, so optimistic)"
    pick = []
    for lab in (1, 0):
        us = sorted(u for u, v in val.items() if v[1] == lab)
        rng.shuffle(us)
        pick += us[:val_per_class]
    return train, {u: val[u] for u in pick}, note


class CallhomeCalls:
    def __init__(self, directory, holdout):
        paths = sorted(glob.glob(os.path.join(directory, "call*_mix.wav")))
        self.all = [(p, sf.info(p).frames) for p in paths]
        k = int(np.ceil(holdout * len(self.all))) if len(self.all) >= 3 else 0
        self.train = self.all[: len(self.all) - k]
        self.held = self.all[len(self.all) - k:]

    def crop(self, rng):
        path, n = self.train[int(rng.integers(len(self.train)))]
        L = int(rng.integers(3 * SR, 6 * SR + 1))
        s = int(rng.integers(0, max(1, n - L)))
        x, _ = sf.read(path, start=s, frames=L, dtype="float32", always_2d=True)
        return x.mean(axis=1)


class MixDataset(torch.utils.data.Dataset):
    """Each item: class drawn 50/50; bonafide comes from CallHome with prob ch_frac (if available)."""

    def __init__(self, train, calls, n, seed, ch_frac):
        self.bona = sorted(u for u, v in train.items() if v[1] == 1)
        self.spoof = sorted(u for u, v in train.items() if v[1] == 0)
        self.train, self.calls, self.n, self.seed, self.ch_frac, self.epoch = train, calls, n, seed, ch_frac, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def _read(self, utt):
        x, _ = sf.read(self.train[utt][0], dtype="float32", always_2d=True)
        return x.mean(axis=1)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        label = int(rng.random() < 0.5)
        phone = False
        if label == 1 and self.calls is not None and self.calls.train and rng.random() < self.ch_frac:
            x, phone = self.calls.crop(rng), True
        else:
            pool = self.bona if label == 1 else self.spoof
            x = self._read(pool[int(rng.integers(len(pool)))])
        return torch.from_numpy(augment.augment(x, rng, already_phone=phone)), label


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def p_spoof(model, dev, arrs, bs=16):
    model.eval()
    out = []
    for i in range(0, len(arrs), bs):
        b = torch.from_numpy(np.stack([augment.crop_or_tile(np.asarray(a, np.float32)) for a in arrs[i:i + bs]])).to(dev)
        o = model(b)
        o = o[1] if isinstance(o, (tuple, list)) else o
        out += torch.softmax(o, dim=1)[:, 0].cpu().tolist()
    return np.array(out)


VAL_CONDITIONS = {
    "asis": lambda x, r: x,
    "no_pauses": lambda x, r: augment.no_pauses(x),
    "pad_edges": lambda x, r: augment.pad_noise(x, r, 1.0, 1.0),
    "phone8k": lambda x, r: augment.phone8k(x),
    "gain-12dB": lambda x, r: augment.gain_db(x, -12.0),
}


def load_val(val):
    arrs, labs = [], []
    for u, (path, lab) in sorted(val.items()):
        x, _ = sf.read(path, dtype="float32", always_2d=True)
        arrs.append(x.mean(axis=1))
        labs.append(lab)
    return arrs, np.array(labs)


def evaluate(model, dev, val_arrs, val_labs, held_calls, max_s=120.0, hop_s=2.0):
    res = {}
    for name, fn in VAL_CONDITIONS.items():
        rng = np.random.default_rng(0)
        ys = []
        for x in val_arrs:
            y = fn(x, rng)
            ys.append(y if len(y) >= SR // 4 else x)
        res["EER_" + name] = 100 * compute_eer(val_labs, 1.0 - p_spoof(model, dev, ys))
    ps = []
    for path, n in held_calls:
        x, _ = sf.read(path, frames=int(max_s * SR), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        ps += list(p_spoof(model, dev, [x[s:s + N] for s in range(0, max(1, len(x) - N + 1), int(hop_s * SR))]))
    if ps:
        ps = np.array(ps)
        res["callhome_FA%"] = 100 * float(np.mean(ps > 0.5))
        res["callhome_mean_p"] = float(np.mean(ps))
    eers = [v for k, v in res.items() if k.startswith("EER_")]
    res["select"] = float(np.mean(eers) / 100 + 0.5 * res.get("callhome_FA%", 0.0) / 100)
    return res


def fmt(res):
    return "  ".join(f"{k}={v:.1f}" if k != "select" and k != "callhome_mean_p" else f"{k}={v:.3f}" for k, v in res.items())


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la-root", required=True, help="folder searched recursively for ASVspoof 2019 LA flac files + protocols")
    ap.add_argument("--protocol", action="append", default=[], help="extra protocol .txt (e.g. the train protocol), repeatable")
    ap.add_argument("--callhome-dir", help="folder with callNNN_mix.wav (from callhome_extract.py)")
    ap.add_argument("--ch-holdout", type=float, default=0.3, help="fraction of CallHome calls (the last ones) held out")
    ap.add_argument("--ch-frac", type=float, default=0.25, help="share of bonafide samples taken from CallHome")
    ap.add_argument("--init", help="checkpoint to start from (default: pretrained AASIST.pth)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-per-class", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out", default="finetune_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    items = discover_la(args.la_root, args.protocol)
    train, val, note = split_la(items, args.val_per_class, args.seed)
    nb = sum(1 for v in train.values() if v[1] == 1)
    print(f"LA clips with labels: {len(items)}   train: {len(train)} (bonafide {nb}, spoof {len(train) - nb})")
    print(f"validation: {len(val)} clips from {note}")
    if nb < 20 or len(train) - nb < 20:
        sys.exit("Not enough LA_T_ training clips with labels. Point --la-root at the folder holding the LA train flacs and "
                 "pass the train protocol with --protocol (ASVspoof2019.LA.cm.train.trn.txt).")
    calls = None
    if args.callhome_dir:
        calls = CallhomeCalls(args.callhome_dir, args.ch_holdout)
        print(f"CallHome: {len(calls.all)} calls -> {len(calls.train)} for training, {len(calls.held)} held out")
        if not calls.all:
            sys.exit("No callNNN_mix.wav files found in --callhome-dir")

    model, _ = aasist_io.load(args.init, dev, train=True)
    val_arrs, val_labs = load_val(val)
    held = calls.held if calls else []
    log = []
    cols = None

    def record(epoch, loss, res):
        nonlocal cols
        row = {"epoch": epoch, "train_loss": loss, **res}
        log.append(row)
        cols = cols or list(row.keys())
        with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(log)

    best = float("inf")
    if not args.skip_baseline:
        t = time.time()
        res = evaluate(model, dev, val_arrs, val_labs, held)
        print(f"epoch 0 (pretrained) [{time.time() - t:.0f}s]  {fmt(res)}", flush=True)
        record(0, float("nan"), res)
        best = res["select"]

    ds = MixDataset(train, calls, args.steps_per_epoch * args.batch_size, args.seed, args.ch_frac)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total), eta_min=args.lr / 20)
    step = 0
    for epoch in range(1, args.epochs + 1):
        ds.set_epoch(epoch)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        model.train()
        t0, losses = time.time(), []
        for xb, yb in dl:
            out = model(xb.to(dev))
            logits = out[1] if isinstance(out, (tuple, list)) else out
            loss = F.cross_entropy(logits, yb.to(dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            if step % 25 == 0 or step == 1:
                print(f"  step {step}/{total}  loss {np.mean(losses[-25:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        res = evaluate(model, dev, val_arrs, val_labs, held)
        print(f"epoch {epoch}  loss {np.mean(losses):.4f}  {fmt(res)}", flush=True)
        record(epoch, float(np.mean(losses)), res)
        torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
        if res["select"] < best:
            best = res["select"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
            print("  -> new best.pth", flush=True)
    if not os.path.isfile(os.path.join(args.out, "best.pth")):
        print("No epoch beat the pretrained weights on the selection score: best.pth was NOT written; "
              "the pretrained model stays the best. last.pth is the final epoch for inspection.")
    print(f"\nMetrics: {os.path.join(args.out, 'metrics.csv')}")


if __name__ == "__main__":
    main()