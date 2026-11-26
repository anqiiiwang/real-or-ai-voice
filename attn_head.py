"""
Stronger detection head: attentive pooling (a small version of the MHFA head of Hurt & Yang, 2026).

Question: is the collapse under edits caused by the simple classifier (mean pooling + logistic
regression), or by the pretrained features themselves?

Same clips, same singer folds and same layer as rotate.py. The only change is the head:
instead of averaging the frames, H attention heads learn which moments of the clip to look at.

  frames (T x D) -> LayerNorm -> Linear(D->256) -> H attention heads, each a softmax over time
                 -> H weighted sums (H x 256) -> LayerNorm -> dropout -> Linear -> real/AI

Per fold: one training singer is held out for early stopping and for the decision threshold.

Run AFTER rotate.py for the same model (it reuses the layer that rotate.py picked in each fold):
    python attn_head.py
    python attn_head.py --model microsoft/wavlm-large
    python attn_head.py --model m-a-p/MERT-v1-95M
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import baseline as B
from rotate import add_common_args, build_pool_and_folds

DOWNSAMPLE = 4  # average every 4 frames (20 ms -> 80 ms) to keep the cache small


# ---------------------------------------------------------------- frame features
def extract_frames(pool, files, ext, edit, layers, args, cache):
    """Frame-level features of the given layers: {layer: (N, T', D) float16}."""
    if cache.exists():
        z = np.load(cache)
        return {int(k[1:]): z[k] for k in z.files}
    rng = np.random.default_rng(args.seed)  # same edits as baseline.extract
    out = {l: [] for l in layers}
    batch = []

    def flush():
        hs = ext.hidden(batch)
        for l in layers:
            h = hs[l].float()
            T = h.shape[1] // DOWNSAMPLE * DOWNSAMPLE
            h = h[:, :T].reshape(h.shape[0], T // DOWNSAMPLE, DOWNSAMPLE, -1).mean(2)
            out[l].append(h.half().cpu().numpy())

    for r in tqdm(pool, desc=f"{edit:>12s}", leave=False):
        y = B.load_clip(files[r["name"]], ext.sr, args.seconds)
        batch.append(B.fix_length(B.apply_edit(y, ext.sr, edit, rng), ext.sr, args.seconds))
        if len(batch) == args.batch:
            flush(); batch = []
    if batch:
        flush()
    res = {l: np.concatenate(v) for l, v in out.items()}
    np.savez(cache, **{f"L{l}": v for l, v in res.items()})
    return res


# ---------------------------------------------------------------- head
class AttentiveHead(nn.Module):
    def __init__(self, d_in, d=256, heads=8, dropout=0.3):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d)
        self.att = nn.Linear(d, heads)          # one attention score per frame per head
        self.norm_out = nn.LayerNorm(heads * d)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(heads * d, 1)

    def forward(self, x):                        # x: (B, T, D)
        v = self.proj(self.norm_in(x))           # (B, T, d)
        w = self.att(v).softmax(dim=1)           # (B, T, H), weights over time
        pooled = torch.einsum("bth,btd->bhd", w, v).flatten(1)
        return self.cls(self.drop(self.norm_out(pooled))).squeeze(-1)


def train_head(Xtr, ytr, Xva, yva, device, seed, epochs=40, patience=8):
    torch.manual_seed(seed)
    model = AttentiveHead(Xtr.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    best, best_state, bad = 1e9, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xtr_t), generator=g).split(64):
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx].to(device)), ytr_t[idx].to(device))
            loss.backward()
            opt.step()
        e = B.eer(yva, predict(model, Xva, device))
        if e < best - 1e-4:
            best, bad = e, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    out = [torch.sigmoid(model(torch.tensor(X[i:i + 256], dtype=torch.float32).to(device))).cpu().numpy()
           for i in range(0, len(X), 256)]
    return np.concatenate(out)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--layer", type=int, default=None,
                    help="use this layer in every fold (default: the layer rotate.py chose per fold)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1]
    edits = args.edits.split(",")

    files, pool, singers, folds = build_pool_and_folds(args, rng)
    y = np.array([r["label"] == "deepfake" for r in pool], int)
    singer_of = np.array([r["singer"] for r in pool])
    rot_singers = {s for src, s in singers if src == args.rotate_source}

    # layer per fold: same as the mean-pooling run, so only the head differs
    if args.layer is not None:
        fold_layer = [args.layer] * len(folds)
    else:
        f = out / f"{tag}_rotate_folds.csv"
        if not f.exists():
            raise SystemExit(f"{f} not found: run rotate.py for this model first, or pass --layer")
        fl = {}
        for r in csv.DictReader(open(f)):
            fl[int(r["fold"])] = int(r["layer"])
        fold_layer = [fl[i + 1] for i in range(len(folds))]
    layers = sorted(set(fold_layer))
    print(f"device={device}  model={args.model}  head=attentive pooling")
    print(f"layer per fold: {fold_layer}")

    ext = B.Extractor(args.model, device)
    t0 = time.time()
    F = {}
    for e in edits:
        F[e] = extract_frames(pool, files, ext, e, layers, args,
                              B.cache_path(out, tag, f"frames{'-'.join(map(str, layers))}", e, pool, args))
        print(f"  frames {e:>11s}: {F[e][layers[0]].shape}  ({time.time() - t0:.0f}s)")
    del ext
    torch.mps.empty_cache() if device == "mps" else None

    lines = ["model,fold,test_singers,edit,accuracy,eer,acc_real,acc_fake,layer"]
    res = {e: [] for e in edits}
    for fi, test_singers in enumerate(folds):
        L = fold_layer[fi]
        te = np.isin(singer_of, test_singers)
        # validation singer: one rotating singer that is not in this test fold
        train_rot = sorted(s for s in rot_singers if s not in test_singers)
        val_singer = train_rot[fi % len(train_rot)]
        va = singer_of == val_singer
        tr = ~te & ~va

        model = train_head(F["clean"][L][tr], y[tr], F["clean"][L][va], y[va], device, args.seed + fi)
        _, thr = B.eer(y[va], predict(model, F["clean"][L][va], device), return_threshold=True)

        for e in edits:
            p = predict(model, F[e][L][te], device)
            pred = (p > thr).astype(int)
            yt = y[te]
            r = dict(acc=(pred == yt).mean(), eer=B.eer(yt, p),
                     real=(pred[yt == 0] == 0).mean(), fake=(pred[yt == 1] == 1).mean())
            res[e].append(r)
            lines.append(f"{tag},{fi + 1},{'|'.join(test_singers)},{e},"
                         f"{r['acc']:.4f},{r['eer']:.4f},{r['real']:.4f},{r['fake']:.4f},{L}")
        print(f"fold {fi + 1}: layer {L:2d}  val singer {val_singer}  clean EER {res['clean'][-1]['eer']:.3f}")

    summary = ["edit,eer_mean,eer_std,acc_mean,acc_std,acc_real_mean,acc_real_std,acc_fake_mean,acc_fake_std"]
    print(f"\n{'edit':>12s}  {'EER (mean ± std)':>17s}  {'acc':>12s}  {'real ok':>12s}  {'fake ok':>12s}")
    for e in edits:
        v = {k: np.array([r[k] for r in res[e]]) for k in ("eer", "acc", "real", "fake")}
        print(f"{e:>12s}  {v['eer'].mean():8.3f} ± {v['eer'].std():.3f}  "
              f"{v['acc'].mean():.3f} ± {v['acc'].std():.3f}  "
              f"{v['real'].mean():.3f} ± {v['real'].std():.3f}  {v['fake'].mean():.3f} ± {v['fake'].std():.3f}")
        summary.append(f"{e},{v['eer'].mean():.4f},{v['eer'].std():.4f},{v['acc'].mean():.4f},{v['acc'].std():.4f},"
                       f"{v['real'].mean():.4f},{v['real'].std():.4f},{v['fake'].mean():.4f},{v['fake'].std():.4f}")

    (out / f"{tag}_attn_folds.csv").write_text("\n".join(lines) + "\n")
    (out / f"{tag}_attn_summary.csv").write_text("\n".join(summary) + "\n")
    print(f"\nsaved {out / (tag + '_attn_summary.csv')}   total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
