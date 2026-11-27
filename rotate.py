"""
Rotated test singers (singer-level cross-validation).

Every M4Singer singer is used as a test singer exactly once:
  - the 10 M4Singer singers are split into 5 folds of 2 singers
  - for each fold: train on all other singers (incl. Opencpop), test on the 2 held-out singers
  - layer and decision threshold are chosen inside each fold, on its training singers only
Results are reported as mean +- std over the 5 folds.

To keep it fast, features are extracted ONCE for a fixed pool of clips
(N real + N AI clips per singer, for clean audio and every edit), then reused by all folds.

Run (from the project folder):
    python rotate.py                                  # wav2vec2-base
    python rotate.py --model microsoft/wavlm-large
    python rotate.py --model m-a-p/MERT-v1-95M

Round-trip control (edit and undo; saved separately so the main results are kept):
    python rotate.py --model microsoft/wavlm-large --suffix _roundtrip \
        --edits clean,pitch:2,pitchrt:2,tempo:1.2,temport:1.2
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

import baseline as B


def build_pool_and_folds(args, rng):
    """Fixed pool (per singer N real + N AI, AI spread over attacks) and the singer folds.
    Shared with attn_head.py so both scripts use exactly the same clips and folds."""
    sources = set(args.sources.split(","))
    files, rows = {}, []
    for txt, audio in zip(args.txt, args.audio):
        files.update({p.stem: p for p in Path(audio).rglob("*.flac")})
        rows += B.read_labels(txt, sources)
    rows = [r for r in rows if r["name"] in files]
    singers = sorted({(r["source"], r["singer"]) for r in rows})
    pool = []
    for src, s in singers:
        pool += B.balanced_subset([r for r in rows if r["singer"] == s], args.per_singer, rng)
    rot = [s for src, s in singers if src == args.rotate_source]
    rng.shuffle(rot)
    folds = [sorted(rot[i::args.folds]) for i in range(args.folds)]
    return files, pool, singers, folds


def add_common_args(ap):
    ap.add_argument("--txt", nargs="+", default=["data/dev.txt", "data/train.txt"])
    ap.add_argument("--audio", nargs="+", default=["data/dev_set", "data/train_set"])
    ap.add_argument("--sources", default="m4singer,opencpop")
    ap.add_argument("--rotate-source", default="m4singer", help="singers of this source take turns as test singers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--per-singer", type=int, default=100, help="real clips per singer (same number of AI clips)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--edits", default=B.DEFAULT_EDITS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--suffix", default="", help="added to the output file names, e.g. _roundtrip")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1]
    edits = args.edits.split(",")

    files, pool, singers, folds = build_pool_and_folds(args, rng)
    y = np.array([r["label"] == "deepfake" for r in pool], int)
    singer_of = np.array([r["singer"] for r in pool])
    print(f"device={device}  model={args.model}")
    print(f"pool: {len(pool)} clips from {len(singers)} singers")
    for i, f in enumerate(folds):
        print(f"  fold {i + 1}: test singers {f}")
    n_cond = len(edits)
    print(f"extracting {len(pool)} clips x {n_cond} conditions (cached after the first run)\n")

    # --- extract features once per condition
    ext = B.Extractor(args.model, device)
    t0 = time.time()
    X = {}
    for e in edits:
        X[e] = B.extract(pool, files, ext, e, args, B.cache_path(out, tag, "pool", e, pool, args))
        print(f"  features {e:>11s}: {X[e].shape}  ({time.time() - t0:.0f}s)")

    # --- folds
    lines = ["model,fold,test_singers,edit,accuracy,eer,acc_real,acc_fake,layer"]
    res = {e: [] for e in edits}
    for fi, test_singers in enumerate(folds):
        te = np.isin(singer_of, test_singers)
        tr = ~te
        groups = singer_of[tr]
        cv = GroupKFold(n_splits=min(5, len(set(groups))))
        Xc = X["clean"][tr]
        layer_acc = [cross_val_score(B.clf(), Xc[:, l], y[tr], groups=groups, cv=cv).mean()
                     for l in range(Xc.shape[1])]
        best = int(np.argmax(layer_acc))
        oof = cross_val_predict(B.clf(), Xc[:, best], y[tr], groups=groups, cv=cv, method="predict_proba")[:, 1]
        _, thr = B.eer(y[tr], oof, return_threshold=True)
        model = B.clf().fit(Xc[:, best], y[tr])

        for e in edits:
            p = model.predict_proba(X[e][te][:, best])[:, 1]
            pred = (p > thr).astype(int)
            yt = y[te]
            r = dict(acc=(pred == yt).mean(), eer=B.eer(yt, p),
                     real=(pred[yt == 0] == 0).mean(), fake=(pred[yt == 1] == 1).mean())
            res[e].append(r)
            lines.append(f"{tag},{fi + 1},{'|'.join(test_singers)},{e},"
                         f"{r['acc']:.4f},{r['eer']:.4f},{r['real']:.4f},{r['fake']:.4f},{best}")
        print(f"fold {fi + 1}: layer {best:2d}  clean EER {res['clean'][-1]['eer']:.3f}")

    # --- summary
    summary = ["edit,eer_mean,eer_std,acc_mean,acc_std,acc_real_mean,acc_real_std,acc_fake_mean,acc_fake_std"]
    print(f"\n{'edit':>12s}  {'EER (mean ± std)':>17s}  {'acc':>12s}  {'real ok':>12s}  {'fake ok':>12s}")
    for e in edits:
        v = {k: np.array([r[k] for r in res[e]]) for k in ("eer", "acc", "real", "fake")}
        print(f"{e:>12s}  {v['eer'].mean():8.3f} ± {v['eer'].std():.3f}  "
              f"{v['acc'].mean():.3f} ± {v['acc'].std():.3f}  "
              f"{v['real'].mean():.3f} ± {v['real'].std():.3f}  {v['fake'].mean():.3f} ± {v['fake'].std():.3f}")
        summary.append(f"{e},{v['eer'].mean():.4f},{v['eer'].std():.4f},{v['acc'].mean():.4f},{v['acc'].std():.4f},"
                       f"{v['real'].mean():.4f},{v['real'].std():.4f},{v['fake'].mean():.4f},{v['fake'].std():.4f}")

    name = f"{tag}_rotate{args.suffix}"
    (out / f"{name}_folds.csv").write_text("\n".join(lines) + "\n")
    (out / f"{name}_summary.csv").write_text("\n".join(summary) + "\n")
    print(f"\nsaved {out / (name + '_summary.csv')} and per-fold results   total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
