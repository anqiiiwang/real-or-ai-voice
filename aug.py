"""
RQ4: can training with edited audio (augmentation) recover the lost performance?

Same clips, singer folds, layer choice and classifier as rotate.py (mean pooling + logistic regression).
Only the TRAINING data changes: the training singers' clips are added again with edits.

Fair test: the training edits are never the same as the test edits.
  test edits  (from rotate.py): pitch +1 +2 +4 +6 | tempo x0.8 x1.2 | noise 20 10 dB | reverb 0.5 1.0 s
  train edits (default)       : pitch -1 -3       | tempo x0.9 x1.1 | noise 30 dB     | reverb 0.3 s
  (other direction / other strengths of the same kind of edit)

Three training settings, all tested on the same unseen singers:
  none : clean audio only (= rotate.py)
  all  : clean + all training edits
  lofo : leave one edit FAMILY out. To test e.g. pitch shift, train with clean + tempo/noise/reverb
         edits only. This asks: does augmentation help against a kind of edit it has never seen?

Run AFTER rotate.py for the same model (the clean and test-edit features are reused from its cache):
    python aug.py --model microsoft/wavlm-large
    python aug.py --model m-a-p/MERT-v1-95M
    python aug.py
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

import baseline as B
from rotate import add_common_args, build_pool_and_folds

DEFAULT_TRAIN_EDITS = "pitch:-1,pitch:-3,tempo:0.9,tempo:1.1,noise:30,reverb:0.3"
family = lambda e: e.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--train-edits", default=DEFAULT_TRAIN_EDITS)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1]
    test_edits = args.edits.split(",")
    train_edits = args.train_edits.split(",")
    overlap = set(test_edits) & set(train_edits)
    assert not overlap, f"training and test edits must differ: {overlap}"
    families = sorted({family(e) for e in test_edits if e != "clean"})

    files, pool, singers, folds = build_pool_and_folds(args, rng)
    y = np.array([r["label"] == "deepfake" for r in pool], int)
    singer_of = np.array([r["singer"] for r in pool])
    print(f"device={device}  model={args.model}")
    print(f"train edits: {train_edits}")
    print(f"test edits : {test_edits}\n")

    ext = B.Extractor(args.model, device)
    t0 = time.time()
    X = {}
    for e in ["clean"] + [e for e in test_edits if e != "clean"] + train_edits:
        X[e] = B.extract(pool, files, ext, e, args, B.cache_path(out, tag, "pool", e, pool, args))
        print(f"  features {e:>11s}: {X[e].shape}  ({time.time() - t0:.0f}s)")

    # training settings: name -> (list of training edits, which test families it is evaluated on)
    settings = {"none": ([], families), "all": (train_edits, families)}
    for f in families:
        settings[f"lofo-{f}"] = ([e for e in train_edits if family(e) != f], [f])

    lines = ["model,setting,fold,edit,accuracy,eer,acc_real,acc_fake"]
    res = {}  # (setting, edit) -> list of fold dicts
    for fi, test_singers in enumerate(folds):
        te = np.isin(singer_of, test_singers)
        tr = ~te
        groups_clean = singer_of[tr]
        cv = GroupKFold(n_splits=min(5, len(set(groups_clean))))
        # same layer choice as rotate.py: CV on clean training audio
        Xc = X["clean"][tr]
        layer_acc = [cross_val_score(B.clf(), Xc[:, l], y[tr], groups=groups_clean, cv=cv).mean()
                     for l in range(Xc.shape[1])]
        L = int(np.argmax(layer_acc))

        for name, (aug_edits, fams) in settings.items():
            Xtr = np.concatenate([X[e][tr][:, L] for e in ["clean"] + aug_edits])
            ytr = np.concatenate([y[tr]] * (1 + len(aug_edits)))
            gtr = np.concatenate([groups_clean] * (1 + len(aug_edits)))
            oof = cross_val_predict(B.clf(), Xtr, ytr, groups=gtr, cv=cv, method="predict_proba")[:, 1]
            _, thr = B.eer(ytr, oof, return_threshold=True)
            model = B.clf().fit(Xtr, ytr)
            for e in ["clean"] + [e for e in test_edits if e != "clean" and family(e) in fams]:
                p = model.predict_proba(X[e][te][:, L])[:, 1]
                pred = (p > thr).astype(int)
                yt = y[te]
                r = dict(acc=(pred == yt).mean(), eer=B.eer(yt, p),
                         real=(pred[yt == 0] == 0).mean(), fake=(pred[yt == 1] == 1).mean())
                res.setdefault((name, e), []).append(r)
                lines.append(f"{tag},{name},{fi + 1},{e},{r['acc']:.4f},{r['eer']:.4f},{r['real']:.4f},{r['fake']:.4f}")
        print(f"fold {fi + 1}: layer {L:2d}  clean EER none {res[('none', 'clean')][-1]['eer']:.3f}"
              f"  all {res[('all', 'clean')][-1]['eer']:.3f}   ({time.time() - t0:.0f}s)")

    # summary: one row per test edit, EER for none / all / lofo
    def ms(name, e, k="eer"):
        v = np.array([r[k] for r in res[(name, e)]])
        return v.mean(), v.std()

    summary = ["edit,setting,eer_mean,eer_std,acc_mean,acc_std,acc_real_mean,acc_real_std,acc_fake_mean,acc_fake_std"]
    print(f"\nEER (mean ± std over folds)")
    print(f"{'edit':>12s}  {'clean only':>15s}  {'+ all aug':>15s}  {'+ aug, family unseen':>21s}")
    for e in test_edits:
        lofo = "lofo-" + family(e) if e != "clean" else None
        cells = []
        for name in ["none", "all"] + ([lofo] if lofo else []):
            m, s = ms(name, e)
            cells.append(f"{m:.3f} ± {s:.3f}")
            row = [f"{ms(name, e, k)[i]:.4f}" for k in ("eer", "acc", "real", "fake") for i in (0, 1)]
            summary.append(",".join([e, "lofo" if name.startswith("lofo") else name] + row))
        print(f"{e:>12s}  {cells[0]:>15s}  {cells[1]:>15s}  {cells[2] if len(cells) > 2 else '':>21s}")

    print(f"\nreal ok / fake ok  (clean only -> + all aug)")
    for e in test_edits:
        a, b = ms("none", e, "real")[0], ms("all", e, "real")[0]
        c, d = ms("none", e, "fake")[0], ms("all", e, "fake")[0]
        print(f"{e:>12s}  real {a:.3f} -> {b:.3f}   fake {c:.3f} -> {d:.3f}")

    (out / f"{tag}_aug_folds.csv").write_text("\n".join(lines) + "\n")
    (out / f"{tag}_aug_summary.csv").write_text("\n".join(summary) + "\n")
    print(f"\nsaved {out / (tag + '_aug_summary.csv')}   total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
