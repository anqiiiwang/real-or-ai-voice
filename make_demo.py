"""
Build the "Listen" demo for the website.

Picks one REAL clip and one AI clip from the test singers of one fold, makes edited versions,
saves them as short WAV files in docs/audio/, and scores every version with two detectors
trained on the other singers (same layer, classifier and threshold method as rotate.py / aug.py):
  - trained on clean audio only
  - trained with edited clips added (augmentation, edits never used in the demo)

Writes docs/data/demo.js, which the website reads.

Run AFTER rotate.py and aug.py for the same model (features are read from their cache):
    python make_demo.py                       # WavLM-Large, fold 1
    python make_demo.py --fold 2              # other test singers
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score

import baseline as B
from aug import DEFAULT_TRAIN_EDITS
from rotate import add_common_args, build_pool_and_folds

VERSIONS = [("clean", "Original"), ("pitch:1", "Pitch +1 st"), ("pitch:2", "Pitch +2 st"),
            ("pitch:4", "Pitch +4 st"), ("pitch:6", "Pitch +6 st"),
            ("pitchrt:2", "+2 st and back"), ("reverb:1.0", "Reverb 1.0 s")]


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.set_defaults(model="microsoft/wavlm-large")
    ap.add_argument("--fold", type=int, default=1, help="which singer fold to take the test clips from (1-5)")
    ap.add_argument("--train-edits", default=DEFAULT_TRAIN_EDITS)
    ap.add_argument("--site", default="docs")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    tag = args.model.split("/")[-1]
    train_edits = args.train_edits.split(",")

    files, pool, singers, folds = build_pool_and_folds(args, rng)
    y = np.array([r["label"] == "deepfake" for r in pool], int)
    singer_of = np.array([r["singer"] for r in pool])
    test_singers = folds[args.fold - 1]
    te = np.isin(singer_of, test_singers)
    tr = ~te
    print(f"model={args.model}  fold {args.fold}: test singers {test_singers}")

    ext = B.Extractor(args.model, device)
    X = {e: B.extract(pool, files, ext, e, args, B.cache_path(out, tag, "pool", e, pool, args))
         for e in ["clean"] + train_edits}

    # layer: the one rotate.py chose for this fold (else choose it the same way)
    L = None
    f = out / f"{tag}_rotate_folds.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            if int(r["fold"]) == args.fold:
                L = int(r["layer"])
    groups = singer_of[tr]
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    if L is None:
        L = int(np.argmax([cross_val_score(B.clf(), X["clean"][tr][:, l], y[tr], groups=groups, cv=cv).mean()
                           for l in range(X["clean"].shape[1])]))
    print(f"layer {L}")

    def fit(edits):
        Xtr = np.concatenate([X[e][tr][:, L] for e in ["clean"] + edits])
        ytr = np.concatenate([y[tr]] * (1 + len(edits)))
        gtr = np.concatenate([groups] * (1 + len(edits)))
        oof = cross_val_predict(B.clf(), Xtr, ytr, groups=gtr, cv=cv, method="predict_proba")[:, 1]
        _, thr = B.eer(ytr, oof, return_threshold=True)
        return B.clf().fit(Xtr, ytr), float(thr)

    m_none, thr_none = fit([])
    m_aug, thr_aug = fit(train_edits)
    print(f"thresholds: clean-trained {thr_none:.3f}   augmented {thr_aug:.3f}")

    # pick a typical correctly-detected clip of each class (median score among the correct ones)
    p_clean = m_none.predict_proba(X["clean"][:, L])[:, 1]
    idx_te = np.where(te)[0]
    picks = []
    for cls, correct in [(0, lambda p: p <= thr_none), (1, lambda p: p > thr_none)]:
        cand = [i for i in idx_te if y[i] == cls and correct(p_clean[i])]
        med = np.median(p_clean[cand])
        picks.append(min(cand, key=lambda i: abs(p_clean[i] - med)))

    audio_dir = Path(args.site) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for old in audio_dir.glob("demo_*.wav"):
        old.unlink()
    clips = []
    for i, kind in zip(picks, ["real", "ai"]):
        r = pool[i]
        y0 = B.load_clip(files[r["name"]], ext.sr, args.seconds)
        edit_rng = np.random.default_rng(args.seed)
        rows = []
        for e, label in VERSIONS:
            wav = B.fix_length(B.apply_edit(y0, ext.sr, e, edit_rng), ext.sr, args.seconds)
            name = f"demo_{kind}_{e.replace(':', '').replace('.', '')}.wav"
            # save at 16 kHz to keep the files small
            out_wav = wav if ext.sr == 16000 else __import__("librosa").resample(wav, orig_sr=ext.sr, target_sr=16000)
            peak = np.max(np.abs(out_wav)) + 1e-9
            sf.write(audio_dir / name, (out_wav / peak * 0.9).astype(np.float32), 16000, subtype="PCM_16")
            feat = ext([wav])[0, L][None]
            rows.append({"label": label, "file": f"audio/{name}",
                         "score_none": round(float(m_none.predict_proba(feat)[0, 1]), 3),
                         "score_aug": round(float(m_aug.predict_proba(feat)[0, 1]), 3)})
        clips.append({"kind": kind, "singer": r["singer"], "attack": r["attack"], "rows": rows})
        print(f"\n{kind.upper()} clip {r['name']} ({r['singer']}, {r['attack']})")
        for row in rows:
            v = lambda s, t: ("AI  " if s > t else "real") + f" {s:.2f}"
            print(f"  {row['label']:>15s}   clean-trained: {v(row['score_none'], thr_none)}"
                  f"   augmented: {v(row['score_aug'], thr_aug)}")

    demo = {"model": tag, "fold": args.fold, "thr_none": round(thr_none, 3), "thr_aug": round(thr_aug, 3),
            "clips": clips}
    dest = Path(args.site) / "data" / "demo.js"
    dest.write_text("// generated by make_demo.py - do not edit by hand\nwindow.DEMO = "
                    + json.dumps(demo, indent=1) + ";\n")
    print(f"\nwrote {dest} and {len(clips) * len(VERSIONS)} audio files in {audio_dir}")


if __name__ == "__main__":
    main()
