"""
Baseline: Chinese singing-voice deepfake detection + robustness to audio edits.

Pipeline
  1. Take the Mandarin clips (m4singer + opencpop) from CtrSVDD dev and/or train set
  2. Split BY SINGER (test singers never seen in training)
  3. Balanced subset: N real + N fake (fakes spread evenly over attacks A01..A08)
  4. Frozen pretrained model (wav2vec2 / MERT) -> mean-pooled features of every layer
  5. Logistic regression per layer; best layer AND decision threshold picked by
     singer-grouped CV on the TRAINING singers only (no peeking at the test set)
  6. Evaluate on the test singer: clean + edited versions (pitch, tempo, noise, reverb)

Install (once):
    pip install torch transformers librosa soundfile scikit-learn scipy tqdm

Run from the project folder (expects data/dev.txt, data/dev_set, data/train.txt, data/train_set):
    python baseline.py

Only the dev set:
    python baseline.py --txt data/dev.txt --audio data/dev_set

Quick try:
    python baseline.py --n-train 300 --n-test 100

Other front ends:
    python baseline.py --model microsoft/wavlm-large
    python baseline.py --model m-a-p/MERT-v1-95M
"""
import argparse
import hashlib
import os
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import librosa
import numpy as np
import torch
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

DEFAULT_EDITS = "clean,pitch:1,pitch:2,pitch:4,pitch:6,tempo:0.8,tempo:1.2,noise:20,noise:10,reverb:0.5,reverb:1.0"


# ---------------------------------------------------------------- data
def read_labels(txt, sources):
    rows = []
    for line in open(txt):
        p = line.split()
        if len(p) == 6 and p[0] in sources:
            rows.append(dict(source=p[0], singer=p[1], name=p[2], attack=p[4], label=p[5]))
    return rows


def balanced_subset(rows, n, rng):
    """n real + n fake; fakes spread evenly over attack types."""
    real = [r for r in rows if r["label"] == "bonafide"]
    by_attack = defaultdict(list)
    for r in rows:
        if r["label"] == "deepfake":
            by_attack[r["attack"]].append(r)
    rng.shuffle(real)
    fake = []
    per = int(np.ceil(n / len(by_attack)))
    for a in sorted(by_attack):
        rng.shuffle(by_attack[a])
        fake += by_attack[a][:per]
    rng.shuffle(fake)
    return real[:n] + fake[:n]


def load_clip(path, sr, seconds):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def fix_length(y, sr, seconds):
    """Center crop / zero-pad to a fixed length."""
    L = int(sr * seconds)
    if len(y) >= L:
        s = (len(y) - L) // 2
        return y[s:s + L]
    return np.pad(y, (0, L - len(y)))


# ---------------------------------------------------------------- edits
def apply_edit(y, sr, edit, rng):
    """edit strings: clean | pitch:<semitones> | tempo:<rate> | noise:<snr_db> | reverb:<rt60_s>"""
    if edit == "clean":
        return y
    kind, val = edit.split(":")
    val = float(val)
    if kind == "pitch":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=val)
    if kind == "tempo":
        return librosa.effects.time_stretch(y, rate=val)
    if kind == "noise":
        p_sig = np.mean(y ** 2) + 1e-12
        noise = rng.standard_normal(len(y)) * np.sqrt(p_sig / 10 ** (val / 10))
        return (y + noise).astype(np.float32)
    if kind == "reverb":
        # synthetic room impulse response: exponentially decaying noise, -60 dB at rt60
        n = int(sr * val)
        t = np.arange(n) / sr
        ir = rng.standard_normal(n) * np.exp(-6.9 * t / val)
        ir[0] = 1.0
        out = fftconvolve(y, ir)[: len(y)]
        return (out / (np.max(np.abs(out)) + 1e-9) * np.max(np.abs(y))).astype(np.float32)
    raise ValueError(f"unknown edit: {edit}")


# ---------------------------------------------------------------- features
class Extractor:
    def __init__(self, name, device):
        self.fe = AutoFeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        # some remote-code models (e.g. MERT) ignore the call argument and only read the config
        self.model.config.output_hidden_states = True
        self.model.config.return_dict = True
        self.sr = self.fe.sampling_rate
        self.device = device
        self.force_hooks = False

    def _layers(self):
        """The ModuleList holding the transformer layers (e.g. model.encoder.layers)."""
        cands = [(n, m) for n, m in self.model.named_modules()
                 if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and "conv" not in n
                 and len(m) >= 2]
        if not cands:
            raise RuntimeError("Could not find the transformer layers of this model.")
        return min(cands, key=lambda c: c[0].count("."))[1]

    def _hidden_states_via_hooks(self, x):
        captured = []
        layers = self._layers()
        # input of the first layer = embedding output ("layer 0"), then each layer's output
        handles = [layers[0].register_forward_pre_hook(
            lambda m, args, kwargs: captured.append(args[0] if args else kwargs["hidden_states"]),
            with_kwargs=True)]
        for layer in layers:
            handles.append(layer.register_forward_hook(
                lambda m, inp, out: captured.append(out[0] if isinstance(out, (tuple, list)) else out)))
        try:
            self.model(**x)
        finally:
            for h in handles:
                h.remove()
        return captured

    @torch.no_grad()
    def __call__(self, wavs):
        hs = self.hidden(wavs)
        # (layers, batch, dim): mean over time for every layer
        return torch.stack([h.mean(dim=1) for h in hs]).permute(1, 0, 2).float().cpu().numpy()

    @torch.no_grad()
    def hidden(self, wavs):
        """List of per-layer hidden states, each (batch, time, dim)."""
        x = self.fe(wavs, sampling_rate=self.sr, return_tensors="pt")
        x = {k: v.to(self.device) for k, v in x.items()}
        if self.force_hooks:
            hs = self._hidden_states_via_hooks(x)
        else:
            out = self.model(**x, output_hidden_states=True)
            hs = getattr(out, "hidden_states", None)
            if hs is None and isinstance(out, (tuple, list)):  # old-style tuple output
                hs = next((o for o in out if isinstance(o, (tuple, list)) and len(o) > 1), None)
            if hs is None:
                # e.g. MERT's own code + transformers 5: the model ignores output_hidden_states.
                # Fall back to reading every transformer layer's output with forward hooks.
                self.force_hooks = True
                hs = self._hidden_states_via_hooks(x)
        return hs


def cache_path(out, tag, split, edit, rows, args):
    """Cache name depends on the exact clips, so changing the subset never reuses stale features."""
    h = hashlib.md5(("|".join(r["name"] for r in rows) + f"{args.seconds}").encode()).hexdigest()[:8]
    return out / f"{tag}_{split}_{edit.replace(':', '')}_{h}.npz"


def extract(rows, files, ext, edit, args, cache):
    if cache.exists():
        return np.load(cache)["X"]
    rng = np.random.default_rng(args.seed)
    feats, batch = [], []
    for r in tqdm(rows, desc=f"{edit:>12s}", leave=False):
        y = load_clip(files[r["name"]], ext.sr, args.seconds)
        y = fix_length(apply_edit(y, ext.sr, edit, rng), ext.sr, args.seconds)
        batch.append(y)
        if len(batch) == args.batch:
            feats.append(ext(batch)); batch = []
    if batch:
        feats.append(ext(batch))
    X = np.concatenate(feats)
    np.savez_compressed(cache, X=X)
    return X


# ---------------------------------------------------------------- metrics
def eer(y, score, return_threshold=False):
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    e = (fpr[i] + fnr[i]) / 2
    return (e, thr[i]) if return_threshold else e


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", nargs="+", default=["data/dev.txt", "data/train.txt"])
    ap.add_argument("--audio", nargs="+", default=["data/dev_set", "data/train_set"])
    ap.add_argument("--sources", default="m4singer,opencpop")
    ap.add_argument("--test-singers", default=None,
                    help="comma-separated singer IDs held out for testing; default: last 2 m4singer singers")
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--n-train", type=int, default=1000, help="per class")
    ap.add_argument("--n-test", type=int, default=300, help="per class")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--edits", default=DEFAULT_EDITS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(exist_ok=True)
    tag = args.model.split("/")[-1]

    # --- data + split by singer
    assert len(args.txt) == len(args.audio), "give one --audio folder per --txt file"
    sources = set(args.sources.split(","))
    files, rows = {}, []
    for txt, audio in zip(args.txt, args.audio):
        files.update({p.stem: p for p in Path(audio).rglob("*.flac")})
        rows += read_labels(txt, sources)
    rows = [r for r in rows if r["name"] in files]
    singers = sorted({(r["source"], r["singer"]) for r in rows})
    if args.test_singers:
        test_singers = set(args.test_singers.split(","))
    else:
        test_singers = {s for _, s in [x for x in singers if x[0] == "m4singer"][-2:]}
    train_rows = balanced_subset([r for r in rows if r["singer"] not in test_singers], args.n_train, rng)
    test_rows = balanced_subset([r for r in rows if r["singer"] in test_singers], args.n_test, rng)
    print(f"device={device}  model={args.model}")
    print(f"{len(singers)} singers: " + ", ".join(f"{s}({src})" for src, s in singers))
    print(f"test singers: {sorted(test_singers)}")
    print(f"train: {len(train_rows)} clips   test: {len(test_rows)} clips\n")

    y_tr = np.array([r["label"] == "deepfake" for r in train_rows], int)
    y_te = np.array([r["label"] == "deepfake" for r in test_rows], int)
    groups = [r["singer"] for r in train_rows]

    ext = Extractor(args.model, device)
    t0 = time.time()
    X_tr = extract(train_rows, files, ext, "clean", args, cache_path(out, tag, "train", "clean", train_rows, args))
    print(f"train features: {X_tr.shape}  ({time.time() - t0:.0f}s)")

    # --- choose layer with leave-one-singer-out CV on TRAIN singers only
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    layer_acc = [cross_val_score(clf(), X_tr[:, l], y_tr, groups=groups, cv=cv).mean()
                 for l in range(X_tr.shape[1])]
    best = int(np.argmax(layer_acc))
    print("CV accuracy per layer: " + " ".join(f"{a:.2f}" for a in layer_acc))
    print(f"-> best layer: {best}\n")

    # --- decision threshold: EER point of out-of-fold predictions on TRAIN singers.
    # (A fixed 0.5 threshold breaks when the test singer sounds different from the training singers.)
    oof = cross_val_predict(clf(), X_tr[:, best], y_tr, groups=groups, cv=cv, method="predict_proba")[:, 1]
    cv_eer, thr = eer(y_tr, oof, return_threshold=True)
    print(f"CV EER on training singers: {cv_eer:.3f}   decision threshold: {thr:.3f}\n")

    model = clf().fit(X_tr[:, best], y_tr)

    # --- evaluate clean + edits on the unseen singer
    lines = ["edit,accuracy,eer,acc_real,acc_fake"]
    print(f"{'edit':>12s}  {'acc':>6s}  {'EER':>6s}  {'real ok':>7s}  {'fake ok':>7s}")
    for edit in args.edits.split(","):
        X_te = extract(test_rows, files, ext, edit, args, cache_path(out, tag, "test", edit, test_rows, args))
        p = model.predict_proba(X_te[:, best])[:, 1]
        pred = (p > thr).astype(int)
        acc = (pred == y_te).mean()
        e = eer(y_te, p)
        a_real = (pred[y_te == 0] == 0).mean()
        a_fake = (pred[y_te == 1] == 1).mean()
        print(f"{edit:>12s}  {acc:6.3f}  {e:6.3f}  {a_real:7.3f}  {a_fake:7.3f}")
        lines.append(f"{edit},{acc:.4f},{e:.4f},{a_real:.4f},{a_fake:.4f}")

        if edit == "clean":  # per-attack breakdown
            for a in sorted({r["attack"] for r in test_rows if r["attack"] != "-"}):
                m = np.array([r["attack"] == a for r in test_rows])
                print(f"{'  ' + a:>12s}  fake detected: {(pred[m] == 1).mean():.3f}")

    csv = out / f"{tag}_results_n{args.n_train}_s{args.seed}.csv"
    csv.write_text("\n".join(lines) + "\n")
    print(f"\nsaved {csv}   total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
