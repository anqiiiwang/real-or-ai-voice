# Real Voice or AI Voice? Robust Detection of AI-Sung Chinese Songs

**Project page (results, figures and audio demo):** https://anqiiiwang.github.io/real-or-ai-voice/

Course project for *Audio Processing and Indexing*, Leiden University, 2026, by Anqi Wang.

## Summary

Singing-voice deepfake detectors are usually evaluated on clean audio. We test three of them on Mandarin
singing after everyday edits: **pitch shift, tempo change, noise and reverb**.

- Every edit hurts every detector. The detector that is best on clean audio (WavLM) collapses: after a
  one-semitone pitch shift it labels almost every real singer as AI.
- A control experiment (edit, then undo it) shows that the detectors react to **processing traces**, not to
  the musical change itself.
- A stronger classifier head does not fix this, but **training with a few edited clips** does, even for
  edit types never seen in training (except noise).

## Method

- **Data:** Mandarin part of [CtrSVDD](https://zenodo.org/records/10467648) (Zang et al., Interspeech 2024):
  11 singers (10 M4Singer + 1 Opencpop), 100 real + 100 AI clips per singer, 4-second segments.
- **Front ends (frozen):** wav2vec2-base, WavLM-Large, MERT-v1-95M. Each layer is mean-pooled over time.
- **Classifier:** logistic regression on one layer. Layer and decision threshold are chosen by
  singer-grouped cross-validation on the training singers only.
- **Evaluation:** 5 folds, each holding out 2 M4Singer singers; mean ± std over folds. Metric: EER, plus
  accuracy on real and AI clips separately.
- **Edits (test):** pitch +1/+2/+4/+6 semitones, tempo ×0.8/×1.2, white noise 20/10 dB SNR,
  synthetic reverb RT60 0.5/1.0 s.

## Code

| Script | What it does |
|---|---|
| `baseline.py` | Shared code (data loading, edits, feature extraction, metrics). Can also run a single quick experiment with one fixed train/test split. |
| `rotate.py` | Main experiment: 5 rotated singer folds, clean + all edits. |
| `attn_head.py` | Same folds, but an attentive pooling head instead of mean pooling + logistic regression. |
| `aug.py` | Augmentation: adds edited training clips (edits never used in testing), incl. leave-one-edit-type-out. |
| `make_demo.py` | Builds the audio demo on the project page. |
| `make_site_data.py` | Collects all result CSVs into `docs/data/results.js` for the project page. |

## Run

Download `dev.txt`, `dev_set.zip`, `train.txt` and `train_set.zip` from Zenodo and unzip into `data/`:

```
data/
├── dev.txt
├── dev_set/
├── train.txt
└── train_set/
```

Then:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

for m in facebook/wav2vec2-base microsoft/wavlm-large m-a-p/MERT-v1-95M; do
  python rotate.py --model $m                                        # main results
  python attn_head.py --model $m                                     # attentive head
  python aug.py --model $m                                           # augmentation
  python rotate.py --model $m --suffix _roundtrip \
      --edits clean,pitch:2,pitchrt:2,tempo:1.2,temport:1.2          # round-trip control
done

python make_demo.py          # audio demo (WavLM-Large, fold 1)
python make_site_data.py     # update the project page
```

Features are cached in `results/*.npz` (not in git), so reruns are fast. Each model takes roughly
10–15 minutes per experiment on an Apple M4 Max.

## References

1. M. Frohmann et al., "AI-Generated Song Detection via Lyrics Transcripts," ISMIR 2025.
2. Y. Zang et al., "SingFake: Singing Voice Deepfake Detection," ICASSP 2024.
3. Y. Zang et al., "CtrSVDD: A Benchmark Dataset and Baseline Analysis for Controlled Singing Voice Deepfake Detection," Interspeech 2024.
4. Y. Zhang et al., "SVDD 2024: The Inaugural Singing Voice Deepfake Detection Challenge," IEEE SLT 2024.
5. X. Wang and J. Yamagishi, "Investigating Self-Supervised Front Ends for Speech Spoofing Countermeasures," Odyssey 2022.
6. B. Hurt and Y. Yang, "Which Layer to Trust? Layer Selection in Frozen Speech Encoders for Singing-Voice Deepfake Detection under Signal Corruption," arXiv:2611.29871, 2026.
7. D. Afchar, G. Meseguer-Brocal and R. Hennequin, "Detecting Music Deepfakes is Easy but Actually Hard," arXiv:2405.04181, 2024.
