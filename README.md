# Real Voice or AI Voice? Robust Detection of AI-Sung Chinese Songs

**Project page:** https://anqiiiwang.github.io/real-or-ai-voice/

Singing-voice deepfake detectors are evaluated on clean audio. This project tests how well they hold up on
Mandarin singing after everyday edits: **pitch shift, tempo change, noise and reverb**.

Course project for *Audio Processing and Indexing*, Leiden University, 2026, by Anqi Wang.

## Method

A frozen pretrained model (wav2vec2-base, WavLM-Large or MERT-95M) extracts features from each 4-second clip.
A logistic regression classifier is trained on top to predict real or AI. The best layer and the decision
threshold are chosen by singer-grouped cross-validation on the training singers. Testing is done on
unseen singers, on clean audio and on edited copies of the same clips.

## Data

Mandarin part (M4Singer + Opencpop) of [CtrSVDD](https://zenodo.org/records/10467648) (Zang et al., Interspeech 2024).
Download `dev.txt`, `dev_set.zip`, `train.txt` and `train_set.zip`, then unzip them into `data/`:

```
data/
├── dev.txt
├── dev_set/
├── train.txt
└── train_set/
```

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python baseline.py                                  # wav2vec2-base
python baseline.py --model microsoft/wavlm-large
python baseline.py --model m-a-p/MERT-v1-95M

python make_site_data.py                            # update the project page
```

Results are written to `results/*.csv`. The project page lives in `docs/` and is served with GitHub Pages.

## References

1. M. Frohmann et al., "AI-Generated Song Detection via Lyrics Transcripts," ISMIR 2025.
2. Y. Zang et al., "SingFake: Singing Voice Deepfake Detection," ICASSP 2024.
3. Y. Zang et al., "CtrSVDD: A Benchmark Dataset and Baseline Analysis for Controlled Singing Voice Deepfake Detection," Interspeech 2024.
4. Y. Zhang et al., "SVDD 2024: The Inaugural Singing Voice Deepfake Detection Challenge," IEEE SLT 2024.
5. X. Wang and J. Yamagishi, "Investigating Self-Supervised Front Ends for Speech Spoofing Countermeasures," Odyssey 2022.
