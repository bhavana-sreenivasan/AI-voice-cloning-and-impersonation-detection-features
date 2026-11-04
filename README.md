# 🎙️ AI-Powered Real-Time Voice Cloning & Impersonation Detection

A prototype that checks whether a voice is AI-generated (cloned) and whether it sounds like a known speaker, then combines both into a risk level on a Streamlit dashboard. Two anti-spoofing models are evaluated: **AASIST** (used in the app) and **SSL-AASIST** (wav2vec 2.0 XLS-R front end + AASIST back end, evaluated offline so far).

> **Status:** prototype. The dashboard works on uploaded clips and call recordings. SSL-AASIST is not connected to the dashboard yet, and live call capture and several planned features are not implemented (see [Roadmap](#roadmap)).

---

## What it does (current version)

One upload gives a spoof check, a speaker-identity check, and a fused risk level.

- **Upload** a clip in WAV, FLAC, MP3, M4A, AAC or OGG. It is converted to WAV automatically (MP3/M4A/AAC need [ffmpeg](https://ffmpeg.org/) installed).
- **Anti-spoofing:** a pretrained **AASIST** model scores overlapping 4-second windows (1-second hop) and flags where synthetic speech starts.
- **Speaker identity:** an **ECAPA-TDNN** model (SpeechBrain) compares the voice with a saved voice library. Voices are saved only with an explicit consent checkbox, and only clips with low spoof risk can be saved.
- **Fused risk:** the two checks are combined into **Low / Uncertain / High**, with a plain-language reason.
- **Phone-call recordings:** speakers are separated and you choose which voice is the caller.
- **Challenge-response:** for *Uncertain* results, the app generates a random phrase, transcribes the caller's reply (faster-whisper) and re-runs the voice checks.
- **Call log and complaint draft:** each analysis is logged with the caller number and can be turned into a complaint draft (user-confirmed, never auto-submitted).
- **Dashboard:** verdict, authenticity, closest saved voice, share of suspicious audio, a timeline of spoof probability and voice similarity, and a voice-swap alert.

## Results

Both models were evaluated on the **same random 400-file subset** of the ASVspoof 2019 LA evaluation set (seed 42; 39 bonafide, 361 spoof).

| Model | Overall EER |
| ----- | ----------- |
| AASIST (pretrained, clovaai) | **2.39%** |
| SSL-AASIST (pretrained, Tak et al.) | **0.00%** |

Per-attack EER on the same subset (about 28 files per attack, so these numbers are noisy):

| Attack | AASIST | SSL-AASIST | Attack | AASIST | SSL-AASIST |
| ------ | ------ | ---------- | ------ | ------ | ---------- |
| A07 | 0.00% | 0.00% | A14 | 0.00% | 0.00% |
| A08 | 0.00% | 0.00% | A15 | 2.75% | 0.00% |
| A09 | 0.00% | 0.00% | A16 | 0.00% | 0.00% |
| A10 | 2.95% | 0.00% | A17 | 3.13% | 0.00% |
| A11 | 0.00% | 0.00% | A18 | 8.58% | 0.00% |
| A12 | 3.01% | 0.00% | A19 | 0.00% | 0.00% |
| A13 | 0.00% | 0.00% |  |  |  |

Raw outputs are in [`results/`](results/). AASIST was first run on a laptop CPU and reproduced with the same numbers on a Colab GPU; SSL-AASIST was run on a Colab T4 GPU (about 33 s for the 400 files).

**Limitations**

- The subset has only 39 bonafide clips, so one misranked clip moves the EER by about 2.6 points. Read the table as "SSL-AASIST separated all 400 files, AASIST misranked a few", not as a final benchmark.
- ASVspoof 2019 contains older spoofing systems recorded as clean wideband audio. Performance on newer generators (modern TTS and voice-cloning tools) and on phone-call audio (8 kHz, codecs, noise) has **not** been tested yet and is likely to be lower for both models.
- Both models are used as pretrained; neither has been fine-tuned on this project's data.
- The EER figures above evaluate only the anti-spoofing models. The speaker check, the fusion rules, the challenge-response step and the risk thresholds are not part of this evaluation, and the thresholds are a starting point that still needs calibration.

## Project structure

```
.
├── app.py                 # Streamlit dashboard (upload, verdict, voice library, challenge, report)
├── pages/                 # extra dashboard pages: Live monitor, Call log
├── pipeline.py            # windows, scores, risk fusion (fuse_risk), voice-swap alert
├── shared.py              # run_checks, split_call, audio conversion to WAV
├── spoof.py               # AASIST model loading and inference
├── spoof_ssl.py           # SSL-AASIST model loading and inference
├── voice_store.py         # saved voice library (ECAPA-TDNN embeddings)
├── challenge.py           # random phrase, speech-to-text, challenge decision
├── call_log.py            # call log with caller numbers
├── report_panel.py        # complaint draft
├── eval.py                # EER evaluation on ASVspoof 2019 LA (--model aasist | ssl)
├── make_subset.py         # extract a small random subset from LA.zip
├── make_long_demo.py      # build a demo clip: real segments then fake segments
├── requirements.txt       # app + AASIST environment
├── requirements-ssl.txt   # SSL-AASIST environment (Python 3.9)
├── aasist/                # AASIST code and pretrained weights (clovaai/aasist)
├── ssl_aasist/            # SSL-AASIST model code (TakHemlata/SSL_Anti-spoofing), weights NOT included
├── results/               # raw evaluation outputs for both models
└── colab/                 # notebook that reproduces the AASIST vs SSL-AASIST comparison
```

The voice library (`voice_store/`), call log (`calls.db`) and downloaded speaker models (`ecapa_model/`) are created locally and are excluded from the repository.

## Environments

The two models need different Python environments, because SSL-AASIST depends on an old version of fairseq. Keep them in separate virtual environments.

| Purpose | Python | File |
| ------- | ------ | ---- |
| Streamlit app + AASIST | 3.10+ | `requirements.txt` |
| SSL-AASIST scoring / evaluation | 3.9 | `requirements-ssl.txt` + fairseq at commit `a54021305d6b3c4c5959ac9395135f63202db8f1` |

`python eval.py --model aasist` works in the main environment; `python eval.py --model ssl` works only in the SSL environment.

## Setup: app and AASIST (Windows / PowerShell)

```
git clone https://github.com/bhavana-sreenivasan/AI-voice-cloning-and-impersonation-detection-features.git
cd AI-voice-cloning-and-impersonation-detection-features
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The AASIST code and weights must be in `aasist/` (`aasist/models/weights/AASIST.pth`, `aasist/config/AASIST.conf`).

## Setup: SSL-AASIST

The SSL-AASIST weights are large (about 1.2 GB each) and are **not** in this repository. A GPU is strongly recommended; the Colab notebook in `colab/` does all of this automatically.

1. Create a **Python 3.9** virtual environment and install the pinned build tools first:
   ```
   pip install "pip<24.1" "setuptools<66" wheel
   pip install numpy==1.23.5 cython
   pip install -r requirements-ssl.txt
   ```
2. Install fairseq at the commit used by the original SSL-AASIST repo (download the archive for commit `a54021305d6b3c4c5959ac9395135f63202db8f1` from `github.com/facebookresearch/fairseq`, then inside the folder run):
   ```
   pip install --no-build-isolation --editable ./
   ```
3. Download into `ssl_aasist/`:
   - `xlsr2_300m.pt`, the XLS-R 300M base model: <https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt>
   - the pretrained SSL-AASIST weights (`LA_model.pth`): link in the README of the original repo, <https://github.com/TakHemlata/SSL_Anti-spoofing>

## Run the dashboard

```
streamlit run app.py
```

Upload an audio file (WAV, FLAC, MP3, M4A, AAC or OGG); it is converted to WAV automatically (compressed formats need [ffmpeg](https://ffmpeg.org/) installed and on your PATH). Clips shorter than 0.5 s are rejected. The dashboard currently uses AASIST.

## Reproduce the evaluation

1. Download **ASVspoof 2019 LA** from the University of Edinburgh DataShare (<https://datashare.ed.ac.uk/handle/10283/3336>). The dataset is **not** included in this repository.
2. Extract a random subset without unzipping everything:
   ```
   python make_subset.py --zip path\to\LA.zip --out data\LA --n 400
   ```
3. Evaluate:
   ```
   python eval.py --root data\LA --n 400 --model aasist
   python eval.py --root data\LA --n 400 --model ssl
   ```
   Both runs use the same seed, so they score the same files.

## Demo: voice-swap detection

```
python make_long_demo.py
```

This builds `long_demo.wav` (3 real clips followed by 3 fake clips). Upload it to the dashboard: the risk line stays low for the real part and rises when the synthetic part begins, triggering the swap alert.

## Roadmap

Done:

- [x] AASIST scoring in the dashboard (windowed risk levels, voice-swap alert)
- [x] Speaker identity check against a saved voice library (ECAPA-TDNN)
- [x] Fused risk level (spoof check + identity check)
- [x] Challenge-response verification for uncertain results
- [x] Call log with caller numbers and complaint draft (user-confirmed, not auto-submitted)
- [x] Offline evaluation of AASIST and SSL-AASIST on an ASVspoof 2019 LA subset

Planned but **not yet implemented**:

- [ ] Use SSL-AASIST in the dashboard / pipeline
- [ ] Replace ECAPA-TDNN with ECAPA2 for speaker verification
- [ ] Remote-speaker isolation: enrol the user's own voice so the app automatically ignores it and analyses only the other party
- [ ] Robustness testing on phone-call quality audio (G.711 / AMR / Opus codecs, noise) for both models
- [ ] Real-time streaming analysis of a live call with live alerts
- [ ] Explainability (spectrogram / Grad-CAM highlighting of suspicious regions)
- [ ] Cross-generator testing on unseen TTS / cloning tools (e.g. In-the-Wild)
- [ ] Human voice imitation (mimicry) detection
- [ ] Fine-tuning with phone-style augmentation

## Privacy and ethics

- Only use voice recordings with the speaker's consent.
- Do not clone or analyse voices of people outside your team without permission.
- Risk levels are decision support, not proof. A "Fake" result can be wrong, so do not use it alone to accuse anyone.
- Saved voices are biometric data and phone numbers are personal data. They are stored locally, are not part of this repository, and voices can only be saved with an explicit consent confirmation. Saved voices can be deleted from the dashboard sidebar.
- Storage of voiceprints and phone numbers is not yet encrypted; encrypt or hash them before any real deployment.

## Acknowledgements and licenses

- **AASIST** (clovaai/aasist): pretrained anti-spoofing model and code. See `aasist/LICENSE` and `aasist/NOTICE`.
- **SSL-AASIST** (TakHemlata/SSL_Anti-spoofing): model code and pretrained weights. See `ssl_aasist/LICENSE`.
- **SpeechBrain**: ECAPA-TDNN speaker embeddings. **faster-whisper**: speech-to-text for the challenge step.
- **XLS-R** and **fairseq** (Meta AI): the wav2vec 2.0 XLS-R front end, downloaded separately and not redistributed here.
- **ASVspoof 2019 LA** dataset (University of Edinburgh): used for evaluation only and not redistributed. Follow the license on the dataset page.

### References

- J.-w. Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022.
- H. Tak et al., "Automatic Speaker Verification Spoofing and Deepfake Detection using wav2vec 2.0 and Data Augmentation," Odyssey 2022.
- B. Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," Interspeech 2020.
- A. Babu et al., "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale," Interspeech 2022.
- X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," Computer Speech & Language, 2020.
- M. Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech 2019.
