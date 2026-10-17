# 🎙️ AI-Powered Real-Time Voice Cloning & Impersonation Detection

A prototype that detects AI-generated (cloned) voices and flags suspicious segments in an audio clip, shown on a simple Streamlit dashboard.

> **Status:** early prototype (~40% of the planned system). The current version works on uploaded audio files. Live call streaming and several planned features are not implemented yet (see [Roadmap](#roadmap)).

---

## What it does (current version)

- Upload a `.wav` or `.flac` clip.
- A pretrained **AASIST** anti-spoofing model scores the audio using overlapping 4-second windows (1-second hop).
- The dashboard shows:
  - **Result:** Real / Fake
  - **Confidence**
  - **Risk level:** Low / Uncertain / High
  - **Risk-over-time chart** of the spoof probability
  - **Voice-swap alert** when 3 consecutive windows look synthetic (an early version of continuous verification)

## Results

Evaluation of the pretrained AASIST model on a **random 400-file subset** of the ASVspoof 2019 LA evaluation set (seed 42; 39 bonafide, 361 spoof), run on CPU.

| Metric | Value |
|---|---|
| Overall EER | **2.39%** |

Per-attack EER on the same subset (about 28 files per attack, so these numbers are noisy):

| Attack | EER | Attack | EER |
|---|---|---|---|
| A07 | 0.00% | A14 | 0.00% |
| A08 | 0.00% | A15 | 2.75% |
| A09 | 0.00% | A16 | 0.00% |
| A10 | 2.95% | A17 | 3.13% |
| A11 | 0.00% | A18 | 8.58% |
| A12 | 3.01% | A19 | 0.00% |
| A13 | 0.00% | | |

**Limitations**
- The small subset has few bonafide clips, so these figures are estimates, not final benchmark results.
- ASVspoof 2019 contains older spoofing systems. Performance on newer generators (e.g. modern TTS and voice-cloning tools) and on phone-call audio has **not** been tested yet and is likely to be lower.

## Project structure

```
.
├── app.py              # Streamlit dashboard
├── pipeline.py         # analyze(): windows, scores, risk level, swap alert
├── spoof.py            # AASIST model loading and inference
├── eval.py             # EER evaluation on ASVspoof 2019 LA
├── make_subset.py      # extract a small random subset from LA.zip
├── make_long_demo.py   # build a demo clip: real segments then fake segments
├── requirements.txt
└── aasist/             # AASIST code and pretrained weights (clovaai/aasist)
```

## Setup (Windows / PowerShell)

```powershell
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The AASIST code and weights must be in `aasist/` (`aasist/models/weights/AASIST.pth`, `aasist/config/AASIST.conf`).

## Run the dashboard

```powershell
streamlit run app.py
```

Upload a `.wav` or `.flac` file. Clips shorter than 0.5 s are rejected.

## Reproduce the evaluation

1. Download **ASVspoof 2019 LA** from the University of Edinburgh DataShare (https://datashare.ed.ac.uk/handle/10283/3336). The dataset is **not** included in this repository.
2. Extract a random subset without unzipping everything:
   ```powershell
   python make_subset.py --zip path\to\LA.zip --out data\LA --n 400
   ```
3. Evaluate:
   ```powershell
   python eval.py --root data\LA --n 400
   ```

## Demo: voice-swap detection

```powershell
python make_long_demo.py
```
This builds `long_demo.wav` (3 real clips followed by 3 fake clips). Upload it to the dashboard: the risk line stays low for the real part and rises when the synthetic part begins, triggering the swap alert.

## Roadmap

Planned but **not yet implemented**:

- [ ] Speaker identity verification (ECAPA-TDNN) to check the claimed speaker
- [ ] Liveness challenge (random phrase) for uncertain cases
- [ ] Explainability (spectrogram / Grad-CAM highlighting of suspicious regions)
- [ ] Real-time streaming analysis with live alerts
- [ ] Robustness testing on phone-call quality audio (codecs, noise)
- [ ] Cross-generator testing on unseen TTS / cloning tools (e.g. In-the-Wild)
- [ ] Human voice imitation (mimicry) detection
- [ ] Fine-tuning on additional data
- [ ] Caller number logging and complaint report generation (user-confirmed, not auto-submitted)

## Privacy and ethics

- Only use voice recordings with the speaker's consent.
- Do not clone or analyse voices of people outside your team without permission.
- Risk levels are decision support, not proof. A "Fake" result can be wrong, so do not use it alone to accuse anyone.
- Any future storage of voiceprints or phone numbers should be encrypted or hashed and require user consent.

## Acknowledgements and licenses

- **AASIST** (clovaai/aasist): pretrained anti-spoofing model and code. See `aasist/LICENSE` and `aasist/NOTICE`.
- **ASVspoof 2019 LA** dataset (University of Edinburgh): used for evaluation only and not redistributed. Follow the license on the dataset page.

### References

- J.-w. Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022.
- X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," Computer Speech & Language, 2020.
- M. Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech 2019.



