"""analyze(audio) -> Real/Fake label, confidence, risk level, per-window timeline."""
import numpy as np
import torch
import librosa

from spoof import load_model, _pad, SR, MAX_LEN

torch.set_num_threads(2)  # keep CPU/memory use modest on small cloud machines

# Risk thresholds on spoof probability. Tune these using your eval scores.
LOW_T = 0.30   # below this -> Low risk
HIGH_T = 0.70  # above this -> High risk; in between -> Uncertain

HOP = SR           # 1 second hop between 4 s windows
BATCH = 1          # one window at a time: low peak memory on Streamlit Cloud
MAX_SECONDS = 60   # cap very long uploads so one file can't crash the app


def risk_level(p_spoof):
    if p_spoof < LOW_T:
        return "Low"
    if p_spoof > HIGH_T:
        return "High"
    return "Uncertain"


def _windows(wav):
    if len(wav) <= MAX_LEN:
        return [_pad(wav)], [0.0]
    starts = list(range(0, len(wav) - MAX_LEN + 1, HOP))
    if starts[-1] + MAX_LEN < len(wav):
        starts.append(len(wav) - MAX_LEN)
    return [wav[s:s + MAX_LEN] for s in starts], [s / SR for s in starts]


@torch.no_grad()
def analyze(path, model, device):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    if len(wav) < SR // 2:
        raise ValueError("Audio too short (need at least 0.5 seconds).")
    if len(wav) > MAX_SECONDS * SR:
        wav = wav[:MAX_SECONDS * SR]

    wins, starts = _windows(wav)
    probs = []
    for i in range(0, len(wins), BATCH):
        batch = torch.tensor(np.stack(wins[i:i + BATCH]), dtype=torch.float32).to(device)
        _, out = model(batch)
        probs.extend(torch.softmax(out, dim=1)[:, 0].cpu().numpy().tolist())

    p_spoof = float(np.mean(probs))
    label = "Fake" if p_spoof >= 0.5 else "Real"
    alert_time = None
    hot = [p > HIGH_T for p in probs]
    for i in range(len(hot) - 2):
        if all(hot[i:i + 3]):  # 3 suspicious windows in a row
            alert_time = starts[i]
            break
    return {
        "label": label,
        "confidence": max(p_spoof, 1 - p_spoof),
        "p_spoof": p_spoof,
        "risk": risk_level(p_spoof),
        "duration": len(wav) / SR,
        "timeline": [(round(s, 1), float(p)) for s, p in zip(starts, probs)],
        "alert_time": alert_time,
    }


if __name__ == "__main__":
    import sys
    m, d = load_model()
    print(analyze(sys.argv[1], m, d))
