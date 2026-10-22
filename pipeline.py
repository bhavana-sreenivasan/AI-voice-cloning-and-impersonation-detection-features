"""analyze(audio) -> Real/Fake label, confidence, risk level, per-window timeline."""
import numpy as np
import torch
import librosa

from spoof import load_model, _pad, SR, MAX_LEN

# Risk thresholds on spoof probability. Tune these using your eval scores.
LOW_T = 0.30   # below this -> Low risk
HIGH_T = 0.70  # above this -> High risk; in between -> Uncertain

HOP = SR  # 1 second hop between 4 s windows


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

    wins, starts = _windows(wav)
    probs = []
    
    for i in range(0, len(wins), 1):
        batch = torch.tensor(
            np.stack(wins[i:i + 1]),
            dtype=torch.float32
        ).to(device)
    
        _, out = model(batch)
        probs.extend(
            torch.softmax(out, dim=1)[:, 0].cpu().numpy().tolist()
        )

        

    p_spoof = float(np.mean(probs))
    alert_time = None
    hot = [p > HIGH_T for p in probs]
    for i in range(len(hot) - 2):
        if all(hot[i:i + 3]):  # 3 suspicious windows in a row
            alert_time = starts[i]
            break

    # A clip that is half real / half fake averages to ~0.5, which hides the
    # fake part. So a sustained run of suspicious windows overrides the mean.
    frac_hot = sum(hot) / len(hot)
    if alert_time is not None:
        label = "Fake" if frac_hot >= 0.8 else "Partially fake"
        hot_probs = [p for p in probs if p > HIGH_T]
        confidence = float(np.mean(hot_probs))
        risk = "High"
    else:
        label = "Fake" if p_spoof >= 0.5 else "Real"
        confidence = max(p_spoof, 1 - p_spoof)
        risk = risk_level(p_spoof)
    return {
        "label": label,
        "confidence": confidence,
        "p_spoof": p_spoof,
        "frac_suspicious": frac_hot,
        "risk": risk,
        "duration": len(wav) / SR,
        "timeline": [(round(s, 1), float(p)) for s, p in zip(starts, probs)],
        "alert_time": alert_time,
    }


def fuse_risk(spoof_risk, identity=None, speaker_change=False):
    """Combine the spoof-detector risk with the speaker-identity decision.

    spoof_risk : "Low" | "Uncertain" | "High"   (from analyze())
    identity   : None | "match" | "uncertain" | "mismatch"  (from speaker.py)
    Returns (risk, reason).
    """
    if spoof_risk == "High":
        return "High", "Synthetic speech detected."
    if identity == "mismatch":
        return "High", ("Voice sounds genuine but does not match the claimed "
                        "speaker (possible human impersonation or wrong person).")
    if speaker_change:
        return "High", "The speaker changed partway through the clip."
    if spoof_risk == "Uncertain" or identity == "uncertain":
        return "Uncertain", "Result is borderline: run an additional verification step."
    if identity == "match":
        return "Low", "Voice looks genuine and matches the claimed speaker."
    return "Low", "Voice looks genuine (speaker identity not checked)."


if __name__ == "__main__":
    import sys
    m, d = load_model()
    print(analyze(sys.argv[1], m, d))
