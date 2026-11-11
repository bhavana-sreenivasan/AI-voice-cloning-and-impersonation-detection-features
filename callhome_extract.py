"""callhome_extract.py - turn CallHome calls into WAV files with ground-truth speakers.

Input (either):
  A) the Hugging Face parquet (talkbank/callhome): one row per call with columns
       audio (bytes, 16 kHz), timestamps_start, timestamps_end, speakers
  B) one audio file (wav / flac / mp3 ...) + its .cha transcript (CHAT format; the turn times are the
     bullets  <0x15>start_end<0x15>  in milliseconds)

Output, per call, in --out:
  callNNN_mix.wav        the whole call as it is (mono, 16 kHz)   <- what the detector would hear
  callNNN_user.wav       the 'user' speaker alone (talk that does NOT overlap the other side), up to
                         --enroll-seconds: use it as --me-file / the saved voice
  callNNN_opponent.wav   the OTHER speaker alone (non-overlapping): the ground-truth 'opponent' voice
  callNNN_segments.csv   start_s, end_s, speaker
and a printed summary (duration, speakers, seconds each, channels).

Examples (PowerShell, project folder):
  python callhome_extract.py --parquet "C:\\...\\eng\\data-00000-of-00005.parquet" --out callhome_out --max-calls 5
  python callhome_extract.py --audio call4065.mp3 --cha call4065.cha --out callhome_out

Then, for call 001:
  python me_similarity.py callhome_out\\call001_mix.wav --me-file callhome_out\\call001_user.wav
  python live_cli.py --file callhome_out\\call001_mix.wav --me-file callhome_out\\call001_user.wav --no-speaker --fast
Needs: numpy, soundfile, scipy; pyarrow for --parquet; ffmpeg on PATH for mp3 and other non-wav/flac audio.
"""
import argparse
import csv
import io
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SR = 16000
MIN_SEG_S = 0.5          # shortest stretch used for the user / opponent files


# ----------------------------------------------------------------------------- audio decoding
def to_16k(x, sr):
    if sr == SR:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), SR)
    return resample_poly(x, SR // g, int(sr) // g, axis=0).astype(np.float32)


def decode_bytes(data, name_hint=".wav"):
    """-> float32 array (n, channels) at 16 kHz. soundfile first (wav/flac/ogg), ffmpeg as a fallback."""
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return to_16k(x, sr)
    except Exception:                                    # noqa: BLE001
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "in" + name_hint), os.path.join(d, "out.wav")
            open(src, "wb").write(data)
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ar", str(SR), dst], capture_output=True)
            if r.returncode or not os.path.isfile(dst):
                raise RuntimeError("Could not decode the audio (is ffmpeg on PATH?): " + r.stderr.decode(errors="ignore")[:200])
            x, _ = sf.read(dst, dtype="float32", always_2d=True)
            return x


# ----------------------------------------------------------------------------- .cha parsing
_BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")


def parse_cha(path):
    """-> (starts_s, ends_s, speakers) from the speaker tiers of a CHAT file."""
    text = open(path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    # an utterance = a line starting with '*' plus the continuation lines that start with a tab
    utts, cur = [], None
    for line in text.split("\n"):
        if line.startswith("*"):
            if cur:
                utts.append(cur)
            cur = line
        elif line.startswith("\t") and cur is not None:
            cur += " " + line.strip()
        else:
            if cur:
                utts.append(cur)
            cur = None
    if cur:
        utts.append(cur)
    starts, ends, spk = [], [], []
    for u in utts:
        m = re.match(r"\*([A-Za-z0-9_]+):", u)
        b = _BULLET.findall(u)
        if m and b:
            starts.append(int(b[0][0]) / 1000.0)
            ends.append(int(b[-1][1]) / 1000.0)
            spk.append(m.group(1))
    return starts, ends, spk


# ----------------------------------------------------------------------------- per call outputs
def write_wav(path, x):
    sf.write(path, np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")


def process_call(idx, audio, starts, ends, spk, out_dir, user, enroll_s):
    """audio: (n, ch) float32 at 16 kHz."""
    ch = audio.shape[1]
    mix = audio.mean(axis=1) if ch > 1 else audio[:, 0]
    n = len(mix)
    dur = n / SR
    names = sorted(set(spk))
    base = os.path.join(out_dir, f"call{idx:03d}")
    write_wav(base + "_mix.wav", mix)

    with open(base + "_segments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s", "speaker"])
        for s, e, k in sorted(zip(starts, ends, spk)):
            w.writerow([f"{s:.3f}", f"{e:.3f}", k])

    masks = {k: np.zeros(n, dtype=bool) for k in names}
    for s, e, k in zip(starts, ends, spk):
        masks[k][max(0, int(s * SR)):min(n, int(e * SR))] = True
    last_end = max(ends) if ends else 0.0
    secs = {k: masks[k].sum() / SR for k in names}
    note = ""
    if last_end > dur + 0.5:
        note = f"  !! last timestamp {last_end:.1f}s is beyond the audio ({dur:.1f}s): times may not match this audio"
    print(f"call{idx:03d}: {dur:6.1f}s, {ch} channel(s), speakers "
          + ", ".join(f"{k}={secs[k]:.0f}s" for k in names) + note)

    if len(names) < 2:
        print("   (only one labelled speaker: no user/opponent files written)")
        return
    me = user if user in names else names[0]
    other = [k for k in names if k != me]
    other_mask = np.zeros(n, dtype=bool)
    for k in other:
        other_mask |= masks[k]

    def alone(mask, limit_s=None):
        """non-overlapping talk of one side, glued together; keeps stretches >= MIN_SEG_S."""
        m = mask.copy()
        edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
        parts, total = [], 0.0
        for a, b in zip(edges[::2], edges[1::2]):
            if (b - a) / SR >= MIN_SEG_S:
                parts.append(mix[a:b])
                total += (b - a) / SR
                if limit_s and total >= limit_s:
                    break
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    u = alone(masks[me] & ~other_mask, enroll_s)
    o = alone(other_mask & ~masks[me])
    write_wav(base + "_user.wav", u)
    write_wav(base + "_opponent.wav", o)
    print(f"         user='{me}' -> {len(u) / SR:.0f}s alone-talk (call{idx:03d}_user.wav);  "
          f"opponent={'+'.join(other)} -> {len(o) / SR:.0f}s (call{idx:03d}_opponent.wav)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", help="Hugging Face parquet file(s) of talkbank/callhome")
    ap.add_argument("--audio", help="audio file of ONE call (with --cha)")
    ap.add_argument("--cha", help="the .cha transcript of that call")
    ap.add_argument("--out", default="callhome_out")
    ap.add_argument("--user", default="A", help="speaker label to treat as 'the user' (default A)")
    ap.add_argument("--enroll-seconds", type=float, default=25.0)
    ap.add_argument("--max-calls", type=int, default=5, help="stop after this many calls (parquet files are large)")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N calls")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.parquet:
        import pyarrow.parquet as pq
        idx = done = 0
        for path in args.parquet:
            pf = pq.ParquetFile(path)
            cols = [c for c in ("audio", "timestamps_start", "timestamps_end", "speakers") if c in pf.schema_arrow.names]
            if len(cols) < 4:
                sys.exit(f"{path}: expected columns audio, timestamps_start, timestamps_end, speakers; found {pf.schema_arrow.names}")
            for batch in pf.iter_batches(batch_size=1, columns=cols):
                row = batch.to_pylist()[0]
                idx += 1
                if idx <= args.skip:
                    continue
                a = row["audio"]
                data = a["bytes"] if isinstance(a, dict) else a
                ext = os.path.splitext((a.get("path") or "") if isinstance(a, dict) else "")[1] or ".wav"
                process_call(idx, decode_bytes(data, ext), list(row["timestamps_start"]),
                             list(row["timestamps_end"]), list(row["speakers"]), args.out, args.user, args.enroll_seconds)
                done += 1
                if done >= args.max_calls:
                    break
            if done >= args.max_calls:
                break
    elif args.audio and args.cha:
        data = open(args.audio, "rb").read()
        s, e, k = parse_cha(args.cha)
        if not s:
            sys.exit("No timed turns found in the .cha (expected bullets like \\x15 1234_5678 \\x15).")
        process_call(1, decode_bytes(data, os.path.splitext(args.audio)[1] or ".wav"), s, e, k,
                     args.out, args.user, args.enroll_seconds)
    else:
        ap.error("give --parquet FILE [FILE ...]   or   --audio FILE --cha FILE")
    print(f"\nFiles are in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()