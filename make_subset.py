"""Extract only a small random subset of ASVspoof 2019 LA eval from LA.zip.

Usage (PowerShell):
    python make_subset.py --zip C:\\path\\to\\LA.zip --out ..\\data\\LA --n 400

Then run:
    python eval.py --root ..\\data\\LA --n 400
"""
import argparse
import os
import random
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="../data/LA")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    names = z.namelist()

    proto_name = next(n for n in names if n.endswith("ASVspoof2019.LA.cm.eval.trl.txt"))
    flac_index = {os.path.basename(n): n for n in names
                  if "ASVspoof2019_LA_eval" in n and n.endswith(".flac")}

    lines = [l for l in z.read(proto_name).decode().splitlines() if l.strip()]
    random.Random(args.seed).shuffle(lines)
    lines = [l for l in lines if (l.split()[1] + ".flac") in flac_index][: args.n]

    proto_dir = os.path.join(args.out, "ASVspoof2019_LA_cm_protocols")
    audio_dir = os.path.join(args.out, "ASVspoof2019_LA_eval", "flac")
    os.makedirs(proto_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    with open(os.path.join(proto_dir, "ASVspoof2019.LA.cm.eval.trl.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    for l in lines:
        fname = l.split()[1] + ".flac"
        with open(os.path.join(audio_dir, fname), "wb") as out:
            out.write(z.read(flac_index[fname]))

    print(f"Done: {len(lines)} files extracted to {args.out}")


if __name__ == "__main__":
    main()