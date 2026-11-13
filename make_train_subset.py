import os, sys, random, shutil, glob, zipfile
root, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
zip_path = sys.argv[4] if len(sys.argv) > 4 else None

protos = glob.glob(os.path.join(root, "**", "*cm.train.trn.txt"), recursive=True)
if protos:
    proto = protos[0]
elif zip_path:
    z = zipfile.ZipFile(zip_path)
    name = next(m for m in z.namelist() if m.endswith("ASVspoof2019.LA.cm.train.trn.txt"))
    os.makedirs(out, exist_ok=True)
    proto = os.path.join(out, os.path.basename(name))
    with open(proto, "wb") as f:
        f.write(z.read(name))
else:
    sys.exit("No train protocol on disk. Run again with the LA.zip path as a 4th argument.")
print("protocol:", proto)

flac = {os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(root, "**", "LA_T_*.flac"), recursive=True)}

by = {"bonafide": [], "spoof": []}
for line in open(proto):
    f = line.split()
    if len(f) >= 5 and f[4] in by and f[1] in flac:
        by[f[4]].append(f[1])
print({k: len(v) for k, v in by.items()}, "available on disk")

k = min(n, len(by["bonafide"]), len(by["spoof"]))
os.makedirs(os.path.join(out, "flac"), exist_ok=True)
if os.path.dirname(os.path.abspath(proto)) != os.path.abspath(out):
    shutil.copy(proto, out)
rng = random.Random(0)
for utts in by.values():
    for u in rng.sample(utts, k):
        shutil.copy(flac[u], os.path.join(out, "flac", u + ".flac"))
print(f"done: {k} clips per class in {out}")