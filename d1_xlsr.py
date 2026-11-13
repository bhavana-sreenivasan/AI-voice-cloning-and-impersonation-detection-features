import os, requests
u = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
f = "xlsr2_300m.pt"
while True:
    s = os.path.getsize(f) if os.path.exists(f) else 0
    try:
        r = requests.get(u, headers={"Range": f"bytes={s}-"}, stream=True, timeout=60)
        print("status", r.status_code, "from MB", s // 2**20, flush=True)
        if r.status_code == 416:
            print("already complete"); break
        d = open(f, "ab" if r.status_code == 206 else "wb")
        n = 0
        for c in r.iter_content(1 << 20):
            d.write(c); n += 1
            if n % 50 == 0:
                print("MB now", os.path.getsize(f) // 2**20, flush=True)
        d.close()
        break
    except Exception as e:
        print("interrupted, retrying:", e, flush=True)
print("done MB", os.path.getsize(f) // 2**20)