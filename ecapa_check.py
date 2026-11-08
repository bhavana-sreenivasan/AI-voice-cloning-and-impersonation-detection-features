import time, torch
from huggingface_hub import hf_hub_download
m = torch.jit.load(hf_hub_download("Jenthe/ECAPA2", "ecapa2.pt"), map_location="cpu")
x = torch.randn(1, 32000)
for _ in range(3):
    t = time.time(); m(x); print(round(time.time() - t, 2), "s")