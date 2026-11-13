"""How long does the speaker model take?  (run from the project folder)

    python time_speaker.py                  # CPU, audio lengths 3 s, 30 s, 120 s
    python time_speaker.py --gpu            # use CUDA (needs a GPU runtime)
    python time_speaker.py --lengths 3 10   # only these lengths (seconds); skip the slow ones
    python time_speaker.py --nojit --lengths 1.5 1.5 3 3
        # --nojit turns off TorchScript's profiling / fusion before the first call (the usual cure for a
        # first call that takes minutes).  Repeated lengths show whether only the FIRST call of each
        # length is slow (shape-dependent warm-up) or every call is.

A line is printed BEFORE each step, so if it seems stuck you can see which step it is in:
  import        loading torch / speaker
  model load    downloading + loading ECAPA2 (first run only downloads about 109 MB)
  first embed   the first call can be much slower than later ones (TorchScript warm-up)
  N s of audio  time for each requested length, measured after the first call
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true", help="set SPEAKER_DEVICE=cuda before importing speaker")
ap.add_argument("--lengths", type=float, nargs="+", default=[3, 30, 120])
ap.add_argument("--nojit", action="store_true", help="disable TorchScript profiling executor / fusers first")
args = ap.parse_args()
if args.gpu:
    os.environ["SPEAKER_DEVICE"] = "cuda"


def say(msg):
    print(msg, flush=True)


say("import ...")
t = time.perf_counter()
import torch   # noqa: E402
import speaker   # noqa: E402
say(f"import: {time.perf_counter() - t:.1f}s   device={speaker.DEVICE}   "
    f"cuda available={torch.cuda.is_available()}   backend={speaker.BACKEND}   threads={torch.get_num_threads()}")

if args.nojit:
    for fn, val in (("_jit_set_profiling_executor", False), ("_jit_set_profiling_mode", False),
                    ("_jit_override_can_fuse_on_cpu", False), ("_jit_set_texpr_fuser_enabled", False)):
        try:
            getattr(torch._C, fn)(val)
            say(f"torch._C.{fn}({val})")
        except AttributeError:
            say(f"(this torch has no {fn})")
say(f"torch {torch.__version__}")
say("model load ... (first run downloads ~109 MB)")
t = time.perf_counter()
speaker._get_model()
say(f"model load: {time.perf_counter() - t:.1f}s")

say("first embed (1.5 s of audio) ... this one includes any warm-up")
t = time.perf_counter()
speaker.embed(torch.randn(int(1.5 * 16000)))
say(f"first embed: {time.perf_counter() - t:.1f}s")

for sec in args.lengths:
    say(f"{sec:g} s of audio ...")
    x = torch.randn(int(sec * 16000))
    t = time.perf_counter()
    speaker.embed(x)
    say(f"{sec:>6g} s of audio: {time.perf_counter() - t:.2f}s")
say("done")