import pathlib
NEW_CLASS = '''class UserVoiceFilter:
    """Removes the LOCAL user's own voice from a mixed stream, keeping only the other party.
    Audio is cut into `slice_s` slices; each slice is embedded and compared with the user's
    enrolled voiceprint. Slices that sound like the user (cosine >= drop_above) are dropped.
    replace=False (old behaviour): dropped slices are deleted, so the kept audio is stitched
        together. Both detectors score stitched, pause-free audio as fake (silence_experiment.py).
    replace=True: a dropped slice is replaced by faint noise of the same length, so the timing
        and pauses of the other party's speech survive. Put a PauseLimiter AFTER this filter so
        a long user turn does not fill the scoring window with silence.
    hysteresis: once dropping, keep dropping until the similarity falls below
        drop_above - hysteresis (stops a turn from flickering on and off).
    silence_db: slices quieter than this (mean power, dB) are passed through untouched without
        being embedded (None = embed everything, the old behaviour).
    embed_fn(slice) -> 1-D vector;  user_ref -> the user's enrolled embedding (same model!).
    """
    def __init__(self, embed_fn, user_ref, drop_above, slice_s=1.0, sr=SR,
                 replace=False, hysteresis=0.0, silence_db=None, seed=0):
        r = np.asarray(user_ref, dtype=np.float64)
        self.embed_fn, self.ref = embed_fn, r / (np.linalg.norm(r) + 1e-12)
        self.drop_above, self.n = drop_above, int(slice_s * sr)
        self.replace, self.hyst, self.silence_db = replace, hysteresis, silence_db
        self._rng = np.random.default_rng(seed)
        self._floor_db = -62.0
        self._dropping = False
        self._rest = np.zeros(0, dtype=np.float32)
        self.kept_samples = self.dropped_samples = 0
        self.last_sim = None
    @property
    def dropped_s(self):
        return self.dropped_samples / SR
    def _quiet(self, piece):
        amp = 10.0 ** (self._floor_db / 20.0)
        return (amp * self._rng.standard_normal(len(piece))).astype(np.float32)
    def process(self, samples):
        x = np.concatenate([self._rest, np.asarray(samples, dtype=np.float32).ravel()])
        out = []
        while len(x) >= self.n:
            piece, x = x[:self.n], x[self.n:]
            if self.silence_db is not None:
                db = 10.0 * np.log10(float(np.mean(piece * piece)) + 1e-10)
                if db < self.silence_db:          # a pause: nothing to identify, keep it as it is
                    self._floor_db = float(np.clip(0.9 * self._floor_db + 0.1 * db, -75.0, -50.0))
                    self.kept_samples += len(piece)
                    out.append(piece)
                    continue
            e = np.asarray(self.embed_fn(piece), dtype=np.float64)
            self.last_sim = float(self.ref @ (e / (np.linalg.norm(e) + 1e-12)))
            limit = self.drop_above - (self.hyst if self._dropping else 0.0)
            self._dropping = self.last_sim >= limit
            if self._dropping:
                self.dropped_samples += len(piece)
                if self.replace:
                    out.append(self._quiet(piece))
            else:
                self.kept_samples += len(piece)
                out.append(piece)
        self._rest = x
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
'''
NEW_BLOCK = '''    removing_user = bool(args.me_file or args.me)
    user_filter = None
    if removing_user:                            # remove the local user's own voice
        import speaker
        import torch
        import voice_store
        if args.me_file:
            from capture import load_audio_16k
            wav = load_audio_16k(args.me_file)[: 30 * 16000]      # up to 30 s is plenty
            ref = speaker.embed(torch.tensor(wav)).numpy()
        else:
            ref = speaker.enroll(voice_store.clips(args.me)).numpy()
        t = speaker.MATCH_THRESHOLD if args.drop_above is None else args.drop_above
        keep_timing = args.gate != "vad"
        user_filter = UserVoiceFilter(lambda w: speaker.embed(w).numpy(), ref, drop_above=t,
                                      replace=keep_timing, hysteresis=0.05 if keep_timing else 0.0,
                                      silence_db=-50.0 if keep_timing else None)
        print(f"Ignoring the local user's voice (similarity >= {t:.2f} is dropped; "
              + ("replaced by silence, timing kept)." if keep_timing else "deleted, audio is stitched)."), flush=True)
    uf = [user_filter] if user_filter else []
    if args.no_vad or args.gate == "none":
        gates = uf
    elif args.gate == "vad":
        gates = [EnergyVAD()] + uf
    else:                                  # pause: your turns become short pauses, the opponent keeps his
        gates = uf + [PauseLimiter()]
'''
def load(p):
    raw = pathlib.Path(p).read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw
def save(p, text, crlf):
    bak = pathlib.Path(p + ".bak_userfilter")
    if not bak.exists():
        bak.write_bytes(pathlib.Path(p).read_bytes())
    pathlib.Path(p).write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
s, crlf = load("live.py")
if "replace=False, hysteresis" in s:
    print("live.py: already patched")
else:
    a, b = s.find("class UserVoiceFilter:"), s.find("class GateChain:")
    if a < 0 or b < a:
        print("live.py: UserVoiceFilter / GateChain not found, nothing changed")
    else:
        save("live.py", s[:a] + NEW_CLASS.rstrip("\n") + "\n\n\n" + s[b:], crlf)
        print("live.py: patched")
s, crlf = load("live_cli.py")
if "user_filter" in s:
    print("live_cli.py: gate block already patched")
else:
    a = s.find("    removing_user = bool(args.me_file or args.me)\n")
    b = s.find("    if args.normalize:\n")
    if a < 0 or b < a:
        print("live_cli.py: block not found, nothing changed")
    else:
        s = s[:a] + NEW_BLOCK + s[b:]
        if "min_s=" not in s:
            s = s.replace("hop_s=args.hop, smooth=args.smooth)", "hop_s=args.hop, smooth=args.smooth, min_s=4.0)", 1)
        save("live_cli.py", s, crlf)
        print("live_cli.py: patched")
