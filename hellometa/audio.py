"""Soundtrack: cleaned interview voice + synthesised futuristic electronic score + SFX, mixed.

  python3 audio.py            -> build/voice.wav, build/music.wav, build/sfx.wav, build/mix.wav
"""
import numpy as np, subprocess, json, os
from scipy import signal
import timeline as T

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
B = lambda *p: os.path.join(HERE, 'build', *p)
RAW = os.path.join(HERE, 'input', 'Raw video.mp4')
N = int(T.DUR * SR)
rng = np.random.default_rng(3)


def load(path, sr=SR):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '2', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def save(path, x):
    x = np.clip(x, -1, 1).astype(np.float32)
    p = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                        '-c:a', 'pcm_s24le', path], input=x.tobytes(), check=True)


# ------------------------------------------------------------------ voice
def build_voice():
    a = load(RAW)
    XF = int(T.XF * SR)
    out = None
    for i, (s, e, _) in enumerate(T.EDL):
        tail = 0.35 if i == len(T.EDL) - 1 else 0  # let the last "Thank you!" ring out
        c = a[int(s * SR) - XF // 2:int((e + tail) * SR) + XF // 2].copy()
        if tail:
            c[-int(0.30 * SR):] *= np.linspace(1, 0, int(0.30 * SR))[:, None]
        if out is None:
            c[:XF] *= np.linspace(0, 1, XF)[:, None]
            out = c
        else:
            r = np.linspace(0, 1, XF)[:, None]
            out[-XF:] = out[-XF:] * np.sqrt(1 - r) + c[:XF] * np.sqrt(r)  # equal-power crossfade
            out = np.vstack([out, c[XF:]])
    tmp = B('voice_edit_raw.wav'); save(tmp, out)
    # enhancement: rumble cut, broadband denoise, de-mud, presence, gentle compression, loudness
    chain = ('highpass=f=85,afftdn=nr=4:nf=-34:tn=1,equalizer=f=280:t=q:w=1.2:g=-2.5,'
             'equalizer=f=3200:t=q:w=1.0:g=3,'
             'acompressor=threshold=-22dB:ratio=3:attack=8:release=120:makeup=3,'
             'loudnorm=I=-16:LRA=6:TP=-1.5')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-af', chain, '-ar', str(SR), B('voice_clean.wav')],
                   check=True)
    v = load(B('voice_clean.wav'))
    v = v.mean(1, keepdims=True).repeat(2, 1)  # centred mono voice
    full = np.zeros((N, 2), np.float32)
    o = int(T.SPEECH_OFF * SR)
    full[o:o + len(v)] = v[:N - o]
    return full


# ------------------------------------------------------------------ synth helpers
def t_(d):
    return np.arange(int(d * SR)) / SR


def adsr(n, a=0.005, d=0.1, s=0.6, r=0.1):
    env = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n); env[:na] = np.linspace(0, 1, na)
    nd = min(nd, max(n - na, 0)); env[na:na + nd] = np.linspace(1, s, nd)
    nr = min(nr, n); env[n - nr:] *= np.linspace(1, 0, nr)
    return env


def saw(f, tt, detune=0.0, phase=0.0):
    return 2 * ((tt * f * (1 + detune) + phase) % 1) - 1


def lp(x, fc, order=2):
    b, a = signal.butter(order, min(fc, SR * 0.45) / (SR / 2), 'low')
    return signal.lfilter(b, a, x, axis=0)


def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high')
    return signal.lfilter(b, a, x, axis=0)


def bp(x, lo, hi, order=2):
    b, a = signal.butter(order, [lo / (SR / 2), min(hi, SR * 0.45) / (SR / 2)], 'band')
    return signal.lfilter(b, a, x, axis=0)


def sweep_lp(x, f0, f1):
    """time-varying one-pole lowpass (exponential cutoff sweep), processed in blocks"""
    x = np.asarray(x, np.float64)
    out = np.zeros_like(x); n = len(x); blk = 256
    yprev = np.zeros((1,) + x.shape[1:])
    for i in range(0, n, blk):
        fc = f0 * (f1 / f0) ** (i / max(n - 1, 1))
        k = 1 - np.exp(-2 * np.pi * fc / SR)
        y, _ = signal.lfilter([k], [1, -(1 - k)], x[i:i + blk], axis=0, zi=yprev * (1 - k))
        out[i:i + blk] = y
        yprev = y[-1:]
    # two passes = 12 dB/oct
    return out


def note(n):
    return 440 * 2 ** ((n - 69) / 12)


def place(buf, x, t0, gain=1.0, pan=0.0):
    i = int(round(t0 * SR))
    if i >= len(buf):
        return
    if x.ndim == 1:
        x = np.stack([x * np.sqrt(0.5 * (1 - pan)), x * np.sqrt(0.5 * (1 + pan))], 1) * np.sqrt(2)
    j = min(len(buf), i + len(x))
    if i < 0:
        x = x[-i:]; i = 0
    buf[i:j] += x[:j - i] * gain


def reverb_ir(dur=2.2, pre=0.01):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-tt / (dur / 6.9))[:, None]
    ir = lp(ir, 7000)
    ir[:int(pre * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(0))


def convolve(x, ir):
    return np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)


# ------------------------------------------------------------------ instruments
def kick(d=0.45, punch=1.0):
    tt = t_(d)
    f = 45 + 110 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-tt * 7) * punch
    x += 0.4 * np.exp(-tt * 300) * rng.standard_normal(len(tt)) * 0.3
    return np.tanh(x * 1.6) * 0.9


def clap(d=0.3):
    tt = t_(d)
    nz = rng.standard_normal(len(tt))
    env = np.exp(-tt * 22)
    for o in (0.0, 0.012, 0.024):
        env += np.exp(-np.clip(tt - o, 0, None) * 140) * (tt >= o) * 0.6
    return bp(nz * env, 900, 6000) * 0.8


def hat(d=0.06, open_=False):
    tt = t_(0.35 if open_ else d)
    x = hp(rng.standard_normal(len(tt)), 7500) * np.exp(-tt * (9 if open_ else 70))
    return x * 0.5


def pluck(f, d=0.25, bright=4000):
    tt = t_(d)
    x = saw(f, tt) * 0.6 + saw(f, tt, 0.004) * 0.4
    x = sweep_lp(x[:, None], bright, 300)[:, 0]
    return x * np.exp(-tt * 9)


def supersaw(freqs, d, cutoff=3500, voices=7):
    tt = t_(d)
    x = np.zeros((len(tt), 2))
    for f in freqs:
        for v in range(voices):
            det = (v - voices // 2) * 0.0045
            ph = rng.random()
            s = saw(f, tt, det, ph)
            pan = (v / (voices - 1)) * 2 - 1
            x[:, 0] += s * (1 - pan) * 0.5; x[:, 1] += s * (1 + pan) * 0.5
    x = lp(x, cutoff)
    return x / (len(freqs) * voices) * 2.2


def sub(f, d):
    tt = t_(d)
    return (np.sin(2 * np.pi * f * tt) + 0.25 * np.tanh(3 * np.sin(2 * np.pi * f * tt))) * adsr(len(tt), 0.01, 0.1, 0.9, 0.05)


def riser(d, f0=200, f1=4000):
    tt = t_(d)
    nz = rng.standard_normal((len(tt), 2))
    x = sweep_lp(nz, f0, f1) * (tt / d)[:, None] ** 2
    f = f0 * 0.5 * (f1 / f0) ** (tt / d) * 0.1
    ph = 2 * np.pi * np.cumsum(f + 60) / SR
    x += (np.sin(ph) * (tt / d) ** 2 * 0.25)[:, None]
    return x * 0.6


def whoosh(d=0.6, f0=300, f1=5000, rev=False):
    tt = t_(d)
    nz = rng.standard_normal((len(tt), 2))
    env = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 2
    x = sweep_lp(nz, f0, f1) * env[:, None]
    x = hp(x, 150)
    if rev:
        x = x[::-1]
    # stereo motion L->R
    pan = np.linspace(-0.7, 0.7, len(tt))
    x[:, 0] *= np.sqrt(0.5 * (1 - pan)); x[:, 1] *= np.sqrt(0.5 * (1 + pan))
    return x * 1.2


def boom(d=2.5):
    tt = t_(d)
    f = 32 + 60 * np.exp(-tt * 6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.6)
    x += lp(rng.standard_normal(len(tt)), 400) * np.exp(-tt * 5) * 0.6
    return np.tanh(x * 1.4)


def ping(f=1760, d=1.2):
    tt = t_(d)
    x = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 2.01 * tt)) * np.exp(-tt * 5)
    return x * 0.5


def glitch(d=0.12):
    tt = t_(d)
    x = np.sign(np.sin(2 * np.pi * 1200 * tt)) * (rng.random(len(tt)) > 0.5)
    return bp(x, 800, 6000) * np.exp(-tt * 30) * 0.4


# ------------------------------------------------------------------ score
CHORDS = [  # A minor: Am, F, C, G  (MIDI notes), one per bar
    (57, [57, 60, 64, 69]), (53, [53, 57, 60, 65]), (48, [55, 60, 64, 67]), (55, [55, 59, 62, 67])]


def build_music():
    mus = np.zeros((N, 2)); drums = np.zeros((N, 2)); pads = np.zeros((N, 2)); fx = np.zeros((N, 2))
    nbars = int(T.DUR / 2) + 1
    beat = T.BEAT

    # --- HOOK (0 - 2.4): impact, fast arp opening, riser into the speech
    place(fx, boom(2.0), 0.0, 0.9)
    place(drums, kick(0.6, 1.3), 0.0, 1.0)
    for k in range(int(2.4 / (beat / 4))):
        t0 = k * beat / 4
        root, ch = CHORDS[0]
        f = note(ch[[0, 2, 3, 1][k % 4]] + 12)
        place(mus, pluck(f, 0.2, 1500 + 4000 * t0 / 2.4), t0, 0.22, pan=0.4 * np.sin(k))
    place(fx, riser(1.4, 300, 6000), 1.0, 0.45)
    for t0 in (0.5, 1.0, 1.5, 2.0):
        place(drums, kick(0.3, 0.8), t0, 0.55)

    # --- INTERVIEW BED (2.4 - 38.4): half-time groove, pads, plucks; ducked later under the voice
    for b in range(1, 20):
        t0 = b * 2.0
        if t0 + 2 > 38.4 + 0.01:
            break
        root, ch = CHORDS[b % 4]
        place(pads, supersaw([note(n) for n in ch], 2.05, cutoff=1800 + 600 * (b % 2)) * adsr(int(2.05 * SR), 0.25, 0.3, 0.8, 0.4)[:, None], t0, 0.30)
        place(mus, sub(note(root - 24), 1.9), t0, 0.32)
        for k in range(8):  # 8th note plucks
            if b < 4 and k % 2:
                continue
            f = note(ch[[0, 2, 1, 3, 2, 0, 3, 1][k]] + 12)
            place(mus, pluck(f, 0.18, 2600), t0 + k * beat / 2, 0.12, pan=0.5 * (-1) ** k)
        for k in range(4):
            if k in (0, 2):
                place(drums, kick(0.4, 0.9), t0 + k * beat, 0.55)
            if b >= 6 and k in (1, 3):
                place(drums, clap(), t0 + k * beat, 0.22)
            for h in range(2):
                place(drums, hat(), t0 + k * beat + h * beat / 2 + 0.25 * beat * (h == 1), 0.10 + 0.05 * h, pan=0.3)

    # --- BUILD (38.4 - 40.0): snare roll + riser
    place(fx, riser(1.6, 200, 9000), 38.4, 0.55)
    tt = 38.4
    k = 0
    while tt < 40.0:
        place(drums, clap(0.12), tt, 0.10 + 0.25 * (tt - 38.4) / 1.6)
        tt += beat / (2 if tt < 39.2 else 4)

    # --- DROP / WEBSITE MONTAGE (40.0 - 47.5)
    for b in range(20, 24):
        t0 = b * 2.0
        dur = min(2.0, T.HILL[0] - t0)
        if dur <= 0:
            break
        root, ch = CHORDS[b % 4]
        for k in range(4):
            tb = t0 + k * beat
            if tb >= T.HILL[0]:
                break
            place(drums, kick(0.45, 1.1), tb, 0.85)
            if k in (1, 3):
                place(drums, clap(), tb, 0.38)
            place(mus, sub(note(root - 24), beat / 2 - 0.02), tb + beat / 2, 0.42)  # offbeat bass
            for h in range(4):
                place(drums, hat(open_=(h == 2)), tb + h * beat / 4, 0.12 if h != 2 else 0.10, pan=0.25)
        stab = supersaw([note(n) for n in ch], dur, cutoff=5000) * adsr(int(dur * SR), 0.01, 0.4, 0.55, 0.2)[:, None]
        place(pads, stab, t0, 0.42)
        for k in range(int(dur / (beat / 4))):  # 16th lead arp
            f = note(ch[[0, 1, 2, 3, 2, 1][k % 6]] + 24)
            place(mus, pluck(f, 0.14, 5000), t0 + k * beat / 4, 0.10, pan=0.6 * np.sin(k * 1.3))
    for t0 in (T.CALC[0], T.HOW[0], T.HOW[0] + 1.0, T.HOW[0] + 2.0):
        place(fx, boom(1.0), t0, 0.35)

    # --- HILLTOP (47.5 - 52.0): breakdown, wide pad, riser, boom on the logo
    root, ch = CHORDS[0]
    wide = supersaw([note(n) for n in ch] + [note(ch[0] + 12)], 4.6, cutoff=2600)
    wide *= adsr(len(wide), 0.6, 0.5, 0.9, 1.2)[:, None]
    place(pads, wide, T.HILL[0], 0.5)
    place(mus, sub(note(root - 24), 4.4), T.HILL[0], 0.35)
    place(fx, riser(T.HILL_HIT - T.HILL[0] - 0.3, 150, 8000), T.HILL[0] + 0.3, 0.5)
    place(fx, boom(3.0), T.HILL_HIT, 1.0)
    place(drums, kick(0.8, 1.4), T.HILL_HIT, 0.9)
    for i, n in enumerate([81, 84, 88, 93]):
        place(fx, ping(note(n), 2.0), T.HILL_HIT + 0.05 + i * 0.09, 0.12, pan=-0.4 + i * 0.27)

    # --- END (52.0 - 55.5): clean bell chord and tail
    for i, n in enumerate([69, 72, 76, 81]):
        place(fx, ping(note(n), 3.2), T.END[0] + 0.02 * i, 0.16, pan=-0.3 + 0.2 * i)
    place(pads, supersaw([note(n) for n in [57, 64, 69, 72]], 3.4, cutoff=1500) * adsr(int(3.4 * SR), 0.05, 0.5, 0.6, 2.5)[:, None], T.END[0], 0.3)
    place(fx, boom(2.0), T.END[0], 0.4)

    ir = reverb_ir(2.4)
    pads_w = pads + convolve(pads, ir) * 0.35
    mus_w = mus + convolve(mus, ir) * 0.25
    fx_w = fx + convolve(fx, ir) * 0.3
    # sidechain pump from the kick onto pads/bass (simple envelope follower)
    kenv = np.abs(drums[:, 0])
    kenv = signal.lfilter([1 - np.exp(-1 / (0.06 * SR))], [1, -np.exp(-1 / (0.06 * SR))], kenv)
    pump = 1 - np.clip(kenv * 1.2, 0, 0.55)
    music = (pads_w + mus_w) * pump[:, None] + drums + fx_w
    music = hp(music, 28)
    return music


# ------------------------------------------------------------------ SFX tied to picture
def build_sfx():
    s = np.zeros((N, 2))
    for t0 in T.transitions():
        if T.HILL[0] - 0.01 < t0 < T.END[1]:
            continue  # the hilltop/end have their own score hits
        place(s, whoosh(0.55, 400, 6000), t0 - 0.30, 0.35)
    # map: travel whoosh and two location markers
    m0, m1 = T.BROLL[0][1], T.BROLL[0][2]
    place(s, ping(1320, 0.9), m0 + 0.55, 0.18)              # Philippines marker
    place(s, whoosh(1.4, 200, 3000), m0 + 0.9, 0.3)          # route travel
    place(s, ping(1760, 0.9), m0 + 2.35, 0.2)               # Sydney marker
    # typography impacts for the kinetic words
    for _, a, b in T.KINETIC:
        place(s, boom(0.8), a, 0.35)
        place(s, glitch(), a, 0.4)
    # small tick for each How-It-Works step
    for i in range(3):
        place(s, glitch(0.08), T.HOW[0] + i * 1.0, 0.35)
    # logo shimmer on the white end slide
    place(s, whoosh(1.2, 2000, 12000, rev=True), T.END[0] - 1.2 + 0.05, 0.25)
    return s


def duck_env(voice):
    """music gain from the voice envelope: -14 dB under speech, smooth attack/release"""
    env = np.abs(voice[:, 0])
    a = np.exp(-1 / (0.25 * SR))
    env = signal.lfilter([1 - a], [1, -a], env)
    act = np.clip(env / 0.02, 0, 1)
    g = 1 - act * (1 - 10 ** (-14 / 20))
    a2 = np.exp(-1 / (0.15 * SR))
    g = signal.lfilter([1 - a2], [1, -a2], g)
    return g


def lufs(x):
    p = subprocess.run(['ffmpeg', '-hide_banner', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                        '-af', 'ebur128', '-f', 'null', '-'], input=np.clip(x, -1, 1).astype(np.float32).tobytes(),
                       capture_output=True)
    for line in p.stderr.decode().splitlines()[::-1]:
        if line.strip().startswith('I:'):
            return float(line.split()[1])


if __name__ == '__main__':
    os.makedirs(B(), exist_ok=True)
    voice = build_voice(); print('voice', lufs(voice))
    music = build_music(); print('music raw', lufs(music))
    sfx = build_sfx()
    # music bed level: speech sections sit under the voice, outro comes up (but not a jump)
    tt = np.arange(N) / SR
    bed = np.where(tt < T.SPEECH_OFF, 1.15, np.where(tt < 38.4, 0.55, 0.74))
    bed = signal.lfilter([1 - np.exp(-1 / (0.3 * SR))], [1, -np.exp(-1 / (0.3 * SR))], bed)
    music *= 10 ** ((-16 - lufs(music)) / 20)
    g = duck_env(voice) * bed
    sfx_g = np.where(np.abs(voice[:, 0]) > 0, 1, 1) * g / bed  # SFX follow the ducking fully
    bg = music * g[:, None] + sfx * (0.25 + 0.75 * sfx_g[:, None]) * 0.7
    mix = voice + bg
    for a, b in [(3, 6), (8, 9.5), (10, 14), (20, 24), (30, 34), (36, 38)]:
        i, j = int(a * SR), int(b * SR)
        print(f"{a:>4}-{b:<4} voice-to-background margin {20*np.log10(np.std(voice[i:j]) / np.std(bg[i:j])):5.1f} dB")
    for a, b in [(0, 2.4), (5, 38), (40, 47.5), (47.5, 52), (52, 55)]:
        print(f"{a:>4}-{b:<4} LUFS {lufs(mix[int(a*SR):int(b*SR)]):6.1f}")
    # final fade, static gain to -14 LUFS, true-peak limiter (no dynamic loudness riding)
    fo = int(1.2 * SR); mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
    mix *= 10 ** ((-14 - lufs(mix)) / 20)
    save(B('voice.wav'), voice); save(B('music.wav'), music); save(B('sfx.wav'), sfx)
    save(B('mix_pre.wav'), mix * 0.5)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', B('mix_pre.wav'), '-af',
                    'volume=6.0206dB,alimiter=limit=0.891:attack=2:release=80:level=false',
                    '-ar', str(SR), '-c:a', 'pcm_s24le', B('mix.wav')], check=True)
    print('mix', lufs(load(B('mix.wav'))))
