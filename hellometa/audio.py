"""Soundtrack: interview voice (neural denoise) + synthesised electronic score, mixed. No sound effects.

  DF_PYTHON=... DF_MODEL=... python3 audio.py   -> build/voice.wav, build/music.wav, build/mix.wav
"""
import numpy as np, subprocess, json, os, sys
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
    # 1) neural noise reduction (DeepFilterNet3) removes the expo-hall crowd. Capped at 20 dB of
    #    attenuation, which keeps every word; around "whole-day" a 12 dB pass is blended in because the
    #    stronger pass blurs that word into "half-day".
    df = {}
    for lim in ('20', '12'):
        p = B(f'voice_df_{lim}.wav')
        if not os.path.exists(p):
            py = os.environ.get('DF_PYTHON'); model = os.environ.get('DF_MODEL')
            if not (py and model):
                sys.exit('set DF_PYTHON (python with deepfilternet) and DF_MODEL (DeepFilterNet3 dir)')
            subprocess.run([py, os.path.join(HERE, 'tools_deepfilter.py'), tmp, p, lim, model], check=True)
        df[lim] = load(p)[:, 0]
    v = df['20'].copy()
    words = json.load(open(os.path.join(HERE, 'data', 'subs_words.json')))
    for w in words:
        if w['w'].startswith('whole-day'):
            a0, a1 = int((w['s'] - 0.25) * SR), int((w['e'] + 0.15) * SR)
            env = np.zeros(len(v)); env[a0:a1] = 1
            env = np.convolve(env, np.ones(int(0.06 * SR)) / int(0.06 * SR), 'same')
            v = v * (1 - env) + df['12'][:len(v)] * env
    save(B('voice_df.wav'), np.stack([v, v], 1))
    # 2) polish: rumble cut, de-mud, presence, gentle compression, loudness
    chain = ('highpass=f=80,equalizer=f=280:t=q:w=1.2:g=-2,'
             'equalizer=f=3200:t=q:w=1.0:g=2.5,'
             'acompressor=threshold=-22dB:ratio=2.5:attack=8:release=120:makeup=2,'
             'loudnorm=I=-16:LRA=6:TP=-1.5')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', B('voice_df.wav'), '-af', chain, '-ar', str(SR),
                    B('voice_clean.wav')], check=True)
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
    """composed on the original 120 BPM grid (bar lines every 2 s from grid 0), then trimmed by
    MUSIC_SHIFT so bar lines still fall on the picture's outro cuts. No impacts, pings or noise risers."""
    SH = T.MUSIC_SHIFT
    NM = N + int(SH * SR)
    HILL0, HILL_HIT, END0, OUT = T.HILL[0] + SH, T.HILL_HIT + SH, T.END[0] + SH, T.OUTRO + SH
    mus = np.zeros((NM, 2)); drums = np.zeros((NM, 2)); pads = np.zeros((NM, 2))
    beat = T.BEAT

    # --- INTERVIEW BED (2.4 - 38.4): half-time groove, pads, plucks; ducked later under the voice
    for b in range(1, 20):
        t0 = b * 2.0
        if t0 + 2 > OUT - 1.6 + 0.01:
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
    tt = OUT - 1.6
    k = 0
    while tt < OUT:
        place(drums, clap(0.12), tt, 0.08 + 0.18 * (tt - (OUT - 1.6)) / 1.6)
        tt += beat / (2 if tt < OUT - 0.8 else 4)

    # --- DROP / WEBSITE MONTAGE (grid 40.0 - 47.5)
    for b in range(int(OUT / 2), int(HILL0 / 2) + 1):
        t0 = b * 2.0
        dur = min(2.0, HILL0 - t0)
        if dur <= 0:
            break
        root, ch = CHORDS[b % 4]
        for k in range(4):
            tb = t0 + k * beat
            if tb >= HILL0:
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

    # --- HILLTOP: breakdown, wide pad; a single soft kick when the logo lands
    root, ch = CHORDS[0]
    wide = supersaw([note(n) for n in ch] + [note(ch[0] + 12)], 4.6, cutoff=2600)
    wide *= adsr(len(wide), 0.6, 0.5, 0.9, 1.2)[:, None]
    place(pads, wide, HILL0, 0.5)
    place(mus, sub(note(root - 24), 4.4), HILL0, 0.35)
    place(drums, kick(0.8, 1.0), HILL_HIT, 0.6)

    # --- END: warm closing chord with a long tail
    place(pads, supersaw([note(n) for n in [57, 64, 69, 72]], 3.6, cutoff=1500) * adsr(int(3.6 * SR), 0.08, 0.5, 0.6, 2.5)[:, None], END0, 0.34)
    place(mus, sub(note(45 - 12), 3.0), END0, 0.25)

    ir = reverb_ir(2.4)
    pads_w = pads + convolve(pads, ir) * 0.35
    mus_w = mus + convolve(mus, ir) * 0.25
    # sidechain pump from the kick onto pads/bass (simple envelope follower)
    kenv = np.abs(drums[:, 0])
    kenv = signal.lfilter([1 - np.exp(-1 / (0.06 * SR))], [1, -np.exp(-1 / (0.06 * SR))], kenv)
    pump = 1 - np.clip(kenv * 1.2, 0, 0.55)
    music = (pads_w + mus_w) * pump[:, None] + drums
    music = hp(music, 28)
    music = music[int(SH * SR):int(SH * SR) + N]  # trim to the picture
    fi = int(0.6 * SR); music[:fi] *= np.linspace(0, 1, fi)[:, None]
    return music


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
    # music bed level: speech sections sit under the voice, outro comes up (but not a jump)
    tt = np.arange(N) / SR
    bed = np.where(tt < T.OUTRO - 1.6, 0.55, 0.74)
    bed = signal.lfilter([1 - np.exp(-1 / (0.3 * SR))], [1, -np.exp(-1 / (0.3 * SR))], bed)
    music *= 10 ** ((-16 - lufs(music)) / 20)
    g = duck_env(voice) * bed
    bg = music * g[:, None]
    mix = voice + bg
    for a, b in [(1, 4), (6, 7.5), (8, 12), (18, 22), (28, 32), (33, 36)]:
        i, j = int(a * SR), int(b * SR)
        print(f"{a:>4}-{b:<4} voice-to-background margin {20*np.log10(np.std(voice[i:j]) / np.std(bg[i:j])):5.1f} dB")
    for a, b in [(0, 36), (T.OUTRO, T.HILL[0]), T.HILL, T.END]:
        print(f"{a:>4}-{b:<4} LUFS {lufs(mix[int(a*SR):int(b*SR)]):6.1f}")
    # final fade, static gain to -14 LUFS, true-peak limiter (no dynamic loudness riding)
    fo = int(1.2 * SR); mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 2
    mix *= 10 ** ((-14 - lufs(mix)) / 20)
    save(B('voice.wav'), voice); save(B('music.wav'), music)
    save(B('mix_pre.wav'), mix * 0.5)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', B('mix_pre.wav'), '-af',
                    'volume=6.0206dB,alimiter=limit=0.891:attack=2:release=80:level=false',
                    '-ar', str(SR), '-c:a', 'pcm_s24le', B('mix.wav')], check=True)
    print('mix', lufs(load(B('mix.wav'))))
