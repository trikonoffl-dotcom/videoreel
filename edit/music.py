"""Synthesise an original festive (Diwali-style) soundtrack for the reel.

100 BPM, D tonic. Sections are aligned to the video edit:
  0.0  - 4.8   intro: temple bell, tanpura drone, pad, santoor arpeggio
  4.8  - 9.6   build: light percussion
  9.6  - 26.4  groove A (Diya painting)
  26.4 - 45.6  groove B (Creative challenge)
  45.6 - 48.0  build-up / drum roll
  48.0 - 55.2  drop (finale)
  55.2 - 61.2  outro
Usage: python3 music.py out.wav
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
BPM = 100
BEAT = 60 / BPM            # 0.6 s
BAR = BEAT * 4             # 2.4 s
TOTAL = 61.2
N = int(SR * (TOTAL + 2))
rng = np.random.default_rng(7)

mix = {k: np.zeros(N) for k in ["drone", "pad", "mel", "perc", "fx", "bell"]}


def tt(d):
    return np.arange(int(d * SR)) / SR


def add(bus, t0, sig, gain=1.0):
    i = int(t0 * SR)
    if i >= N:
        return
    j = min(N, i + len(sig))
    mix[bus][i:j] += sig[: j - i] * gain


def hz(n, base=146.83):  # n semitones from D3
    return base * 2 ** (n / 12)


def lp(x, fc, order=2):
    return sosfilt(butter(order, fc, "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


# ---------------------------------------------------------------- instruments
def pluck(f, dur, bright=0.5, decay=0.996):
    """Karplus-Strong, vectorised per delay-line period."""
    L = max(2, int(SR / f))
    n = int(dur * SR)
    buf = rng.uniform(-1, 1, L)
    buf = lp(buf, 1500 + 6000 * bright, 1) if L > 8 else buf
    out = np.empty(n + L)
    out[:L] = buf
    pos = L
    while pos < n + L:
        seg = out[pos - L: pos]
        prev = np.concatenate(([out[pos - L - 1] if pos - L - 1 >= 0 else 0.0], seg[:-1]))
        newseg = decay * 0.5 * (seg + prev)
        k = min(L, n + L - pos)
        out[pos: pos + k] = newseg[:k]
        pos += k
    y = out[L:]
    y *= np.minimum(1, tt(dur)[: len(y)] * 400)
    return y / (np.abs(y).max() + 1e-9)


def santoor(f, dur=1.6):
    y = pluck(f, dur, 0.8, 0.9975) + 0.45 * pluck(f * 2.003, dur, 0.9, 0.995)
    return y * np.exp(-tt(dur)[: len(y)] * 2.2)


def bell(f, dur=4.0):
    t = tt(dur)
    parts = [(1, 1, 1.2), (2.0, 0.6, 1.8), (2.76, 0.5, 2.4), (5.4, 0.25, 3.5), (8.93, 0.12, 5)]
    y = sum(a * np.sin(2 * np.pi * f * r * t + rng.uniform(0, 6)) * np.exp(-t * d) for r, a, d in parts)
    y *= np.minimum(1, t * 800)
    return y / 2.5


def dhol_bass(dur=0.45, f0=125, f1=58):
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 9) + 0.25 * np.sin(2 * ph) * np.exp(-t * 18)
    click = lp(rng.normal(0, 1, len(t)), 3000) * np.exp(-t * 120) * 0.4
    return np.tanh(1.6 * (y + click))


def dhol_slap(dur=0.18):
    t = tt(dur)
    n = bp(rng.normal(0, 1, len(t)), 900, 5000) * np.exp(-t * 38)
    tone = np.sin(2 * np.pi * 420 * t) * np.exp(-t * 30)
    return 0.7 * n / (np.abs(n).max() + 1e-9) + 0.5 * tone


def tabla_na(dur=0.35, f=720):
    t = tt(dur)
    y = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.02 * t) + 0.3 * np.sin(2 * np.pi * f * 3.1 * t))
    y *= np.exp(-t * 14)
    y += bp(rng.normal(0, 1, len(t)), 2000, 8000) * np.exp(-t * 90) * 0.3
    return y / 1.8


def shaker(dur=0.06):
    t = tt(dur)
    y = hp(rng.normal(0, 1, len(t)), 6000) * np.exp(-t * 60) * np.minimum(1, t * 2000)
    return y / (np.abs(y).max() + 1e-9)


def ghungroo(dur=0.25):
    t = tt(dur)
    y = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in rng.uniform(4500, 9000, 8))
    y *= np.exp(-t * 18) * (1 + 0.5 * np.sin(2 * np.pi * 38 * t))
    return y / 8


def pad_chord(notes, dur):
    t = tt(dur)
    y = np.zeros(len(t))
    for n in notes:
        for det in (-0.08, 0.0, 0.07):
            f = hz(n) * 2 ** (det / 12)
            y += 2 * ((t * f + rng.uniform()) % 1) - 1
    y = lp(y, 1400, 2)
    att, rel = 0.5, 0.6
    env = np.minimum(1, t / att) * np.minimum(1, (dur - t) / rel)
    return y * env / (len(notes) * 3)


def tanpura(n, dur=2.2):
    y = pluck(hz(n - 12), dur, 0.95, 0.9992)
    t = tt(dur)[: len(y)]
    return y * np.exp(-t * 0.8) * (1 + 0.15 * np.sin(2 * np.pi * 5.5 * t))


def noise_sweep(dur, f0, f1, up=True):
    t = tt(dur)
    noise = rng.normal(0, 1, len(t))
    out = np.zeros(len(t))
    blk = 1024
    for i in range(0, len(t), blk):
        fr = i / len(t)
        fc = f0 * (f1 / f0) ** fr
        seg = bp(noise[max(0, i - 2048): i + blk], fc * 0.7, min(fc * 1.4, SR / 2 - 100))
        out[i: i + blk] = seg[-len(out[i: i + blk]):]
    env = (t / dur) ** 2 if up else np.sin(np.pi * t / dur)
    return out * env / (np.abs(out).max() + 1e-9)


def boom(dur=2.5):
    t = tt(dur)
    f = 35 + 60 * np.exp(-t * 6)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.5)
    cr = hp(rng.normal(0, 1, len(t)), 4000) * np.exp(-t * 2.2) * 0.35
    return np.tanh(2 * y) + cr


def chime_run(t0, notes, step=0.07, gain=0.5):
    for k, n in enumerate(notes):
        add("fx", t0 + k * step, bell(hz(n, 587.33), 1.5), gain * 0.5)


# ---------------------------------------------------------------- arrangement
chords = [[0, 4, 7], [-2, 2, 5], [-5, 0, 4], [-3, 2, 5]]  # D, Bm/D-ish, G, A (voiced near D3)
chords = [[0, 4, 7], [-3, 0, 4], [-7, -3, 0], [-5, -1, 2]]  # D  Bm  G  A

# Tanpura drone throughout (P S S S cycle)
t = 0.0
cyc = [7 - 12, 0, 0, -12]
k = 0
while t < TOTAL - 1:
    add("drone", t, tanpura(cyc[k % 4] + 12), 0.22)
    t += BEAT
    k += 1

# Pads
for b in range(int(TOTAL / BAR) + 1):
    t0 = b * BAR
    g = 0.30 if (t0 < 4.8 or t0 >= 55.2) else 0.17
    add("pad", t0, pad_chord(chords[b % 4], BAR + 0.6), g)

# Melody (santoor). Values are semitones from D4 (one octave above D3); None = rest.
phraseA = [[12, None, 11, 9, 7, 9, 11, 12], [14, 12, 11, 9, 7, None, 9, None],
           [7, 9, 7, 4, 5, 4, 2, 0], [2, 4, 7, None, 4, 2, 0, None]]
phraseB = [[7, None, 9, 11, 12, 11, 9, 7], [9, 7, 4, 7, 9, None, None, None],
           [12, 14, 16, 14, 12, 11, 9, 11], [12, None, 7, None, 12, None, None, None]]
intro_arp = [[0, 4, 7, 12, 7, 4, 7, 12], [-3, 0, 4, 9, 4, 0, 4, 9]]


def play_bars(t0, bars, gain, octave=12):
    for bi, bar in enumerate(bars):
        for si, n in enumerate(bar):
            if n is None:
                continue
            add("mel", t0 + bi * BAR + si * BEAT / 2, santoor(hz(n + octave)), gain)


play_bars(0.0, intro_arp, 0.30)
play_bars(4.8, intro_arp, 0.28)
for i, t0 in enumerate(np.arange(9.6, 26.39, 4 * BAR)):
    play_bars(t0, phraseA, 0.34)
for t0 in np.arange(26.4, 45.59, 4 * BAR):
    play_bars(t0, phraseB, 0.34)
play_bars(48.0, phraseA[:2], 0.40)
play_bars(48.0 + 2 * BAR, phraseB[2:], 0.40)
play_bars(55.2, [[12, 7, 4, 0, 4, 7, 12, None], [7, 4, 0, None, None, None, None, None]], 0.30)

# Percussion
def groove(t0, t1, density=1.0, gain=1.0):
    # 16th-step bhangra-ish chaal
    bass = {0, 3, 6, 8, 11, 14}
    slap = {4, 12}
    na = {2, 7, 10, 15}
    step = BEAT / 4
    t = t0
    s = 0
    while t < t1 - 1e-6:
        p = s % 16
        if p in bass:
            add("perc", t, dhol_bass(), 0.55 * gain)
        if p in slap:
            add("perc", t, dhol_slap(), 0.45 * gain)
        if density > 0.6 and p in na:
            add("perc", t, tabla_na(), 0.22 * gain)
        if density > 0.3:
            add("perc", t, shaker(), (0.10 if p % 2 else 0.05) * gain)
        if p == 0 and density > 0.6:
            add("perc", t, ghungroo(), 0.18 * gain)
        t += step
        s += 1


groove(4.8, 9.6, density=0.4, gain=0.6)
groove(9.6, 45.6, density=1.0)
groove(48.0, 55.2, density=1.0, gain=1.1)

# Build-up roll 45.6-48.0
steps = int((48.0 - 45.6) / (BEAT / 4))
for i in range(steps):
    tt0 = 45.6 + i * BEAT / 4
    g = 0.2 + 0.7 * i / steps
    add("perc", tt0, dhol_slap(), 0.35 * g)
    if i % 2 == 0:
        add("perc", tt0, dhol_bass(0.3, 140, 70), 0.4 * g)
add("fx", 45.6, noise_sweep(2.4, 300, 9000, up=True), 0.22)

# Bells & impacts
add("bell", 0.05, bell(293.66), 0.55)
add("bell", 0.05, bell(587.33), 0.25)
add("bell", 48.0, bell(293.66), 0.5)
add("fx", 48.0, boom(), 0.55)
add("bell", 55.2, bell(293.66), 0.45)
add("bell", 55.2, bell(440.0), 0.25)

# Whooshes for the rangoli transitions & flashes
for tw in (4.8, 26.4, 55.2):
    add("fx", tw - 0.6, noise_sweep(1.1, 400, 7000, up=False), 0.28)
for tf in (9.6, 19.8, 36.0, 45.6):
    add("fx", tf - 0.25, noise_sweep(0.5, 1500, 9000, up=False), 0.12)

# Sparkle chimes on titles
chime_run(1.0, [0, 4, 7, 12, 16, 19, 24], gain=0.35)
chime_run(9.7, [7, 12, 16, 19], gain=0.3)
chime_run(26.6, [7, 12, 16, 19], gain=0.3)
chime_run(55.6, [0, 4, 7, 12, 16, 19, 24], gain=0.35)

# ---------------------------------------------------------------- mixdown
def reverb(x, secs=1.8, wet=0.18):
    ir_t = tt(secs)
    ir = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t * 3.5)
    ir = lp(ir, 5000)
    ir /= np.sqrt((ir ** 2).sum())
    return x + wet * fftconvolve(x, ir)[: len(x)]


levels = {"drone": 0.55, "pad": 0.6, "mel": 0.75, "perc": 0.85, "fx": 0.7, "bell": 0.6}
out = np.zeros(N)
for k, v in mix.items():
    v = reverb(v, wet=0.25 if k in ("mel", "bell", "pad", "fx") else 0.06)
    out += levels[k] * v

# stereo: tiny Haas widening on melodic content
left = out.copy()
right = out.copy()
wide = reverb(mix["mel"] + mix["pad"], 1.2, 0.3) * 0.15
d = int(0.012 * SR)
right[d:] += wide[:-d]
left += wide

st = np.stack([left, right], 1)[: int(TOTAL * SR)]
st = np.tanh(st / (np.abs(st).max() + 1e-9) * 1.6)
# fade in/out
fi = int(0.05 * SR)
fo = int(1.6 * SR)
st[:fi] *= np.linspace(0, 1, fi)[:, None]
st[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
st = st / np.abs(st).max() * 0.89

import wave
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((st * 32767).astype("<i2").tobytes())
print("wrote", sys.argv[1], st.shape[0] / SR, "s")
