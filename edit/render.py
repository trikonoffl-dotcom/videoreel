"""Diwali reel renderer v2: synced to "Oorum Blood" (129 BPM), 9:16 1080x1920 @30fps.

* Every cut sits on a beat/kick of the song section 2:41.1 - 3:48.6.
* Virtual camera (crop/zoom/pan/dutch) from the 4K source; on people shots the camera
  follows the detected face (YuNet) so faces always stay in frame.
* Kick-synced zoom pulses, drop shakes and flashes.
* Cinematic festive grade: per-shot white balance + exposure normalisation, filmic curve,
  warm-gold highlights / cool shadows, skin-safe vibrance, bloom, vignette, grain.
* Modern titles (Bebas Neue + Outfit), rangoli transitions, bokeh, fireworks, sparkles.

usage: python3 render.py SRC.mp4 OUT.mp4 [--audio SONG] [--preview t1,t2,..]
(run from the edit/ directory; uses assets/, fonts/, models/, song_beats.json)
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
SRC_W, SRC_H = 2160, 3840

args = sys.argv[1:]
SRC, OUT = args[0], args[1]
PREVIEW = [float(x) for x in args[args.index("--preview") + 1].split(",")] if "--preview" in args else None
AUDIO = args[args.index("--audio") + 1] if "--audio" in args else None

# ------------------------------------------------------------------- song sync
SONG_A = 161.100          # song time that maps to video t=0
SONG_END = 228.600
TOTAL = SONG_END - SONG_A
BEATS = np.array(json.load(open(f"{HERE}/song_beats.json"))["beats"])


def every_other(lo, hi):
    b = BEATS[(BEATS > lo) & (BEATS < hi)]
    return list(b[::2])


KICKS_S = [175.427, 177.284, 179.142, 180.976, 182.811, 184.668, 186.526] + every_other(190.1, 203.2) + \
          every_other(205.8, 219.3)
KICKS = np.array(KICKS_S) - SONG_A
DROPS = np.array([190.195, 205.891]) - SONG_A

# ------------------------------------------------------------------- EDL (song times)
# name, song_start, song_end, src_start, speed, keyframes[(frac, cx, cy, zoom, rot)], bokeh, face
S = [
    ("intro",    161.100, 167.137, None,   1.0, None, 0, False),
    ("door",     167.137, 171.270, 0.6,    1.0, [(0, .5, .5, 1.0, 0), (1, .5, .44, 1.3, 0)], .45, False),
    ("banner",   171.270, 173.105, 5.6,    1.0, [(0, .44, .37, 1.75, -2), (1, .56, .37, 1.6, 1)], .35, False),
    ("paint1",   173.105, 175.427, 14.0,   1.0, [(0, .5, .6, 1.5, 0), (1, .48, .62, 1.8, 0)], .25, False),
    ("paint2",   175.427, 177.284, 39.8,   1.0, [(0, .62, .55, 1.6, 3), (1, .64, .55, 1.85, 0)], .2, False),
    ("paint4",   177.284, 179.142, 74.2,   1.0, [(0, 0, 0, 1.15, 0), (1, 0, 0, 1.3, 0)], .2, True),
    ("paint3",   179.142, 180.976, 61.6,   1.0, [(0, .5, .58, 1.25, 0), (1, .52, .6, 1.45, 0)], .2, False),
    ("paint5",   180.976, 182.811, 87.4,   1.0, [(0, .56, .64, 1.8, -3), (1, .56, .62, 1.55, 0)], .2, False),
    ("hero",     182.811, 186.526, 98.0,   1.0, [(0, .5, .45, 1.0, 0), (1, .5, .43, 1.12, 3)], .5, False),
    ("write3",   186.526, 188.360, 30.5,   1.0, [(0, 0, 0, 1.25, 0), (1, 0, 0, 1.45, 0)], .15, True),
    ("funny",    188.360, 190.195, 112.05, 1.0, [(0, .5, .48, 1.0, 0), (.6, .5, .47, 1.03, 0),
                                                 (.9, .5, .4, 1.5, 0), (1, .5, .4, 1.55, 0)], .15, False),
    ("write1",   190.195, 192.029, 19.6,   1.0, [(0, 0, 0, 1.25, 0), (1, 0, 0, 1.45, 0)], .15, True),
    ("write2",   192.029, 193.887, 27.0,   1.0, [(0, 0, 0, 1.1, 0), (1, 0, 0, 1.3, 0)], .15, True),
    ("write4",   193.887, 195.721, 52.4,   1.0, [(0, 0, 0, 1.15, 0), (1, 0, 0, 1.35, 0)], .2, True),
    ("laugh1",   195.721, 197.579, 78.9,   1.0, [(0, 0, 0, 1.45, 0), (1, 0, 0, 1.2, 0)], .15, True),
    ("laugh2",   197.579, 199.413, 82.0,   1.0, [(0, 0, 0, 1.2, -3), (1, 0, 0, 1.45, 0)], .15, True),
    ("write5",   199.413, 201.271, 108.6,  1.0, [(0, 0, 0, 1.15, 0), (1, 0, 0, 1.35, 0)], .15, True),
    ("smile",    201.271, 203.128, 114.9,  1.0, [(0, 0, 0, 1.45, 0), (1, 0, 0, 1.7, 0)], .15, True),
    ("write7",   203.128, 205.891, 124.74, 0.97, [(0, 0, 0, 1.15, 0), (1, 0, 0, 1.4, 0)], .2, True),
    ("raise",    205.891, 209.583, 44.9,   0.95, [(0, .5, .52, 1.0, 0), (1, .5, .45, 1.22, 0)], .5, False),
    ("write6",   209.583, 211.418, 119.0,  1.0, [(0, 0, 0, 1.0, 3), (1, 0, 0, 1.08, 3)], .2, True),
    ("raise2",   211.418, 213.275, 48.7,   1.0, [(0, 0, 0, 1.55, 0), (1, 0, 0, 1.75, 0)], .3, True),
    ("hero2",    213.275, 215.110, 103.6,  1.0, [(0, .5, .45, 1.2, -4), (1, .5, .45, 1.35, 0)], .4, False),
    ("write4b",  215.110, 216.967, 56.0,   1.0, [(0, 0, 0, 1.3, 0), (1, 0, 0, 1.5, 0)], .3, True),
    ("lights",   216.967, 218.802, 6.0,    1.0, [(0, .5, .4, 1.45, -3), (1, .5, .42, 1.25, 0)], .5, False),
    ("banner2",  218.802, 219.730, 2.4,    1.0, [(0, .5, .37, 1.5, 0), (1, .5, .37, 1.7, 0)], .5, False),
    ("outro",    219.730, 228.600, None,   1.0, None, 0, False),
]
TRANS = {
    "door": ("rangoli1", 1.0), "banner": ("zoom", 0.4), "paint1": ("flash", 0.4), "hero": ("zoom", 0.4),
    "write3": ("rangoli2", 1.0), "write1": ("flash", 0.6), "write7": ("fade", 0.4), "raise": ("flash", 0.7),
    "raise2": ("zoom", 0.35), "lights": ("flash", 0.4), "outro": ("rangoli1", 1.0),
}

TL = []
for name, s0, s1, src, speed, kf, bokeh, face in S:
    TL.append(dict(name=name, t0=s0 - SONG_A, t1=s1 - SONG_A, src=src, speed=speed, kf=kf, bokeh=bokeh, face=face))
for s in TL:
    tr = TRANS.get(s["name"], ("cut", 0))
    s["tin"] = tr
    s["pre"] = tr[1] / 2 if tr[0] in ("fade", "zoom") else 0.0
for i, s in enumerate(TL):
    s["post"] = TL[i + 1]["pre"] if i + 1 < len(TL) else 0.0


def T(song_t):
    return song_t - SONG_A


# ------------------------------------------------------------------- helpers
def clamp01(x):
    return min(1.0, max(0.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def interp_kf(kf, f):
    if f <= kf[0][0]:
        return kf[0][1:]
    for a, b in zip(kf, kf[1:]):
        if f <= b[0]:
            u = smooth((f - a[0]) / (b[0] - a[0]))
            return tuple(a[i] + (b[i] - a[i]) * u for i in range(1, 5))
    return kf[-1][1:]


def kick_env(t):
    """0..1 envelope that spikes on every kick."""
    d = t - KICKS
    d = d[(d >= 0) & (d < 0.5)]
    return float(np.exp(-d.min() * 9)) if len(d) else 0.0


def drop_env(t):
    d = t - DROPS
    d = d[(d >= 0) & (d < 0.6)]
    return float(np.exp(-d.min() * 6)) if len(d) else 0.0


def load_rgba(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    a[:, :, :3] *= a[:, :, 3:4]
    return a


R1 = load_rgba(f"{HERE}/assets/rangoli1.png")
R2 = load_rgba(f"{HERE}/assets/rangoli2.png")
R1_small = cv2.resize(R1, (340, 340), interpolation=cv2.INTER_AREA)
R2_small = cv2.resize(R2, (300, 300), interpolation=cv2.INTER_AREA)


def pick(spr, small, size):
    return small if size < small.shape[0] * 1.2 else spr


def draw_sprite(dst, spr, cx, cy, size, angle, alpha=1.0, additive=False):
    if alpha <= 0.002 or size < 2:
        return
    sh = spr.shape[0]
    s = size / sh
    half = size * 0.7072 + 2
    x0, x1 = int(max(0, cx - half)), int(min(dst.shape[1], cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(dst.shape[0], cy + half))
    if x1 <= x0 or y1 <= y0:
        return
    M = cv2.getRotationMatrix2D((sh / 2, sh / 2), angle, s)
    M[0, 2] += cx - sh / 2 - x0
    M[1, 2] += cy - sh / 2 - y0
    w = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
    roi = dst[y0:y1, x0:x1]
    if additive:
        roi += w[:, :, :3] * alpha
    else:
        roi *= 1 - w[:, :, 3:4] * alpha
        roi += w[:, :, :3] * alpha


# ------------------------------------------------------------------- typography (modern)
F_BEBAS = f"{HERE}/fonts/Bebas_Neue.ttf"
F_LIGHT = f"{HERE}/fonts/Outfit_wght_300.ttf"
F_SEMI = f"{HERE}/fonts/Outfit_wght_600.ttf"
F_BOLD = f"{HERE}/fonts/Outfit_wght_800.ttf"
GOLD = (np.array([175, 240, 255]), np.array([40, 195, 255]), np.array([15, 125, 225]))  # BGR top/mid/bottom


def make_text(text, fpath, size, tracking=0, style="gold", glow=True, maxw=960, box=False):
    f = ImageFont.truetype(fpath, size)
    chars = list(text)
    adv = [f.getlength(c) for c in chars]
    tw = sum(adv) + tracking * (len(chars) - 1)
    if tw > maxw:
        return make_text(text, fpath, int(size * maxw / tw), tracking * maxw / tw, style, glow, maxw, box)
    asc, desc = f.getmetrics()
    pad = int(size * 0.6) + (30 if box else 0)
    img = Image.new("L", (int(tw) + 2 * pad, asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    x = pad
    for c, a in zip(chars, adv):
        d.text((x, pad), c, font=f, fill=255)
        x += a + tracking
    mask = np.asarray(img).astype(np.float32) / 255
    hh, ww = mask.shape
    ys = np.where(mask.max(1) > 0.05)[0]
    ink0, ink1 = (ys.min(), ys.max()) if len(ys) else (0, hh)
    if style == "gold":
        y1 = np.clip((np.arange(hh) - ink0) / max(1, ink1 - ink0), 0, 1)[:, None]
        top, mid, bot = [c / 255 for c in GOLD]
        g = np.where(y1 < 0.5, top + (mid - top) * (y1 / 0.5), mid + (bot - mid) * ((y1 - 0.5) / 0.5))
        col = np.broadcast_to(g[:, None, :], (hh, ww, 3)).astype(np.float32)
    elif style == "white":
        col = np.ones((hh, ww, 3), np.float32) * np.array([245, 250, 255], np.float32) / 255
    else:
        col = np.ones((hh, ww, 3), np.float32) * np.array(style, np.float32) / 255
    out = np.zeros((hh, ww, 4), np.float32)
    if box:
        # frosted dark pill with thin gold outline
        bx = Image.new("L", (ww, hh), 0)
        bd = ImageDraw.Draw(bx)
        r = (ink1 - ink0) // 2 + 26
        cyb = (ink0 + ink1) / 2
        bd.rounded_rectangle([pad - 34, cyb - r, ww - pad + 34, cyb + r], radius=r, fill=255)
        bm = np.asarray(bx).astype(np.float32) / 255
        edge = Image.new("L", (ww, hh), 0)
        ImageDraw.Draw(edge).rounded_rectangle([pad - 34, cyb - r, ww - pad + 34, cyb + r], radius=r, outline=255,
                                               width=3)
        em = np.asarray(edge).astype(np.float32) / 255
        out[:, :, :3] = np.array([35, 10, 25], np.float32) / 255 * bm[..., None] * 0.62
        out[:, :, 3] = bm * 0.62
        gold = np.array([60, 190, 250], np.float32) / 255
        out[:, :, :3] = out[:, :, :3] * (1 - em[..., None]) + gold * em[..., None]
        out[:, :, 3] = out[:, :, 3] * (1 - em) + em
    elif glow:
        sh = np.clip(cv2.GaussianBlur(mask, (0, 0), max(2, size * 0.06)) * 2.2, 0, 1)
        out[:, :, :3] = np.array([25, 5, 30], np.float32) / 255 * sh[..., None] * 0.8
        out[:, :, 3] = sh * 0.8
        gl = cv2.GaussianBlur(mask, (0, 0), size * 0.22)
        out[:, :, :3] += np.array([40, 150, 255], np.float32) / 255 * gl[..., None] * 0.45
        out[:, :, 3] = np.clip(out[:, :, 3] + gl * 0.2, 0, 1)
    a = mask[..., None]
    out[:, :, :3] = out[:, :, :3] * (1 - a) + col * a
    out[:, :, 3] = out[:, :, 3] * (1 - mask) + mask
    return out


TEXT_CACHE = {}


def txt(key, *a, **k):
    if key not in TEXT_CACHE:
        TEXT_CACHE[key] = make_text(*a, **k)
    return TEXT_CACHE[key]


def draw_text(dst, spr, cx, cy, alpha=1.0, scale=1.0, dy=0.0, reveal=1.0):
    if alpha <= 0.002 or reveal <= 0:
        return
    if abs(scale - 1) > 1e-3:
        spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    if reveal < 1:  # soft left-to-right wipe
        xs = np.arange(w) / w
        edge = 0.08
        m = np.clip((reveal * (1 + edge) - xs) / edge, 0, 1)[None, :, None]
        spr = spr * m
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2 + dy))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return
    s = spr[sy0: sy0 + (y1c - y0c), sx0: sx0 + (x1c - x0c)]
    a = s[:, :, 3:4] * alpha
    roi = dst[y0c:y1c, x0c:x1c]
    roi *= 1 - a
    roi += s[:, :, :3] * alpha


def anim(t, t_in, t_out, din=0.45, dout=0.35):
    """alpha, reveal, dy for a title shown between t_in and t_out."""
    if t < t_in or t > t_out:
        return 0.0, 0.0, 0.0
    r = ease_out((t - t_in) / din)
    o = 1 - smooth((t - (t_out - dout)) / dout)
    return min(1.0, r * 1.5) * o, r, (1 - o) * -30 + (1 - r) * 18


def flank_lines(dst, spr, cy, alpha, length=120, gap=10):
    """gold lines either side of a text sprite (spr includes padding)."""
    a = spr[:, :, 3].max(0)
    xs = np.where(a > 0.3)[0]
    half = (xs.max() - xs.min()) / 2 if len(xs) else spr.shape[1] / 2
    gold_line(dst, W / 2 - half - gap - length / 2, cy, length, alpha)
    gold_line(dst, W / 2 + half + gap + length / 2, cy, length, alpha)


def gold_line(dst, cx, cy, length, alpha):
    if alpha <= 0:
        return
    x0, x1 = int(cx - length / 2), int(cx + length / 2)
    y = int(cy)
    xs = np.linspace(-1, 1, x1 - x0)
    a = (1 - xs ** 2)[None, :, None] * alpha
    col = np.array([60, 195, 255], np.float32) / 255
    roi = dst[y - 1:y + 2, x0:x1]
    roi *= 1 - a
    roi += col * a


def scrim(dst, cy, hgt, amt):
    if amt <= 0.002:
        return
    y0, y1 = int(max(0, cy - hgt)), int(min(H, cy + hgt))
    yy = (np.arange(y0, y1) - cy) / hgt
    m = (np.cos(np.clip(yy, -1, 1) * np.pi) * 0.5 + 0.5)[:, None, None] * amt
    dst[y0:y1] *= 1 - 0.32 * m


# ------------------------------------------------------------------- particles
rng = np.random.default_rng(11)
NB = 46
BK = dict(x=rng.uniform(0, 1, NB), y=rng.uniform(0, 1, NB), r=rng.uniform(6, 26, NB), vy=rng.uniform(0.015, 0.05, NB),
          ph=rng.uniform(0, 6.28, NB), fr=rng.uniform(0.5, 1.6, NB), c=rng.choice(4, NB))
BCOL = np.array([[60, 190, 255], [40, 140, 255], [120, 90, 255], [150, 230, 255]], np.float32) / 255
HW, HH = W // 2, H // 2
_yy = np.linspace(-1, 1, HH)[:, None]
_xx = np.linspace(-1, 1, HW)[None, :]
EDGE = (0.25 + 0.75 * np.clip(np.maximum(np.abs(_yy) * 1.15 - 0.35, 0) + np.abs(_xx) * 0.5, 0, 1))[..., None].astype(
    np.float32)


def bokeh_layer(t, strength, edge=True):
    if strength <= 0:
        return None
    lay = np.zeros((HH, HW, 3), np.float32)
    for i in range(NB):
        y = (BK["y"][i] - t * BK["vy"][i]) % 1.2 - 0.1
        x = BK["x"][i] + 0.02 * math.sin(t * 0.7 + BK["ph"][i])
        tw = 0.55 + 0.45 * math.sin(t * BK["fr"][i] * 3 + BK["ph"][i])
        cv2.circle(lay, (int(x * HW), int(y * HH)), int(BK["r"][i]), (BCOL[BK["c"][i]] * tw).tolist(), -1,
                   cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 2.2)
    if edge:
        lay *= EDGE
    return lay * strength


def make_fw(t0, x, y, col, n=70, spd=260):
    r = np.random.default_rng(int(t0 * 1000 + x * 7))
    ang = r.uniform(0, 2 * np.pi, n)
    sp = spd * (0.55 + 0.45 * r.uniform(0, 1, n))
    return dict(t0=t0, x=x, y=y, col=np.array(col, np.float32) / 255, vx=np.cos(ang) * sp, vy=np.sin(ang) * sp,
                tw=r.uniform(0, 6, n))


FW_COLS = [(80, 200, 255), (60, 120, 255), (200, 120, 255), (120, 255, 200), (255, 200, 120), (180, 255, 255)]
FW_SPEC = [(0.4, .25, .16, 0), (1.3, .78, .1, 2), (2.3, .55, .07, 5), (3.4, .2, .28, 1), (4.6, .82, .25, 3)]
# drop-2 fireworks land on kicks
for k, kt in enumerate(every_other(205.8, 212.0)):
    FW_SPEC.append((kt - SONG_A, [.28, .74, .52, .18, .82, .45, .3][k % 7], [.12, .08, .18, .06, .2, .1, .15][k % 7],
                    k % 6))
for k, kt in enumerate([219.73, 220.66, 221.6, 222.5, 223.6, 224.6, 225.5]):
    FW_SPEC.append((kt - SONG_A, [.2, .8, .5, .25, .78, .5, .3][k], [.1, .12, .06, .25, .28, .14, .08][k], k % 6))
FIREWORKS = [make_fw(t0, x * W, y * H, FW_COLS[c]) for t0, x, y, c in FW_SPEC]


def fireworks_layer(t):
    lay = None
    for fw in FIREWORKS:
        a = t - fw["t0"]
        if a < 0 or a > 1.8:
            continue
        if lay is None:
            lay = np.zeros((HH, HW, 3), np.float32)
        fade = (1 - a / 1.8) ** 1.6
        g = 60

        def pos(tt):
            d = (1 - math.exp(-2.2 * tt)) / 2.2
            return (fw["x"] + fw["vx"] * d) / 2, (fw["y"] + fw["vy"] * d + g * tt * tt) / 2

        px0, py0 = pos(a)
        px1, py1 = pos(max(0, a - 0.09))
        tw = 0.6 + 0.4 * np.sin(fw["tw"] + a * 25)
        for j in range(len(px0)):
            cv2.line(lay, (int(px1[j]), int(py1[j])), (int(px0[j]), int(py0[j])), (fw["col"] * fade * tw[j]).tolist(), 2,
                     cv2.LINE_AA)
        if a < 0.15:
            cv2.circle(lay, (int(fw["x"] / 2), int(fw["y"] / 2)), int(30 * (1 - a / 0.15)) + 1, (0.8, 0.95, 1.0), -1,
                       cv2.LINE_AA)
    if lay is None:
        return None
    return lay * 1.3 + cv2.GaussianBlur(lay, (0, 0), 6) * 1.6


def star_sprite(size=64):
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    rays = np.exp(-np.abs(xx) * 40) * np.exp(-np.abs(yy) * 3) + np.exp(-np.abs(yy) * 40) * np.exp(-np.abs(xx) * 3)
    return np.clip(np.exp(-r ** 2 * 60) + rays * 0.8, 0, 1).astype(np.float32)


STAR = star_sprite()


def sparkles(dst, t, region, n, seed, strength=1.0, size=40):
    if strength <= 0.01:
        return
    r = np.random.default_rng(seed)
    xs, ys = r.uniform(region[0], region[2], n), r.uniform(region[1], region[3], n)
    ph, fr, sz = r.uniform(0, 6.28, n), r.uniform(1.5, 3.5, n), r.uniform(0.5, 1.2, n)
    for i in range(n):
        v = math.sin(t * fr[i] + ph[i])
        if v <= 0:
            continue
        s = int(size * sz[i] * (0.5 + 0.5 * v))
        x0, y0 = int(xs[i] - s / 2), int(ys[i] - s / 2)
        if s < 4 or x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H:
            continue
        dst[y0:y0 + s, x0:x0 + s] += cv2.resize(STAR, (s, s))[..., None] * np.array([0.75, 0.95, 1.0],
                                                                                      np.float32) * v * strength


# ------------------------------------------------------------------- colour grade
def build_look():
    x = np.arange(256) / 255.0
    # filmic: gentle toe, S-contrast, soft highlight shoulder
    sm = x * x * (3 - 2 * x)
    c = 0.62 * x + 0.38 * sm
    c = np.where(c > 0.8, 0.8 + (c - 0.8) * (1 - 0.35 * ((c - 0.8) / 0.2)), c)
    c = 0.004 + c * 0.99  # rich blacks
    sh = np.clip(1 - x / 0.45, 0, 1) ** 2
    hi = np.clip((x - 0.45) / 0.55, 0, 1) ** 1.5
    b = c + 0.035 * sh - 0.05 * hi        # cool-teal shadows, warm highlights
    g = c + 0.012 * sh - 0.008 * hi
    r = c - 0.02 * sh + 0.04 * hi
    return [np.clip(v * 255, 0, 255).astype(np.uint8) for v in (b, g, r)]


LOOK = build_look()
_vy = np.linspace(-1, 1, H)[:, None]
_vx = np.linspace(-1, 1, W)[None, :] * 0.9
VIG = (1 - 0.10 * np.clip((np.sqrt(_vx ** 2 + (_vy * 0.9) ** 2) - 0.55) / 0.85, 0, 1) ** 1.6)[..., None].astype(
    np.float32)
# hue-selective saturation (OpenCV hue 0..180): skin (5..25) gently, decor colours strongly
_h = np.arange(180)
_skin = np.exp(-((_h - 14) / 9.0) ** 2)
SAT_GAIN = (1.18 - 0.12 * _skin).astype(np.float32)
SKIN = _skin.astype(np.float32)


# --- shot matching: every frame is normalised to the same targets (smoothed over time) ---
TGT_L, TGT_SPREAD, TGT_A, TGT_B, TGT_SAT, TGT_FACE_L = 47.0, 76.0, 1.5, 6.0, 0.46, 50.0
CLAHE = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 7))


class Tone:
    def __init__(self):
        self.st = None

    def update(self, lab_small, hsv_small, face_xy=None):
        L, A, B = lab_small[:, :, 0], lab_small[:, :, 1], lab_small[:, :, 2]
        p5, p50, p95 = np.percentile(L, [4, 50, 96])
        if face_xy is not None:
            hh_, ww_ = L.shape
            fx, fy, fr = face_xy
            reg = L[int(max(0, (fy - fr) * hh_)):int(min(1, fy + fr) * hh_), int(max(0, (fx - fr * 0.56) * ww_)):
                    int(min(1, fx + fr * 0.56) * ww_)]
            if reg.size > 30:
                # exposure anchored on the face (mapped to its own target)
                p50 = 0.6 * p50 + 0.4 * (np.median(reg) - (TGT_FACE_L - TGT_L))
        C = np.sqrt(A ** 2 + B ** 2)
        mid = (L > 15) & (L < 92)
        neutral = mid & (C < 22)
        if neutral.sum() > 0.08 * L.size:
            am, bm, k = A[neutral].mean(), B[neutral].mean(), 0.8
        else:
            am, bm, k = A[mid].mean() if mid.any() else 0, B[mid].mean() if mid.any() else 0, 0.5
        da, db = (am - TGT_A) * k, (bm - TGT_B) * k
        sm = hsv_small[:, :, 1].mean()
        new = np.array([p50, max(p95 - p5, 20), da, db, sm], np.float64)
        self.st = new if self.st is None else self.st * 0.86 + new * 0.14
        return self.st


def tone_match(f, state):
    """f: float BGR 0..1 (W x H). Returns matched float BGR."""
    p50, spread, da, db, sm = state
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    g = float(np.clip((TGT_SPREAD / spread) ** 0.85, 0.85, 1.45))
    L = (lab[:, :, 0] - p50) * g + TGT_L
    # soft toe & shoulder so nothing clips
    L = np.where(L > 80, 80 + 20 * np.tanh((L - 80) / 20), L)
    L = np.where(L < 6, 6 * np.exp((L - 6) / 6), L) - 1.5
    # gentle local tone mapping: evens out bright windows vs. darker faces
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    Lc = CLAHE.apply(L8).astype(np.float32) / 2.55
    lab[:, :, 0] = L * 0.78 + Lc * 0.22
    lab[:, :, 1] -= float(np.clip(da, -18, 18))
    lab[:, :, 2] -= float(np.clip(db, -22, 22))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out, 0, 1), float(np.clip((TGT_SAT / max(sm, 0.05)) ** 0.8, 0.8, 1.55))


def grade(img_u8, tone, t, face_xy=None):
    small = cv2.resize(img_u8, (135, 240), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    st = tone.update(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), cv2.cvtColor(small, cv2.COLOR_BGR2HSV), face_xy)
    f, sgain = tone_match(img_u8.astype(np.float32) / 255, st)
    u8 = (f * 255 + 0.5).astype(np.uint8)
    b, g, r = cv2.split(u8)
    u8 = cv2.merge([cv2.LUT(b, LOOK[0]), cv2.LUT(g, LOOK[1]), cv2.LUT(r, LOOK[2])])
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255
    gain = 1 + (sgain * SAT_GAIN[hsv[:, :, 0]] - 1) * (1 - 0.45 * SKIN[hsv[:, :, 0]])
    hsv[:, :, 1] = (np.clip(s * gain, 0, 1) * 255).astype(np.uint8)
    f = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    small = cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = cv2.GaussianBlur(np.clip(small - 0.82, 0, 1), (0, 0), 6)
    f += cv2.resize(br, (W, H)) * np.array([0.45, 0.75, 1.0], np.float32) * 0.3
    f *= VIG
    gr = np.random.default_rng(int(t * 1000)).normal(0, 0.008, (H // 2, W // 2)).astype(np.float32)
    f += cv2.resize(gr, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    return f


# ------------------------------------------------------------------- face tracking
FACE_MODEL = f"{HERE}/models/yunet.onnx"


def face_track(src_t0, n_frames, speed):
    DW, DH = 540, 960
    det = cv2.FaceDetectorYN.create(FACE_MODEL, "", (DW, DH), 0.6, 0.3, 5000)
    dur = n_frames / FPS * speed + 0.2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{max(0, src_t0):.3f}", "-i", SRC, "-t", f"{dur:.3f}", "-vf",
                          f"scale={DW}:{DH}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    pts = []
    i = 0
    while True:
        b = p.stdout.read(DW * DH * 3)
        if len(b) < DW * DH * 3:
            break
        if i % 2 == 0:
            _, faces = det.detect(np.frombuffer(b, np.uint8).reshape(DH, DW, 3))
            if faces is not None:
                # prefer big, central-ish faces
                best = max(faces, key=lambda r: r[2] * r[3] * (1.2 - abs((r[0] + r[2] / 2) / DW - 0.5)))
                pts.append((i, (best[0] + best[2] / 2) / DW, (best[1] + best[3] / 2) / DH, best[3] / DH))
        i += 1
    p.wait()
    if len(pts) < 3:
        return None
    a = np.array(pts)
    idx = np.arange(i)
    tr = np.stack([np.interp(idx, a[:, 0], a[:, k]) for k in (1, 2, 3)], 1)
    for k in range(3):
        tr[:, k] = cv2.GaussianBlur(tr[:, k].reshape(-1, 1).astype(np.float64), (1, 0), 6).ravel()
    return tr


# ------------------------------------------------------------------- shot reader
class Shot:
    def __init__(self, s):
        self.s = s
        self.proc = None
        self.tone = Tone()
        self.face_xy = None
        self.track = None
        self.tracked = False

    def open(self):
        s = self.s
        self.t_lo = s["t0"] - s["pre"]
        t_hi = s["t1"] + s["post"]
        self.src0 = s["src"] - s["pre"] * s["speed"]
        dur = (t_hi - self.t_lo) * s["speed"] + 0.2
        if s["face"] and not self.tracked:
            self.track = face_track(self.src0, int(dur * FPS / s["speed"]) + 5, s["speed"])
            self.tracked = True
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{max(0, self.src0):.3f}", "-i", SRC, "-t",
                                      f"{dur:.3f}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                     stdout=subprocess.PIPE, bufsize=SRC_W * SRC_H * 3 * 2)
        self.idx = -1
        self.cur = None

    def frame_at(self, t):
        if self.proc is None:
            self.open()
        want = int(math.floor((t - self.t_lo) * self.s["speed"] * FPS + 1e-6))
        while self.idx < want:
            b = self.proc.stdout.read(SRC_W * SRC_H * 3)
            if len(b) < SRC_W * SRC_H * 3:
                break
            self.cur = np.frombuffer(b, np.uint8).reshape(SRC_H, SRC_W, 3)
            self.idx += 1
        return self.cur

    def camera(self, t):
        s = self.s
        frac = (t - s["t0"]) / (s["t1"] - s["t0"])
        cx, cy, z, rot = interp_kf(s["kf"], frac)
        if s["face"]:
            if self.track is not None:
                k = min(len(self.track) - 1, max(0, self.idx))
                fx, fy, fh = self.track[k]
                self._face = (fx, fy, fh)
                z = min(z, max(1.0, 0.36 / max(fh, 1e-3)))   # face never bigger than ~36% of frame height
                cx = fx
                cy = fy + 0.13 / z                            # face sits on the upper third line
            else:
                cx, cy = 0.5, 0.45
        # beat pulse & drop shake
        z *= 1 + 0.035 * kick_env(t)
        de = drop_env(t)
        if de > 0:
            cx += 0.012 * de * math.sin(t * 90)
            cy += 0.010 * de * math.cos(t * 77)
            rot += 1.2 * de * math.sin(t * 60)
        return cx, cy, z, rot

    def render(self, t):
        src = self.frame_at(t)
        cx, cy, z, rot = self.camera(t)
        rr = math.radians(abs(rot))
        z = max(z, 1.0) * ((math.cos(rr) + (H / W) * math.sin(rr)) if rot else 1.0)
        k = z * W / SRC_W
        hw = 0.5 / z
        cx = min(max(cx, hw), 1 - hw)
        cy = min(max(cy, hw), 1 - hw)
        if self.s["face"] and self.track is not None and getattr(self, "_face", None) is not None:
            fx, fy, fh = self._face
            self.face_xy = ((fx - cx) * z + 0.5, (fy - cy) * z + 0.5, fh * z * 0.5)
        if k < 0.98:
            sw, shh = int(round(SRC_W * k)), int(round(SRC_H * k))
            srcs = cv2.resize(src, (sw, shh), interpolation=cv2.INTER_AREA)
            k2, ccx, ccy = 1.0, cx * sw, cy * shh
        else:
            srcs, k2, ccx, ccy = src, k, cx * SRC_W, cy * SRC_H
        M = cv2.getRotationMatrix2D((ccx, ccy), rot, k2)
        M[0, 2] += W / 2 - ccx
        M[1, 2] += H / 2 - ccy
        out = cv2.warpAffine(srcs, M, (W, H), flags=cv2.INTER_LINEAR if k2 < 1.3 else cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT)
        return grade(out, self.tone, t, self.face_xy)

    def close(self):
        if self.proc:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc = None


READERS = {i: Shot(s) for i, s in enumerate(TL) if s["src"] is not None}

# ------------------------------------------------------------------- intro / outro
_gy = np.linspace(0, 1, H)[:, None, None]
_BG_TOP = np.array([72, 20, 82], np.float32) / 255
_BG_BOT = np.array([92, 24, 52], np.float32) / 255
_BG_BASE = _BG_TOP + (_BG_BOT - _BG_TOP) * _gy
_gx = np.linspace(-1, 1, W)[None, :]
_gy2 = np.linspace(-1, 1, H)[:, None]
_GLOW = (np.exp(-(_gx ** 2 + ((_gy2 + 0.3) * 0.6) ** 2) * 1.3)[..., None] *
         np.array([40, 120, 230], np.float32) / 255 * 0.5)


def festive_bg(t):
    return np.broadcast_to(_BG_BASE, (H, W, 3)) + _GLOW * (0.85 + 0.15 * math.sin(t * 2))


LAST_LIGHTS = [None]


def compose_intro(t):
    f = festive_bg(t).astype(np.float32)
    draw_sprite(f, R2, W / 2, H * 0.36, 1900, t * 6, alpha=0.10)
    fw = fireworks_layer(t)
    if fw is not None:
        f += cv2.resize(fw, (W, H)) * 0.9
    f += cv2.resize(bokeh_layer(t, 0.9, edge=False), (W, H))
    a = ease_out(t / 1.4)
    size = 720 * (0.2 + 0.8 * a)
    spr = pick(R1, R1_small, size)
    draw_sprite(f, spr, W / 2, H * 0.33, size + 10, -200 * (1 - a) - t * 10, alpha=0.35 * a, additive=True)
    draw_sprite(f, spr, W / 2, H * 0.33, size, -200 * (1 - a) - t * 10, alpha=min(1, a * 1.5))
    al, rv, dy = anim(t, 1.2, 5.8)
    draw_text(f, txt("happy", "HAPPY", F_LIGHT, 64, tracking=34, style="white", glow=True), W / 2, H * 0.575, al,
              dy=dy, reveal=rv)
    al, rv, dy = anim(t, 1.5, 5.8)
    draw_text(f, txt("diwali", "DIWALI", F_BEBAS, 330, tracking=10), W / 2, H * 0.665, al, dy=dy, reveal=rv)
    al, rv, dy = anim(t, 2.2, 5.8)
    flank_lines(f, txt("trikon", "TRIKON TELESOFT", F_SEMI, 40, tracking=12, style="white"), H * 0.765, al * rv,
                110, 28)
    draw_text(f, txt("trikon", "TRIKON TELESOFT", F_SEMI, 40, tracking=12, style="white"), W / 2, H * 0.765, al,
              dy=dy, reveal=rv)
    al, rv, dy = anim(t, 2.8, 5.8)
    draw_text(f, txt("fol", "FESTIVAL OF LIGHTS", F_LIGHT, 34, tracking=16), W / 2, H * 0.81, al, dy=dy, reveal=rv)
    sparkles(f, t, (120, 1050, W - 120, 1600), 14, 3, 0.9 * smooth((t - 1.5) / 0.5), 56)
    return f


def compose_outro(t, lt):
    lf = LAST_LIGHTS[0]
    if lf is not None:
        bgv = cv2.resize(cv2.GaussianBlur(cv2.resize(lf, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 3),
                         (W, H))
        f = bgv * 0.35 + festive_bg(t) * 0.65
    else:
        f = festive_bg(t).astype(np.float32)
    f = f.astype(np.float32)
    fw = fireworks_layer(t)
    if fw is not None:
        f += cv2.resize(fw, (W, H)) * 0.8
    f += cv2.resize(bokeh_layer(t, 0.8, edge=False), (W, H))
    a = ease_out(lt / 1.0)
    size = 680 * (0.3 + 0.7 * a)
    spr = pick(R2, R2_small, size)
    draw_sprite(f, R1, W / 2, H * 0.31, 1500 * (0.6 + 0.4 * a), lt * 8, alpha=0.16 * a)
    draw_sprite(f, spr, W / 2, H * 0.31, size + 30, -lt * 10, alpha=0.4 * a, additive=True)
    draw_sprite(f, spr, W / 2, H * 0.31, size, -lt * 10, alpha=min(1, a * 1.4))
    end_t = TL[-1]["t1"] - TL[-1]["t0"]
    al, rv, dy = anim(lt, 0.5, end_t - 0.6)
    draw_text(f, txt("happy2", "HAPPY", F_LIGHT, 60, tracking=34, style="white"), W / 2, H * 0.545, al, dy=dy,
              reveal=rv)
    al, rv, dy = anim(lt, 0.75, end_t - 0.6)
    draw_text(f, txt("diwali2", "DIWALI", F_BEBAS, 300, tracking=10), W / 2, H * 0.625, al, dy=dy, reveal=rv)
    al, rv, dy = anim(lt, 1.5, end_t - 0.6)
    draw_text(f, txt("w1", "May the festival of lights fill", F_LIGHT, 44, style="white"), W / 2, H * 0.715, al,
              dy=dy, reveal=rv)
    draw_text(f, txt("w2", "your life with joy & prosperity", F_LIGHT, 44, style="white"), W / 2, H * 0.745, al,
              dy=dy, reveal=rv)
    al, rv, dy = anim(lt, 2.3, end_t - 0.6)
    flank_lines(f, txt("team", "TEAM TRIKON TELESOFT", F_SEMI, 40, tracking=12), H * 0.81, al * rv, 90, 28)
    draw_text(f, txt("team", "TEAM TRIKON TELESOFT", F_SEMI, 40, tracking=12), W / 2, H * 0.81, al, dy=dy, reveal=rv)
    sparkles(f, t, (100, 250, W - 100, 1000), 12, 9, 0.9 * smooth((lt - 0.4) / 0.5), 60)
    return f * (1 - smooth((lt - (end_t - 0.9)) / 0.9))


# ------------------------------------------------------------------- titles on footage (timeline-level)
def chapter(f, t, t_in, t_out, num, title, sub, key):
    al, rv, dy = anim(t, t_in, t_out)
    if al <= 0:
        return
    scrim(f, H * 0.2, 300, al)
    draw_text(f, txt(key + "n", f"CHAPTER {num}", F_LIGHT, 34, tracking=18), W / 2, H * 0.12, al, dy=dy, reveal=rv)
    al2, rv2, dy2 = anim(t, t_in + 0.12, t_out)
    pulse = 1 + 0.04 * kick_env(t)
    draw_text(f, txt(key + "t", title, F_BEBAS, 190, tracking=6), W / 2, H * 0.185, al2, scale=pulse, dy=dy2,
              reveal=rv2)
    al3, rv3, dy3 = anim(t, t_in + 0.3, t_out)
    gold_line(f, W / 2, H * 0.245, 300 * rv3, al3)
    draw_text(f, txt(key + "s", sub, F_LIGHT, 40, tracking=6, style="white"), W / 2, H * 0.27, al3, dy=dy3,
              reveal=rv3)


def caption(f, t, t_in, t_out, text, key, y=0.83):
    al, rv, dy = anim(t, t_in, t_out)
    if al <= 0:
        return
    draw_text(f, txt(key, text, F_SEMI, 48, tracking=3, style="white", box=True), W / 2, H * y, al, dy=dy, reveal=rv)


def big_word(f, t, t_in, t_out, text, key, y=0.8):
    al, rv, dy = anim(t, t_in, t_out)
    if al <= 0:
        return
    scrim(f, H * y, 230, al)
    pulse = 1 + 0.06 * kick_env(t)
    draw_text(f, txt(key, text, F_BEBAS, 220, tracking=8), W / 2, H * y, al, scale=pulse, dy=dy, reveal=rv)


def timeline_titles(f, t):
    caption(f, t, T(167.6), T(173.0), "Our office, all lit up", "cap_office")
    chapter(f, t, T(173.25), T(177.2), "01", "DIYA PAINTING", "colours of light", "c1")
    caption(f, t, T(183.2), T(186.4), "Handcrafted with love", "cap_hand")
    chapter(f, t, T(186.7), T(190.1), "02", "CREATIVE CHALLENGE", "pens out, ideas on", "c2")
    big_word(f, t, T(206.0), T(209.5), "SHINE BRIGHT", "shine")
    big_word(f, t, T(216.98), T(219.7), "HAPPY DIWALI", "hd_end", y=0.8)


def shot_overlays(f, s, t):
    lt = t - s["t0"]
    bk = bokeh_layer(t, s["bokeh"])
    if bk is not None:
        f += cv2.resize(bk, (W, H))
    fw = fireworks_layer(t)
    if fw is not None:
        f += cv2.resize(fw, (W, H)) * 0.85
    if s["name"] in ("hero", "hero2"):
        sparkles(f, t, (200, 350, W - 200, 1250), 14, 33, smooth(lt / 0.6), 64)
    if s["name"] in ("raise", "raise2"):
        sparkles(f, t, (450, 150, W - 60, 900), 16, 51, smooth(lt / 0.5), 70)
    # rangoli corner accent during chapter titles
    for (ti, to, spr, x) in ((T(173.25), T(177.2), R2_small, W - 30), (T(186.7), T(190.1), R1_small, 30)):
        if ti <= t <= to:
            a = min(smooth((t - ti) / 0.4), 1 - smooth((t - (to - 0.35)) / 0.35))
            draw_sprite(f, spr, x, 60, 380, (t - ti) * 25 * (1 if x > W / 2 else -1), alpha=0.95 * a)
    return f


# ------------------------------------------------------------------- transitions
_lx = np.linspace(-1, 1, W)[None, :]
_ly = np.linspace(-1, 1, H)[:, None]
LEAK = (np.exp(-(_lx ** 2 + _ly ** 2) * 0.8)[..., None] * np.array([0.6, 0.85, 1.0], np.float32)).astype(np.float32)


def gold_flash(f, amt):
    return f if amt <= 0 else f + LEAK * amt * 1.0 + amt * 0.12


def scaled(img, s):
    if abs(s - 1) < 1e-3:
        return img
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((W / 2, H / 2), 0, s), (W, H), borderMode=cv2.BORDER_REFLECT)


def zoom_blur(img, strength):
    if strength < 0.01:
        return img
    acc = img.copy()
    for i in range(1, 5):
        acc += scaled(img, 1 + strength * 0.08 * i)
    return acc / 5


def mix_trans(kind, a, b, u):
    if kind == "fade":
        w = smooth(u)
        return a * (1 - w) + b * w
    w = smooth((u - 0.35) / 0.3)
    A = zoom_blur(scaled(a, 1 + 0.35 * ease_in(min(1, u / 0.5))), 2 * min(1, u / 0.5))
    B = zoom_blur(scaled(b, 1 + 0.35 * (1 - ease_out(max(0, (u - 0.5) / 0.5)))), 2 * (1 - max(0, (u - 0.5) / 0.5)))
    return A * (1 - w) + B * w


def render_segment(i, t):
    s = TL[i]
    if s["name"] == "intro":
        f = compose_intro(t)
    elif s["name"] == "outro":
        return compose_outro(t, t - s["t0"])
    else:
        f = READERS[i].render(t)
        if s["name"] == "lights":
            LAST_LIGHTS[0] = f
        f = shot_overlays(f, s, t)
    return f


def frame(t):
    i = max(k for k, s in enumerate(TL) if s["t0"] <= t + 1e-9)
    s = TL[i]
    ttype, td = s["tin"]
    lt = t - s["t0"]
    out = None
    if i + 1 < len(TL):
        ntype, ntd = TL[i + 1]["tin"]
        if ntype in ("fade", "zoom") and t >= TL[i + 1]["t0"] - ntd / 2:
            out = mix_trans(ntype, render_segment(i, t), render_segment(i + 1, t), (t - (TL[i + 1]["t0"] - ntd / 2)) / ntd)
    if out is None and ttype in ("fade", "zoom") and lt < td / 2 and i > 0:
        out = mix_trans(ttype, render_segment(i - 1, t), render_segment(i, t), (t - (s["t0"] - td / 2)) / td)
    if out is None:
        out = render_segment(i, t)
    if ttype == "flash" and lt < td:
        out = gold_flash(out, (1 - lt / td) ** 2 * (1.3 if t - s["t0"] < 0.01 and False else 1.0))
    if i + 1 < len(TL) and TL[i + 1]["tin"][0] == "flash":
        dt = TL[i + 1]["t0"] - t
        if dt < 0.12:
            out = gold_flash(out, (1 - dt / 0.12) * 0.6)
    # subtle brightness lift on kicks
    ke = kick_env(t)
    if ke > 0 and s["name"] not in ("intro", "outro"):
        pass
    for j in (i, i + 1):
        if j >= len(TL) or j == 0:
            continue
        rt, rd = TL[j]["tin"]
        if not rt.startswith("rangoli"):
            continue
        u = (t - (TL[j]["t0"] - rd / 2)) / rd
        if 0 <= u <= 1:
            spr = R1 if rt == "rangoli1" else R2
            if u < 0.5:
                sz, al = 200 + 4000 * ease_in(u / 0.5), smooth(u / 0.15)
            else:
                sz, al = 4200 + 6000 * ease_out((u - 0.5) / 0.5), 1 - smooth((u - 0.5) / 0.5)
            cover = smooth((1 - abs(u - 0.5) / 0.5) * 1.6 - 0.6)
            out = out * (1 - 0.85 * cover) + np.array([30, 5, 45], np.float32) / 255 * 0.85 * cover
            draw_sprite(out, spr, W / 2, H / 2, sz, 120 * u, alpha=al)
    timeline_titles(out, t)
    return out


def to_u8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    if PREVIEW:
        for t in sorted(PREVIEW):
            for r in READERS.values():
                r.close()
                r.tracked = False
                r.tone = Tone()
            cv2.imwrite(f"{OUT}_{t:06.2f}.jpg", to_u8(frame(t)), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("preview", t, flush=True)
        return
    nfr = int(round(TOTAL * FPS))
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-"]
    if AUDIO:
        enc += ["-ss", f"{SONG_A:.3f}", "-t", f"{TOTAL:.3f}", "-i", AUDIO, "-map", "0:v", "-map", "1:a",
                "-af", f"afade=t=in:d=0.05,afade=t=out:st={TOTAL - 1.8:.3f}:d=1.8", "-c:a", "aac", "-b:a", "256k"]
    enc += ["-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-movflags", "+faststart", OUT]
    p = subprocess.Popen(enc, stdin=subprocess.PIPE)
    for n in range(nfr):
        t = n / FPS
        p.stdin.write(to_u8(frame(t)).tobytes())
        for k, r in READERS.items():
            if r.proc is not None and t > TL[k]["t1"] + TL[k]["post"] + 0.05:
                r.close()
        if n % 150 == 0:
            print(f"frame {n}/{nfr}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", OUT)


if __name__ == "__main__":
    main()
