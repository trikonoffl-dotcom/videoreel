"""Diwali reel renderer (9:16, 1080x1920, 30 fps).

Cuts the source footage, applies virtual camera moves (crop/zoom/pan/dutch
angle from the 4K source), a warm festive colour grade, Diwali overlays
(bokeh lights, fireworks, sparkles, rangoli), titles and transitions.

Usage:
  python3 render.py SRC.mp4 ASSETS_DIR FONTS_DIR OUT.mp4 [--preview t1,t2,...] [--audio music.wav]
"""
import math
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
SRC_W, SRC_H = 2160, 3840
TOTAL = 61.2

args = sys.argv[1:]
SRC, ASSETS, FONTS, OUT = args[:4]
PREVIEW = None
AUDIO = None
if "--preview" in args:
    PREVIEW = [float(x) for x in args[args.index("--preview") + 1].split(",")]
if "--audio" in args:
    AUDIO = args[args.index("--audio") + 1]

# ------------------------------------------------------------------- EDL
# Each shot: (src_start, src_speed, keyframes[(frac, cx, cy, zoom, rot_deg)], bokeh, label)
# Timeline times are derived from DUR list (beats at 100 BPM = 0.6 s).
SHOTS = [
    # name,     dur, src,    speed, keyframes, bokeh
    ("intro",   4.8, None,   1.0, None, 0.0),
    ("door",    3.0, 0.6,    1.0, [(0, .5, .5, 1.0, 0), (1, .5, .44, 1.28, 0)], 0.45),
    ("banner",  1.8, 5.6,    1.0, [(0, .44, .37, 1.75, -2), (1, .56, .37, 1.6, 1)], 0.35),
    ("paint1",  3.0, 14.0,   1.0, [(0, .5, .6, 1.5, 0), (1, .48, .62, 1.85, 0)], 0.25),
    ("paint2",  2.4, 39.8,   1.0, [(0, .62, .55, 1.65, 3), (1, .64, .55, 1.9, 0)], 0.2),
    ("paint3",  2.4, 61.6,   1.0, [(0, .5, .58, 1.2, 0), (1, .52, .6, 1.45, 0)], 0.2),
    ("paint4",  2.4, 74.2,   1.0, [(0, .36, .6, 1.6, 0), (1, .55, .42, 1.4, 0)], 0.2),
    ("paint5",  1.8, 87.4,   1.0, [(0, .56, .64, 1.8, -3), (1, .56, .62, 1.55, 0)], 0.2),
    ("hero",    4.8, 98.0,   1.0, [(0, .5, .45, 1.0, 0), (1, .5, .43, 1.12, 3)], 0.5),
    ("write1",  2.4, 19.6,   1.0, [(0, .45, .45, 1.3, 0), (1, .42, .4, 1.5, 0)], 0.15),
    ("write2",  2.4, 27.0,   1.0, [(0, .45, .5, 1.05, 0), (1, .45, .48, 1.25, 0)], 0.15),
    ("write3",  2.4, 30.5,   1.0, [(0, .6, .45, 1.3, 0), (1, .62, .38, 1.55, 0)], 0.15),
    ("write4",  2.4, 52.4,   1.0, [(0, .55, .5, 1.0, 0), (1, .55, .47, 1.22, 0)], 0.2),
    ("laugh1",  1.8, 78.9,   1.0, [(0, .55, .35, 1.45, 0), (1, .55, .4, 1.2, 0)], 0.15),
    ("laugh2",  1.8, 82.0,   1.0, [(0, .5, .4, 1.2, -3), (1, .5, .38, 1.4, 0)], 0.15),
    ("write5",  1.8, 108.6,  1.0, [(0, .55, .45, 1.25, 0), (1, .58, .4, 1.45, 0)], 0.15),
    ("smile",   1.8, 114.9,  1.0, [(0, .5, .38, 1.5, 0), (1, .5, .36, 1.72, 0)], 0.15),
    ("write6",  1.2, 119.0,  1.0, [(0, .5, .45, 1.2, 4), (1, .5, .43, 1.3, 4)], 0.15),
    ("write7",  1.2, 125.0,  1.0, [(0, .5, .42, 1.3, 0), (1, .5, .4, 1.4, 0)], 0.15),
    ("funny",   2.4, 112.05, 1.0, [(0, .5, .48, 1.0, 0), (.42, .5, .47, 1.04, 0),
                                   (.62, .5, .36, 1.65, 0), (1, .5, .36, 1.7, 0)], 0.15),
    ("raise",   4.8, 44.9,   0.9, [(0, .5, .52, 1.0, 0), (1, .5, .44, 1.25, 0)], 0.5),
    ("lights",  2.4, 6.0,    1.0, [(0, .5, .4, 1.45, -3), (1, .5, .42, 1.2, 0)], 0.5),
    ("outro",   6.0, None,   1.0, None, 0.0),
]
# transition INTO shot i (between i-1 and i): (type, duration)
TRANS = {
    "door": ("rangoli1", 1.1), "banner": ("zoom", 0.4), "paint1": ("flash", 0.5),
    "paint2": ("cut", 0), "paint3": ("zoom", 0.4), "paint4": ("cut", 0),
    "paint5": ("flash", 0.4), "hero": ("fade", 0.5), "write1": ("rangoli2", 1.1),
    "write2": ("cut", 0), "write3": ("zoom", 0.4), "write4": ("cut", 0),
    "laugh1": ("flash", 0.4), "laugh2": ("cut", 0), "write5": ("cut", 0),
    "smile": ("zoom", 0.4), "write6": ("cut", 0), "write7": ("cut", 0),
    "funny": ("flash", 0.4), "raise": ("flash", 0.7), "lights": ("fade", 0.6),
    "outro": ("rangoli1", 1.1),
}

starts = np.cumsum([0] + [s[1] for s in SHOTS])[:-1]
TL = []
for (name, dur, src, speed, kf, bokeh), st in zip(SHOTS, starts):
    TL.append(dict(name=name, t0=float(st), t1=float(st + dur), src=src, speed=speed, kf=kf, bokeh=bokeh))
for i, s in enumerate(TL):
    tr = TRANS.get(s["name"], ("cut", 0))
    s["tin"] = tr
    # overlap handles for crossfading transitions
    s["pre"] = tr[1] / 2 if tr[0] in ("fade", "zoom") else 0.0
for i, s in enumerate(TL[:-1]):
    nxt = TL[i + 1]
    s["post"] = nxt["pre"]
TL[-1]["post"] = 0.0

# ------------------------------------------------------------------- helpers
def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = min(1.0, max(0.0, x))
    return x ** 3


def interp_kf(kf, f):
    if f <= kf[0][0]:
        return kf[0][1:]
    for a, b in zip(kf, kf[1:]):
        if f <= b[0]:
            u = smooth((f - a[0]) / (b[0] - a[0]))
            return tuple(a[i] + (b[i] - a[i]) * u for i in range(1, 5))
    return kf[-1][1:]


def load_rgba(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    a[:, :, :3] *= a[:, :, 3:4]  # premultiply
    return a


R1 = load_rgba(f"{ASSETS}/rangoli1.png")
R2 = load_rgba(f"{ASSETS}/rangoli2.png")


def draw_sprite(dst, spr, cx, cy, size, angle, alpha=1.0, additive=False):
    """Composite premultiplied BGRA sprite centred at (cx,cy) with given size (px) and rotation."""
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
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    w = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=interp if s >= 1 else cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if s < 0.6:  # pre-filter for heavy minification
        pass
    roi = dst[y0:y1, x0:x1]
    a = w[:, :, 3:4] * alpha
    if additive:
        roi += w[:, :, :3] * alpha
    else:
        roi *= 1 - a
        roi += w[:, :, :3] * alpha


# small, pre-minified rangoli copies for crisp small rendering
def mip(spr, size):
    return cv2.resize(spr, (size, size), interpolation=cv2.INTER_AREA)


R1_small = mip(R1, 340)
R2_small = mip(R2, 300)


def rangoli_for(spr, small, size):
    return small if size < small.shape[0] * 1.2 else spr


# ------------------------------------------------------------------- text
def font(name, size):
    return ImageFont.truetype(f"{FONTS}/{name}", size)


F_SCRIPT = "Great_Vibes.ttf"
F_DECO = "Yatra_One.ttf"
F_SERIF = "Cinzel_wght_700.ttf"
F_SANS = "Poppins_wght_500.ttf"


def make_text(text, fname, size, tracking=0, style="gold", glow=True, maxw=960):
    f = font(fname, size)
    bb = f.getbbox(text)
    est = bb[2] - bb[0] + tracking * (len(text) - 1)
    if est > maxw:
        return make_text(text, fname, int(size * maxw / est), tracking, style, glow, maxw)
    # measure
    if tracking:
        widths = [f.getbbox(c)[2] - f.getbbox(c)[0] if c != " " else size * 0.3 for c in text]
        tw = int(sum(widths) + tracking * (len(text) - 1))
    else:
        bb = f.getbbox(text)
        tw = bb[2] - bb[0]
    pad = int(size * 0.8)
    asc, desc = f.getmetrics()
    th = asc + desc
    img = Image.new("L", (tw + 2 * pad, th + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    if tracking:
        x = pad
        for c, wdt in zip(text, widths):
            if c != " ":
                bb = f.getbbox(c)
                d.text((x - bb[0], pad), c, font=f, fill=255)
            x += wdt + tracking
    else:
        bb = f.getbbox(text)
        d.text((pad - bb[0], pad), text, font=f, fill=255)
    mask = np.asarray(img).astype(np.float32) / 255
    hh, ww = mask.shape
    yy = np.linspace(0, 1, hh)[:, None]
    if style == "gold":
        # vertical gold gradient (BGR)
        top = np.array([170, 240, 255]) / 255
        mid = np.array([30, 190, 255]) / 255
        bot = np.array([10, 120, 215]) / 255
        y1 = np.linspace(0, 1, hh)[:, None]
        g = np.where(y1 < 0.5, top + (mid - top) * (y1 / 0.5), mid + (bot - mid) * ((y1 - 0.5) / 0.5))
        col = np.broadcast_to(g[:, None, :], (hh, ww, 3)).astype(np.float32).copy()
    elif style == "white":
        col = np.ones((hh, ww, 3), np.float32) * np.array([235, 248, 255]) / 255
    else:
        col = np.ones((hh, ww, 3), np.float32) * np.array(style) / 255
    out = np.zeros((hh, ww, 4), np.float32)
    if glow:
        # dark maroon soft shadow + warm outer glow
        sh = cv2.GaussianBlur(mask, (0, 0), size * 0.08)
        sh = np.clip(sh * 2.6, 0, 1)
        shadow_col = np.array([20, 5, 45]) / 255
        out[:, :, :3] = shadow_col * sh[..., None] * 0.85
        out[:, :, 3] = sh * 0.92
        gl = cv2.GaussianBlur(mask, (0, 0), size * 0.25)
        out[:, :, :3] += (np.array([40, 140, 255]) / 255) * gl[..., None] * 0.35
        out[:, :, 3] = np.clip(out[:, :, 3] + gl * 0.15, 0, 1)
    a = mask[..., None]
    out[:, :, :3] = out[:, :, :3] * (1 - a) + col * a
    out[:, :, 3] = out[:, :, 3] * (1 - mask) + mask
    return out


TEXT_CACHE = {}


def text_sprite(key, *a, **k):
    if key not in TEXT_CACHE:
        TEXT_CACHE[key] = make_text(*a, **k)
    return TEXT_CACHE[key]


def draw_text(dst, spr, cx, cy, alpha=1.0, scale=1.0, dy=0.0):
    if alpha <= 0.002:
        return
    if abs(scale - 1) > 1e-3:
        spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
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
    roi += s[:, :, :3] * a


def scrim(dst, cy, hgt, amt):
    if amt <= 0.002:
        return
    y0, y1 = int(max(0, cy - hgt)), int(min(H, cy + hgt))
    yy = (np.arange(y0, y1) - cy) / hgt
    m = (np.cos(np.clip(yy, -1, 1) * np.pi) * 0.5 + 0.5)[:, None, None] * amt
    dst[y0:y1] *= 1 - 0.55 * m


def title_anim(t, t_in, t_out, rise=40, dur_in=0.6, dur_out=0.4):
    """returns (alpha, dy, scale)"""
    if t < t_in or t > t_out:
        return 0, 0, 1
    a_in = ease_out((t - t_in) / dur_in)
    a_out = 1 - smooth((t - (t_out - dur_out)) / dur_out)
    a = min(a_in, a_out)
    return a, (1 - a_in) * rise, 0.94 + 0.06 * a_in


# ------------------------------------------------------------------- particles
rng = np.random.default_rng(11)
NB = 46
BOKEH = dict(
    x=rng.uniform(0, 1, NB), y=rng.uniform(0, 1, NB), r=rng.uniform(6, 26, NB),
    vy=rng.uniform(0.015, 0.05, NB), ph=rng.uniform(0, 6.28, NB), fr=rng.uniform(0.5, 1.6, NB),
    c=rng.choice(4, NB),
)
BCOL = np.array([[60, 190, 255], [40, 140, 255], [120, 90, 255], [150, 230, 255]], np.float32) / 255
HW, HH = W // 2, H // 2
EDGE = None


def edge_mask():
    yy = np.linspace(-1, 1, HH)[:, None]
    xx = np.linspace(-1, 1, HW)[None, :]
    m = np.clip(np.maximum(np.abs(yy) * 1.15 - 0.35, 0) + np.abs(xx) * 0.5, 0, 1)
    return (0.25 + 0.75 * m)[..., None].astype(np.float32)


EDGE = edge_mask()


def bokeh_layer(t, strength, edge=True):
    lay = np.zeros((HH, HW, 3), np.float32)
    if strength <= 0:
        return None
    for i in range(NB):
        y = (BOKEH["y"][i] - t * BOKEH["vy"][i]) % 1.2 - 0.1
        x = BOKEH["x"][i] + 0.02 * math.sin(t * 0.7 + BOKEH["ph"][i])
        tw = 0.55 + 0.45 * math.sin(t * BOKEH["fr"][i] * 3 + BOKEH["ph"][i])
        col = BCOL[BOKEH["c"][i]] * tw
        cv2.circle(lay, (int(x * HW), int(y * HH)), int(BOKEH["r"][i]), col.tolist(), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 2.2)
    if edge:
        lay *= EDGE
    return lay * strength


# fireworks: (t0, x, y, hue BGR, n)
def make_fw(t0, x, y, col, n=70, spd=260):
    r = np.random.default_rng(int(t0 * 1000 + x * 7))
    ang = r.uniform(0, 2 * np.pi, n)
    sp = spd * (0.55 + 0.45 * r.uniform(0, 1, n))
    return dict(t0=t0, x=x, y=y, col=np.array(col, np.float32) / 255, vx=np.cos(ang) * sp,
                vy=np.sin(ang) * sp, tw=r.uniform(0, 6, n))


FW_COLS = [(80, 200, 255), (60, 120, 255), (200, 120, 255), (120, 255, 200), (255, 200, 120), (180, 255, 255)]
FIREWORKS = []
for (t0, x, y, c) in [(0.3, .25, .18, 0), (0.9, .78, .12, 2), (1.6, .55, .08, 5), (2.4, .2, .3, 1), (3.1, .82, .28, 3),
                      (48.05, .3, .14, 0), (48.4, .75, .1, 5), (49.0, .55, .2, 2), (49.7, .2, .08, 1),
                      (50.5, .8, .18, 4), (51.3, .45, .06, 0), (52.0, .25, .2, 3),
                      (55.4, .2, .12, 5), (56.2, .8, .1, 0), (57.1, .5, .07, 2), (58.0, .22, .26, 1),
                      (58.8, .78, .3, 3), (59.6, .5, .14, 0)]:
    FIREWORKS.append(make_fw(t0, x * W, y * H, FW_COLS[c]))


def fireworks_layer(t):
    lay = None
    for fw in FIREWORKS:
        a = t - fw["t0"]
        if a < 0 or a > 1.8:
            continue
        if lay is None:
            lay = np.zeros((HH, HW, 3), np.float32)
        fade = (1 - a / 1.8) ** 1.6
        drag = (1 - math.exp(-2.2 * a)) / 2.2
        g = 60
        for k, a2 in enumerate((a, max(0, a - 0.05), max(0, a - 0.1))):
            d2 = (1 - math.exp(-2.2 * a2)) / 2.2
            px = (fw["x"] + fw["vx"] * d2) / 2
            py = (fw["y"] + fw["vy"] * d2 + g * a2 * a2) / 2
            if k == 0:
                prev = (px, py)
            else:
                pass
        px0 = (fw["x"] + fw["vx"] * drag) / 2
        py0 = (fw["y"] + fw["vy"] * drag + g * a * a) / 2
        a2 = max(0, a - 0.09)
        d2 = (1 - math.exp(-2.2 * a2)) / 2.2
        px1 = (fw["x"] + fw["vx"] * d2) / 2
        py1 = (fw["y"] + fw["vy"] * d2 + g * a2 * a2) / 2
        tw = 0.6 + 0.4 * np.sin(fw["tw"] + a * 25)
        for j in range(len(px0)):
            c = (fw["col"] * fade * tw[j]).tolist()
            cv2.line(lay, (int(px1[j]), int(py1[j])), (int(px0[j]), int(py0[j])), c, 2, cv2.LINE_AA)
        if a < 0.15:  # initial flash
            cv2.circle(lay, (int(fw["x"] / 2), int(fw["y"] / 2)), int(30 * (1 - a / 0.15)) + 1,
                       (0.8, 0.95, 1.0), -1, cv2.LINE_AA)
    if lay is None:
        return None
    glow = cv2.GaussianBlur(lay, (0, 0), 6)
    return lay * 1.3 + glow * 1.6


# sparkles (4-point stars)
def star_sprite(size=48):
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    core = np.exp(-r ** 2 * 60)
    rays = np.exp(-np.abs(xx) * 40) * np.exp(-np.abs(yy) * 3) + np.exp(-np.abs(yy) * 40) * np.exp(-np.abs(xx) * 3)
    s = np.clip(core + rays * 0.8, 0, 1).astype(np.float32)
    return s


STAR = star_sprite(64)


def sparkles(dst, t, region, n, seed, strength=1.0, size=40):
    """region = (x0,y0,x1,y1) in px"""
    r = np.random.default_rng(seed)
    xs = r.uniform(region[0], region[2], n)
    ys = r.uniform(region[1], region[3], n)
    ph = r.uniform(0, 6.28, n)
    fr = r.uniform(1.5, 3.5, n)
    sz = r.uniform(0.5, 1.2, n)
    for i in range(n):
        v = math.sin(t * fr[i] + ph[i])
        if v <= 0:
            continue
        s = int(size * sz[i] * (0.5 + 0.5 * v))
        if s < 4:
            continue
        sp = cv2.resize(STAR, (s, s))
        x0, y0 = int(xs[i] - s / 2), int(ys[i] - s / 2)
        if x0 < 0 or y0 < 0 or x0 + s > W or y0 + s > H:
            continue
        dst[y0:y0 + s, x0:x0 + s] += sp[..., None] * np.array([0.75, 0.95, 1.0], np.float32) * v * strength


# ------------------------------------------------------------------- grading
def build_luts():
    x = np.arange(256) / 255.0
    s = x + 0.10 * np.sin(2 * np.pi * x) * -0.5 * 0  # placeholder
    # gentle S-curve
    sc = x + 0.08 * np.sin(np.pi * (2 * x - 1)) * -1 * -1
    sc = np.clip(0.5 + (x - 0.5) * 1.08 + 0.05 * np.sin(2 * np.pi * (x - 0.5)) * 0, 0, 1)
    sc = np.clip(sc ** 1.0, 0, 1)
    # S curve via smoothstep blend
    sm = x * x * (3 - 2 * x)
    curve = 0.72 * x + 0.28 * sm
    # warm split-tone: shadows toward deep maroon, highlights toward gold
    sh = (1 - x) ** 2
    hi = x ** 2
    b = np.clip(curve - 0.035 * hi + 0.025 * sh * 0.6, 0, 1)
    g = np.clip(curve + 0.006 * hi - 0.006 * sh, 0, 1)
    r = np.clip(curve + 0.05 * hi * (1 - x) * 3 * 0.5 + 0.02 * sh, 0, 1)
    return [(c * 255).astype(np.uint8) for c in (b, g, r)]


LUTS = build_luts()
VIG = None


def vignette():
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :] * 0.9
    d = np.sqrt(xx ** 2 + (yy * 0.9) ** 2)
    return (1 - 0.38 * np.clip((d - 0.55) / 0.85, 0, 1) ** 1.6)[..., None].astype(np.float32)


VIG = vignette()


def grade(img_u8, gamma):
    if abs(gamma - 1) > 0.01:
        lut = (np.clip((np.arange(256) / 255.0) ** gamma, 0, 1) * 255).astype(np.uint8)
        img_u8 = cv2.LUT(img_u8, lut)
    b, g, r = cv2.split(img_u8)
    img_u8 = cv2.merge([cv2.LUT(b, LUTS[0]), cv2.LUT(g, LUTS[1]), cv2.LUT(r, LUTS[2])])
    # saturation / vibrance in HSV
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255
    s = np.clip(s + 0.30 * s * (1 - s) * 2.2 + 0.04, 0, 1)
    hsv[:, :, 1] = s * 255
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    f = img_u8.astype(np.float32) / 255
    # bloom
    small = cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.7, 0, 1)
    br = cv2.GaussianBlur(br, (0, 0), 6)
    f += cv2.resize(br, (W, H)) * np.array([0.55, 0.8, 1.0], np.float32) * 0.9
    f *= VIG
    return f


# ------------------------------------------------------------------- shot reader
class Shot:
    def __init__(self, s):
        self.s = s
        self.proc = None
        self.idx = -1
        self.cur = None
        self.gamma = 1.0
        self.fr_src_t0 = None

    def open(self):
        s = self.s
        t_lo = s["t0"] - s["pre"]
        t_hi = s["t1"] + s["post"]
        src0 = s["src"] - s["pre"] * s["speed"]
        dur = (t_hi - t_lo) * s["speed"] + 0.2
        self.src0 = src0
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0, src0):.3f}", "-i", SRC, "-t", f"{dur:.3f}",
               "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=SRC_W * SRC_H * 3 * 2)
        self.idx = -1
        self.t_lo = t_lo

    def frame_at(self, t):
        """source frame for timeline time t (monotonic)."""
        if self.proc is None:
            self.open()
        want = int(math.floor((t - self.t_lo) * self.s["speed"] * FPS + 1e-6))
        while self.idx < want:
            b = self.proc.stdout.read(SRC_W * SRC_H * 3)
            if len(b) < SRC_W * SRC_H * 3:
                break
            self.cur = np.frombuffer(b, np.uint8).reshape(SRC_H, SRC_W, 3)
            self.idx += 1
            if self.idx == 0:
                lum = cv2.cvtColor(cv2.resize(self.cur, (216, 384)), cv2.COLOR_BGR2GRAY).mean() / 255
                target = 0.47
                self.gamma = float(np.clip(math.log(target) / math.log(max(lum, 0.05)), 0.75, 1.15))
        return self.cur

    def render(self, t):
        s = self.s
        src = self.frame_at(t)
        frac = (t - s["t0"]) / (s["t1"] - s["t0"])
        cx, cy, z, rot = interp_kf(s["kf"], frac)
        # extra dutch-angle needs extra zoom so corners stay covered
        rr = math.radians(abs(rot))
        z_cover = (math.cos(rr) + (H / W) * math.sin(rr)) if rot else 1.0
        z = max(z, 1.0) * z_cover
        # output px per source px
        k = z * W / SRC_W
        # clamp centre so crop stays inside
        hw, hh = 0.5 / z, 0.5 / z
        cx = min(max(cx, hw), 1 - hw)
        cy = min(max(cy, hh), 1 - hh)
        if k < 0.98:
            # pre-shrink with area filter for clean downscale
            sw, shh = int(round(SRC_W * k)), int(round(SRC_H * k))
            srcs = cv2.resize(src, (sw, shh), interpolation=cv2.INTER_AREA)
            k2 = 1.0
            ccx, ccy = cx * sw, cy * shh
        else:
            srcs = src
            k2 = k
            ccx, ccy = cx * SRC_W, cy * SRC_H
        M = cv2.getRotationMatrix2D((ccx, ccy), rot, k2)
        M[0, 2] += W / 2 - ccx
        M[1, 2] += H / 2 - ccy
        out = cv2.warpAffine(srcs, M, (W, H), flags=cv2.INTER_LINEAR if k2 < 1.3 else cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT)
        return grade(out, self.gamma)

    def close(self):
        if self.proc:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc = None


# ------------------------------------------------------------------- backgrounds for intro/outro
def festive_bg(t, seed=0):
    yy = np.linspace(0, 1, H)[:, None, None]
    top = np.array([40, 8, 55], np.float32) / 255     # deep maroon (BGR)
    bot = np.array([70, 10, 30], np.float32) / 255     # deep purple-blue
    bg = top + (bot - top) * yy
    bg = np.broadcast_to(bg, (H, W, 3)).copy()
    # warm radial glow behind centre
    xx = np.linspace(-1, 1, W)[None, :]
    yy2 = np.linspace(-1, 1, H)[:, None]
    d = np.sqrt(xx ** 2 + ((yy2 + 0.25) * 0.6) ** 2)
    glow = np.exp(-d ** 2 * 2.2)[..., None] * np.array([30, 110, 230], np.float32) / 255 * 0.55
    bg += glow * (0.85 + 0.15 * math.sin(t * 2))
    return bg


BG_CACHE = {}


def get_bg(t):
    key = int(t * 6)
    if key not in BG_CACHE:
        BG_CACHE.clear()
        BG_CACHE[key] = festive_bg(key / 6)
    return BG_CACHE[key].copy()


def outro_bg_video(t, lights_frame):
    """Blurred, darkened, warm frame of the lights as outro backdrop."""
    if lights_frame is None:
        return get_bg(t)
    f = cv2.resize(lights_frame, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    f = cv2.GaussianBlur(f, (0, 0), 3)
    f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
    return f * 0.35 + get_bg(t) * 0.65


# ------------------------------------------------------------------- per-frame compose
LAST_LIGHTS = [None]


def compose_intro(t):
    f = get_bg(t)
    # big faint rangoli2 rotating behind
    draw_sprite(f, R2, W / 2, H * 0.40, 1900, t * 6, alpha=0.10)
    fw = fireworks_layer(t)
    if fw is not None:
        f += cv2.resize(fw, (W, H)) * 0.9
    bk = bokeh_layer(t, 0.9, edge=False)
    f += cv2.resize(bk, (W, H))
    # main rangoli: spins in
    a = ease_out(t / 1.2)
    size = 760 * (0.2 + 0.8 * a)
    draw_sprite(f, rangoli_for(R1, R1_small, size), W / 2, H * 0.36, size + 8, -200 * (1 - a) - t * 12, alpha=0.35 * a,
                additive=True)  # glow halo
    draw_sprite(f, rangoli_for(R1, R1_small, size), W / 2, H * 0.36, size, -200 * (1 - a) - t * 12, alpha=min(1, a * 1.5))
    # titles
    al, dy, sc = title_anim(t, 1.0, 4.4)
    draw_text(f, text_sprite("hd", "Happy Diwali", F_SCRIPT, 210), W / 2, H * 0.635, al, sc, dy)
    al, dy, sc = title_anim(t, 1.6, 4.4, rise=25)
    draw_text(f, text_sprite("tt", "TRIKON TELESOFT", F_SERIF, 64, tracking=14), W / 2, H * 0.735, al, sc, dy)
    al, dy, sc = title_anim(t, 2.1, 4.4, rise=20)
    draw_text(f, text_sprite("fol", "Festival of Lights", F_SANS, 46, tracking=4, style="white"), W / 2, H * 0.79, al * 0.9,
              sc, dy)
    sparkles(f, t, (120, 1100, W - 120, 1600), 14, 3, 0.9 * smooth((t - 1.0) / 0.5), 56)
    return f


def compose_outro(t, lt):
    f = outro_bg_video(t, LAST_LIGHTS[0])
    fw = fireworks_layer(t)
    if fw is not None:
        f += cv2.resize(fw, (W, H)) * 0.8
    f += cv2.resize(bokeh_layer(t, 0.8, edge=False), (W, H))
    a = ease_out(lt / 1.0)
    size = 700 * (0.3 + 0.7 * a)
    spr = rangoli_for(R2, R2_small, size)
    draw_sprite(f, R1, W / 2, H * 0.33, 1500 * (0.6 + 0.4 * a), lt * 8, alpha=0.18 * a)
    draw_sprite(f, spr, W / 2, H * 0.33, size + 30, -lt * 10, alpha=0.4 * a, additive=True)
    draw_sprite(f, spr, W / 2, H * 0.33, size, -lt * 10, alpha=min(1, a * 1.4))
    al, dy, sc = title_anim(lt, 0.5, 5.6, dur_out=0.5)
    draw_text(f, text_sprite("hd2", "Happy Diwali", F_SCRIPT, 200), W / 2, H * 0.6, al, sc, dy)
    al, dy, sc = title_anim(lt, 1.2, 5.6, rise=20, dur_out=0.5)
    draw_text(f, text_sprite("w1", "May the festival of lights fill", F_SANS, 44, style="white"), W / 2, H * 0.685,
              al * 0.95, sc, dy)
    draw_text(f, text_sprite("w2", "your life with joy & prosperity", F_SANS, 44, style="white"), W / 2, H * 0.715,
              al * 0.95, sc, dy)
    al, dy, sc = title_anim(lt, 1.9, 5.6, rise=20, dur_out=0.5)
    draw_text(f, text_sprite("tt2", "TEAM TRIKON TELESOFT", F_SERIF, 54, tracking=10), W / 2, H * 0.79, al, sc, dy)
    sparkles(f, t, (100, 300, W - 100, 1000), 12, 9, 0.9 * smooth((lt - 0.4) / 0.5), 60)
    # fade to black at end
    end = 1 - smooth((lt - 5.2) / 0.8)
    return f * end


def shot_overlays(f, s, t):
    lt = t - s["t0"]
    name = s["name"]
    bk = bokeh_layer(t, s["bokeh"])
    if bk is not None:
        f += cv2.resize(bk, (W, H))
    fw = fireworks_layer(t)
    if fw is not None:
        f += cv2.resize(fw, (W, H)) * 0.85
    # chapter titles
    if name == "paint1":
        al, dy, sc = title_anim(lt, 0.15, 2.9)
        draw_sprite(f, R2_small, W - 40, 120, 420, lt * 20, alpha=0.95 * al)
        scrim(f, H * 0.23, 260, al)
        draw_text(f, text_sprite("c1a", "Diya Painting", F_DECO, 120), W / 2, H * 0.2, al, sc, dy)
        al2, dy2, sc2 = title_anim(lt, 0.5, 2.9, rise=20)
        draw_text(f, text_sprite("c1b", "colours of light", F_SCRIPT, 110), W / 2, H * 0.27, al2, sc2, dy2)
        sparkles(f, t, (150, 250, W - 150, 600), 8, 21, al, 48)
    if name == "hero":
        # golden glow around the finished diya + sparkles
        sparkles(f, t, (200, 350, W - 200, 1250), 14, 33, smooth(lt / 0.6), 64)
        al, dy, sc = title_anim(lt, 0.8, 4.8 + 0.3)
        scrim(f, H * 0.8, 220, al)
        draw_text(f, text_sprite("hl", "handcrafted with love", F_SCRIPT, 128), W / 2, H * 0.80, al, sc, dy)
    if name == "write1":
        al, dy, sc = title_anim(lt, 0.4, 2.35)
        draw_sprite(f, R1_small, 40, 120, 420, -lt * 20, alpha=0.95 * al)
        scrim(f, H * 0.23, 260, al)
        draw_text(f, text_sprite("c2a", "Creative Challenge", F_DECO, 104), W / 2, H * 0.2, al, sc, dy)
        al2, dy2, sc2 = title_anim(lt, 0.7, 2.35, rise=20)
        draw_text(f, text_sprite("c2b", "pens out, ideas on", F_SCRIPT, 110), W / 2, H * 0.27, al2, sc2, dy2)
        sparkles(f, t, (150, 250, W - 150, 600), 8, 41, al, 48)
    if name == "raise":
        sparkles(f, t, (500, 150, W - 60, 900), 16, 51, smooth(lt / 0.5), 70)
        al, dy, sc = title_anim(lt, 0.9, 4.9)
        scrim(f, H * 0.8, 200, al)
        draw_text(f, text_sprite("sb", "Shine Bright!", F_DECO, 130), W / 2, H * 0.80, al, sc, dy)
    if name == "door":
        al, dy, sc = title_anim(lt, 0.9, 3.0 + 1.8)
    if name in ("door", "banner"):
        T = t - TL[1]["t0"]
        al, dy, sc = title_anim(T, 0.8, 4.7)
        scrim(f, H * 0.83, 220, al)
        draw_text(f, text_sprite("cap1", "our office, all lit up", F_SCRIPT, 124), W / 2, H * 0.83, al, sc, dy)
    return f


def gold_flash(f, amt):
    if amt <= 0:
        return f
    yy = np.linspace(-1, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    leak = np.exp(-(xx ** 2 + yy ** 2) * 0.8)[..., None] * np.array([0.6, 0.85, 1.0], np.float32)
    return f + leak * amt * 1.6 + amt * 0.25


FLASH_LEAK = None


def zoom_blur(img, strength):
    if strength < 0.01:
        return img
    acc = img.copy()
    n = 5
    for i in range(1, n):
        s = 1 + strength * 0.08 * i
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        acc += cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    return acc / n


def scaled(img, s):
    if abs(s - 1) < 1e-3:
        return img
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def render_segment(i, t):
    """Fully composed frame of TL[i] at time t (may be outside its nominal range by handles)."""
    s = TL[i]
    if s["name"] == "intro":
        return compose_intro(t)
    if s["name"] == "outro":
        return compose_outro(t, t - s["t0"])
    f = READERS[i].render(t)
    if s["name"] == "lights":
        LAST_LIGHTS[0] = f
    return shot_overlays(f, s, t)


READERS = {i: Shot(s) for i, s in enumerate(TL) if s["src"] is not None}


def frame(t):
    # active segment
    i = max(k for k, s in enumerate(TL) if s["t0"] <= t + 1e-9)
    s = TL[i]
    out = None
    # incoming transition of s
    ttype, td = s["tin"]
    lt = t - s["t0"]
    # outgoing handle: next segment's transition may need us
    if i + 1 < len(TL):
        nxt = TL[i + 1]
        ntype, ntd = nxt["tin"]
        if ntype in ("fade", "zoom") and t >= nxt["t0"] - ntd / 2:
            u = (t - (nxt["t0"] - ntd / 2)) / ntd  # 0..0.5 here
            a = render_segment(i, t)
            b = render_segment(i + 1, t)
            return mix_trans(ntype, a, b, u)
    if ttype in ("fade", "zoom") and lt < td / 2 and i > 0:
        u = (t - (s["t0"] - td / 2)) / td  # 0.5..1
        a = render_segment(i - 1, t)
        b = render_segment(i, t)
        return mix_trans(ttype, a, b, u)
    out = render_segment(i, t)
    if ttype == "flash" and lt < td:
        amt = (1 - lt / td) ** 2
        out = gold_flash(out, amt)
    # flash also slightly before the cut
    if i + 1 < len(TL) and TL[i + 1]["tin"][0] == "flash":
        dt = TL[i + 1]["t0"] - t
        if dt < 0.12:
            out = gold_flash(out, (1 - dt / 0.12) * 0.6)
    # rangoli wipes
    for j in (i, i + 1):
        if j >= len(TL) or j == 0:
            continue
        rt, rd = TL[j]["tin"]
        if not rt.startswith("rangoli"):
            continue
        T = TL[j]["t0"]
        u = (t - (T - rd / 2)) / rd
        if 0 <= u <= 1:
            spr = R1 if rt == "rangoli1" else R2
            if u < 0.5:
                sz = 200 + (4200 - 200) * ease_in(u / 0.5)
                al = smooth(u / 0.15)
            else:
                sz = 4200 + 6000 * ease_out((u - 0.5) / 0.5)
                al = 1 - smooth((u - 0.5) / 0.5)
            ang = 120 * u
            # dark tint behind at peak so gaps are covered
            cover = 1 - abs(u - 0.5) / 0.5
            cover = smooth(cover * 1.6 - 0.6)
            out = out * (1 - 0.85 * cover) + np.array([30, 5, 45], np.float32) / 255 * 0.85 * cover
            draw_sprite(out, spr, W / 2, H / 2, sz, ang, alpha=al)
    return out


def mix_trans(kind, a, b, u):
    if kind == "fade":
        w = smooth(u)
        return a * (1 - w) + b * w
    if kind == "zoom":
        # outgoing pushes in with zoom blur, incoming starts pushed and settles
        w = smooth((u - 0.35) / 0.3)
        sa = 1 + 0.35 * ease_in(min(1, u / 0.5))
        sb = 1 + 0.35 * (1 - ease_out(max(0, (u - 0.5) / 0.5)))
        A = zoom_blur(scaled(a, sa), 2 * min(1, u / 0.5))
        B = zoom_blur(scaled(b, sb), 2 * (1 - max(0, (u - 0.5) / 0.5)))
        return A * (1 - w) + B * w
    return b


def to_u8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    if PREVIEW:
        # preview frames: process the whole timeline lazily only up to each time (readers need monotonic time)
        for t in sorted(PREVIEW):
            for r in READERS.values():
                r.close()
            img = to_u8(frame(t))
            cv2.imwrite(f"{OUT}_{t:06.2f}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print("preview", t, flush=True)
        return
    nfr = int(round(TOTAL * FPS))
    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-"]
    if AUDIO:
        enc += ["-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    enc += ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-movflags", "+faststart", OUT]
    p = subprocess.Popen(enc, stdin=subprocess.PIPE)
    active = set()
    for n in range(nfr):
        t = n / FPS
        img = to_u8(frame(t))
        p.stdin.write(img.tobytes())
        # close readers that are done
        for k, r in READERS.items():
            if r.proc is not None and t > TL[k]["t1"] + TL[k]["post"] + 0.05:
                r.close()
        if n % 60 == 0:
            print(f"frame {n}/{nfr}", flush=True)
    p.stdin.close()
    p.wait()
    print("done", OUT)


if __name__ == "__main__":
    main()
