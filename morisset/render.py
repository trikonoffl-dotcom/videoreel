#!/usr/bin/env python3
"""Morisset Mowers & Chainsaws - premium kinetic typography edit.

Pipeline per frame (1080x1920):
  1. source frame (upscaled 720x1280 -> 1080x1920)
  2. BEHIND text layer  : oversized lime 3D type, parallax + perspective
  3. rotoscope matte    : product pixels from the source re-composited OVER the text
  4. FRONT text layer   : supporting white subtitles placed in negative space
  5. logo (top-right) if assets/logo.png is supplied

usage: render.py <scratch> <out.mp4> [--still t1,t2,...] [--chunks N]
"""
import sys, os, json, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 24
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"   # stand-in for SF Pro Display Bold
LIME = (141, 198, 63)          # eyeballed from the supplied logo (~#8DC63F) - swap for an exact sample if needed
LIME_DEEP = tuple(int(c * 0.45) for c in LIME)
WHITE = (255, 255, 255)
LOGO = os.path.join(HERE, "assets", "logo.png")

BEATS = json.load(open(os.path.join(HERE, "beats.json")))["beats"]
BEAT = 60 / 76.0

# ---------------------------------------------------------------- shot plan
# layer: 'behind' (under the rotoscoped product) | 'front'
# y: line centre as fraction of frame height.  fit: line width as fraction of frame width
# words: list of (text, colour) ; t: seconds after shot start (snapped to the beat grid)
def ln(words, times, **kw):
    """words: (text,colour[,size-multiplier]) ; times: absolute seconds each word is SPOKEN (kinetic word-by-word sync)"""
    return dict(words=words, times=times, **kw)
L, Wh = LIME, WHITE
SHOTS = [
 dict(t0=0.00, t1=3.54, mask=False, lines=[       # sky = negative space
   ln([("WHAT",Wh),("IF",Wh),("I",Wh),("TOLD",Wh),("YOU",Wh)], [0.05,.28,.40,.56,.74], layer="front", y=0.075, fit=0.70, shadow_dark=True),
   ln([("BLACK",L),("FRIDAY",L)], [0.96,1.16], layer="front", y=0.150, fit=0.94, ext=True),
   ln([("COULD",Wh),("SAVE",Wh),("YOU",Wh),("UP",Wh),("TO",Wh)], [1.46,1.72,1.94,2.12,2.24], layer="front", y=0.222, fit=0.74, shadow_dark=True),
   ln([("$1,000?",L)], [2.50], layer="front", y=0.300, fit=0.62, ext=True),
 ]),
 dict(t0=3.54, t1=6.50, mask=False, lines=[       # dark glass top
   ln([("THAT'S",Wh),("RIGHT.",L)], [4.04,4.44], layer="front", y=0.072, fit=0.78, x=0.57, end=5.0, shadow_dark=True),
   ln([("MORISSET",L),("MOWERS",L)], [5.20,5.50], layer="front", y=0.072, fit=0.80, x=0.55, ext=True),
   ln([("IS",Wh),("BRINGING",Wh),("YOU",Wh)], [5.74,5.96,6.20], layer="front", y=0.133, fit=0.66, x=0.57, shadow_dark=True),
 ]),
 dict(t0=6.50, t1=9.12, mask=True, mask_front=True, lines=[
   ln([("SOME",Wh,.55),("MASSIVE",L),("DEALS",L)], [6.53,6.62,7.02], layer="behind", y=0.40, fit=0.93, end=7.78, ext=True, alpha=.95),
   ln([("GET",Wh,.5),("$1,000",L,1.7),("OFF",Wh,.5)], [7.88,8.04,8.76], layer="behind", y=0.40, fit=0.92, ext=True, alpha=.95),
 ]),
 dict(t0=9.12, t1=11.40, mask=False, lines=[      # close-up: bottom-left grass only
   ln([("CUB",Wh),("CADET",Wh),("AND",Wh)], [9.14,9.38,9.64], layer="front", y=0.800, fit=0.50, x=0.275, shadow_dark=True),
   ln([("GRAVELY",Wh)], [9.96], layer="front", y=0.853, fit=0.40, x=0.225, shadow_dark=True),
   ln([("ZERO-TURN",Wh),("MOWERS",Wh)], [10.28,10.90], layer="front", y=0.906, fit=0.52, x=0.28, shadow_dark=True),
 ]),
 dict(t0=11.40, t1=13.70, mask=True, mask_front=True, lines=[
   ln([("SAVE",Wh,.5),("$500",L,1.7)], [11.76,11.90], layer="behind", y=0.375, fit=0.92, ext=True, alpha=.95),
   ln([("ON",Wh),("SELECTED",Wh)], [12.50,13.02], layer="front", y=0.875, fit=0.62, shadow_dark=True),
 ]),
 dict(t0=13.70, t1=16.46, mask=True, mask_front=True, lines=[
   ln([("CUB",Wh),("CADET,",Wh),("GRAVELY",Wh)], [13.74,13.78,14.32], layer="front", y=0.775, fit=0.88),
   ln([("AND",Wh),("VICTA",Wh),("RIDE-ON",Wh),("MOWERS",Wh)], [14.62,15.00,15.36,15.70], layer="front", y=0.862, fit=0.80, shadow_dark=True),
 ]),
 dict(t0=16.46, t1=18.50, mask=False, lines=[
   ln([("PLUS,",Wh),("GET",Wh)], [16.62,17.28], layer="front", y=0.775, fit=0.36, x=0.76, shadow_dark=True),
   ln([("10%",L,1.5),("OFF",Wh)], [17.40,17.90], layer="front", y=0.835, fit=0.50, x=0.74, ext=True, shadow_dark=True),
 ]),
 dict(t0=18.50, t1=21.21, mask=False, lines=[
   ln([("SMALL",Wh),("GOODS,",Wh),("MOWERS,",L)], [18.53,18.57,19.12], layer="front", y=0.052, fit=0.86, shadow_dark=True),
   ln([("STRIMMERS",L),("AND",Wh),("CHAINSAWS",L)], [19.74,19.92,20.40], layer="front", y=0.112, fit=0.92, shadow_dark=True),
 ]),
 dict(t0=21.21, t1=25.71, mask=True, mask_front=True, lines=[
   ln([("THE",Wh),("SALE",Wh),("RUNS",Wh),("FROM",Wh)], [21.38,21.50,21.68,22.02], layer="front", y=0.915, fit=0.60, end=22.35, shadow_dark=True),
   ln([("NOV",Wh,.5),("27TH",L,1.5),("TO",Wh,.5),("30TH",L,1.5)], [22.28,22.56,23.30,23.40], layer="behind", y=0.125, fit=0.92, ext=True, alpha=1.0),
   ln([("WITH",Wh),("LIMITED",Wh),("FLOOR",Wh),("STOCK",Wh)], [24.06,24.18,24.58,24.84], layer="front", y=0.905, fit=0.78, shadow_dark=True),
   ln([("AVAILABLE",L)], [25.10], layer="front", y=0.945, fit=0.40, shadow_dark=True),
 ]),
 dict(t0=25.71, t1=27.63, mask=False, lines=[
   ln([("DON'T",Wh),("MISS",Wh),("OUT.",L)], [25.74,26.28,26.40], layer="front", y=0.072, fit=0.78, x=0.57, end=26.95, shadow_dark=True),
   ln([("HEAD",Wh),("TO",Wh),("MORISSET",L)], [27.02,27.12,27.30], layer="front", y=0.072, fit=0.80, x=0.57, shadow_dark=True),
 ]),
 dict(t0=27.63, t1=30.05, mask=True, mask_front=True, lines=[
   ln([("BLACK",L),("FRIDAY",L)], [28.92,29.18], layer="behind", y=0.25, fit=0.92, ext=True, alpha=.95),
   ln([("MOWERS",Wh),("AND",Wh),("GRAB",Wh)], [27.70,27.98,28.38], layer="front", y=0.905, fit=0.74),
   ln([("YOUR",Wh),("SALE.",L,1.25)], [28.66,29.54], layer="front", y=0.958, fit=0.56),
 ]),
]

# ---------------------------------------------------------------- helpers
def ease_out_expo(x): return 1 - 2 ** (-10 * x) if x < 1 else 1.0
def ease_out_back(x, s=1.4):
    x = min(max(x, 0), 1) - 1; return 1 + (s + 1) * x ** 3 + s * x ** 2
def smooth(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)

def snap(t_abs):
    """snap to the nearest half-beat if close, never earlier than requested-0.12"""
    grid = [b + k * BEAT / 2 for b in BEATS[:1] for k in range(-1, 100)]
    best = min(grid, key=lambda g: abs(g - t_abs))
    return best if abs(best - t_abs) < 0.2 else t_abs

_font_cache = {}
def font(sz):
    sz = int(sz)
    if sz not in _font_cache: _font_cache[sz] = ImageFont.truetype(FONT, sz)
    return _font_cache[sz]

def word_width(text, size, tracking):
    f = font(size)
    return sum(f.getlength(c) for c in text) + tracking * size * (len(text) - 1)

_sprite_cache = {}
def sprite(text, size, color, tracking, ext, shadow_dark):
    """premultiplied RGBA float sprite (h,w,4) and its centre. 3D extrusion + glow + shadow baked in."""
    key = (text, int(size), color, round(tracking, 3), ext, shadow_dark)
    if key in _sprite_cache: return _sprite_cache[key]
    f = font(size)
    tw = int(word_width(text, size, tracking)) + 2
    pad = int(size * 0.55)
    asc, desc = f.getmetrics()
    cw, ch = tw + 2 * pad, asc + desc + 2 * pad
    m = Image.new("L", (cw, ch), 0); d = ImageDraw.Draw(m)
    x = pad
    for c in text:
        d.text((x, pad), c, font=f, fill=255); x += f.getlength(c) + tracking * size
    face = np.asarray(m, np.float32) / 255
    # vertical bevel gradient on the face (lighter top, richer bottom)
    yy = np.linspace(0, 1, ch, dtype=np.float32)[:, None]
    col = np.array(color, np.float32)[None, None, :] / 255
    grad = (1.12 - 0.22 * yy)[..., None]
    face_rgb = np.clip(col * grad, 0, 1)
    out = np.zeros((ch, cw, 4), np.float32)
    # extrusion (3D depth) - stacked darker copies down-right
    if ext:
        depth = max(4, int(size * 0.035))
        deep = np.array(LIME_DEEP if color == LIME else tuple(int(c*.45) for c in color), np.float32) / 255
        for i in range(depth, 0, -1):
            sh = np.roll(np.roll(face, i, 0), int(i * .6), 1)
            a = sh[..., None]
            out[..., :3] = out[..., :3] * (1 - a) + deep * a
            out[..., 3:] = out[..., 3:] * (1 - a) + a
    # glow
    glow = cv2.GaussianBlur(face, (0, 0), size * 0.07)
    ga = (glow * (0.55 if color == LIME else 0.28))[..., None]
    gcol = col
    out[..., :3] = out[..., :3] * (1 - ga) + gcol * ga
    out[..., 3:] = np.maximum(out[..., 3:], ga)
    # face
    a = face[..., None]
    out[..., :3] = out[..., :3] * (1 - a) + face_rgb * a
    out[..., 3:] = out[..., 3:] * (1 - a) + a
    # soft contact shadow underneath everything
    sh = cv2.GaussianBlur(np.roll(np.roll(face, int(size * .05), 0), int(size * .02), 1), (0, 0), size * 0.045)
    sa = (sh * (0.62 if shadow_dark else 0.38))[..., None]
    a_out = out[..., 3:]
    out[..., :3] = out[..., :3] + 0 * sa      # shadow is black -> only alpha changes
    out[..., 3:] = a_out + sa * (1 - a_out)
    # premultiply (colour currently straight where alpha>0 for face; shadow adds black)
    prem = out.copy()
    prem[..., :3] = out[..., :3] * a_out        # approximate: colour weighted by its own coverage
    res = (prem, (cw / 2, ch / 2))
    if len(_sprite_cache) > 400: _sprite_cache.clear()
    _sprite_cache[key] = res
    return res

def layout(line):
    """final (settled) layout: list of dict(word,color,xc,size) - supports per-word size multipliers"""
    ws = [(w[0], w[1], w[2] if len(w) > 2 else 1.0) for w in line["words"]]
    gap_em, base = 0.26, 100
    tw = sum(word_width(t, base * k, -0.02) for t, _, k in ws) + gap_em * base * (len(ws) - 1)
    size = base * line["fit"] * W / tw
    total = sum(word_width(t, size * k, -0.02) for t, _, k in ws) + gap_em * size * (len(ws) - 1)
    x = line.get("x", 0.5) * W - total / 2
    out = []
    for t, c, k in ws:
        ww = word_width(t, size * k, -0.02)
        out.append(dict(word=t, color=c, xc=x + ww / 2, size=size * k))
        x += ww + gap_em * size
    return out

def motion_blur(img, length, angle):
    if length < 1.5: return img
    k = int(length) | 1
    ker = np.zeros((k, k), np.float32)
    cv2.line(ker, (0, k // 2), (k - 1, k // 2), 1, 1)
    M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), angle, 1)
    ker = cv2.warpAffine(ker, M, (k, k)); ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(img, -1, ker)

def paste(layer, spr, cx, cy, scale, alpha):
    """add premultiplied sprite onto premultiplied layer, centred at (cx,cy)"""
    if alpha <= 0.002 or scale <= 0.01: return
    if abs(scale - 1) > 0.002:
        spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    h, w = spr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(w, W - x0), min(h, H - y0)
    if sx1 <= sx0 or sy1 <= sy0: return
    dst = layer[y0 + sy0:y0 + sy1, x0 + sx0:x0 + sx1]
    s = spr[sy0:sy1, sx0:sx1] * alpha
    dst[:] = s + dst * (1 - s[..., 3:4])

def persp(layer, shot_u, ax, ay):
    """gentle 3D tilt of the whole text plane about its centre (ax,ay degrees) + parallax drift"""
    f = 1800.0
    rx, ry = math.radians(ax), math.radians(ay)
    cxm, cym = W / 2, H * 0.4
    pts = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    out = []
    for px, py in pts:
        x, y, z = px - cxm, py - cym, 0.0
        y2 = y * math.cos(rx) - z * math.sin(rx); z2 = y * math.sin(rx) + z * math.cos(rx)
        x3 = x * math.cos(ry) + z2 * math.sin(ry); z3 = -x * math.sin(ry) + z2 * math.cos(ry)
        k = f / (f + z3)
        out.append([x3 * k + cxm, y2 * k + cym])
    Hm = cv2.getPerspectiveTransform(pts, np.array(out, np.float32))
    return cv2.warpPerspective(layer, Hm, (W, H), flags=cv2.INTER_LINEAR)

# ---------------------------------------------------------------- masks
_mask_cache = {}
def raw_mask(i):
    if i not in _mask_cache:
        m = cv2.imread(f"{SCR}/mk/{i:04d}.png", 0)
        _mask_cache[i] = m
        if len(_mask_cache) > 12: _mask_cache.pop(next(iter(_mask_cache)))
    return _mask_cache[i]

def matte(i, i0, i1):
    """rotoscope matte for frame i : 3-frame temporal average inside the shot, hole fill, soft edge"""
    idx = [j for j in (i - 1, i, i + 1) if i0 <= j <= i1 and os.path.exists(f"{SCR}/mk/{j:04d}.png")]
    m = np.mean([raw_mask(j).astype(np.float32) for j in idx], 0) / 255
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)
    b = (m > 0.5).astype(np.uint8)
    # keep only substantial blobs, close small gaps, fill interior holes
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros_like(b)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] > 0.012 * W * H: keep[lab == k] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    ff = keep.copy(); cv2.floodFill(ff, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 2)
    filled = (ff != 2).astype(np.float32)
    # slight outward bias so text never rides over the product edge, then feather
    filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    soft = cv2.GaussianBlur(filled, (0, 0), 1.6)
    # preserve real soft detail from the network where it agrees
    out = np.maximum(soft, np.clip((m - .35) / .4, 0, 1) * cv2.dilate(keep, np.ones((9, 9), np.uint8)).astype(np.float32))
    return np.clip(out, 0, 1)

# ---------------------------------------------------------------- frame
SCR = None
_logo = None
def get_logo():
    global _logo
    if _logo is None:
        if os.path.exists(LOGO):
            im = cv2.imread(LOGO, cv2.IMREAD_UNCHANGED)
            if im.shape[2] == 3: im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            lw = 230; s = lw / im.shape[1]
            im = cv2.resize(im, (lw, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            rgb = im[..., [2, 1, 0]].astype(np.float32) / 255; a = im[..., 3:].astype(np.float32) / 255
            _logo = np.concatenate([rgb * a, a], -1)
        else: _logo = False
    return _logo

def render_frame(i):
    t = (i - 1) / FPS
    si = max(k for k, s in enumerate(SHOTS) if t >= s["t0"] - 1e-6)
    shot = SHOTS[si]
    f0 = int(round(shot["t0"] * FPS)) + 1; f1 = min(721, int(round(shot["t1"] * FPS)))
    bg = cv2.imread(f"{SCR}/fr/{i:04d}.jpg").astype(np.float32)[..., ::-1] / 255
    u = (t - shot["t0"]) / (shot["t1"] - shot["t0"])             # shot progress 0..1
    st = t - shot["t0"]
    layers = {"behind": np.zeros((H, W, 4), np.float32), "front": np.zeros((H, W, 4), np.float32)}
    for li, line in enumerate(shot["lines"]):
        lay = layers[line["layer"]]
        lo = layout(line)
        line_end = line.get("end", shot["t1"])
        beat_ph = ((t - BEATS[0]) % BEAT) / BEAT
        pulse = 1 + 0.018 * math.exp(-beat_ph * 6) if line["layer"] == "behind" else 1
        par = (u - .5)                                             # parallax progress
        for wi, w in enumerate(lo):
            tw = max(line["times"][wi], shot["t0"] + 0.03)      # word appears exactly when spoken
            a = (t - tw) / 0.5
            if a <= 0: continue
            e = ease_out_expo(min(a, 1));
            sc = (1.28 - 0.28 * e) if a < 1 else 1.0
            trk = -0.02 + 0.16 * (1 - e)                          # tracking tightens in
            dirx = line.get("dx", 1 if (li + wi) % 2 == 0 else -1)
            slide_x = dirx * 70 * (1 - e)
            slide_y = 38 * (1 - e)
            fade = smooth(a * 1.6)
            # exit: fade & drift 0.22s before the cut
            ex = (line_end - t) / 0.2
            if ex < 1: fade *= smooth(ex); sc *= 1 + 0.04 * (1 - smooth(ex))
            alpha = fade * line.get("alpha", 1.0)
            spr, (scx, scy) = sprite(w["word"], w["size"], w["color"], trk, line.get("ext", False) and w["color"] == LIME, True)
            # recentre on word centre
            drift_x = par * (60 if line["layer"] == "behind" else 24) * (1 if li % 2 == 0 else -1)
            drift_y = par * (26 if line["layer"] == "behind" else 10)
            cx = w["xc"] + slide_x + drift_x; cy = line["y"] * H + slide_y + drift_y
            if a < 1:
                blur_len = (1 - e) * 46
                spr = motion_blur(spr, blur_len, 0)
            paste(lay, spr, cx, cy, sc * pulse, alpha)
    out = bg
    m = matte(i, f0, f1)[..., None] if shot["mask"] else None
    if layers["behind"][..., 3].max() > 0:
        tilt_y = 5.0 * math.sin((u - .5) * 2.2); tilt_x = 2.2 * (u - .5)
        L = persp(layers["behind"], u, tilt_x, tilt_y)
        out = out * (1 - L[..., 3:4]) + L[..., :3]
        if m is not None and not shot.get("mask_front"):
            out = out * (1 - m) + bg * m
    L = layers["front"]
    if L[..., 3].max() > 0:
        L = persp(L, u, 0.8 * (u - .5), 2.0 * (u - .5))
        out = out * (1 - L[..., 3:4]) + L[..., :3]
    if m is not None and shot.get("mask_front"):
        out = out * (1 - m) + bg * m          # product always stays in front of ALL text
    lg = get_logo()
    if lg is not False:
        h, w = lg.shape[:2]; x0, y0 = W - w - 56, 70
        out[y0:y0 + h, x0:x0 + w] = out[y0:y0 + h, x0:x0 + w] * (1 - lg[..., 3:4]) + lg[..., :3]
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)[..., ::-1]

# ---------------------------------------------------------------- driver
def work(args):
    a, b, path = args
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-g", "24", "-bf", "2", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", path], stdin=subprocess.PIPE)
    for i in range(a, b + 1): p.stdin.write(render_frame(i).tobytes())
    p.stdin.close(); p.wait()
    return path

if __name__ == "__main__":
    SCR = sys.argv[1]; out = sys.argv[2]
    if "--still" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")]
        os.makedirs(f"{SCR}/stills", exist_ok=True)
        for t in ts:
            i = min(721, int(t * FPS) + 1); cv2.imwrite(f"{SCR}/stills/{t:05.2f}.jpg", render_frame(i), [cv2.IMWRITE_JPEG_QUALITY, 92])
        sys.exit()
    n = int(sys.argv[sys.argv.index("--chunks") + 1]) if "--chunks" in sys.argv else 4
    per = math.ceil(721 / n)
    jobs = [(k * per + 1, min(721, (k + 1) * per), f"{SCR}/seg{k}.mp4") for k in range(n)]
    with Pool(n) as p: segs = p.map(work, jobs)
    open(f"{SCR}/list.txt", "w").write("".join(f"file '{s}'\n" for s in segs))
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{SCR}/list.txt", "-i", f"{SCR}/mix.wav",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
