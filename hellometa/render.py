"""HelloMeta.ai brand film renderer (1080x1920, 30 fps).

  python3 render.py --stills 1.0,5.2,12.5        # QC stills -> build/still_*.jpg
  python3 render.py --jobs 4                      # full render -> ../output/hellometa_brand_film_9x16.mp4

Needs build/ assets from assets.py, audio from audio.py and data/faces.json from faces.py.
"""
import argparse, json, os, subprocess, sys, time
import cv2, numpy as np
import timeline as T
import webrects as WR
from core import *
from core import _font

cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
B = lambda *p: os.path.join(HERE, 'build', *p)
FPS = T.FPS

# ================================================================== shared assets (lazy)
_A = {}


def asset(key):
    if key in _A:
        return _A[key]
    if key == 'raw':
        path = B('raw_frames.npy')
        if not os.path.exists(path):
            cap = cv2.VideoCapture(os.path.join(HERE, 'input', 'Raw video.mp4'))
            fr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                fr.append(f)
            np.save(path, np.stack(fr))
        v = np.load(path, mmap_mode='r')
    elif key == 'faces':
        v = np.array(json.load(open(os.path.join(HERE, 'data', 'faces.json'))), np.float32)
    elif key == 'words':
        v = json.load(open(os.path.join(HERE, 'data', 'subs_words.json')))
    elif key == 'map':
        v = Pyr(cv2.imread(B('map_dark.png')).astype(np.float32) / 255)
    elif key in ('mask_ph', 'mask_au'):
        v = Pyr(cv2.imread(os.path.join(HERE, 'assets', key + '.png'), 0).astype(np.float32) / 255)
    elif key == 'logo':  # the supplied HelloMeta logo, used exactly as provided
        v = cv2.imread(os.path.join(HERE, 'assets', 'HelloMeta-Logo-1.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    elif key.startswith('logo'):
        v = cv2.imread(B(key + '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    elif key == 'land_meta':
        v = json.load(open(B('land_meta.json')))
    elif key.startswith('land_'):
        v = cv2.imread(B(key + '.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    elif key.startswith('web:'):
        k = key[4:]
        v = WR.crop(k).astype(np.float32) / 255
    elif key == 'techbg':
        v = make_tech_bg()
    else:
        raise KeyError(key)
    _A[key] = v
    return v


def web_card(k, radius=26):
    key = 'card:' + k
    if key not in _A:
        img = asset('web:' + k)
        _A[key] = to_rgba(img, radius * img.shape[1] / 1000)
    return _A[key]


# ================================================================== backgrounds
def make_tech_bg():
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), np.float32) + np.array([0.018, 0.018, 0.02], np.float32)
    d1 = np.sqrt(((xs - 540) / 900) ** 2 + ((ys - 620) / 1000) ** 2)
    bg += np.exp(-d1 ** 2 * 2.2)[..., None] * ORANGE * 0.16
    d2 = np.sqrt(((xs - 900) / 700) ** 2 + ((ys - 1750) / 600) ** 2)
    bg += np.exp(-d2 ** 2 * 2.0)[..., None] * ORANGE * 0.08
    return bg


def grid_layer(t):
    """perspective floor grid that scrolls toward camera + faint HUD lines"""
    g = np.zeros((H, W), np.float32)
    vx, vy = 540, 980
    for k in range(-14, 15):
        x_far = vx + k * 18
        x_near = vx + k * 260
        cv2.line(g, (int(x_far), vy), (int(x_near), H), 0.5, 1, cv2.LINE_AA)
    off = (t * 0.6) % 1.0
    for i in range(14):
        z = (i + 1 - off)
        y = vy + 940 / z ** 1.0 * 0.9
        if vy < y < H:
            cv2.line(g, (0, int(y)), (W, int(y)), 0.5 * min(1, (y - vy) / 300), 1, cv2.LINE_AA)
    fade = np.clip((np.arange(H) - vy) / 700, 0, 1)[:, None]
    return g * fade


_particles = np.random.default_rng(5).random((70, 4)).astype(np.float32)


def particles(dst, t, color=ORANGE, amt=1.0, speed=40):
    for x, y, s, ph in _particles:
        px = x * W + 20 * np.sin(t * 0.7 + ph * 6)
        py = (y * H - t * speed * (0.5 + s)) % H
        r = 1.2 + 2.2 * s
        a = (0.25 + 0.5 * s) * (0.6 + 0.4 * np.sin(t * 3 + ph * 9)) * amt
        x0, y0 = int(px) - 8, int(py) - 8
        if 0 <= x0 < W - 16 and 0 <= y0 < H - 16:
            yy, xx = np.mgrid[0:16, 0:16]
            blob = np.exp(-((xx - (px - x0)) ** 2 + (yy - (py - y0)) ** 2) / (2 * r * r)).astype(np.float32)
            dst[y0:y0 + 16, x0:x0 + 16] += blob[..., None] * color * a


def tech_bg(t, grid=True):
    f = asset('techbg').copy()
    if grid:
        g = grid_layer(t)
        f += g[..., None] * (GREY * 0.25 + ORANGE * 0.10)
    particles(f, t, amt=0.7)
    return f


# ================================================================== speaker
def speaker_shots():
    """list of (t0, t1, zoom_class) for the visible interview pieces"""
    cuts = [T.E(te) for te, _ in T.seg_bounds_edit()][1:]
    vis = [(T.SPEECH_OFF, T.OUTRO)]
    for _, a, b in T.BROLL:
        nv = []
        for s, e in vis:
            if b <= s or a >= e:
                nv.append((s, e)); continue
            if a > s: nv.append((s, a))
            if b < e: nv.append((b, e))
        vis = nv
    pieces = []
    for s, e in vis:
        pts = [s] + [c for c in cuts if s + 0.35 < c < e - 0.35] + [e]
        for a, b in zip(pts[:-1], pts[1:]):
            pieces.append((a, b))
    shots = []
    for i, (a, b) in enumerate(pieces):
        shots.append((a, b, i % 2))
    return shots


SHOTS = speaker_shots()


def src_time(t):
    return T.edit_to_source(t - T.SPEECH_OFF)


def speaker(t, fi):
    raw = asset('raw'); faces = asset('faces')
    ts = src_time(t)
    k = int(np.clip(round(ts * FPS), 0, len(raw) - 1))
    frame = raw[k]
    shot = next((s for s in SHOTS if s[0] <= t < s[1] + 1e-6), SHOTS[-1] if t >= SHOTS[-1][0] else SHOTS[0])
    a, b, cls = shot
    u = lin(t, a, b)
    zoom = (1.035 if cls == 0 else 1.24) + 0.035 * ease_io(u)
    # kinetic punch-in
    dark = 0.0
    for _, ka, kb in T.KINETIC:
        if ka - 0.05 <= t <= kb + 0.1:
            p = ease_expo_out(lin(t, ka - 0.05, ka + 0.25)) * (1 - ease_io(lin(t, kb - 0.05, kb + 0.12)))
            zoom += 0.10 * p
            dark = max(dark, 0.58 * p)
    # framing: follow the smoothed face, blended with the shot's mean so the camera does not wobble
    k0, k1 = int(src_time(a) * FPS), int(src_time(max(a, b - 0.01)) * FPS)
    k0, k1 = np.clip([k0, k1], 0, len(faces) - 1)
    mean = faces[min(k0, k1):max(k0, k1) + 1].mean(0)
    fx, fy = 0.4 * faces[k] [:2] + 0.6 * mean[:2]
    sh_, sw_ = frame.shape[:2]
    cw, ch = sw_ / zoom, sh_ / zoom
    x0 = np.clip(fx - cw / 2, 0, sw_ - cw)
    y0 = np.clip(fy - 0.37 * ch, 0, sh_ - ch)
    s = W / cw
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    img = cv2.warpAffine(np.ascontiguousarray(frame), M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    img = img.astype(np.float32) / 255
    # detail: gentle unsharp after the 2.66x upscale (no face reconstruction)
    img = np.clip(img + 0.55 * (img - cv2.GaussianBlur(img, (0, 0), 2.0)), 0, 1)
    img = grade_speaker(img)
    if dark > 0:
        img = img * (1 - dark)  # darken under full-screen typography
    return img


def grade_speaker(img):
    l = img.mean(2, keepdims=True)
    # filmic contrast with rich blacks
    x = np.clip((img - 0.02) / 0.98, 0, 1)
    x = x + 0.10 * (x - x * x) * (2 * x - 1) * -1  # soft S
    x = np.clip((x - 0.5) * 1.06 + 0.5, 0, 1)
    # split tone: warm orange highlights, neutral-cool shadows
    hi = np.clip((l - 0.45) / 0.55, 0, 1) ** 1.5
    lo = np.clip(1 - l / 0.35, 0, 1) ** 2
    x = x + hi * np.array([-0.025, 0.008, 0.035], np.float32) + lo * np.array([0.012, 0.004, -0.008], np.float32)
    # saturation (skin-safe, mild)
    g = x.mean(2, keepdims=True)
    x = g + (x - g) * 1.06
    return np.clip(x * get_vig(), 0, 1)


# ================================================================== map (Philippines -> Sydney)
MAPM = json.load(open(os.path.join(HERE, 'assets', 'map_meta.json')))


def geo(lon, lat):
    return np.array([(lon - MAPM['L0']) * MAPM['P'], (MAPM['A0'] - lat) * MAPM['P']], np.float64)


PH = geo(121.0, 14.6)       # Manila
SYD = geo(151.21, -33.87)   # Sydney
CTRL = (PH + SYD) / 2 + np.array([900.0, -350.0])  # route bows east over the Pacific


def bez(u):
    return (1 - u) ** 2 * PH + 2 * (1 - u) * u * CTRL + u ** 2 * SYD


def map_scene(t, fi, t0, t1):
    u = t - t0
    D = t1 - t0
    # camera keyframes: (time, centre, crop width)
    keys = [(0.0, PH + np.array([150, 250]), 2600), (0.55, PH, 1250), (0.95, PH + np.array([120, 200]), 1450),
            (1.9, bez(0.55), 2650), (2.45, SYD, 1300), (D, SYD, 1150)]
    for (ta, ca, wa), (tb, cb, wb) in zip(keys[:-1], keys[1:]):
        if ta <= u <= tb or u > keys[-1][0]:
            e = ease_io(lin(u, ta, tb))
            c = ca + (cb - ca) * e
            cw = wa * (wb / wa) ** e
            if ta <= u <= tb:
                break
    ch = cw * H / W
    rect = (c[0] - cw / 2, c[1] - ch / 2, cw, ch)
    f = asset('map').crop_to(rect, (W, H))
    # previous-frame camera for motion blur
    sc = W / cw

    def to_frame(p):
        return (p - np.array([rect[0], rect[1]])) * sc

    # country highlight glow (Philippines first, Australia on arrival)
    for key, ta in (('mask_ph', 0.35), ('mask_au', 2.3)):
        a = ease_out(lin(u, ta, ta + 0.5))
        if a > 0:
            m = asset(key).crop_to(rect, (W, H))
            edge = np.clip(m - cv2.erode(m, np.ones((5, 5), np.uint8)), 0, 1)
            f += m[..., None] * ORANGE * 0.10 * a
            f += edge[..., None] * ORANGE * 0.9 * a
            glow(f, edge[..., None] * ORANGE, 10, 1.2 * a)
    # lat/long HUD grid
    hud = np.zeros((H, W), np.float32)
    for lon in range(95, 180, 5):
        x = to_frame(geo(lon, 0))[0]
        if 0 <= x < W: cv2.line(hud, (int(x), 0), (int(x), H), 1, 1, cv2.LINE_AA)
    for lat in range(-50, 35, 5):
        y = to_frame(geo(100, lat))[1]
        if 0 <= y < H: cv2.line(hud, (0, int(y)), (W, int(y)), 1, 1, cv2.LINE_AA)
    f += hud[..., None] * GREY * 0.10
    # route
    pr = ease_io(lin(u, 0.85, 2.4))
    lay = np.zeros((H, W, 3), np.float32)
    if pr > 0:
        us = np.linspace(0, pr, max(2, int(200 * pr)))
        pts = np.array([to_frame(bez(x)) for x in us])
        cv2.polylines(lay, [np.round(pts * 4).astype(np.int32)], False, tuple(float(c) for c in ORANGE), 5,
                      cv2.LINE_AA, shift=2)
        # dashed faint full path preview
        f += lay * 1.0
        glow(f, lay, 14, 2.2)
        if pr < 1:
            hp_ = to_frame(bez(pr))
            dot = np.zeros((H, W, 3), np.float32)
            cv2.circle(dot, (int(hp_[0]), int(hp_[1])), 9, (1, 1, 1), -1, cv2.LINE_AA)
            f += dot
            glow(f, dot * ORANGE * 3, 22, 2.5)
    # markers
    for P, ta, label, sub in ((PH, 0.45, 'PHILIPPINES', ''), (SYD, 2.35, 'SYDNEY', 'AUSTRALIA')):
        a = lin(u, ta, ta + 0.45)
        if a <= 0:
            continue
        q = to_frame(P)
        mk = np.zeros((H, W, 3), np.float32)
        r = 12 * ease_expo_out(a)
        cv2.circle(mk, (int(q[0]), int(q[1])), max(1, int(r)), tuple(float(c) for c in ORANGE), -1, cv2.LINE_AA)
        cv2.circle(mk, (int(q[0]), int(q[1])), max(1, int(r * 0.45)), (1, 1, 1), -1, cv2.LINE_AA)
        for k in range(2):  # pulse rings
            pu = ((u - ta) * 0.9 + k * 0.5) % 1.0
            cv2.circle(mk, (int(q[0]), int(q[1])), int(14 + 60 * pu), tuple(float(c) * (1 - pu) for c in ORANGE), 2, cv2.LINE_AA)
        f += mk
        glow(f, mk, 16, 1.6)
        e = ease_expo_out(lin(u, ta + 0.1, ta + 0.6))
        spr = text_sprite(label, 64, 800, (1.0, 1.0, 1.0), 10)
        lx, ly = q[0], q[1] - 150
        lx = np.clip(lx, sprite_size(spr)[0] / 2 + 30, W - sprite_size(spr)[0] / 2 - 30)
        place(f, spr, lx, ly + 25 * (1 - e), 1, e)
        if sub:
            spr2 = text_sprite(sub, 38, 600, tuple(ORANGE), 12)
            place(f, spr2, lx, ly + 58 + 25 * (1 - e), 1, e)
        else:
            spr2 = text_sprite('ORIGIN', 30, 600, tuple(ORANGE), 12)
            place(f, spr2, lx, ly + 52 + 25 * (1 - e), 1, e)
    # atmosphere: vignette + soft top haze
    f = f * vignette_cache('map')
    return np.clip(f, 0, 1.5)


_vig = {}


def vignette_cache(k):
    if k not in _vig:
        _vig[k] = vignette(0.55, 1.8)
    return _vig[k]


# ================================================================== browser window
def browser_sprite(content, w):
    """RGBA browser window: dark chrome bar with url pill + content (BGR float)"""
    bar = int(w * 0.055)
    ch = int(content.shape[0] * w / content.shape[1])
    img = np.zeros((bar + ch, w, 3), np.float32)
    img[:bar] = np.array([0.11, 0.105, 0.10], np.float32)
    img[bar:] = cv2.resize(content, (w, ch), interpolation=cv2.INTER_AREA if content.shape[1] > w else cv2.INTER_LINEAR)
    for i, col in enumerate([(0.35, 0.37, 0.95), (0.25, 0.75, 0.98), (0.4, 0.8, 0.35)]):
        cv2.circle(img, (int(bar * (0.55 + 0.42 * i)), bar // 2), int(bar * 0.13), col, -1, cv2.LINE_AA)
    px0, px1 = int(w * 0.28), int(w * 0.72)
    cv2.rectangle(img, (px0, int(bar * 0.22)), (px1, int(bar * 0.78)), (0.2, 0.19, 0.18), -1, cv2.LINE_AA)
    url = text_sprite('hellometa.ai', max(10, int(bar * 0.42)), 500, (0.85, 0.85, 0.85))
    rgba = np.dstack([img, rounded_mask(w, bar + ch, w * 0.018)])
    place(rgba[..., :3], url, w / 2, bar / 2, 1.0, 1.0)
    return rgba


def web_hero(t, fi, t0, t1):
    u = t - t0; D = t1 - t0
    f = tech_bg(t)
    hero = asset('web:hero_view')  # 1440x900 css @3x
    e_in = ease_expo_out(lin(u, 0, 0.9))
    e_push = ease_io(lin(u, 0.85, D - 0.1))
    # content rect in css px: whole viewport -> headline
    r0 = np.array([0, 0, 1440, 900.]); r1 = np.array([350, 70, 740, 462.5])
    r = r0 + (r1 - r0) * e_push
    win_w = int(mix(960, 1110, e_push))
    rs = r * 3
    content = crop_resample(hero, rs, win_w)
    spr = browser_sprite(content, win_w)
    rx = mix(16, 5, e_in) * (1 - e_push)
    ry = mix(-24, -8, e_in) * (1 - e_push)
    z = mix(900, 0, e_in)
    m = card3d(f, spr, (0, mix(-230, -300, e_push), z), win_w, rx=rx, ry=ry, alpha=ease_out(lin(u, 0, 0.3)),
               shadow=0.6, glow_col=ORANGE, glow_amt=0.55)
    # orange accent sweep under the headline once it fills the frame
    a = lin(u, D - 0.9, D - 0.25)
    if a > 0:
        xs = np.arange(W, dtype=np.float32)
        band = np.exp(-((xs - (-300 + 1700 * a)) / 120) ** 2)[None, :, None]
        msk = np.zeros((H, W, 1), np.float32)
        if m is not None: msk[..., 0] = m
        f += band * msk * 0.18
    return f


def crop_resample(img, rect, out_w):
    x, y, w, h = rect
    out_h = int(out_w * h / w)
    s = out_w / w
    src = img
    f = 1
    while s * f < 0.55 and min(src.shape[:2]) > 64:
        src = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2), interpolation=cv2.INTER_AREA); f *= 2
    sf = s * f
    M = np.array([[sf, 0, -x / f * sf], [0, sf, -y / f * sf]], np.float32)
    return cv2.warpAffine(src, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ================================================================== web scenes with floating cards
def cam_dolly(u, D, z0=-1800, z1=-1500, x0=0, x1=0, y0=0, y1=0):
    e = ease_io(lin(u, 0, D))
    return (mix(x0, x1, e), mix(y0, y1, e), mix(z0, z1, e))


def web_tech(t, fi, t0, t1):
    u = t - t0; D = t1 - t0
    f = tech_bg(t)
    # camera tilts down from phone+Plug to the "answers" cards
    tilt = ease_expo_io(lin(u, 1.30, 1.80))
    cam = (mix(-30, 30, lin(u, 0, D)), 2100 * tilt, mix(-1800, -1560, ease_io(lin(u, 0, D))))
    ein = ease_expo_out(lin(u, 0, 0.7))
    card3d(f, web_card('plug', 20), (230, -380, mix(700, 260, ein)), 600, ry=-14, rx=4, cam=cam,
           alpha=ein, glow_col=ORANGE, glow_amt=0.35)
    card3d(f, web_card('phone', 30), (-190, -250, mix(500, 0, ein)), 540, ry=12, rx=3, cam=cam,
           alpha=ein, glow_col=ORANGE, glow_amt=0.45)
    e2 = ease_expo_out(lin(u, 1.45, 2.1))
    card3d(f, web_card('card_smarter', 22), (250, 2100 - 120, mix(600, 0, e2)), 480, ry=-6, cam=cam,
           alpha=e2, glow_col=ORANGE, glow_amt=0.3)
    card3d(f, web_card('card_understand', 22), (-250, 2100 - 300, mix(400, 0, e2)), 480, ry=6, cam=cam,
           alpha=e2, glow_col=ORANGE, glow_amt=0.45)
    if 1.3 < u < 1.8:
        f = motion_blur(f, 0, 60 * np.sin(np.pi * lin(u, 1.3, 1.8)))
    return f


def kinetic_title(f, u, lines, y0=300, gap=8):
    """section heading set in the brand font, revealed line by line. lines: [(text, size, colour)]"""
    y = y0
    for i, (txt, size, col) in enumerate(lines):
        e = ease_expo_out(lin(u, 0.05 + 0.12 * i, 0.5 + 0.12 * i))
        tr = round(12 * (1 - e))
        spr = text_sprite(txt, size, 800 if col is not None and col != (1.0, 1.0, 1.0) else 700,
                          col if col is not None else (1.0, 1.0, 1.0), tr)
        place(f, spr, 540, y + size / 2 + 30 * (1 - e), 1, e)
        y += size * 1.05 + gap


def web_ind(t, fi, t0, t1):
    u = t - t0; D = t1 - t0
    f = tech_bg(t)
    ein = ease_expo_out(lin(u, 0, 0.6))
    hc = ease_expo_out(lin(u, D - 1.0, D - 0.5))
    cam = (mix(-560, 560, ease_io(lin(u, 0.05, D - 0.55))), 0, -1800)
    card3d(f, web_card('ind_cards', 10), (0, -150 + 90 * hc, mix(500, 0, ein)), 2000, ry=mix(-10, 0, ein), cam=cam,
           alpha=ein, glow_col=ORANGE, glow_amt=0.25)
    # site heading as kinetic type, swapped for the Healthcare call card at the end
    g = 1 - hc
    if g > 0.01:
        tmp = np.zeros((H, W, 4), np.float32)
        kinetic_title(tmp, u, [('REAL-WORLD', 64, (1.0, 1.0, 1.0)), ('AI ASSISTANTS', 92, tuple(ORANGE)),
                               ('FOR EVERY INDUSTRY', 50, (0.8, 0.8, 0.8))], y0=150)
        f[:] = f * (1 - tmp[..., 3:] * g) + tmp[..., :3] * g
    card3d(f, web_card('healthcare', 30), (0, -700 + 40 * (1 - hc), -100), 620, rx=-6, alpha=hc,
           glow_col=ORANGE, glow_amt=0.6)
    return f


def web_dash(t, fi, t0, t1):
    u = t - t0; D = t1 - t0
    f = tech_bg(t)
    e = ease_expo_out(lin(u, 0, 0.8))
    push = ease_io(lin(u, 0.6, D))
    cam = (mix(0, 30, push), mix(0, 40, push), mix(-1800, -1150, push))
    g = 1 - ease_in(lin(u, 1.2, 1.8), 2)
    if g > 0.01:
        tmp = np.zeros((H, W, 4), np.float32)
        kinetic_title(tmp, u, [('TRUST WHAT YOU SEE', 70, (1.0, 1.0, 1.0)), ('POWERFUL BACKEND INSIGHT', 66, tuple(ORANGE))], y0=170)
        f[:] = f * (1 - tmp[..., 3:] * g) + tmp[..., :3] * g
    card3d(f, web_card('dash_login', 26), (-400, -70, mix(900, 420, e)), 520, ry=28, cam=cam, alpha=e,
           glow_col=ORANGE, glow_amt=0.25)
    card3d(f, web_card('dash_logs', 22), (410, -50, mix(900, 420, e)), 520, ry=-28, cam=cam, alpha=e,
           glow_col=ORANGE, glow_amt=0.25)
    card3d(f, web_card('dash_main', 26), (0, -110, mix(600, 0, e)), 560, cam=cam, alpha=e,
           glow_col=ORANGE, glow_amt=0.55)
    return f


FEAT = [('f_recordings', 805, 800), ('f_calllogs', 275, 800), ('f_voiceflows', 275, 960),
        ('f_oneclick', 805, 960), ('f_forwarding', 805, 1115), ('f_bulk', 275, 1115)]


def web_feat(t, fi, t0, t1):
    u = t - t0; D = t1 - t0
    f = tech_bg(t)
    tmp = np.zeros((H, W, 4), np.float32)
    kinetic_title(tmp, u, [('SMART AUTOMATED', 58, (1.0, 1.0, 1.0)), ('CALL ASSISTANTS', 84, tuple(ORANGE))], y0=150)
    f[:] = f * (1 - tmp[..., 3:]) + tmp[..., :3]
    push = 1 + 0.05 * ease_io(lin(u, 0, D))

    def P(x, y):
        return ((x - 540) * push, (y - 960) * push)

    # connector lines from Plug to each card
    lines = np.zeros((H, W, 3), np.float32)
    hub = (540, 610)
    for i, (k, x, y) in enumerate(FEAT):
        p = ease_io(lin(u, 0.45 + 0.16 * i, 0.85 + 0.16 * i))
        if p > 0:
            hx, hy = 540 + (hub[0] - 540) * push, 960 + (hub[1] - 960) * push
            cx, cy = 540 + (x - 540) * push, 960 + (y - 960) * push
            mx = hx + (cx - hx) * p; my = hy + (cy - hy) * p
            cv2.line(lines, (int(hx), int(hy)), (int(mx), int(my)), tuple(float(c) * 0.8 for c in ORANGE), 2, cv2.LINE_AA)
    f += lines * 0.6
    glow(f, lines, 8, 0.8)
    ep = ease_expo_out(lin(u, 0.1, 0.6))
    X, Y = P(540, 560)
    card3d(f, web_card('plug', 26), (X, Y, mix(500, 0, ep)), 330, alpha=ep, glow_col=ORANGE, glow_amt=0.45)
    for i, (k, x, y) in enumerate(FEAT):
        a = ease_expo_out(lin(u, 0.55 + 0.16 * i, 1.0 + 0.16 * i))
        X, Y = P(x, y)
        hl = np.exp(-((u - (1.9 + 0.35 * (i % 2))) / 0.25) ** 2) if k in ('f_calllogs', 'f_recordings') else 0
        card3d(f, web_card(k, 16), (X, Y, mix(400, -40 * hl, a)), 470 * (1 + 0.05 * hl),
               ry=mix(20 if x < 540 else -20, 0, a), alpha=a, shadow=0.4,
               glow_col=ORANGE, glow_amt=0.25 + 1.2 * hl)
    return f


# ================================================================== outro montage
LANGS = ['English', 'Hindi', 'Chinese', 'Arabic', 'Filipino', 'French', 'Greek', 'Italian', 'Spanish', 'Vietnamese']


def lang_scene(t, fi):
    t0, t1 = T.LANG; u = t - t0
    f = tech_bg(t)
    # headline from the site's section title
    e = ease_expo_out(lin(u, 0.0, 0.5))
    place(f, text_sprite('AI VOICE AGENTS IN', 54, 600, (1.0, 1.0, 1.0), 8), 540, 300 + 30 * (1 - e), 1, e)
    e2 = ease_expo_out(lin(u, 0.12, 0.6))
    place(f, text_sprite('10+ LANGUAGES', 128, 800, tuple(ORANGE)), 540, 410, 1.15 - 0.15 * e2, e2)
    hl = ease_io(lin(u, 1.15, 1.55))
    push = 1 + 0.04 * ease_io(lin(u, 0, 2.0))
    for i, name in enumerate(LANGS):
        col, row = i % 2, i // 2
        x = 540 + (col - 0.5) * 470
        y = 690 + row * 235
        a = ease_expo_out(lin(u, 0.15 + i * 0.07, 0.55 + i * 0.07))
        is_fil = name == 'Filipino'
        sc = (0.6 + 0.4 * a) * (1 + 0.14 * hl * is_fil)
        dim = 1 - 0.55 * hl * (not is_fil)
        X = (x - 540) * push; Y = (y - 960) * push
        card3d(f, web_card('lang_' + name, 18), (X, Y, mix(600, 0, a) - 200 * hl * is_fil), 440 * sc,
               ry=mix(25 * (1 if col else -1), 0, a), alpha=a * dim, shadow=0.4,
               glow_col=ORANGE, glow_amt=(0.25 + 1.1 * hl) if is_fil else 0.15 * a)
    return f


def calc_scene(t, fi):
    t0, t1 = T.CALC; u = t - t0; D = t1 - t0
    f = tech_bg(t)
    e = ease_expo_out(lin(u, 0, 0.5))
    push = ease_io(lin(u, 0.0, D))
    cam = (mix(-60, 120, push), mix(-40, 40, push), mix(-1800, -1250, push))
    card3d(f, web_card('calc_phone', 36), (-150, -40, mix(500, 0, e)), 600, ry=10, cam=cam, alpha=e,
           glow_col=ORANGE, glow_amt=0.45)
    e2 = ease_expo_out(lin(u, 0.45, 0.95))
    card3d(f, web_card('calc_result', 26), (175, 140, mix(-100, -260, e2)), 500, ry=-14, cam=cam, alpha=e2,
           glow_col=ORANGE, glow_amt=0.6)
    # scanning highlight over the input fields
    sc = lin(u, 0.2, 0.9)
    if 0 < sc < 1:
        y = 700 + 520 * sc
        band = np.exp(-((np.arange(H, dtype=np.float32) - y) / 22) ** 2)[:, None, None]
        f += band * ORANGE * 0.10
    return f


STEPS = [('01', 'DEFINE', 'Define what the AI voice agent should do.'),
         ('02', 'DESIGN', 'Design the script, voice and actions.'),
         ('03', 'DEPLOY', 'Test, deploy and track the AI agent.')]


def how_scene(t, fi):
    t0, t1 = T.HOW; u = t - t0
    f = tech_bg(t, grid=False)
    lift = -60 * ease_io(lin(u, 0, 4.0))
    e = ease_expo_out(lin(u, 0, 0.45))
    place(f, text_sprite('HOW IT WORKS', 46, 700, tuple(ORANGE), 16), 540, 330 + lift + 20 * (1 - e), 1, e)
    ys = [700, 1010, 1320]
    # connecting line between numbers (draws as each step lands)
    line = np.zeros((H, W, 3), np.float32)
    for i in range(2):
        p = ease_io(lin(u, 0.55 + i * 1.0, 1.0 + i * 1.0))
        if p > 0:
            ya, yb = ys[i] + 95 + lift, ys[i + 1] - 95 + lift
            cv2.line(line, (190, int(ya)), (190, int(ya + (yb - ya) * p)), tuple(float(c) for c in ORANGE), 4, cv2.LINE_AA)
            if p < 1:
                cv2.circle(line, (190, int(ya + (yb - ya) * p)), 8, (1, 1, 1), -1, cv2.LINE_AA)
    f += line
    glow(f, line, 12, 1.5)
    for i, (num, title, desc) in enumerate(STEPS):
        ta = i * 1.0
        a = lin(u, ta, ta + 0.45)
        if a <= 0:
            continue
        y = ys[i] + lift
        en = ease_expo_out(a)
        nspr = num_sprite(num)
        place(f, nspr, 190, y, 1.35 - 0.35 * en, en)
        tr = 30 * (1 - ease_out(lin(u, ta + 0.05, ta + 0.6)))
        tspr = text_sprite(title, 150, 800, (1.0, 1.0, 1.0), round(tr / 3) * 3)
        tw = sprite_size(tspr)[0]
        place(f, tspr, 330 + tw / 2 + 80 * (1 - en), y - 28, 1, en)
        d = ease_out(lin(u, ta + 0.2, ta + 0.6))
        dspr = text_sprite(desc, 40, 500, (0.78, 0.78, 0.78))
        place(f, dspr, 330 + sprite_size(dspr)[0] / 2 - 18, y + 80 + 15 * (1 - d), 1, d)
        # active step glow pulse
        if u > ta:
            pulse = np.exp(-((u - ta) / 0.25) ** 2)
            if pulse > 0.02:
                m = np.zeros((H, W, 3), np.float32)
                cv2.circle(m, (190, int(y)), 70, tuple(float(c) for c in ORANGE), -1, cv2.LINE_AA)
                glow(f, m, 30, 0.6 * pulse)
    return f


def num_sprite(num):
    key = 'num:' + num
    if key not in _A:
        from PIL import Image, ImageDraw
        fnt = _font(150, 800)
        img = Image.new('L', (300, 240), 0)
        d = ImageDraw.Draw(img)
        d.text((150, 120), num, font=fnt, fill=0, stroke_width=4, stroke_fill=255, anchor='mm')
        a = np.asarray(img, np.float32) / 255
        _A[key] = np.dstack([np.broadcast_to(ORANGE, a.shape + (3,)), a]).astype(np.float32)
    return _A[key]


# ================================================================== hilltop logo reveal
def hill_scene(t, fi):
    t0, t1 = T.HILL; u = t - t0; D = t1 - t0
    meta = asset('land_meta')
    LW, LH = meta['W'], meta['H']
    F = np.array([meta['summit_x'], meta['summit_y'] - 160])  # focus: just above the summit
    Z = 1 + 1.45 * ease_io(lin(u, 0, D)) + 0.3 * ease_in(lin(u, D - 1.2, D), 2)
    base = 0.62
    P0, P1 = np.array([540, 1240.]), np.array([540, 900.])
    P = P0 + (P1 - P0) * ease_io(lin(u, 0, D * 0.8))
    f = np.zeros((H, W, 3), np.float32)
    fg_alpha = np.zeros((H, W), np.float32)

    def layer(img, d, extra_dy=0.0):
        zd = 1 + (Z - 1) * d
        s = base * zd
        M = np.array([[s, 0, P[0] - s * F[0]], [0, s, P[1] - s * F[1] + extra_dy]], np.float32)
        out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE if img.shape[2] == 3 else cv2.BORDER_CONSTANT)
        return out, s, M

    sky, _, _ = layer(asset('land_sky'), 0.12)
    f[:] = sky
    for name, d, haze in (('far', 0.35, 0.35), ('mid', 0.55, 0.22), ('near', 0.75, 0.12)):
        img, _, _ = layer(asset('land_' + name), d)
        over(f, img[..., :3], img[..., 3])
        # atmospheric haze band in front of each ridge
        hz = np.exp(-((np.arange(H, dtype=np.float32) - (P[1] + 120 * d)) / (260 + 200 * d)) ** 2)[:, None, None]
        f += hz * np.array([0.10, 0.25, 0.45], np.float32) * haze * 0.5
    # logo rises from behind the summit and stands on the plateau. It is composited LAST, after all
    # lighting effects, so its colours and shape are exactly the supplied artwork; only the hill
    # in front of it masks it while it rises.
    logo = asset('logo')
    lw_world = 600
    lh_world = lw_world * logo.shape[0] / logo.shape[1]
    rise = ease_out(lin(u, 1.15, T.HILL_HIT - t0), 3)
    hill, s_hill, M = layer(asset('land_hill'), 1.0)
    lx = meta['summit_x']; ly_bottom = meta['summit_y'] + 10 + lh_world * (1 - rise) * 1.05
    p = M @ np.array([lx, ly_bottom, 1.0])
    ll = np.zeros((H, W, 4), np.float32)
    if rise > 0:
        place(ll, logo, p[0], p[1] - lh_world * s_hill / 2, lw_world * s_hill / logo.shape[1], 1.0)
    over(f, hill[..., :3], hill[..., 3])
    fg_alpha = np.maximum(hill[..., 3], ll[..., 3] * (1 - hill[..., 3]))
    # god rays from the sun, shaded by the hill and the logo silhouette
    sun = M @ np.array([meta['summit_x'], meta['horizon'] - 250, 1.0])
    bright = np.clip(f.mean(2) - 0.55, 0, 1) * (1 - fg_alpha)
    small = cv2.resize(bright, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    rays = zoom_blur(small[..., None].repeat(3, 2), 0.55, sun[0] / 4, sun[1] / 4, n=10)
    rays = cv2.resize(rays, (W, H), interpolation=cv2.INTER_LINEAR)
    f += rays * np.array([0.35, 0.65, 1.0], np.float32) * 0.9
    particles(f, t, color=np.array([0.4, 0.75, 1.0], np.float32), amt=0.8, speed=25)
    f *= vignette_cache('map') ** 0.6
    f = np.clip(f, 0, 1)
    # the logo itself, unaltered, in front of the sky and behind the hill
    la = ll[..., 3] * (1 - hill[..., 3])
    f = f * (1 - la[..., None]) + ll[..., :3] * (1 - hill[..., 3])[..., None]
    # bloom into the white end slide
    wb = ease_in(lin(u, D - 0.45, D), 2)
    f = f * (1 - wb) + wb
    return f


# ================================================================== white end slide
def end_scene(t, fi):
    """white slide: the supplied logo, unaltered, fades in with a gentle scale-up"""
    t0, t1 = T.END; u = t - t0
    f = np.ones((H, W, 3), np.float32)
    logo = asset('logo')
    a = ease_out(lin(u, 0.05, 0.8))
    sc = 0.92 + 0.08 * ease_expo_out(lin(u, 0, 1.6)) + 0.015 * lin(u, 1.6, t1 - t0)
    place(f, logo, 540, 960, 880 / logo.shape[1] * sc, a, interp=cv2.INTER_CUBIC)
    return f


# ================================================================== subtitles
GROUPS = ("I'm *Zarina* | from the *Philippines,* | and I have *visited* | the *booth* | for *Hello* *Meta* *AI.* | "
          "And I've *tried* | the *call* *bot.* | I *really* like it. | !It's the *best* | because it *features* | a lot of | "
          "*answerable* *questions.* | They answered *everything* | that I *asked* | about my *childcare* center | "
          "in the *Philippines.* | For *example,* | I asked if there are | *packages* | for a *five-year-old.* | "
          "I'm a *mom,* | and I would *want* | to *enroll* my kid, | and it says there's | a *half-day* | "
          "and a *whole-day* | *package,* | so I *like* it | *very* *much.* | And it's just a bit of | all the *information* | "
          "that I have *asked* | in their *bot,* | !and *excellent.* | *Thank* *you!*")


def build_groups():
    words = asset('words')
    groups, k = [], 0
    for g in GROUPS.split('|'):
        g = g.strip()
        big = g.startswith('!')
        toks = g.lstrip('!').split()
        items = []
        for tok in toks:
            hl = tok.startswith('*')
            w = words[k]
            assert w['w'] == tok.strip('*'), (w['w'], tok)
            items.append(dict(text=tok.strip('*'), hl=hl, s=T.E(w['s']), e=T.E(w['e'])))
            k += 1
        groups.append(dict(big=big, items=items))
    assert k == len(words), (k, len(words))
    for i, g in enumerate(groups):
        g['s'] = g['items'][0]['s']
        nxt = groups[i + 1]['items'][0]['s'] if i + 1 < len(groups) else T.OUTRO - 0.3
        g['end'] = min(nxt, g['items'][-1]['e'] + 0.9)
    return groups


_GROUPS = None


def layout(g):
    """lines of words; a size change starts a new line. returns [(spr, x, y, item)]"""
    if 'layout' in g:
        return g['layout']
    big = g['big']
    lines, cur = [], []
    for it in g['items']:
        if cur and cur[-1]['hl'] != it['hl']:
            lines.append(cur); cur = []
        cur.append(it)
    lines.append(cur)
    out, y = [], 0
    sizes = []
    for ln in lines:
        hl = ln[0]['hl']
        if big:
            size, wg = (250, 800) if hl else (96, 700)
        else:
            size, wg = (112, 800) if hl else (66, 700)
        txt = [w['text'].upper() if big else w['text'] for w in ln]
        sprs = [text_sprite(x, size, wg, tuple(ORANGE) if hl else (1.0, 1.0, 1.0)) for x in txt]
        space = size * 0.26
        tw = sum(sprite_size(s)[0] - 2 * int(size * 0.35) for s in sprs) + space * (len(sprs) - 1)
        fit = min(1.0, 960 / tw)
        sizes.append((ln, sprs, size * fit, fit, tw * fit, space * fit, size))
    total_h = sum(sz[2] * 0.98 for sz in sizes)
    y = -total_h / 2
    for ln, sprs, lh, fit, tw, space, size in sizes:
        x = -tw / 2
        for it, s in zip(ln, sprs):
            wpx = (sprite_size(s)[0] - 2 * int(size * 0.35)) * fit
            out.append((s, x + wpx / 2, y + lh / 2, it, fit))
            x += wpx + space
        y += lh * 0.98
    g['layout'] = out
    return out


def draw_subs(f, t):
    global _GROUPS
    if _GROUPS is None:
        _GROUPS = build_groups()
    for g in _GROUPS:
        if not (g['s'] - 0.01 <= t < g['end']):
            continue
        out = ease_in(lin(t, g['end'] - 0.12, g['end']), 2)
        big = g['big']
        cy = 1240 if big else 1330
        drift = -14 * lin(t, g['s'], g['end'])
        layer = np.zeros((H, W, 4), np.float32)
        for spr, x, y, it, fit in layout(g):
            if t < it['s']:
                continue  # never before the word is spoken
            u = lin(t, it['s'], it['s'] + (0.20 if it['hl'] else 0.14))
            e = ease_expo_out(u)
            sc = fit * (0.62 + 0.38 * e) * (1 - 0.06 * out)
            if big and it['hl']:
                sc = fit * (1.35 - 0.35 * e)
            a = e * (1 - out)
            px, py = 540 + x, cy + y + drift + (18 if not big else 0) * (1 - e)
            place(layer, spr, px, py, sc, a)
        if layer[..., 3].max() <= 0:
            continue
        al = layer[..., 3]
        ys_, xs_ = np.nonzero(al > 0.05)
        if len(ys_):
            reg = f[max(ys_.min() - 20, 0):ys_.max() + 20, max(xs_.min() - 20, 0):xs_.max() + 20]
            bright = clamp01((float(reg.mean()) - 0.28) / 0.35)
        else:
            bright = 0.0
        sh = blur_fast(al, 10)
        sh = np.roll(sh, 5, axis=0)
        f *= (1 - 0.55 * np.clip(sh * 1.3, 0, 1))[..., None]
        if bright > 0:  # soft dark backplate only where the background is bright (web pages)
            bp = blur_fast(cv2.dilate(al, np.ones((31, 31), np.uint8)), 22)
            f *= (1 - 0.82 * bright * np.clip(bp * 1.4, 0, 1))[..., None]
        if big:
            f *= (1 - 0.25 * blur_fast(al, 40))[..., None]
        # orange glow on highlight words
        orange_part = layer[..., :3] * (layer[..., 0:1] < 0.5)  # orange has low blue channel
        glow(f, orange_part, 16, 0.5)
        f[:] = f * (1 - al[..., None]) + layer[..., :3] * (al[..., None] > 0)
    return f


# ================================================================== timeline assembly
def base_scene(t):
    for name, a, b in T.BROLL:
        if a <= t < b:
            return name, (a, b)
    if t < T.OUTRO:
        return 'speaker', None
    if t < T.LANG[1]: return 'lang', None
    if t < T.CALC[1]: return 'calc', None
    if t < T.HOW[1]: return 'how', None
    if t < T.HILL[1]: return 'hill', None
    return 'end', None


def render_scene(name, win, t, fi):
    if name == 'speaker': return speaker(min(t, T.OUTRO - 0.001), fi)
    if name == 'map': return map_scene(t, fi, *win)
    if name == 'web_hero': return web_hero(t, fi, *win)
    if name == 'web_tech': return web_tech(t, fi, *win)
    if name == 'web_ind': return web_ind(t, fi, *win)
    if name == 'web_dash': return web_dash(t, fi, *win)
    if name == 'web_feat': return web_feat(t, fi, *win)
    if name == 'lang': return lang_scene(t, fi)
    if name == 'calc': return calc_scene(t, fi)
    if name == 'how': return how_scene(t, fi)
    if name == 'hill': return hill_scene(t, fi)
    if name == 'end': return end_scene(t, fi)
    raise KeyError(name)


TRANS = 0.14  # half-length of a cut transition (zoom-blur + light flash)
NO_TRANS = {T.HILL[1]}  # hill -> white end is a light bloom, not a cut


def scene_at(t):
    name, win = base_scene(t)
    return name, win


def render_frame(t, fi):
    # transition handling
    bounds = sorted(set([x for _, a, b in T.BROLL for x in (a, b)] +
                        [T.OUTRO, T.CALC[0], T.HOW[0], T.HILL[0]]))
    tc = next((b for b in bounds if abs(t - b) < TRANS and b not in NO_TRANS), None)
    if tc is None:
        n, w = scene_at(t)
        f = render_scene(n, w, t, fi)
    else:
        p = (t - (tc - TRANS)) / (2 * TRANS)
        na, wa = scene_at(tc - 1e-3)
        nb, wb = scene_at(tc + 1e-3)
        ta, tb = min(t, tc - 1e-3), max(t, tc + 1e-3)
        A = render_scene(na, wa, ta, fi)
        Bf = render_scene(nb, wb, tb, fi)
        e = smoothstep(p)
        A = zoom_blur(A, 0.22 * e, n=6)
        Bf = zoom_blur(Bf, 0.22 * (1 - e), n=6)
        sa = 1 + 0.18 * e; sb = 1.18 - 0.18 * e
        A = cv2.warpAffine(A, np.array([[sa, 0, 540 - sa * 540], [0, sa, 960 - sa * 960]], np.float32), (W, H), borderMode=cv2.BORDER_REFLECT)
        Bf = cv2.warpAffine(Bf, np.array([[sb, 0, 540 - sb * 540], [0, sb, 960 - sb * 960]], np.float32), (W, H), borderMode=cv2.BORDER_REFLECT)
        f = A * (1 - e) + Bf * e
        flash = np.sin(np.pi * p) ** 2
        f += flash * 0.22 * (np.array([0.75, 0.88, 1.0], np.float32))
    # subtitles over the interview and its B-roll
    if T.SPEECH_OFF <= t < T.OUTRO:
        draw_subs(f, t)
    # finishing: light grain (not on the white end slide)
    if t < T.END[0]:
        f = f + grain(fi, 0.014)
    if t < 0.2:  # soft open on the speaker
        f = f * ease_out(t / 0.2, 2)
    return np.clip(f, 0, 1)


def to_u8(f):
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


# ================================================================== drivers
def render_range(a, b, out_path):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                          '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '12',
                          '-pix_fmt', 'yuv420p', out_path], stdin=subprocess.PIPE)
    t0 = time.time()
    for fi in range(a, b):
        f = render_frame(fi / FPS, fi)
        p.stdin.write(to_u8(f).tobytes())
        if (fi - a) % 60 == 0:
            print(f'[{a}-{b}] frame {fi} ({(fi - a + 1) / (time.time() - t0 + 1e-9):.2f} fps)', flush=True)
    p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stills', default='')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--range', default='')
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'output', 'hellometa_brand_film_9x16.mp4'))
    args = ap.parse_args()
    asset('raw')  # make sure the frame cache exists before forking
    if args.stills:
        for s in args.stills.split(','):
            t = float(s); fi = int(round(t * FPS))
            cv2.imwrite(B(f'still_{t:06.2f}.jpg'), to_u8(render_frame(t, fi)), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print('still', t, flush=True)
        return
    n = int(round(T.DUR * FPS))
    if args.range:
        a, b = map(int, args.range.split(':'))
        render_range(a, b, B(f'seg_{a:05d}.mp4'))
        return
    chunk = int(np.ceil(n / args.jobs))
    procs, segs = [], []
    for j in range(args.jobs):
        a, b = j * chunk, min(n, (j + 1) * chunk)
        segs.append(B(f'seg_{a:05d}.mp4'))
        procs.append(subprocess.Popen([sys.executable, __file__, '--range', f'{a}:{b}']))
    for p in procs:
        p.wait()
        if p.returncode:
            sys.exit('render worker failed')
    lst = B('segs.txt')
    open(lst, 'w').write(''.join(f"file '{s}'\n" for s in segs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-i', B('mix.wav'),
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                    '-movflags', '+faststart', '-shortest', args.out], check=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
