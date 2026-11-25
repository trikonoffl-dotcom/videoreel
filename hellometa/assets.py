"""Build derived assets for the HelloMeta film.

  build/logo_metal.png      original logo, orange -> black metallic treatment (shape untouched)
  build/logo_metal_rim.png  same, with an orange rim light for dark backgrounds
  build/map_dark.png        Natural Earth relief, graded dark graphite for the route animation
  build/land_*.png          procedural dusk landscape layers for the hilltop reveal

Run from this folder:  python3 assets.py
"""
import cv2, numpy as np, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
A = lambda *p: os.path.join(HERE, 'assets', *p)
B = lambda *p: os.path.join(HERE, 'build', *p)
os.makedirs(B(), exist_ok=True)

ORANGE = np.array([0x1E, 0x89, 0xF6], np.float32) / 255  # BGR of #F6891E


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- logo
def build_logo():
    src = cv2.imread(A('HelloMeta-Logo-1.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
    h, w = src.shape[:2]
    alpha = src[..., 3]
    col = src[..., :3]
    # which pixels are the orange "hello" vs the black "Meta.ai" (keep the original split exactly)
    is_orange = (col[..., 2] - col[..., 0] > 0.3).astype(np.float32)
    is_orange = cv2.GaussianBlur(is_orange, (0, 0), 1.0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    v = ys / h
    # metallic vertical profile: bright upper band, darker belly, a fine specular streak
    prof = 0.86 + 0.30 * np.exp(-((v - 0.30) / 0.16) ** 2) - 0.22 * smooth((v - 0.55) / 0.4) \
        + 0.18 * np.exp(-((v - 0.40) / 0.025) ** 2)
    # orange metal: warm highlight -> hello orange -> burnt copper at the bottom
    o_hi = np.array([0x6A, 0xC8, 0xFF], np.float32) / 255
    o_lo = np.array([0x0A, 0x45, 0x9A], np.float32) / 255
    t = smooth((v - 0.15) / 0.75)[..., None]
    orange_metal = (o_hi * (1 - t) * 0.35 + ORANGE * 0.65 * (1 - t) + o_lo * t * 1.0)
    orange_metal = orange_metal * prof[..., None]
    # black metal: graphite highlight -> deep black
    g_hi = np.array([0.42, 0.42, 0.43], np.float32)
    g_lo = np.array([0.02, 0.02, 0.02], np.float32)
    tb = smooth((v - 0.12) / 0.6)[..., None]
    black_metal = (g_hi * (1 - tb) + g_lo * tb) * prof[..., None]
    # orange -> black gradient: the tail of "hello" glows from orange into graphite before "Meta.ai"
    xg = smooth((xs / w - 0.40) / 0.12)[..., None]  # 0 on "hello", 1 at its tail
    orange_side = orange_metal * (1 - 0.55 * xg) + black_metal * 0.55 * xg
    base = orange_side * is_orange[..., None] + black_metal * (1 - is_orange[..., None])
    # bevel lighting from the blurred alpha (light from upper-left)
    a_bl = cv2.GaussianBlur(alpha, (0, 0), 4)
    gx = cv2.Sobel(a_bl, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a_bl, cv2.CV_32F, 0, 1, ksize=3)
    lit = np.clip(-(gx * -0.6 + gy * -0.8) * 2.2, -1, 1)
    edge = np.clip(np.hypot(gx, gy) * 2.5, 0, 1)
    spec = np.clip(lit, 0, 1) ** 1.5 * edge
    shade = np.clip(-lit, 0, 1) * edge
    out = base * (1 - 0.35 * shade[..., None]) + spec[..., None] * np.where(
        is_orange[..., None] > 0.5, np.array([0.75, 0.9, 1.0]), np.array([0.8, 0.8, 0.82])) * 0.65
    out = np.clip(out, 0, 1)
    rgba = np.dstack([out, alpha])
    cv2.imwrite(B('logo_metal.png'), (rgba * 255 + 0.5).astype(np.uint8))
    # rim-lit version (backlit by an orange sun) for dark scenes
    a_out = cv2.GaussianBlur(alpha, (0, 0), 3)
    inner = np.clip(alpha - cv2.erode(alpha, np.ones((7, 7), np.uint8)), 0, 1)
    rim_w = np.clip(1 - v * 1.4, 0.15, 1)[..., None]
    rim = inner[..., None] * rim_w * np.array([0.35, 0.72, 1.0])
    out2 = np.clip(out + rim * 0.9, 0, 1)
    cv2.imwrite(B('logo_metal_rim.png'), (np.dstack([out2, alpha]) * 255 + 0.5).astype(np.uint8))
    # white-background check sheet
    bg = np.ones((h + 200, w + 200, 3), np.float32)
    bg[100:100 + h, 100:100 + w] = bg[100:100 + h, 100:100 + w] * (1 - alpha[..., None]) + out * alpha[..., None]
    cv2.imwrite(B('logo_check_white.jpg'), (cv2.resize(bg, None, fx=0.5, fy=0.5) * 255).astype(np.uint8))


# ---------------------------------------------------------------- map
def build_map():
    m = cv2.imread(A('map_region.jpg')).astype(np.float32) / 255
    b, g, r = m[..., 0], m[..., 1], m[..., 2]
    water = smooth((b - r - 0.05) / 0.15)
    water = cv2.GaussianBlur(water, (0, 0), 1.5)
    lum = 0.3 * r + 0.55 * g + 0.15 * b
    # relief: high-pass of luminance carries the hillshade
    relief = lum - cv2.GaussianBlur(lum, (0, 0), 25)
    land = np.clip(0.30 + relief * 1.9 + (lum - 0.6) * 0.5, 0, 1)
    land_col = np.dstack([land * 0.93, land * 0.95, land * 1.0])  # neutral graphite, faintly warm
    sea = np.clip(0.055 + relief * 1.2, 0, 1)
    sea_col = np.dstack([sea * 1.25, sea * 1.05, sea * 0.9])  # near-black, faint cool tint
    out = land_col * (1 - water[..., None]) + sea_col * water[..., None]
    # coastline glow edge
    coast = np.abs(cv2.Sobel(cv2.GaussianBlur(water, (0, 0), 2), cv2.CV_32F, 1, 0)) + \
        np.abs(cv2.Sobel(cv2.GaussianBlur(water, (0, 0), 2), cv2.CV_32F, 0, 1))
    out += np.clip(coast * 0.6, 0, 0.25)[..., None] * np.array([0.6, 0.62, 0.65])
    cv2.imwrite(B('map_dark.png'), (np.clip(out, 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(B('map_dark_preview.jpg'), (cv2.resize(np.clip(out, 0, 1), (800, 800)) * 255).astype(np.uint8))


# ---------------------------------------------------------------- landscape
def fbm1d(n, seed, octaves=6, base=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        k = base * 2 ** o
        pts = rng.uniform(-1, 1, k + 3)
        xp = np.linspace(0, 1, k + 3)
        from scipy.interpolate import CubicSpline
        y += amp * CubicSpline(xp, pts)(x)
        amp *= 0.5
    return y


def build_landscape():
    W, H = 2400, 3200  # oversized so the camera can push in and drift
    ys = np.arange(H, dtype=np.float32)[:, None]
    horizon = 0.56 * H
    # sky: deep graphite top -> burnt orange -> bright glow at horizon
    t = np.clip(ys / horizon, 0, 1)
    top = np.array([0.035, 0.03, 0.03]); mid = np.array([0.06, 0.16, 0.42]); low = np.array([0.30, 0.62, 0.98])
    sky = np.where(t[..., None] < 0.62, top + (mid - top) * smooth(t / 0.62)[..., None],
                   mid + (low - mid) * smooth((t - 0.62) / 0.38)[..., None])
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    # sun glow centred behind the summit
    sx, sy = W * 0.5, horizon - 250
    xx = np.arange(W, dtype=np.float32)[None, :]
    d = np.sqrt((xx - sx) ** 2 + ((ys - sy) * 1.15) ** 2)
    glow = np.exp(-(d / 360) ** 2) * 0.9 + np.exp(-(d / 1100) ** 2) * 0.35
    sky += glow[..., None] * np.array([0.45, 0.75, 1.0])
    sun = smooth((95 - d) / 8)
    sky = sky * (1 - sun[..., None]) + sun[..., None] * np.array([0.88, 0.97, 1.0])
    # soft cloud streaks
    rng = np.random.default_rng(7)
    cl = cv2.resize(rng.random((24, 6)).astype(np.float32), (W, int(horizon)), interpolation=cv2.INTER_CUBIC)
    cl = cv2.GaussianBlur(cl, (0, 0), 30)
    band = np.exp(-((np.arange(int(horizon))[:, None] - horizon * 0.72) / (horizon * 0.12)) ** 2)
    sky[:int(horizon)] *= (1 - 0.25 * np.clip(cl - 0.45, 0, 1) * 3 * band)[..., None]
    cv2.imwrite(B('land_sky.png'), (np.clip(sky, 0, 1) * 255).astype(np.uint8))

    def ridge(seed, base_y, amp, color, haze, rim, name, smooth_k=1, summit=None):
        prof = fbm1d(W, seed)
        prof = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (0, 0), smooth_k).ravel()
        y = base_y - amp * prof
        if summit is not None:  # big central hill with a clear summit
            sxp, sh, swid = summit
            dd = np.maximum(np.abs(np.arange(W) - sxp) - 300, 0)
            hill = sh * np.exp(-(dd / swid) ** 2) - 0.00025 * np.minimum(np.abs(np.arange(W) - sxp), 300) ** 2
            hill = cv2.GaussianBlur(hill.reshape(1, -1).astype(np.float32), (0, 0), 140).ravel()
            rough = fbm1d(W, seed + 1, octaves=6, base=6)
            y = base_y - hill - (8 + 40 * np.clip(np.abs(np.arange(W) - sxp) / 900 - 0.3, 0, 1)) * rough
        mask = (ys >= y[None, :]).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
        depth = np.clip((ys - y[None, :]) / 500, 0, 1)
        col = np.array(color)[None, None, :] * (1 - haze * 0.0) * (0.9 + 0.1 * (1 - depth[..., None]))
        # rim light along the ridge (sun behind)
        dist = np.clip(ys - y[None, :], 0, None)
        sunw = np.exp(-((np.arange(W) - W * 0.5) / (W * 0.45)) ** 2)[None, :]
        rimv = np.exp(-dist / 5.0) * rim * sunw
        img = col + rimv[..., None] * np.array([0.35, 0.65, 1.0])
        # fine texture so it does not read as a flat cut-out
        tex = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 1.2)
        img = img * (0.92 + 0.16 * tex[..., None])
        rgba = np.dstack([np.clip(img, 0, 1), mask])
        cv2.imwrite(B(f'land_{name}.png'), (rgba * 255).astype(np.uint8))
        return y

    ridge(11, horizon + 30, 120, [0.20, 0.30, 0.48], 0.8, 0.5, 'far', 6)
    ridge(23, horizon + 120, 150, [0.11, 0.15, 0.24], 0.5, 0.7, 'mid', 4)
    ridge(37, horizon + 300, 110, [0.05, 0.06, 0.09], 0.2, 0.9, 'near', 3)
    y = ridge(41, H * 0.98, 0, [0.022, 0.022, 0.028], 0.0, 1.0, 'hill', 2, summit=(W * 0.5, 1450, W * 0.30))
    json.dump(dict(W=W, H=H, horizon=float(horizon), summit_y=float(y[W // 2]), summit_x=W / 2),
              open(B('land_meta.json'), 'w'))


if __name__ == '__main__':
    build_map(); print('map')
    build_landscape(); print('landscape')
