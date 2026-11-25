"""Rendering primitives: easing, text sprites, compositing, 3D card projection, blur effects.

Frames are float32 BGR in [0, 1], 1080x1920.
"""
import cv2, numpy as np, os
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
FONT = os.path.join(HERE, 'assets', 'BalooThambi2.ttf')

ORANGE = np.array([0x1E, 0x89, 0xF6], np.float32) / 255   # #F6891E (BGR)
GREY = np.array([0x61, 0x61, 0x60], np.float32) / 255     # #606161
WHITE = np.ones(3, np.float32)
BLACK = np.zeros(3, np.float32)


# ------------------------------------------------------------------ easing
def clamp01(x):
    return float(min(max(x, 0.0), 1.0))


def lin(t, a, b):
    """0..1 progress of t through [a, b]"""
    return clamp01((t - a) / (b - a)) if b > a else float(t >= a)


def ease_io(x):  # cubic in-out (bezier-like)
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=3):
    return clamp01(x) ** p


def ease_expo_out(x):
    x = clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_expo_io(x):
    x = clamp01(x)
    if x in (0.0, 1.0):
        return x
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def mix(a, b, x):
    return a + (b - a) * x


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ text sprites
@lru_cache(maxsize=64)
def _font(size, wght):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_axes([wght])
    except Exception:
        pass
    return f


@lru_cache(maxsize=2048)
def text_sprite(text, size, wght=800, color=(1.0, 1.0, 1.0), track=0.0, pad=None):
    """RGBA float32 sprite (BGR + alpha), premultiplied=False. `track` is extra letter spacing in px."""
    f = _font(int(size), int(wght))
    pad = int(size * 0.35) if pad is None else pad
    # measure
    if track == 0:
        l, t, r, b = f.getbbox(text)
        w = r - l
    else:
        adv = [f.getlength(ch) for ch in text]
        w = sum(adv) + track * (len(text) - 1)
        l, t, r, b = f.getbbox(text)
    asc, desc = f.getmetrics()
    hgt = asc + desc
    img = Image.new('L', (int(w + 2 * pad + abs(min(l, 0))), int(hgt + 2 * pad)), 0)
    d = ImageDraw.Draw(img)
    if track == 0:
        d.text((pad - l, pad), text, font=f, fill=255)
    else:
        x = pad - min(l, 0)
        for ch, a in zip(text, adv):
            d.text((x, pad), ch, font=f, fill=255)
            x += a + track
    a = np.asarray(img, np.float32) / 255
    col = np.array(color, np.float32)
    rgba = np.dstack([np.broadcast_to(col, a.shape + (3,)), a]).astype(np.float32)
    return rgba


def sprite_size(spr):
    return spr.shape[1], spr.shape[0]


# ------------------------------------------------------------------ compositing
def over(dst, src_rgb, alpha):
    """in-place alpha over: dst = dst*(1-a) + src*a"""
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    dst *= (1 - a)
    dst += src_rgb * a
    return dst


def place(dst, spr, cx, cy, scale=1.0, alpha=1.0, rot=0.0, shear=0.0, interp=cv2.INTER_LINEAR):
    """composite an RGBA sprite centred at (cx, cy) with scale/rotation; sub-pixel accurate."""
    if alpha <= 0.003 or scale <= 0.001:
        return dst
    sh, sw = spr.shape[:2]
    c, s = np.cos(np.radians(rot)) * scale, np.sin(np.radians(rot)) * scale
    M = np.array([[c, -s + shear * scale, 0], [s, c, 0]], np.float32)
    M[:, 2] = np.array([cx, cy]) - M[:, :2] @ np.array([sw / 2, sh / 2])
    # bounding box in dst
    corners = np.array([[0, 0, 1], [sw, 0, 1], [0, sh, 1], [sw, sh, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, dst.shape[1]), min(y1, dst.shape[0])
    if x1c <= x0c or y1c <= y0c:
        return dst
    M2 = M.copy(); M2[:, 2] -= [x0c, y0c]
    src = spr
    if scale < 0.5:  # pre-shrink to avoid aliasing
        k = max(1, int(0.5 / scale))
        src = cv2.resize(spr, (max(1, sw // (k + 1)), max(1, sh // (k + 1))), interpolation=cv2.INTER_AREA)
        f = sw / src.shape[1]
        M2 = M2.copy(); M2[:, :2] *= f
    roi = cv2.warpAffine(src, M2, (x1c - x0c, y1c - y0c), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = roi[..., 3] * alpha
    reg = dst[y0c:y1c, x0c:x1c]
    if reg.shape[2] == 4:  # layer with its own alpha
        over(reg[..., :3], roi[..., :3], a)
        reg[..., 3] += a * (1 - reg[..., 3])
    else:
        over(reg, roi[..., :3], a)
    return dst


def glow(dst, mask_rgb, sigma, strength):
    """additive glow from an RGB light layer"""
    small = cv2.resize(mask_rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    g = cv2.resize(g, (dst.shape[1], dst.shape[0]), interpolation=cv2.INTER_LINEAR)
    dst += g * strength
    return dst


def blur_fast(img, sigma):
    if sigma < 0.5:
        return img
    f = 4 if sigma > 12 else 2 if sigma > 4 else 1
    if f == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def zoom_blur(img, amount, cx=W / 2, cy=H / 2, n=6):
    """radial (zoom) blur: average of scaled copies"""
    if amount < 0.004:
        return img
    acc = np.zeros_like(img)
    for i in range(n):
        s = 1 + amount * i / (n - 1)
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        acc += cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def motion_blur(img, dx, dy, n=7):
    if abs(dx) + abs(dy) < 1.5:
        return img
    acc = np.zeros_like(img)
    for i in range(n):
        f = i / (n - 1) - 0.5
        M = np.array([[1, 0, dx * f], [0, 1, dy * f]], np.float32)
        acc += cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def vignette(strength=0.35, power=2.2):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xs - W / 2) / (W * 0.62)) ** 2 + ((ys - H / 2) / (H * 0.62)) ** 2)
    return (1 - strength * np.clip(d, 0, 1.6) ** power).astype(np.float32)[..., None]


VIG = None


def get_vig():
    global VIG
    if VIG is None:
        VIG = vignette()
    return VIG


_grain_cache = {}


def grain(frame_idx, amount=0.018):
    k = frame_idx % 12
    if k not in _grain_cache:
        rng = np.random.default_rng(k)
        g = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
        _grain_cache[k] = g[..., None]
    return _grain_cache[k] * amount


# ------------------------------------------------------------------ image pyramids & 3D cards
class Pyr:
    """image + half-res levels; sample() picks the right level for the scale to avoid aliasing"""

    def __init__(self, img):
        self.levels = [img]
        while min(self.levels[-1].shape[:2]) > 256 and len(self.levels) < 5:
            l = self.levels[-1]
            self.levels.append(cv2.resize(l, (l.shape[1] // 2, l.shape[0] // 2), interpolation=cv2.INTER_AREA))

    @property
    def shape(self):
        return self.levels[0].shape

    def level_for(self, scale):
        lv = 0
        while scale < 0.55 and lv < len(self.levels) - 1:
            scale *= 2; lv += 1
        return lv

    def crop_to(self, rect, size, interp=cv2.INTER_LINEAR):
        """resample rect=(x,y,w,h) in level-0 px to output size (w,h)"""
        x, y, w, h = rect
        ow, oh = size
        lv = self.level_for(ow / w)
        f = 2 ** lv
        img = self.levels[lv]
        sx, sy = ow / (w / f), oh / (h / f)
        M = np.array([[sx, 0, -x / f * sx], [0, sy, -y / f * sy]], np.float32)
        return cv2.warpAffine(img, M, (int(ow), int(oh)), flags=interp, borderMode=cv2.BORDER_REPLICATE)


def rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    r = int(r)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)]:
        cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    return m.astype(np.float32) / 255


def project(pts3, cam_z=-1800.0, f=1800.0, cx=W / 2, cy=H / 2):
    """pinhole projection; camera at z=cam_z looking +z. pts3: (N,3)"""
    z = pts3[:, 2] - cam_z
    return np.stack([cx + f * pts3[:, 0] / z, cy + f * pts3[:, 1] / z], 1)


def rot_mat(rx, ry, rz=0.0):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def card3d(dst, rgba, center, size_w, rx=0, ry=0, rz=0, alpha=1.0, cam=(0, 0, -1800), f=1800.0,
           shadow=0.5, glow_col=None, glow_amt=0.0):
    """draw an RGBA image as a 3D card. center=(X,Y,Z) world, size_w = card width in world units.
    returns the projected alpha mask (for glow etc.)"""
    if alpha <= 0.003:
        return None
    ih, iw = rgba.shape[:2]
    hw, hh = size_w / 2, size_w * ih / iw / 2
    corners = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], np.float64)
    R = rot_mat(rx, ry, rz)
    wpts = corners @ R.T + np.array(center) - np.array([cam[0], cam[1], 0])
    if np.any(wpts[:, 2] - cam[2] < 50):
        return None
    dpts = project(wpts, cam_z=cam[2], f=f).astype(np.float32)
    # choose resolution of source so projected size ~ source size
    proj_w = np.linalg.norm(dpts[1] - dpts[0])
    scale = proj_w / iw
    src = rgba
    if scale < 0.55:
        k = int(np.log2(0.55 / scale)) + 1
        for _ in range(k):
            if min(src.shape[:2]) < 32:
                break
            src = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2), interpolation=cv2.INTER_AREA)
    sh_, sw_ = src.shape[:2]
    spts = np.array([[0, 0], [sw_, 0], [sw_, sh_], [0, sh_]], np.float32)
    x0, y0 = np.floor(dpts.min(0)).astype(int) - 60
    x1, y1 = np.ceil(dpts.max(0)).astype(int) + 60
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1c <= x0c or y1c <= y0c:
        return None
    Hm = cv2.getPerspectiveTransform(spts, dpts - np.array([x0c, y0c], np.float32))
    roi = cv2.warpPerspective(src, Hm, (x1c - x0c, y1c - y0c), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = roi[..., 3] * alpha
    reg = dst[y0c:y1c, x0c:x1c]
    if shadow > 0:
        sh = cv2.GaussianBlur(a, (0, 0), 18)
        sh = np.roll(sh, 24, axis=0)
        reg *= (1 - shadow * sh)[..., None]
    if glow_col is not None and glow_amt > 0:
        g = cv2.GaussianBlur(a, (0, 0), 28)
        reg += (g * glow_amt)[..., None] * glow_col
    over(reg, roi[..., :3], a)
    full = np.zeros((H, W), np.float32)
    full[y0c:y1c, x0c:x1c] = a
    return full


def to_rgba(bgr, radius=0):
    h, w = bgr.shape[:2]
    a = rounded_mask(w, h, radius) if radius > 0 else np.ones((h, w), np.float32)
    return np.dstack([bgr, a]).astype(np.float32)
