#!/usr/bin/env python3
"""Morisset - reference-style captions (1-3 words at a time, sentence case, white, tight tracking, centred)
+ white logo end card.   usage: render3.py <scratch> <out.mp4> [--still t,...]
Logo: put the original at assets/logo.png (transparent PNG preferred) - it is centred on the white end card untouched."""
import sys, os, json, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 24
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"   # stand-in for SF Pro Display Bold
SIZE, TRACK, CY = 88, -0.035, 0.58
LOGO = os.path.join(HERE, "assets", "logo.png")
SRC_FRAMES, END_FRAMES = 721, 72                      # 30.04 s footage + 3 s white end card
TOTAL = SRC_FRAMES + END_FRAMES

# user's script (spelling/punctuation as supplied), chunked 1-3 words; token indices follow the voiceover word timing
WORDS = json.load(open(os.path.join(HERE, "words.json")))      # whisper word timings (index = spoken word)
IDX = [  # (display text, first spoken-word index, last spoken-word index)
 ("What if I",0,2),("told you",3,4),("Black Friday",5,6),("could save you",7,9),("up to",10,11),("$1,000?",12,13),
 ("That's right!",14,15),("Morisset Mowers",16,17),("is bringing you",18,20),("some massive",21,22),("deals!",23,23),
 ("Get $1,000 off",24,27),("Cub Cadet",28,29),("and Gravely",30,31),("zero-turn mowers.",32,34),
 ("Save $500",35,36),("on selected",37,38),("Cub Cadet,",39,40),("Gravely and",41,42),("Victa ride-on",43,45),("mowers.",46,46),
 ("Plus, get",47,48),("10% off",49,51),("small goods,",52,53),("mowers,",54,54),("trimmers",55,55),("and chainsaws!",56,57),
 ("The sale runs",58,60),("from November",61,62),("27th to 30th,",63,65),("with limited",66,67),("floor stock",68,69),("available.",70,70),
 ("Don't miss out!",71,73),("Head to",74,75),("Morisset Mowers",76,77),("and grab your",78,80),("Black Friday",81,82),("sale!",83,83),
]
def times(i0, i1):
    # merged token spans in words.json: "$1"+",000?" (12,13)  "$1"+",000" (25,26)  "zero"+"-turn" (32,33)  "ride"+"-on" (44,45)  "10"+"%" (49,50)
    return WORDS[i0][1], WORDS[i1][2]
CAPS = []
for k, (txt, a, b) in enumerate(IDX):
    s, e = times(a, b)
    CAPS.append([txt, s, e])
for k in range(len(CAPS)):                              # continuous captions like the reference: hold until next starts
    nxt = CAPS[k + 1][1] if k + 1 < len(CAPS) else SRC_FRAMES / FPS
    CAPS[k][2] = nxt if nxt - CAPS[k][2] < 0.55 else CAPS[k][2] + 0.35

_font = ImageFont.truetype(FONT, SIZE)
def adv(c1, c2):
    return _font.getlength(c1 + c2) - _font.getlength(c2) + TRACK * SIZE
def sprite(text):
    pad = 70
    w = sum(adv(text[i], text[i + 1]) for i in range(len(text) - 1)) + _font.getlength(text[-1])
    cw, ch = int(w) + 2 * pad, int(SIZE * 1.5) + 2 * pad
    m = Image.new("L", (cw, ch), 0); d = ImageDraw.Draw(m); x = pad
    for i, c in enumerate(text):
        d.text((x, pad), c, font=_font, fill=255)
        x += adv(c, text[i + 1]) if i + 1 < len(text) else 0
    a = np.asarray(m, np.float32) / 255
    sh = cv2.GaussianBlur(np.roll(a, 4, 0), (0, 0), 9) * 0.55 + cv2.GaussianBlur(a, (0, 0), 22) * 0.22
    sh = np.clip(sh, 0, 1)
    o = np.zeros((ch, cw, 4), np.float32); o[..., 3] = a + sh * (1 - a); o[..., :3] = a[..., None]   # white premultiplied, black shadow
    return o
_spr = {}
def spr(k):
    if k not in _spr: _spr[k] = sprite(CAPS[k][0])
    return _spr[k]

_logo = None
def logo():
    global _logo
    if _logo is None:
        if os.path.exists(LOGO):
            im = cv2.imread(LOGO, cv2.IMREAD_UNCHANGED)
            if im.shape[2] == 3: im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            lw = int(W * 0.66); im = cv2.resize(im, (lw, int(im.shape[0] * lw / im.shape[1])), interpolation=cv2.INTER_AREA)
            a = im[..., 3:].astype(np.float32) / 255
            _logo = (im[..., [2, 1, 0]].astype(np.float32) / 255, a)
        else: _logo = False
    return _logo

def render_frame(i, scr):
    if i <= SRC_FRAMES:
        out = cv2.imread(f"{scr}/fr/{i:04d}.jpg").astype(np.float32)[..., ::-1] / 255
        t = (i - 1) / FPS
        for k, (txt, s, e) in enumerate(CAPS):
            if s - 0.03 <= t < e:
                sp = spr(k); u = min(1, (t - (s - 0.03)) / 0.13)
                al = min(1, u * 1.6); sc = 0.92 + 0.08 * (1 - (1 - u) ** 3)
                if sc < 0.999: sp = cv2.resize(sp, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                h, w = sp.shape[:2]; x0, y0 = int(W / 2 - w / 2), int(H * CY - h / 2)
                reg = out[y0:y0 + h, x0:x0 + w]; s_ = sp * al
                reg[:] = s_[..., :3] + reg * (1 - s_[..., 3:4])
                break
        fade = max(0, (i - (SRC_FRAMES - 6)) / 6.0)               # last 0.25 s dissolves to the white end card
        if fade > 0: out = out * (1 - fade) + fade
    else:
        out = np.ones((H, W, 3), np.float32)
        lg = logo()
        if lg is not False:
            u = min(1, (i - SRC_FRAMES) / 10.0); u = u * u * (3 - 2 * u)            # logo eases in
            rgb, a = lg; h, w = a.shape[:2]; x0, y0 = (W - w) // 2, (H - h) // 2
            sc = 0.96 + 0.04 * u
            if sc < 0.999:
                rgb = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA); a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)[..., None] if a.ndim == 3 else cv2.resize(a, None, fx=sc, fy=sc)[..., None]
                h, w = rgb.shape[:2]; x0, y0 = (W - w) // 2, (H - h) // 2
            a = a.reshape(h, w, 1) * u
            out[y0:y0 + h, x0:x0 + w] = out[y0:y0 + h, x0:x0 + w] * (1 - a) + rgb * a
    return (np.clip(out, 0, 1) * 255 + .5).astype(np.uint8)[..., ::-1]

def work(a):
    lo, hi, path, scr = a
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", "24", "-bf", "2",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", path], stdin=subprocess.PIPE)
    for i in range(lo, hi + 1): p.stdin.write(render_frame(i, scr).tobytes())
    p.stdin.close(); p.wait(); return path

if __name__ == "__main__":
    scr, out = sys.argv[1], sys.argv[2]
    if "--still" in sys.argv:
        os.makedirs(f"{scr}/stills", exist_ok=True)
        for t in [float(x) for x in sys.argv[sys.argv.index("--still") + 1].split(",")]:
            cv2.imwrite(f"{scr}/stills/{t:05.2f}.jpg", render_frame(min(TOTAL, int(t * FPS) + 1), scr), [cv2.IMWRITE_JPEG_QUALITY, 92])
        sys.exit()
    n = 4; per = math.ceil(TOTAL / n)
    jobs = [(k * per + 1, min(TOTAL, (k + 1) * per), f"{scr}/seg{k}.mp4", scr) for k in range(n)]
    with Pool(n) as p: segs = p.map(work, jobs)
    open(f"{scr}/list.txt", "w").write("".join(f"file '{s}'\n" for s in segs))
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{scr}/list.txt", "-i", f"{scr}/mix3.wav",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{TOTAL / FPS:.3f}", "-movflags", "+faststart", out])
