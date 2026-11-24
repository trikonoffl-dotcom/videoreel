#!/usr/bin/env python3
"""Morisset - simple centred subtitles (medium size, no boxes) over the original footage.
usage: render2.py <scratch> <out.mp4> [--still t1,t2,...]
Needs <scratch>/fr/*.jpg (1080x1920 frames), <scratch>/mix2.wav (new music mix), words.json (voiceover timings).
Optional: assets/logo.png -> placed top-right at original proportions.
"""
import sys, os, json, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 24
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"   # stand-in for SF Pro Display Bold
LIME = (141, 198, 63)        # eyeballed from the supplied logo (~#8DC63F)
WHITE = (255, 255, 255)
SIZE = 66                    # medium
MAXW = 880
CENTER_Y = 0.5               # vertical centre of the frame
LOGO = os.path.join(HERE, "assets", "logo.png")
L, Wh = LIME, WHITE

# phrase = (word-index range in words.json merged list handled below by text, colour map)
# Each phrase: list of (token, colour); start/end are taken from the voiceover timing of its first/last word.
PHRASES = [
 ([("WHAT",Wh),("IF",Wh),("I",Wh),("TOLD",Wh),("YOU",Wh)], 0.00, 0.95),
 ([("BLACK",L),("FRIDAY",L),("COULD",Wh),("SAVE",Wh),("YOU",Wh),("UP",Wh),("TO",Wh),("$1,000?",L)], 0.96, 3.40),
 ([("THAT'S",Wh),("RIGHT,",Wh)], 4.04, 4.95),
 ([("MORISSET",L),("MOWERS",L),("IS",Wh),("BRINGING",Wh),("YOU",Wh),("SOME",Wh),("MASSIVE",Wh),("DEALS.",L)], 5.20, 7.70),
 ([("GET",Wh),("$1,000",L),("OFF",L)], 7.88, 9.05),
 ([("CUB",L),("CADET",L),("AND",Wh),("GRAVELY",L),("ZERO-TURN",Wh),("MOWERS.",Wh)], 9.14, 11.45),
 ([("SAVE",Wh),("$500",L),("ON",Wh),("SELECTED",Wh)], 11.76, 13.35),
 ([("CUB",L),("CADET,",L),("GRAVELY",L),("AND",Wh),("VICTA",L),("RIDE-ON",Wh),("MOWERS.",Wh)], 13.40, 16.35),
 ([("PLUS,",Wh),("GET",Wh),("10%",L),("OFF",L)], 16.62, 18.05),
 ([("SMALL",Wh),("GOODS,",Wh),("MOWERS,",L),("STRIMMERS",L),("AND",Wh),("CHAINSAWS.",L)], 18.12, 21.15),
 ([("THE",Wh),("SALE",Wh),("RUNS",Wh),("FROM",Wh),("NOVEMBER",L),("27TH",L),("TO",Wh),("30TH",L)], 21.38, 24.00),
 ([("WITH",Wh),("LIMITED",Wh),("FLOOR",Wh),("STOCK",Wh),("AVAILABLE.",L)], 24.06, 25.50),
 ([("DON'T",Wh),("MISS",Wh),("OUT.",L)], 25.58, 26.95),
 ([("HEAD",Wh),("TO",Wh),("MORISSET",L),("MOWERS",L)], 27.02, 28.30),
 ([("AND",Wh),("GRAB",Wh),("YOUR",Wh),("BLACK",L),("FRIDAY",L),("SALE.",L)], 28.38, 30.05),
]

_font = ImageFont.truetype(FONT, SIZE)

def build(tokens):
    """wrap to <=MAXW (balanced into <=2-3 lines), return premultiplied RGBA float sprite"""
    sp = _font.getlength(" ")
    ws = [(t, c, _font.getlength(t)) for t, c in tokens]
    total = sum(w for _, _, w in ws) + sp * (len(ws) - 1)
    def wrap(limit):
        ls, cur, cw = [], [], 0
        for t, c, w in ws:
            add = w + (sp if cur else 0)
            if cur and cw + add > limit: ls.append(cur); cur, cw = [], 0; add = w
            cur.append((t, c, w)); cw += add
        ls.append(cur); return ls
    nl = max(1, math.ceil(total / MAXW))
    lines = wrap(total / nl * 1.12)
    if any(sum(w for _, _, w in l) + sp * (len(l) - 1) > MAXW for l in lines): lines = wrap(MAXW)
    lh = int(SIZE * 1.22); pad = 60
    lw = [sum(w for _, _, w in l) + sp * (len(l) - 1) for l in lines]
    cw_, ch_ = int(max(lw)) + 2 * pad, lh * len(lines) + 2 * pad
    face = np.zeros((ch_, cw_, 3), np.float32); mask = Image.new("L", (cw_, ch_), 0)
    cols = Image.new("RGB", (cw_, ch_), (0, 0, 0))
    md, cd = ImageDraw.Draw(mask), ImageDraw.Draw(cols)
    for li, l in enumerate(lines):
        x = (cw_ - lw[li]) / 2; y = pad + li * lh
        for t, c, w in l:
            md.text((x, y), t, font=_font, fill=255, stroke_width=3, stroke_fill=255)
            cd.text((x, y), t, font=_font, fill=c, stroke_width=3, stroke_fill=(8, 18, 4))
            x += w + sp
    a = np.asarray(mask, np.float32) / 255
    rgb = np.asarray(cols, np.float32) / 255
    # subtle shadow + soft dark halo for readability (no box)
    sh = cv2.GaussianBlur(np.roll(np.roll(a, 4, 0), 2, 1), (0, 0), 6) * 0.9
    halo = cv2.GaussianBlur(a, (0, 0), 18) * 0.55
    shadow_a = np.clip(sh + halo, 0, 1)
    out = np.zeros((ch_, cw_, 4), np.float32)
    out[..., 3] = a + shadow_a * (1 - a)
    out[..., :3] = rgb * a[..., None]                    # premultiplied; shadow is black
    return out

SPR = None
def sprites():
    global SPR
    if SPR is None: SPR = [build(p[0]) for p in PHRASES]
    return SPR

_logo = None
def logo():
    global _logo
    if _logo is None:
        if os.path.exists(LOGO):
            im = cv2.imread(LOGO, cv2.IMREAD_UNCHANGED)
            if im.shape[2] == 3: im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
            lw = 230; im = cv2.resize(im, (lw, int(im.shape[0] * lw / im.shape[1])), interpolation=cv2.INTER_AREA)
            a = im[..., 3:].astype(np.float32) / 255
            _logo = np.concatenate([im[..., [2, 1, 0]].astype(np.float32) / 255 * a, a], -1)
        else: _logo = False
    return _logo

def render_frame(i, scr):
    t = (i - 1) / FPS
    out = cv2.imread(f"{scr}/fr/{i:04d}.jpg").astype(np.float32)[..., ::-1] / 255
    for k, (toks, t0, t1) in enumerate(PHRASES):
        nxt = PHRASES[k + 1][1] if k + 1 < len(PHRASES) else 99
        s, e = t0 - 0.04, min(t1 + 0.30, nxt - 0.04)       # appears with the first word, lingers briefly
        if not (s <= t < e): continue
        spr = sprites()[k]
        a_in = min(1, (t - s) / 0.14); a_out = min(1, (e - t) / 0.12)
        alpha = min(a_in, a_out); alpha = alpha * alpha * (3 - 2 * alpha)
        sc = 0.94 + 0.06 * min(1, (t - s) / 0.18) ** 0.5
        if abs(sc - 1) > 0.003: spr = cv2.resize(spr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        h, w = spr.shape[:2]
        x0, y0 = int(W / 2 - w / 2), int(H * CENTER_Y - h / 2)
        reg = out[y0:y0 + h, x0:x0 + w]
        s_ = spr * alpha
        reg[:] = s_[..., :3] + reg * (1 - s_[..., 3:4])
    lg = logo()
    if lg is not False:
        h, w = lg.shape[:2]; x0, y0 = W - w - 56, 70
        out[y0:y0 + h, x0:x0 + w] = out[y0:y0 + h, x0:x0 + w] * (1 - lg[..., 3:4]) + lg[..., :3]
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
            cv2.imwrite(f"{scr}/stills/{t:05.2f}.jpg", render_frame(min(721, int(t * FPS) + 1), scr), [cv2.IMWRITE_JPEG_QUALITY, 92])
        sys.exit()
    n = 4; per = math.ceil(721 / n)
    jobs = [(k * per + 1, min(721, (k + 1) * per), f"{scr}/seg{k}.mp4", scr) for k in range(n)]
    with Pool(n) as p: segs = p.map(work, jobs)
    open(f"{scr}/list.txt", "w").write("".join(f"file '{s}'\n" for s in segs))
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{scr}/list.txt", "-i", f"{scr}/mix2.wav",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out])
