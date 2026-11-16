"""Detect faces (YuNet) over source ranges; print normalized face tracks and write a check sheet.
usage: faces.py SRC MODEL OUT_JSON SHEET name:start:dur ..."""
import cv2, numpy as np, subprocess, sys, json
SRC, MODEL, OUTJ, SHEET = sys.argv[1:5]
W, H = 540, 960
det = cv2.FaceDetectorYN.create(MODEL, "", (W, H), 0.55, 0.3, 5000)
res = {}; tiles = []
for spec in sys.argv[5:]:
    name, st, du = spec.split(":"); st = float(st); du = float(du)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(st), "-i", SRC, "-t", str(du), "-vf", f"scale={W}:{H}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    tr = []; i = 0; keep = []
    while True:
        b = p.stdout.read(W * H * 3)
        if len(b) < W * H * 3: break
        f = np.frombuffer(b, np.uint8).reshape(H, W, 3)
        if i % 3 == 0:
            _, faces = det.detect(f)
            best = None
            if faces is not None:
                best = max(faces, key=lambda r: r[2] * r[3])
                tr.append([i / 30, (best[0] + best[2] / 2) / W, (best[1] + best[3] / 2) / H, best[3] / H, float(best[-1])])
            if i % 15 == 0:
                g = f.copy()
                if faces is not None:
                    for r in faces:
                        x, y, w, h = map(int, r[:4]); cv2.rectangle(g, (x, y), (x + w, y + h), (0, 255, 0), 3)
                keep.append(g)
        i += 1
    res[name] = tr
    a = np.array(tr) if tr else None
    if a is not None:
        print(f"{name:10s} hits={len(tr):3d}/{i//3:3d} cx={np.median(a[:,1]):.2f} cy={np.median(a[:,2]):.2f} h={np.median(a[:,3]):.3f} cx_rng=({a[:,1].min():.2f},{a[:,1].max():.2f}) cy_rng=({a[:,2].min():.2f},{a[:,2].max():.2f})")
    else:
        print(f"{name:10s} no faces")
    sel = [keep[0], keep[len(keep) // 2], keep[-1]]
    row = np.hstack([cv2.resize(k, (180, 320)) for k in sel])
    cv2.putText(row, name, (5, 25), 0, 0.8, (0, 255, 255), 2)
    tiles.append(row)
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(SHEET, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
json.dump(res, open(OUTJ, "w"))
