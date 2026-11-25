"""Face track of the raw interview (YuNet) -> data/faces.json: per-frame face box, smoothed."""
import cv2, numpy as np, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cap = cv2.VideoCapture(os.path.join(HERE, 'input', 'Raw video.mp4'))
det = cv2.FaceDetectorYN.create(os.path.join(HERE, '..', 'edit', 'models', 'yunet.onnx'), '', (406, 720), 0.6)
out = []
while True:
    ok, f = cap.read()
    if not ok: break
    det.setInputSize((f.shape[1], f.shape[0]))
    _, faces = det.detect(f)
    if faces is None: out.append(None); continue
    # main speaker = largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])[:4]
    out.append([float(x + w / 2), float(y + h / 2), float(w), float(h)])
arr = np.array([o if o else [np.nan] * 4 for o in out])
for c in range(4):  # fill gaps, then smooth ~0.5 s
    v = arr[:, c]; idx = np.arange(len(v)); good = ~np.isnan(v)
    v = np.interp(idx, idx[good], v[good])
    k = np.ones(15) / 15; arr[:, c] = np.convolve(np.pad(v, 7, mode='edge'), k, 'valid')
json.dump(arr.round(1).tolist(), open(os.path.join(HERE, 'data', 'faces.json'), 'w'))
print(len(out), 'frames; missing', sum(o is None for o in out))
print('cx range', arr[:, 0].min(), arr[:, 0].max(), 'cy range', arr[:, 1].min(), arr[:, 1].max(), 'h', arr[:, 3].mean())
