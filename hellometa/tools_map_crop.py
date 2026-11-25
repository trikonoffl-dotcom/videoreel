"""One-off: cut the map region from Natural Earth NE1_HR_LC_SR_W_DR (public domain) + country masks.
usage: python3 tools_map_crop.py <NE1_HR_LC_SR_W_DR.tif> <ne_10m_admin_0_countries.shp>"""
import cv2, numpy as np, shapefile, json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
L0, L1, A0, A1, P = 95, 175, 50, -50, 60
im = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
c = im[(90 - A0) * P:(90 - A1) * P, (L0 + 180) * P:(L1 + 180) * P].copy()
cv2.imwrite(os.path.join(HERE, 'assets', 'map_region.jpg'), c, [cv2.IMWRITE_JPEG_QUALITY, 93])
sf = shapefile.Reader(sys.argv[2])
ia = [f[0] for f in sf.fields[1:]].index('ADM0_A3')
for code, name in [('PHL', 'ph'), ('AUS', 'au')]:
    m = np.zeros(c.shape[:2], np.uint8)
    for sr in sf.iterShapeRecords():
        if sr.record[ia] != code: continue
        pts = sr.shape.points; parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            p = np.array(pts[a:b]); px = np.stack([(p[:, 0] - L0) * P, (A0 - p[:, 1]) * P], 1)
            cv2.fillPoly(m, [np.round(px * 4).astype(np.int32)], 255, shift=2)
    cv2.imwrite(os.path.join(HERE, 'assets', f'mask_{name}.png'), m)
json.dump(dict(L0=L0, A0=A0, P=P), open(os.path.join(HERE, 'assets', 'map_meta.json'), 'w'))
print(c.shape)
