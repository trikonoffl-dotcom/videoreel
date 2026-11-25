"""CSS-pixel rectangles (x, y, w, h) of live-site elements used in the film, and a crop helper.
Captures in assets/web/s_<section>.png are 1440 CSS px wide at 3x density."""
import cv2, os
HERE = os.path.dirname(os.path.abspath(__file__))
DSF = 3
SECTIONS = dict(hero=0, building=860, industries=1480, missed=2290, humanlike=3120, dashboard=3870,
                features=4700, howitworks=5690, languages=6430)
R = {
    'hero_view':   ('hero', 0, 0, 1440, 900),
    'headline':    ('hero', 380, 118, 680, 115),
    'phone':       ('hero', 530, 300, 384, 512),
    'plug':        ('hero', 905, 296, 400, 510),
    'card_understand': ('humanlike', 40, 3322, 410, 445),
    'card_smarter':    ('humanlike', 990, 3322, 410, 445),
    'card_speak':      ('humanlike', 515, 3322, 410, 445),
    'ind_title':   ('industries', 120, 1500, 1200, 140),
    'ind_cards':   ('industries', 0, 1655, 1440, 560),
    'healthcare':  ('building', 840, 1262, 345, 140),
    'dash_title':  ('dashboard', 220, 3880, 1000, 110),
    'dash_login':  ('dashboard', 10, 4127, 446, 412),
    'dash_main':   ('dashboard', 496, 4049, 446, 567),
    'dash_logs':   ('dashboard', 983, 4149, 446, 367),
    'feat_all':    ('features', 0, 4730, 1440, 900),
    'feat_plug':   ('features', 500, 5150, 440, 450),
    'f_voiceflows': ('features', 42, 4998, 317, 95),
    'f_recordings': ('features', 552, 4923, 336, 95),
    'f_oneclick':   ('features', 1042, 4998, 317, 95),
    'f_calllogs':   ('features', 10, 5178, 304, 95),
    'f_forwarding': ('features', 1126, 5178, 304, 95),
    'f_bulk':       ('features', 38, 5353, 304, 120),
    'calc_phone':  ('missed', 580, 2566, 370, 487),
    'calc_result': ('missed', 1000, 2699, 370, 307),
    'calc_title':  ('missed', 0, 2600, 560, 330),
    'lang_title':  ('languages', 0, 6455, 760, 160),
}
for i, name in enumerate(['English', 'Hindi', 'Chinese', 'Arabic', 'Filipino', 'French', 'Greek', 'Italian', 'Spanish', 'Vietnamese']):
    R['lang_' + name] = ('languages', 10 + 290 * (i % 5), 6668 + 140 * (i // 5), 260, 120)

_cache = {}


def section(name):
    if name not in _cache:
        _cache[name] = cv2.imread(os.path.join(HERE, 'assets', 'web', f's_{name}.png'))
    return _cache[name]


def crop(key, pad=0):
    sec, x, y, w, h = R[key]
    img = section(sec)
    oy = SECTIONS[sec]
    x0, y0 = int((x - pad) * DSF), int((y - oy - pad) * DSF)
    x1, y1 = int((x + w + pad) * DSF), int((y - oy + h + pad) * DSF)
    x0, y0 = max(x0, 0), max(y0, 0)
    return img[y0:y1, x0:x1].copy()


if __name__ == '__main__':
    import numpy as np
    tiles = []
    for k in R:
        c = crop(k)
        s = 300 / max(c.shape[:2])
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA)
        t = np.full((330, 310, 3), 60, np.uint8); t[:c.shape[0], :c.shape[1]] = c
        cv2.putText(t, k, (4, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 6: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(os.path.join(HERE, 'build', 'rects_check.jpg'), np.vstack(rows))
