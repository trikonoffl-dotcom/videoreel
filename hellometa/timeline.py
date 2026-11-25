"""Shared timeline for picture and sound (all times in output seconds).

Music runs at 120 BPM (beat = 0.5 s, bar = 2 s). The speech edit starts at SPEECH_OFF.
Edit times (E) in data/subs_words.json are relative to the start of the speech edit.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
W, H = 1080, 1920
BPM = 120
BEAT = 60 / BPM

SPEECH_OFF = 2.4
EDL = json.load(open(os.path.join(HERE, 'data', 'edl.json')))
XF = 0.025  # audio crossfade between segments
SPEECH_LEN = sum(e - s for s, e, _ in EDL) - XF * (len(EDL) - 1)

OUTRO = 40.0          # music drop, website montage after the interview
LANG = (40.0, 42.0)   # multilingual section
CALC = (42.0, 43.5)   # missed revenue calculator
HOW = (43.5, 47.5)    # Define / Design / Deploy, steps land on beats 43.5, 44.5, 45.5
HILL = (47.5, 52.0)   # hilltop logo reveal; logo lands at HILL_HIT
HILL_HIT = 49.5
END = (52.0, 55.5)    # white closing slide
DUR = END[1]


def E(t):
    """edit time -> output time"""
    return SPEECH_OFF + t


# B-roll windows over the interview (output times). Speech keeps running underneath.
BROLL = [
    ('map',        E(1.00),  E(4.20)),   # "...from the Philippines, and I have visited the booth"
    ('web_hero',   E(5.90),  E(8.55)),   # "And I've tried the call bot."
    ('web_tech',   E(10.40), E(13.30)),  # "because it features a lot of answerable questions."
    ('web_ind',    E(15.50), E(17.97)),  # "about my childcare center in the Philippines."
    ('web_dash',   E(24.55), E(28.15)),  # "and it says there's a half-day and a whole-day package,"
    ('web_feat',   E(31.40), E(34.45)),  # "all the information that I have asked in their bot,"
]

# full-screen kinetic word moments (output times)
KINETIC = [
    ('THE BEST', E(9.60), E(10.40)),
    ('EXCELLENT.', E(34.85), E(35.88)),
]


def seg_bounds_edit():
    """edit-time start of every speech segment (jump-cut points)"""
    out, t = [], 0.0
    for s, e, _ in EDL:
        out.append((t, s))  # (edit time, source time)
        t += e - s - XF
    return out


def edit_to_source(te):
    """map an edit time to the source time of the raw interview"""
    t = 0.0
    for i, (s, e, _) in enumerate(EDL):
        d = e - s - (XF if i < len(EDL) - 1 else 0)
        if te < t + d or i == len(EDL) - 1:
            return s + (te - t)
        t += d


# transitions that get a whoosh (output times)
def transitions():
    ts = [SPEECH_OFF]
    for _, a, b in BROLL:
        ts += [a, b]
    ts += [OUTRO, CALC[0], HOW[0], HILL[0], END[0]]
    return sorted(ts)
