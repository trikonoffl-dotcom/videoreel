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

SPEECH_OFF = 0.0   # the film opens directly on the speaker (no title animation)
MUSIC_SHIFT = 2.4  # the score is composed on a grid that starts 2.4 s before the picture
EDL = json.load(open(os.path.join(HERE, 'data', 'edl.json')))
XF = 0.025  # audio crossfade between segments
SPEECH_LEN = sum(e - s for s, e, _ in EDL) - XF * (len(EDL) - 1)

OUTRO = 37.6          # music drop, website montage after the interview
LANG = (37.6, 39.6)   # multilingual section
CALC = (39.6, 41.1)   # missed revenue calculator
HOW = (41.1, 45.1)    # Define / Design / Deploy, one step per beat
HILL = (45.1, 49.6)   # hilltop logo reveal; logo lands at HILL_HIT
HILL_HIT = 47.1
END = (49.6, 53.1)    # white closing slide
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
    ts = []
    for _, a, b in BROLL:
        ts += [a, b]
    ts += [OUTRO, CALC[0], HOW[0], HILL[0], END[0]]
    return sorted(ts)
