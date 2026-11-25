# HelloMeta.ai brand film (9:16)

**Output:** `../output/hellometa_brand_film_9x16.mp4`. It is 1080x1920, 30 fps (same as the source), H.264 with AAC 320k stereo, about 55.5 s long, with subtitles burned in.

## Structure

| Time | Scene | What's on screen |
|---|---|---|
| 0.0–2.4 | Hook | Kinetic type from the live site's hero headline: **MEET YOUR / AI PHONE CALL / AGENT**, an orange light line and a grid, landing on the music hit, then a zoom-blur whip into the speaker |
| 2.4–40.0 | Interview | Zarina's testimonial: filler removed, face-tracked framing, word-by-word subtitles |
| 3.4–6.6 | ↳ Map | Philippines → Sydney: dark relief map, glowing country outlines, curved orange route with a travel point, pulsing markers and labels |
| 8.3–11.0 | ↳ Website intro | The live homepage in a floating browser window, pushing in on "Meet Your AI Phone Call Agent" |
| 12.0–12.8 | ↳ Kinetic | **IT'S THE BEST** |
| 12.8–15.7 | ↳ AI phone tech | The phone mockup and the "Meet Plug" robot, then a tilt down to "Understand First, Respond Later" and "Smarter Responses" |
| 17.9–20.4 | ↳ Industries | Camera tracks across the Angela, Dave and Alex cards (Appointment Booking, Real Estate, Debt Collection), then the Healthcare call card |
| 27.0–30.6 | ↳ Dashboard | CRM login, dashboard and call-log images in 3D, pushing in on the Overview panel |
| 33.8–36.9 | ↳ Features | Plug with Call Recordings, Smart Call Logs, Interactive Voice Flows (the script editor), One-click AI Calling and others |
| 37.3–38.3 | ↳ Kinetic | **AND EXCELLENT.** |
| 40.0–42.0 | Multilingual | The site's 10 language cards fly in, and Filipino (Tagalog) is highlighted |
| 42.0–43.5 | Calculator | The Missed Revenue Calculator, showing the site's own $0.00 placeholders (no invented numbers) |
| 43.5–47.5 | How it works | 01 DEFINE, 02 DESIGN and 03 DEPLOY land on the beat, joined by an orange line |
| 47.5–52.0 | Hilltop | Dusk landscape with parallax layers. The logo rises from behind the summit, rim-lit and light-wrapped, with god rays, then the scene blooms to white |
| 52.0–55.5 | End slide | The original logo with the orange-to-black metallic treatment on white, a scale-up and a light sweep |

## Speech edit (`data/edl.json`)

The interview was cut down to 13 segments (36.3 s of speech, from 52 s of raw footage). These were removed:
- the crew chatter before the take ("Yeah. Good? Yes. Just again. Okay.")
- "So yeah,"
- "or I don't know how you call it"
- "Like,"
- a "like" inside a 1.9 s pause
- "You know,"
- a trailing "and…"
- the false start "if I would be like,"
- the crowd's thank-yous at the end

Every meaningful statement is kept. Cut points snap to nearby energy minima and are joined with 25 ms equal-power crossfades. The edit was checked by transcribing it again with Whisper. Subtitle timings come from wav2vec2 CTC forced alignment (torchaudio MMS_FA) on the edited audio (`data/subs_words.json`). A word never appears before it is spoken.

## Sound (`audio.py`)

**Voice processing:** high-pass filter, light FFT denoise (kept gentle because heavier settings made the voice less intelligible), EQ to reduce mud and add presence, compression, then loudness normalisation.

**Score:** an original synthesised electronic track at 120 BPM in A minor. Sections:
- a hook impact with a riser
- a half-time bed under the interview
- a snare build
- a four-on-the-floor drop for the website montage
- a cinematic breakdown with a boom when the logo lands
- a bell chord on the end slide

**SFX:** whooshes on transitions, map pings, typography hits and a logo shimmer.

**Mix:** the music ducks 14 dB under the voice. The voice sits 9–15 dB above the background throughout, and the final mix is −14 LUFS with a true-peak limiter.

## Rebuild

```bash
cd hellometa
python3 assets.py        # metallic logo, graded map, landscape layers -> build/
python3 faces.py         # face track -> data/faces.json
python3 audio.py         # voice, score, sfx, mix -> build/mix.wav
python3 render.py --jobs 4
```

Other options:
- `python3 render.py --stills 3.6,12.5,49.8` renders QC stills to `build/`.
- `tools_map_crop.py` re-cuts the map region from the Natural Earth raster (not stored in the repo because it is 300 MB).

Requirements: `ffmpeg`, `numpy`, `scipy`, `opencv-python-headless`, `Pillow`. The speech tools also need `pyshp`, `faster-whisper` and `torch`/`torchaudio`.

## Sources

| Source | What it is | Licence / origin |
|---|---|---|
| `input/Raw video.mp4` | The interview | Supplied |
| `input/reference.mp4` | Editing style reference | Supplied |
| `input/map_reference.mp4` | Map style reference | Supplied |
| `assets/HelloMeta-Logo-1.png` | The original logo, downloaded from hellometa.ai. Its shape is never altered; only the colour treatment changes | HelloMeta |
| `assets/web/` | Captures of the live https://hellometa.ai (desktop, 3x density). The sticky header and back-to-top button are hidden on section captures. No forms were touched | Captured from the live site |
| `assets/map_region.jpg` | Natural Earth I raster with shaded relief, water and drainage (NE1_HR_LC_SR_W_DR) | Public domain |
| Country outlines | Natural Earth 10m admin 0 | Public domain |
| `assets/BalooThambi2.ttf` | Baloo Thambi 2 (brand font) | Google Fonts, SIL OFL |
| Face detection | YuNet, from `../edit/models/yunet.onnx` | OpenCV Zoo, MIT |

The hilltop landscape and all music are generated procedurally, so no stock footage or AI-generated imagery is used.
