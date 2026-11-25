# HelloMeta.ai brand film (9:16)

**Output:** `../output/hellometa_brand_film_9x16.mp4`. It is 1080x1920, 30 fps (same as the source), H.264 with AAC 320k stereo, 53.1 s long, with subtitles burned in.

## Structure

The film opens directly on the speaker. There is no title animation.

| Time | Scene | What's on screen |
|---|---|---|
| 0.0–37.6 | Interview | Zarina's testimonial: filler removed, face-tracked framing, word-by-word subtitles |
| 1.0–4.2 | ↳ Map | Philippines → Sydney: dark relief map, glowing country outlines, curved orange route, pulsing markers and labels |
| 5.9–8.6 | ↳ Website intro | The live homepage in a floating browser window, pushing in on "Meet Your AI Phone Call Agent" |
| 9.6–10.4 | ↳ Kinetic | **IT'S THE BEST** |
| 10.4–13.3 | ↳ AI phone tech | The phone mockup and "Meet Plug", then "Understand First, Respond Later" and "Smarter Responses" |
| 15.5–18.0 | ↳ Industries | Angela, Dave and Alex cards (Appointment Booking, Real Estate, Debt Collection), then the Healthcare call card |
| 24.6–28.2 | ↳ Dashboard | CRM login, dashboard and call-log images in 3D |
| 31.4–34.5 | ↳ Features | Plug with Call Recordings, Smart Call Logs, Interactive Voice Flows, One-click AI Calling and others |
| 34.9–35.9 | ↳ Kinetic | **AND EXCELLENT.** |
| 37.6–39.6 | Multilingual | The site's 10 language cards, with Filipino (Tagalog) highlighted |
| 39.6–41.1 | Calculator | The Missed Revenue Calculator, showing the site's own $0.00 placeholders |
| 41.1–45.1 | How it works | 01 DEFINE, 02 DESIGN, 03 DEPLOY |
| 45.1–49.6 | Hilltop | Dusk landscape with parallax. The logo rises from behind the summit. The supplied logo is composited last and unaltered, so its colours and shape are exactly the original |
| 49.6–53.1 | End slide | The supplied logo, unaltered, on white, with a fade-in and gentle scale-up |

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

**Voice:** neural noise reduction with DeepFilterNet3 at 48 kHz removes the expo-hall crowd.
- The speech-to-noise-floor range rises from 8 dB in the raw audio to about 20 dB.
- Attenuation is capped at 20 dB so no words are lost.
- A 12 dB pass is blended in around "whole-day", because the stronger pass blurs that word.
- After the denoise come a high-pass filter, EQ to reduce mud and add presence, gentle compression and loudness normalisation.

**Score:** an original synthesised electronic track at 120 BPM in A minor, with no sound effects (no whooshes, impacts, pings or risers). It has a soft bed under the interview, a four-on-the-floor section for the website montage, a breakdown for the hilltop and a closing chord.

**Mix:** the music ducks 14 dB under the voice, so the voice sits 13–17 dB above the music. The final mix is −14 LUFS with a true-peak limiter.

## Rebuild

```bash
cd hellometa
python3 assets.py        # graded map, landscape layers -> build/
python3 faces.py         # face track -> data/faces.json
DF_PYTHON=/path/to/venv/bin/python DF_MODEL=/path/to/DeepFilterNet3 python3 audio.py   # -> build/mix.wav
python3 render.py --jobs 4
```

Other options:
- `python3 render.py --stills 3.6,12.5,49.8` renders QC stills to `build/`.
- `tools_map_crop.py` re-cuts the map region from the Natural Earth raster (not stored in the repo because it is 300 MB).

Requirements: `ffmpeg`, `numpy`, `scipy`, `opencv-python-headless`, `Pillow`. The speech tools also need `pyshp`, `faster-whisper` and `torch`/`torchaudio`. Run DeepFilterNet (`deepfilternet`, which needs numpy<2) in its own venv. The model is at raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip.

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
