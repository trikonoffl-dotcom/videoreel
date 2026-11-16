# Diwali reel edit

| File | What it is |
|---|---|
| `output/diwali_reel_oorum_blood_9x16.mp4` | **v2 (current)**: synced to "Oorum Blood" (song section 2:41.1 – 3:48.6), 1080x1920, 30 fps, 67.5 s |
| `output/diwali_reel_9x16.mp4` | v1: original synthesised music, 61.2 s |

## v2 structure (cut on the song's beats, 129 BPM)

| Video time | Song | Section |
|---|---|---|
| 0.0–6.0 | quiet bridge | Intro: rangoli + **HAPPY DIWALI / TRIKON TELESOFT** |
| 6.0–14.3 | soft build | Office lights door, then banner close-up, "Our office, all lit up" |
| 12.0–25.4 | half-time kicks | **Chapter 01: Diya Painting**, one shot per kick, hero diya "Handcrafted with love" |
| 25.4–29.1 | break | **Chapter 02: Creative Challenge**, then the funny pose (suspense) |
| 29.1–42.0 | DROP 1 | Fast cuts of faces writing, laughing and smiling, every 2nd beat |
| 42.0–44.8 | break | Calm shot |
| 44.8–58.6 | DROP 2 | Diya raised high, **SHINE BRIGHT** + fireworks on kicks, close-up angle, **HAPPY DIWALI** |
| 58.6–67.5 | quiet outro | Rangoli greeting, "Team Trikon Telesoft", fade out |

What changed from v1:
- **Faces always in frame:** people shots use YuNet face detection (`models/yunet.onnx`). The virtual camera follows the smoothed face track and puts the face on the upper-third line. Exposure is also set from the face so it's well lit.
- **Music sync:** cuts land on beats and kicks. Every kick adds a small zoom pulse and brightness lift, and the drops add camera shake and a gold flash.
- **Modern typography:** Bebas Neue (gold gradient) and Outfit (light, wide tracking), with wipe reveals, frosted caption pills and gold rules.
- **Even tone (shot matching):** every frame is measured in LAB colour space and normalised to the same targets: brightness 47, contrast spread 76, neutral colour balance and saturation 0.46. Exposure is anchored on the face when there is one. Corrections are smoothed over about 0.25 s so nothing flickers. A gentle local tone map (CLAHE at 22%) evens out bright windows against darker faces, highlights roll off softly so nothing clips, and the vignette is kept very light so brightness is even across the frame.
- **Look on top:** filmic S-curve with rich blacks, warm-gold highlights and slightly cool shadows, and skin-safe saturation, with subtle bloom and fine grain.

## Rebuild

```bash
cd edit
python3 render.py ../downloads/1112.mp4 ../output/diwali_reel_oorum_blood_9x16.mp4 --audio /path/to/Oorum_Blood.mp3
```

The song file is not stored in the repo. Pass your own copy with `--audio`.
`--preview 4,13,30` renders stills at those times. The edit decision list is the `S` table at the top of `render.py`, written in song times.

Requires `ffmpeg`, `numpy`, `opencv-python-headless` (4.8+ for `FaceDetectorYN`) and `Pillow`. `music.py` (v1 soundtrack) also needs `scipy`.
`faces.py` is a helper that checks face positions over source ranges.

Fonts: Google Fonts (SIL OFL). Face model: OpenCV Zoo YuNet (MIT).
