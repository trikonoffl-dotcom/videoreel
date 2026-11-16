# Diwali reel edit

Final output: `output/diwali_reel_9x16.mp4`: 1080x1920 (9:16), 30 fps, 61.2 s, H.264 + AAC.

## Structure

| Time | Section | Footage / content |
|---|---|---|
| 0.0–4.8 | Intro | Rangoli 1 spins in over fireworks, "Happy Diwali · Trikon Telesoft · Festival of Lights" |
| 4.8–9.6 | Office lights | Entrance door with lights, then a close-up of the "Happy Diwali" banner |
| 9.6–26.4 | Chapter 1: Diya Painting | 5 painting close-ups, then a top-down hero shot of the finished diya ("handcrafted with love") |
| 26.4–48.0 | Chapter 2: Creative Challenge | Writing contest shots, reaction shots and the funny pose (punch-in zoom) |
| 48.0–55.2 | Finale | Diya raised high with fireworks ("Shine Bright!"), then the entrance lights |
| 55.2–61.2 | Outro | Rangoli 2 with the greeting and "Team Trikon Telesoft", fade to black |

Edit techniques:
- **Virtual camera angles:** push-ins, pull-outs, pans, punch-ins and tilted (dutch) angles, cropped from the 4K source with no loss of quality.
- **Warm festive grade:** S-curve, maroon shadows, gold highlights, more vibrance, bloom and vignette.
- **Overlays:** bokeh lights, fireworks and twinkling sparkles.
- **Transitions:** rangoli zoom-throughs between chapters, plus gold flashes, zoom-blur and crossfades.
- **Music:** original and synthesised (`music.py`) at 100 BPM, with cuts on the beat. It uses tanpura drone, santoor melody, dhol/tabla groove, temple bells and a build-up into the finale.

## Rebuild

```bash
python3 music.py /tmp/music.wav
python3 render.py ../downloads/1112.mp4 assets fonts ../output/diwali_reel_9x16.mp4 --audio /tmp/music.wav
```

Requires `ffmpeg`, `numpy`, `scipy`, `opencv-python-headless` and `Pillow`. The shot list (EDL) and transitions are the `SHOTS` / `TRANS` tables at the top of `render.py`. Change the source in-points, durations or camera keyframes there.
Use `--preview 3.5,11,24` to render still frames at given times for a quick check.

Fonts are from Google Fonts (SIL Open Font License): Great Vibes, Yatra One, Cinzel and Poppins.
