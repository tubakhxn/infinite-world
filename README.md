# Gesture World Builder (OpenCV + MediaPipe)

An interactive prototype inspired by "an infinite world builder with gestures": point your hand at on-screen dials and pinch to set three parameters — biology, tech, culture — then hold an open palm for ~1s to generate a new speculative world title and blurb. Runs locally with OpenCV + MediaPipe and optionally calls an LLM (Gemini or OpenAI) if you set an API key.

## Features
- Real-time hand tracking with MediaPipe Hands
- Three neon-style HUD dials controllable by pinch-and-rotate
- Open-palm hold to trigger generation
- LLM support: Gemini 1.5 Flash or OpenAI (fallback deterministic text if no key)

## Requirements
- Python 3.10–3.12 recommended
- Windows, macOS, or Linux with a webcam

## Setup (Windows PowerShell)
```powershell
# 1) Create and activate a virtual env (recommended)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Set an API key for LLM text
# For Gemini:
$env:GEMINI_API_KEY = "<your_key>"
# Or for OpenAI:
$env:OPENAI_API_KEY = "<your_key>"
```

## Run
```powershell
python -m app.main --camera 0 --width 1280 --height 720
```
Keys: press Q or Esc to quit. Use `--no-flip` to disable mirroring if needed.

## Gestures
- Cursor: index fingertip
- Grab: pinch thumb + index (shows green cursor)
- Adjust a dial: pinch near a dial and rotate around it to change the value
- Generate world: open palm (not pinching) and hold steady ~1 second

## Notes & Troubleshooting
- If the window is black or slow, reduce resolution: `--width 640 --height 480`.
- If hand isn’t detected, ensure good lighting and your whole hand is visible.
- MediaPipe may auto-download models on first run; allow a few seconds.
- If you don’t set an API key, the app uses a deterministic fallback generator.

## Project Layout
```
app/
  main.py          # camera loop, gesture logic, HUD wiring
  hand_tracking.py # MediaPipe wrapper and helpers
  hud.py           # dials and overlay rendering
  worldgen.py      # LLM + fallback world generator
  utils.py         # timers, helpers
requirements.txt
README.md
```
 License & Attribution
This project is open source but please respect the creator's work:

✅ Fork it and give it a ⭐
✅ Share it with proper attribution to @tubakhxn
✅ Modify it for your own projects (with credit)
✅ Learn from it and build something even cooler!
Please DON'T:

❌ Copy without giving credit to the original creator
❌ Claim it as your own work
❌ Remove attribution from the code
🙏 Credits & Acknowledgments
Original Concept & Implementation: @tubakhxn
AI Computer Vision: MediaPipe by Google
Graphics Processing: OpenCV
Inspired by: Cyberpunk 2077, Ghost in the Shell, The Matrix
🤝 Contributing
Want to make this even more awesome? Contributions are welcome!

Fork this repository
Create a feature branch
Make your epic improvements
Submit a pull request
Get featured as a contributor!
Please maintain attribution to @tubakhxn as the original creator.

Made with ❤️ by @tubakhxn for cyberpunk enthusiasts and AR developers!

"I would rather be a cyborg than a goddess." - Donna Haraway

⭐ Don't forget to star this repo if you found it awesome! ⭐
