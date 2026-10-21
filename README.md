# Posture Coach

A webcam sitting-posture coach for Windows. It learns what *your* good posture looks like, then tells you
(out loud and on screen) when you drift and exactly how to fix it. Built for people who sit on chairs,
sofas and bean bags, not just at a desk.

## Quick start

Double-click **`Run Posture Coach.bat`**, or in PowerShell:

```powershell
cd "C:\Users\Nayab\Desktop\Projects\Posture"
venv\Scripts\python.exe posture_detector.py
```

First-time setup (only if the `venv` folder is missing). It needs **Python 3.11**, because MediaPipe's pose
API isn't available for newer versions:

```powershell
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Where to put the camera

- In front of you, at about eye level, with your head and both shoulders in view. Your hips are **not** needed.
- Slightly off to one side (30-40 degrees at most) also works.
- A true side-on view is not supported. It needs to see both ears and both shoulders.

## What happens when you run it

1. **Framing:** tells you how to position yourself ("Sit back a little or tilt your screen down").
2. **Get into position:** an 8-second countdown while you settle into your seat.
3. **Guided setup (about a minute):** five spoken steps (seat, shoulders, head, grow tall, relax). It watches
   you live and corrects you ("Your right shoulder is lower. Lift it a little"). It waits for you to get each
   one right, and waits for each sentence to finish before moving on. You never need to look at the screen.
4. **Locked in:** it confirms out loud and on screen, then coaches you back to that posture.
5. **Coaching:** a live 0-100 score, one instruction at a time, a green target outline with an arrow, spoken
   reminders, praise when you recover, and a stand-up break reminder every 30 minutes.

Small movements are ignored (dead zones) and readings are smoothed, so it won't nag over an inch of movement.

## Keys (in the camera window)

| Key | Action |
|-----|--------|
| `r` | Redo setup (do this when you change seats: chair, sofa, bean bag) |
| `s` | Skip ahead / take your current posture as the target right now |
| `v` | Voice on / off |
| `p` | Pause (e.g. on a call) |
| `b` | Snooze the break reminder for 10 minutes |
| `d` | Show debug numbers (what it measures and how far off each is) |
| `q` | Quit and print a session summary |

## Your history

Saved next to the app, and charted in the notebook:

- `posture_sessions.csv`: one row per session (minutes, average score, % good, % poor, main issue)
- `posture_episodes.csv`: each stretch of 10+ seconds of poor posture and what caused it

## Notebook

`posture_detector.ipynb` runs the same coach, lets you tune thresholds in a cell, and charts your history.

```powershell
venv\Scripts\jupyter.exe notebook posture_detector.ipynb
```

Choose the kernel **"Python 3.11 (Posture venv)"**. The camera still opens in its own window, because
OpenCV can't show live video inside a notebook.

## Tuning

Everything is at the top of `posture_core.py`. The main ones:

- `RULES[...]["dead"]`: how far you can move before it cares (bigger = more forgiving)
- `BAD_GRACE_S`: seconds of poor posture before the spoken nudge (default 10)
- `BREAK_EVERY_MIN`: stand-up reminder interval (default 30)
- `TALL_TARGET`: how relaxed "relaxed" is compared with your tallest (default 90-99.5%)

## Tests

```powershell
cd tests
..\venv\Scripts\python.exe -m unittest discover -s . -v
```

The tests use simulated people and a simulated speaker, so they need no camera and make no sound.

## Files

| File | What it is |
|------|------------|
| `posture_core.py` | The coaching engine: measurements, scoring, setup flow, cues. No camera or UI. |
| `posture_detector.py` | Camera, on-screen display, voice, keys, logging |
| `posture_detector.ipynb` | Notebook version with tuning and history charts |
| `tests/` | Automated tests |
| `Run Posture Coach.bat` | Double-click launcher |

## Limits

- Windows only for the spoken voice (it uses Windows' built-in speech). Everything else is cross-platform.
- It is a coaching aid, not medical advice. The thresholds are sensible defaults tuned on simulated data,
  not clinically validated. Adjust them to suit you.
- It judges how you sit against *your own* target posture, so a poor target means poor coaching. Redo setup
  with `r` if the target wasn't your best.
- One person in frame at a time. Bright, even lighting works best.
