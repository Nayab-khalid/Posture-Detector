"""
Posture Coach: sitting posture guidance from your webcam.

  1. Framing   - tells you exactly how to sit/position so it can see you
  2. Setup     - ~20s guided routine that captures YOUR good posture
  3. Coaching  - live score, one clear instruction at a time, and a green target
                 outline to line up with. Break reminder every 30 min.

Run:  python posture_detector.py
Keys: [r] redo setup  [s] skip to capture  [p] pause  [b] snooze break  [d] debug  [q] quit
"""
import contextlib
import csv
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np

import posture_core as core

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
EPISODES_CSV = os.path.join(HERE, "posture_episodes.csv")
SESSIONS_CSV = os.path.join(HERE, "posture_sessions.csv")
WINDOW = "Posture Coach"

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL = {
    "good": (110, 200, 100), "warn": (40, 175, 255), "bad": (85, 85, 235),
    "info": (235, 190, 90), "white": (245, 245, 245), "grey": (175, 175, 175),
}

try:
    import winsound

    def _tones(seq):
        for freq, ms in seq:
            winsound.Beep(freq, ms)
except ImportError:
    def _tones(seq):
        print("\a", end="", flush=True)

SOUNDS = {
    "nudge": [(660, 140), (520, 200)],
    "break": [(523, 150), (659, 150), (784, 260)],
    "ready": [(523, 120), (784, 220)],
}


def play(name):
    seq = SOUNDS.get(name)
    if seq:
        threading.Thread(target=_tones, args=(seq,), daemon=True).start()


class Speaker:
    """Text-to-speech through Windows' built-in voice (no install). One long-lived
    PowerShell process, so there is no start-up delay per sentence."""

    SCRIPT = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; {null}$s.Rate = 0; "
        "while (($l = [Console]::In.ReadLine()) -ne $null) { "
        "if ($l.StartsWith('!')) { $s.SpeakAsyncCancelAll() } else { [void]$s.SpeakAsync($l) } }"
    )

    def __init__(self, enabled=True, silent=False):
        self.enabled = enabled
        self.proc = None
        self.busy_until = 0.0
        try:
            script = self.SCRIPT.replace("{null}", "$s.SetOutputToNull(); " if silent else "")
            self.proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self.proc = None

    @property
    def available(self):
        return self.proc is not None and self.proc.poll() is None

    def is_busy(self):
        return self.enabled and self.available and time.monotonic() < self.busy_until

    def say(self, text, interrupt=False):
        if not (self.enabled and self.available):
            return
        now = time.monotonic()
        try:
            if interrupt:
                self.proc.stdin.write("!\n")
                self.busy_until = now
            self.proc.stdin.write(text.replace("\n", " ") + "\n")
            self.proc.stdin.flush()
            self.busy_until = max(self.busy_until, now) + len(text.split()) / 2.2 + 0.6
        except Exception:
            pass

    def stop(self):
        try:
            self.proc.stdin.write("!\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.busy_until = 0.0

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            pass


class Voice:
    """Decides WHAT to say and WHEN, so the coach is useful without looking at the screen."""

    def __init__(self, speaker):
        self.sp = speaker
        self.key = None
        self.last_say = -99.0
        self.live = None
        self.live_since = 0.0
        self.spoken_live = None
        self.spoken_live_t = -99.0
        self.fixing = False
        self.cue = None
        self.cue_since = 0.0
        self.spoken_cue = None
        self.spoken_cue_t = -99.0
        self.prev_band = None
        self.hint = None
        self.hint_since = 0.0

    def _say(self, text, now, interrupt=False):
        self.sp.say(text, interrupt)
        self.last_say = now

    def update(self, st, now):
        if not self.sp.enabled or st["paused"]:
            return
        phase = st["phase"]
        key = (phase, st["step"] if phase == "guide" else 0)
        if key != self.key:
            self.key = key
            self.live = self.spoken_live = self.cue = self.spoken_cue = None
            self.fixing = False
            if phase == "getready":
                self._say("I can see you. Sit how you normally sit and look straight ahead. "
                          "I'll talk you through it.", now)
            elif phase == "guide":
                spec = core.SETUP_STEPS[st["step"]]
                self._say(f"Step {st['step'] + 1} of {len(core.SETUP_STEPS)}. {spec['say']}", now)
            return

        if phase == "framing":
            hint = st["detail"] if not st["framing_ok"] else None
            if hint != self.hint:
                self.hint, self.hint_since = hint, now
            if hint and now - self.hint_since > 1.5 and now - self.last_say > 7:
                self._say(hint, now, True)
                self.hint_since = now + 5
        elif phase == "guide" and st["live"] and st["step"] > 0:
            norm = re.sub(r"\d+", "#", st["live"])
            if norm != self.live:
                self.live, self.live_since = norm, now
            if now - self.live_since < 1.2:
                return
            if self.sp.is_busy():
                return
            if st["live_ok"]:
                if self.fixing:
                    self._say("Good. Hold it.", now)
                    self.fixing = False
            elif (norm != self.spoken_live and now - self.last_say > 3.0) or now - self.spoken_live_t > 9.0:
                self._say(st["live"], now)
                self.spoken_live, self.spoken_live_t, self.fixing = norm, now, True
        elif phase == "coach":
            band = st["band"]
            if st["tracking"] and band:
                cue = st["detail"] if band != "good" else None
                if cue != self.cue:
                    self.cue, self.cue_since = cue, now
                if cue and now - self.cue_since >= 2.0 and not self.sp.is_busy() and (
                        cue != self.spoken_cue or now - self.spoken_cue_t >= 25.0) and now - self.last_say > 4.0:
                    self._say(cue, now)
                    self.spoken_cue, self.spoken_cue_t = cue, now
                if band == "good" and self.prev_band in ("adjust", "bad"):
                    self._say("Nice. That's good posture. Hold it.", now)
                    self.spoken_cue = None
                self.prev_band = band
            elif not st["tracking"] and now - self.last_say > 15 and st["detail"]:
                self._say("I lost sight of you. " + st["detail"], now)

    def on_event(self, ev, st, now):
        if not self.sp.enabled:
            return
        if ev == "ready":
            self._say("Got it. Posture locked in. That's your good posture. "
                      "I'll guide you back to it whenever you drift.", now)
        elif ev.startswith("streak:"):
            self._say(f"Great job. {ev.split(':')[1]} minutes of good posture. Keep it up.", now)
        elif ev == "break":
            self._say("Time for a one minute stand and stretch.", now, True)
        elif ev == "nudge" and now - self.last_say > 4.0 and st["detail"]:
            self._say("Posture check. " + st["detail"], now, True)


@contextlib.contextmanager
def quiet_stderr():
    """Hide the TensorFlow/absl start-up chatter."""
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def extract_points(result, w, h):
    """Pixel points sorted left-to-right on screen (the view is mirrored, so screen
    left/right is YOUR left/right)."""
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark

    def P(i):
        return (lm[i].x * w, lm[i].y * h, lm[i].visibility)

    sh = sorted([P(11), P(12)], key=lambda p: p[0])
    ears = sorted([P(7), P(8)], key=lambda p: p[0])
    return {"nose": P(0), "sh_a": sh[0], "sh_b": sh[1], "ear_a": ears[0], "ear_b": ears[1]}


# ----------------------------------------------------------------- drawing
def panel(img, x1, y1, x2, y2, color=(24, 24, 24), alpha=0.72):
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def put(img, s, x, y, scale, color, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def wrap_lines(s, scale, thick, max_w):
    lines, cur = [], ""
    for word in s.split():
        test = (cur + " " + word).strip()
        if cv2.getTextSize(test, FONT, scale, thick)[0][0] <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def score_color(score):
    if score >= 78:
        return COL["good"]
    return COL["warn"] if score >= 55 else COL["bad"]


def bar(img, x, y, w_, h_, frac, color):
    cv2.rectangle(img, (x, y), (x + w_, y + h_), (70, 70, 70), -1)
    cv2.rectangle(img, (x, y), (x + int(w_ * max(0.0, min(1.0, frac))), y + h_), color, -1)
    cv2.rectangle(img, (x, y), (x + w_, y + h_), (200, 200, 200), 1)


def dashed_line(img, p1, p2, color, thick=2, dash=14):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    length = np.linalg.norm(p2 - p1)
    if length < 1:
        return
    d = (p2 - p1) / length
    pos = 0.0
    while pos < length:
        a = p1 + d * pos
        b = p1 + d * min(pos + dash, length)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)
        pos += dash * 2


def mmss(sec):
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def draw_framing_guide(img, st, w, h):
    color = COL["good"] if st["framing_ok"] else COL["warn"]
    cv2.ellipse(img, (w // 2, int(h * 0.36)), (int(w * 0.085), int(h * 0.16)), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.ellipse(img, (w // 2, int(h * 0.98)), (int(w * 0.24), int(h * 0.40)), 0, 200, 340, color, 2, cv2.LINE_AA)


def draw_live(img, pts, st):
    if not pts or st["paused"]:
        return
    band = st["band"]
    color = {"good": COL["good"], "adjust": COL["warn"], "bad": COL["bad"]}.get(band, COL["info"])
    sa, sb = pts["sh_a"], pts["sh_b"]
    cv2.line(img, (int(sa[0]), int(sa[1])), (int(sb[0]), int(sb[1])), color, 3, cv2.LINE_AA)
    for key in ("sh_a", "sh_b", "ear_a", "ear_b", "nose"):
        p = pts[key]
        cv2.circle(img, (int(p[0]), int(p[1])), 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 0, 0), 1, cv2.LINE_AA)


def draw_ghost(img, st):
    g, live = st["ghost"], st["live_head"]
    if not g:
        return
    sw = g["sw"]
    col = COL["good"]
    dashed_line(img, g["sh_a"], g["sh_b"], col, 2)
    hx, hy = int(g["head"][0]), int(g["head"][1])
    axes = (int(sw * 0.22), int(sw * 0.30))
    for a in range(0, 360, 30):
        cv2.ellipse(img, (hx, hy), axes, 0, a, a + 16, col, 2, cv2.LINE_AA)
    if live and np.hypot(live[0] - hx, live[1] - hy) > 0.14 * sw:
        cv2.arrowedLine(img, (int(live[0]), int(live[1])), (hx, hy), COL["warn"], 3, cv2.LINE_AA, tipLength=0.25)


def draw_top(img, st, w, fs):
    pad = int(22 * fs)
    show_score = st["score"] is not None
    right_w = int(310 * fs) if show_score else 0
    max_w = w - 2 * pad - right_w
    col = COL[st["tone"]]
    lines = wrap_lines(st["detail"], 0.72 * fs, 1, max_w) if st["detail"] else []
    lines2 = wrap_lines(st["detail2"], 0.56 * fs, 1, max_w) if st["detail2"] else []
    ph = pad + int(52 * fs) + len(lines) * int(32 * fs) + len(lines2) * int(26 * fs)
    with_bar = not show_score and st["phase"] in ("framing", "getready", "guide")
    if with_bar:
        ph += int(26 * fs)
    if show_score:
        ph = max(ph, int(122 * fs))
    panel(img, 0, 0, w, ph)
    y = pad + int(22 * fs)
    put(img, st["headline"], pad, y, 1.0 * fs, col, 2)
    y += int(14 * fs)
    for ln in lines:
        y += int(32 * fs)
        put(img, ln, pad, y, 0.72 * fs, COL["white"], 1)
    for ln in lines2:
        y += int(26 * fs)
        put(img, ln, pad, y, 0.56 * fs, COL["grey"], 1)
    if show_score:
        x0 = w - right_w
        sc = st["score"]
        put(img, "POSTURE SCORE", x0, pad + int(8 * fs), 0.5 * fs, COL["grey"], 1)
        put(img, str(sc), x0, pad + int(62 * fs), 1.9 * fs, score_color(sc), 3)
        bar(img, x0, pad + int(76 * fs), int(270 * fs), int(14 * fs), sc / 100.0, score_color(sc))
    elif with_bar:
        bar(img, pad, ph - int(26 * fs), int(w * 0.35), int(10 * fs), st["progress"], col)
    return ph


def draw_guide(img, st, w, h, fs, top_h):
    pos, step = st["pos"], st["step"]
    if pos:
        msx, msy, sw = pos["msx"], pos["msy"], pos["sw"]
        g = COL["good"]
        if step == 1:
            dashed_line(img, (msx - 0.85 * sw, msy), (msx + 0.85 * sw, msy), g, 2)
        elif step == 2:
            dashed_line(img, (msx, msy), (msx, msy - 1.0 * sw), g, 2)
        elif step in (3, 4) and st["peak"] > 0:
            ty = msy - st["peak"] * sw
            if step == 3:
                dashed_line(img, (msx - 0.6 * sw, ty), (msx + 0.6 * sw, ty), g, 2)
                put(img, "your tallest", msx + 0.62 * sw, ty + 5, 0.55 * fs, g, 1)
            else:
                lo, hi = core.TALL_TARGET
                y_hi, y_lo = int(msy - st["peak"] * hi * sw), int(msy - st["peak"] * lo * sw)
                roi = img[max(0, y_hi):max(0, y_lo), max(0, int(msx - 0.6 * sw)):int(msx + 0.6 * sw)]
                if roi.size:
                    cv2.addWeighted(np.full_like(roi, g), 0.28, roi, 0.72, 0, roi)
                put(img, "target band", msx + 0.62 * sw, y_hi + 5, 0.55 * fs, g, 1)
    # checklist
    x0, y0 = int(20 * fs), top_h + int(16 * fs)
    rows = st["titles"]
    panel(img, x0, y0, x0 + int(250 * fs), y0 + int(40 * fs) + len(rows) * int(30 * fs))
    put(img, "SETUP CHECKLIST", x0 + int(12 * fs), y0 + int(26 * fs), 0.5 * fs, COL["grey"], 1)
    for i, title in enumerate(rows):
        yy = y0 + int(26 * fs) + (i + 1) * int(30 * fs)
        mark, color = ("[x]", COL["good"]) if i < step else (("> ", COL["info"]) if i == step else ("[ ]", COL["grey"]))
        put(img, f"{mark} {title}", x0 + int(12 * fs), yy, 0.62 * fs, color, 2 if i == step else 1)
    # live feedback
    if st["live"]:
        ph = int(86 * fs) if st["meter"] is not None else int(58 * fs)
        y1 = h - int(52 * fs)
        panel(img, int(20 * fs), y1 - ph, w - int(20 * fs), y1, (24, 24, 24), 0.82)
        color = COL["good"] if st["live_ok"] else COL["warn"]
        put(img, st["live"], int(38 * fs), y1 - ph + int(36 * fs), 0.8 * fs, color, 2)
        if st["meter"] is not None:
            bx, by, bw = int(38 * fs), y1 - int(24 * fs), w - int(76 * fs)
            frac = lambda r: max(0.0, min(1.0, (r - 0.7) / 0.35))
            cv2.rectangle(img, (bx, by), (bx + bw, by + int(10 * fs)), (70, 70, 70), -1)
            if step == 4:
                lo, hi = core.TALL_TARGET
                cv2.rectangle(img, (bx + int(bw * frac(lo)), by), (bx + int(bw * frac(hi)), by + int(10 * fs)), (90, 160, 90), -1)
            cv2.circle(img, (bx + int(bw * frac(st["meter"])), by + int(5 * fs)), int(9 * fs), color, -1, cv2.LINE_AA)
            put(img, "slumped", bx, by - 6, 0.45 * fs, COL["grey"], 1)


def draw_bottom(img, st, w, h, fs, debug):
    notice_lines = wrap_lines(st["notice"], 0.7 * fs, 1, w - 80) if st["notice"] else []
    base_y = h - int(16 * fs)
    if st["phase"] == "coach" and not st["paused"]:
        nxt = "Break due now" if st["break_due"] else f"Next break in {st['break_in_min']} min"
        chip = f"Sitting {mmss(st['sit_s'])}   Good streak {mmss(st['streak_s'])}   {nxt}"
        panel(img, 0, h - int(44 * fs), w, h)
        put(img, chip, int(20 * fs), base_y, 0.58 * fs, COL["white"], 1)
        keys = "[r] redo setup  [v] voice  [p] pause  [q] quit"
        tw = cv2.getTextSize(keys, FONT, 0.5 * fs, 1)[0][0]
        put(img, keys, w - tw - int(20 * fs), base_y, 0.5 * fs, COL["grey"], 1)
    else:
        panel(img, 0, h - int(44 * fs), w, h)
        keys = "[s] skip ahead  [r] restart setup  [v] voice  [p] pause  [q] quit"
        put(img, keys, int(20 * fs), base_y, 0.5 * fs, COL["grey"], 1)

    y = h - int(64 * fs)
    if st["phase"] == "guide":
        y -= int(100 * fs) if st["meter"] is not None else int(72 * fs)
    if st["break_due"]:
        panel(img, int(20 * fs), y - int(46 * fs), w - int(20 * fs), y + int(8 * fs), (20, 70, 110), 0.85)
        put(img, "Time for a 1-minute stand and stretch. Step away and I'll reset the timer. [b] snooze 10 min",
            int(36 * fs), y - int(12 * fs), 0.62 * fs, COL["white"], 1)
        y -= int(62 * fs)
    for ln in reversed(notice_lines):
        panel(img, int(20 * fs), y - int(30 * fs), w - int(20 * fs), y + int(8 * fs), (60, 50, 20), 0.85)
        put(img, ln, int(36 * fs), y - int(4 * fs), 0.7 * fs, COL["white"], 1)
        y -= int(42 * fs)

    if debug and st["debug"] and st["debug"]["dev"]:
        d = st["debug"]
        yy = int(180 * fs)
        panel(img, 10, yy - 22, int(430 * fs), yy + len(d["dev"]) * int(22 * fs) + 8)
        for k, v in d["dev"].items():
            pen = d["pen"][k]
            put(img, f"{k:<9} dev {v:+.2f}  pen {pen:.2f}", 20, yy, 0.5 * fs,
                COL["bad"] if pen > 0.5 else COL["warn"] if pen > 0 else COL["good"], 1)
            yy += int(22 * fs)


def draw_locked(img, w, h, fs):
    cx, cy = w // 2, int(h * 0.42)
    panel(img, cx - int(300 * fs), cy - int(130 * fs), cx + int(300 * fs), cy + int(130 * fs), (20, 60, 25), 0.88)
    g = COL["good"]
    cv2.circle(img, (cx, cy - int(48 * fs)), int(44 * fs), g, 4, cv2.LINE_AA)
    pts = np.array([(cx - 20 * fs, cy - 46 * fs), (cx - 5 * fs, cy - 30 * fs), (cx + 24 * fs, cy - 68 * fs)], np.int32)
    cv2.polylines(img, [pts], False, g, 6, cv2.LINE_AA)
    for text, dy, sc, col, th in (("POSTURE LOCKED IN", 40, 1.1, g, 2),
                                  ("This is your good posture.", 82, 0.7, COL["white"], 1),
                                  ("I'll guide you back to it whenever you drift.", 112, 0.6, COL["grey"], 1)):
        tw = cv2.getTextSize(text, FONT, sc * fs, th)[0][0]
        put(img, text, cx - tw // 2, cy + int(dy * fs), sc * fs, col, th)


def draw(frame, st, pts, debug=False):
    h, w = frame.shape[:2]
    fs = max(0.55, w / 1280.0)
    if st["phase"] == "framing":
        draw_framing_guide(frame, st, w, h)
    draw_live(frame, pts, st)
    draw_ghost(frame, st)
    top_h = draw_top(frame, st, w, fs)
    if st["phase"] == "guide":
        draw_guide(frame, st, w, h, fs, top_h)
    draw_bottom(frame, st, w, h, fs, debug)
    if st["celebrate"]:
        draw_locked(frame, w, h, fs)
    return frame


# --------------------------------------------------------------------- app
def open_camera(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None


def _append_csv(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(header)
        wr.writerow(row)


def run(max_seconds=None, camera=0, voice=True):
    cap = open_camera(camera)
    if cap is None:
        print("Could not open the webcam. Close other apps using it and try again.")
        return

    engine = core.Engine()
    speaker = Speaker(enabled=voice, silent=bool(os.environ.get("POSTURE_VOICE_SILENT")))
    talker = Voice(speaker)
    wall0, t0 = time.time(), time.monotonic()
    debug = False
    print("Posture Coach is starting. A camera window will open.")
    print("Tip: put the camera at about eye level and sit where you normally sit.")
    print("Voice guidance is " + ("ON (press v in the window to mute)." if speaker.available else "OFF (speech not available)."))
    speaker.say("Posture coach starting. Make sure the camera can see your head and shoulders.", True)

    with quiet_stderr():
        pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose.process(np.zeros((64, 64, 3), np.uint8))

    def log_episodes(eps):
        for ts, te, key in eps:
            s, e = wall0 + (ts - t0), wall0 + (te - t0)
            _append_csv(EPISODES_CSV, ["date", "start_time", "end_time", "duration_seconds", "issue"], [
                datetime.fromtimestamp(s).strftime("%Y-%m-%d"), datetime.fromtimestamp(s).strftime("%H:%M:%S"),
                datetime.fromtimestamp(e).strftime("%H:%M:%S"), f"{te - ts:.0f}", core.ISSUE_LABELS.get(key, "")])

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.monotonic()
            if max_seconds and now - t0 > max_seconds:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pts = extract_points(result, w, h)
            st = engine.update(pts, w, h, now, speaker.is_busy())
            for ev in st["events"]:
                play(ev)
                talker.on_event(ev, st, now)
            talker.update(st, now)
            log_episodes(engine.drain_episodes())

            cv2.imshow(WINDOW, draw(frame, st, pts, debug))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("r"):
                speaker.stop()
                talker.key = None
                engine.redo_setup()
            elif key == ord("s"):
                speaker.stop()
                if engine.phase == "framing":
                    engine.frame_ok_for = core.FRAMING_HOLD_S
                else:
                    engine.capture_now()
            elif key == ord("p"):
                engine.toggle_pause()
            elif key == ord("b"):
                engine.dismiss_break()
            elif key == ord("d"):
                debug = not debug
            elif key == ord("v"):
                if speaker.enabled:
                    speaker.stop()
                speaker.enabled = not speaker.enabled
                if speaker.enabled:
                    speaker.say("Voice on.", True)
    finally:
        log_episodes(engine.finish())
        speaker.close()
        pose.close()
        cap.release()
        cv2.destroyAllWindows()

    sm = engine.summary()
    if sm["total_s"] > 10:
        pct = lambda x: 100 * x / sm["total_s"]
        print(f"\nSession: {sm['total_s'] / 60:.1f} min coached | avg score {sm['avg_score']:.0f} | "
              f"{pct(sm['good_s']):.0f}% good, {pct(sm['adjust_s']):.0f}% okay, {pct(sm['bad_s']):.0f}% poor")
        print(f"Longest good streak {mmss(sm['best_streak_s'])}"
              + (f" | most common issue: {sm['top_issue']}" if sm["top_issue"] else ""))
        _append_csv(SESSIONS_CSV,
                    ["date", "start_time", "minutes", "avg_score", "pct_good", "pct_poor", "top_issue"],
                    [datetime.fromtimestamp(wall0).strftime("%Y-%m-%d"), datetime.fromtimestamp(wall0).strftime("%H:%M"),
                     f"{sm['total_s'] / 60:.1f}", f"{sm['avg_score']:.0f}", f"{pct(sm['good_s']):.0f}",
                     f"{pct(sm['bad_s']):.0f}", sm["top_issue"] or ""])


if __name__ == "__main__":
    run()
