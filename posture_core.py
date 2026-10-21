"""
Posture coaching engine (no camera / OpenCV dependency, so it can be tested headlessly).

Designed for a front-facing laptop/desk webcam and SITTING only. It needs just
your shoulders, ears and nose -- never your hips -- so it works at a desk, on a
sofa or in a bean bag.

Flow:  framing -> guided setup (captures YOUR upright posture) -> live coaching.

Every measurement is divided by your shoulder width, so it does not depend on how
far you sit from the camera. Coaching is judged against your own captured posture,
with dead zones (small movements are ignored) and smoothing (no flicker).
"""
import math
import statistics

VIS_SHOULDER = 0.5
VIS_EAR = 0.3
VIS_NOSE = 0.4

# weight: importance | dead: ignored below this | full: counts as 100% wrong at this
RULES = {
    "height":    dict(weight=0.30, dead=0.07, full=0.30, two_sided=False),  # slumping / hunching
    "pitch":     dict(weight=0.15, dead=0.05, full=0.15, two_sided=False),  # head tipped down
    "scale":     dict(weight=0.15, dead=0.05, full=0.18, two_sided=False),  # head pushed forward
    "lean":      dict(weight=0.15, dead=0.06, full=0.25, two_sided=False),  # torso leaning in
    "offset":    dict(weight=0.10, dead=0.07, full=0.25, two_sided=True),   # head off to one side
    "sh_tilt":   dict(weight=0.10, dead=2.5,  full=9.0,  two_sided=True),   # uneven shoulders (deg)
    "head_tilt": dict(weight=0.05, dead=4.0,  full=14.0, two_sided=True),   # tilted head (deg)
}
SCORE_GAIN = 2.5
GOOD_ENTER, GOOD_EXIT = 78, 68
BAD_ENTER, BAD_EXIT = 55, 65
CUE_ON, CUE_OFF = 0.03, 0.012

BAD_GRACE_S = 10.0         # must stay in "bad" this long before an audible nudge
NUDGE_GAP_S = 45.0
BREAK_EVERY_MIN = 30
BREAK_SNOOZE_MIN = 10
AWAY_RESET_S = 20.0        # out of view this long = you stood up / took a break
LOST_TOLERANCE_S = 0.8
FRAMING_HOLD_S = 1.2
READY_S = 8.0              # "get into position" countdown before setup starts

# min/max = seconds. A step ends when its live check has held (1s; 2s for the last) or max is reached.
# "text" is shown on screen; "say" is the spoken version (no references to things you'd have to look at).
SETUP_STEPS = [
    dict(title="Seat", min=6.0, max=6.0,
         text="Sit all the way back: hips against the back of the seat, feet flat on the floor, lower back supported.",
         say="Sit all the way back, feet flat, lower back supported.",
         why="A supported pelvis is the foundation. Everything above it stacks on top."),
    dict(title="Shoulders", min=6.0, max=16.0,
         text="Level your shoulders: roll them up and back, then let them drop down and relax. Watch the green line.",
         say="Level your shoulders. Roll them up and back, then drop and relax. I'll tell you if one is higher.",
         why="Aim for the shoulder line to sit flat against the green guide."),
    dict(title="Head", min=6.0, max=16.0,
         text="Stack your head over your shoulders: ears above shoulders, chin level (not poking forward, not tipped down).",
         say="Stack your head over your shoulders. Chin level, not poking forward.",
         why="Keep your head on the vertical green line rising from between your shoulders."),
    dict(title="Grow tall", min=6.0, max=16.0,
         text="Grow tall: imagine a string lifting the crown of your head. Go as tall as you can.",
         say="Grow tall. Imagine a string lifting the crown of your head.",
         why="Your tallest point is the marker line. This sets the top of your range."),
    dict(title="Relax", min=4.0, max=18.0,
         text="Now relax about 10 percent: tall but comfortable, not stiff. Stay inside the green band and hold still.",
         say="Now relax about ten percent. Tall but comfortable. Hold still.",
         why="This is the posture I'll coach you back to. You can sit like this for hours."),
]
TALL_TARGET = (0.90, 0.995)   # relaxed height as a fraction of your tallest

ISSUE_LABELS = {
    "height": "slumping", "pitch": "head tipped down", "scale": "head pushed forward",
    "lean": "leaning toward the screen", "offset": "head off to one side",
    "sh_tilt": "uneven shoulders", "head_tilt": "head tilted",
}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def tracking_check(pts, w, h):
    """(ok, hint). Only needs shoulders, ears and nose to be visible."""
    if not pts:
        return False, "I can't see you. Sit in front of the camera."
    sa, sb, ea, eb, nose = pts["sh_a"], pts["sh_b"], pts["ear_a"], pts["ear_b"], pts["nose"]
    for s in (sa, sb):
        if s[2] < VIS_SHOULDER or not (0.01 * w <= s[0] <= 0.99 * w) or s[1] > 0.995 * h:
            return False, "I can't see both shoulders. Sit back a little or tilt your screen down."
    if min(ea[2], eb[2]) < VIS_EAR:
        return False, "Face the screen so I can see both of your ears."
    if nose[2] < VIS_NOSE:
        return False, "Face the screen so I can see your face."
    if math.hypot(sb[0] - sa[0], sb[1] - sa[1]) < 0.05 * w:
        return False, "You're very far from the camera. Move closer."
    return True, ""


def framing_check(pts, w, h):
    """Stricter than tracking_check; used only during setup. Returns (ok, hint)."""
    ok, hint = tracking_check(pts, w, h)
    if not ok:
        return False, hint
    sw = math.hypot(pts["sh_b"][0] - pts["sh_a"][0], pts["sh_b"][1] - pts["sh_a"][1])
    if sw / w < 0.16:
        return False, "Move a little closer to the camera."
    if sw / w > 0.70:
        return False, "Sit back a little. You're very close to the camera."
    if min(pts["ear_a"][1], pts["ear_b"][1]) < 0.03 * h:
        return False, "Tilt your screen back a little. Your head is at the top edge of the picture."
    return True, ""


def compute_metrics(pts, frame_w):
    sa, sb = pts["sh_a"], pts["sh_b"]
    ea, eb = pts["ear_a"], pts["ear_b"]
    nose = pts["nose"]
    sw = math.hypot(sb[0] - sa[0], sb[1] - sa[1])
    msx, msy = (sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2
    mex, mey = (ea[0] + eb[0]) / 2, (ea[1] + eb[1]) / 2
    metrics = {
        "height": (msy - mey) / sw,
        "pitch": (nose[1] - mey) / sw,
        "scale": math.hypot(eb[0] - ea[0], eb[1] - ea[1]) / sw,
        "lean": sw / frame_w,
        "offset": (mex - msx) / sw,
        "sh_tilt": math.degrees(math.atan2(sb[1] - sa[1], sb[0] - sa[0])),
        "head_tilt": math.degrees(math.atan2(eb[1] - ea[1], eb[0] - ea[0])),
    }
    pos = {"msx": msx, "msy": msy, "sw": sw, "mex": mex, "mey": mey}
    return metrics, pos


def deviations(m, b):
    """Positive = worse for the one-sided metrics. Signed for the two-sided ones."""
    return {
        "height": (b["height"] - m["height"]) / b["height"],
        "pitch": m["pitch"] - b["pitch"],
        "scale": (m["scale"] - b["scale"]) / b["scale"],
        "lean": (m["lean"] - b["lean"]) / b["lean"],
        "offset": m["offset"] - b["offset"],
        "sh_tilt": m["sh_tilt"] - b["sh_tilt"],
        "head_tilt": m["head_tilt"] - b["head_tilt"],
    }


def penalty(dev, rule):
    d = abs(dev) if rule["two_sided"] else max(dev, 0.0)
    return _clamp((d - rule["dead"]) / (rule["full"] - rule["dead"]))


def cue_text(key, dev):
    side = "right" if dev > 0 else "left"
    return {
        "height": "Grow taller: lift your chest and lengthen the back of your neck.",
        "pitch": "Lift your gaze, your head is tipping down. Raise your screen if you can.",
        "scale": "Draw your chin back a little, your head is drifting toward the screen.",
        "lean": "Sit back a bit, you're leaning toward the screen.",
        "offset": f"Bring your head back over your shoulders, it has drifted to your {side}.",
        "sh_tilt": f"Level your shoulders, your {side} one is lower.",
        "head_tilt": f"Level your head, it is tilted to your {side}.",
    }[key]


class Engine:
    def __init__(self):
        self.paused = False
        self.baseline = None
        self.phase = "framing"
        self.m_s = None
        self.pos_s = None
        self._last_t = None
        self.t = 0.0
        self.tracking = False
        self.lost_for = 0.0

        self.score_s = 100.0
        self.band = "good"
        self.time_in = {"good": 0.0, "adjust": 0.0, "bad": 0.0}
        self.score_sum = 0.0
        self.score_time = 0.0
        self.issue_time = {}
        self.good_streak = 0.0
        self.best_streak = 0.0
        self.recovered_until = 0.0

        self.cue_key = None
        self.cue_since = 0.0
        self.cand_key = None
        self.cand_since = 0.0

        self.bad_for = 0.0
        self.last_nudge = -1e9
        self._ep = None
        self.episodes = []

        self.sit_s = 0.0
        self.break_due = False

        self.notice = ""
        self.notice_until = 0.0
        self.speaking = False
        self.locked_until = 0.0
        self.next_mark = 300.0
        self._skip = False
        self._reset_setup()

    # ---- public controls -------------------------------------------------
    def redo_setup(self):
        self.phase = "framing"
        self.baseline = None
        self._reset_setup()

    def capture_now(self):
        self._skip = True

    def toggle_pause(self):
        self.paused = not self.paused

    def dismiss_break(self):
        self.sit_s = (BREAK_EVERY_MIN - BREAK_SNOOZE_MIN) * 60.0
        self.break_due = False

    def drain_episodes(self):
        eps, self.episodes = self.episodes, []
        return eps

    def finish(self):
        """Close any open bad episode (call when the app exits)."""
        self._close_episode()
        return self.drain_episodes()

    def summary(self):
        total = sum(self.time_in.values())
        top = max(self.issue_time, key=self.issue_time.get) if self.issue_time else None
        return {
            "total_s": total,
            "good_s": self.time_in["good"],
            "adjust_s": self.time_in["adjust"],
            "bad_s": self.time_in["bad"],
            "avg_score": (self.score_sum / self.score_time) if self.score_time else None,
            "top_issue": ISSUE_LABELS.get(top),
            "best_streak_s": self.best_streak,
        }

    # ---- internals -------------------------------------------------------
    def _reset_setup(self):
        self.frame_ok_for = 0.0
        self.step = 0
        self.step_t = 0.0
        self.samples = []
        self.extensions = 0
        self.retries = 0
        self.ok_t = 0.0
        self.peak_h = 0.0
        self.ready_t = 0.0
        self.ready_lost = 0.0

    def _say(self, text, seconds=4.0):
        self.notice = text
        self.notice_until = self.t + seconds

    def _smooth(self, raw, pos, dt):
        if self.m_s is None:
            self.m_s, self.pos_s = dict(raw), dict(pos)
            return
        a_m = 1 - math.exp(-dt / 0.5) if dt > 0 else 0.0
        a_p = 1 - math.exp(-dt / 0.2) if dt > 0 else 0.0
        for k in raw:
            self.m_s[k] += (raw[k] - self.m_s[k]) * a_m
        for k in pos:
            self.pos_s[k] += (pos[k] - self.pos_s[k]) * a_p

    def _out(self, **kw):
        d = dict(
            phase=self.phase, tracking=self.tracking, hint="", headline="", detail="",
            detail2="", tone="info", score=None, band=None, progress=0.0, step=self.step,
            steps=len(SETUP_STEPS), notice=self.notice if self.t < self.notice_until else "",
            ghost=None, live_head=None, sit_s=self.sit_s, streak_s=self.good_streak,
            break_due=self.break_due,
            break_in_min=max(0, math.ceil((BREAK_EVERY_MIN * 60 - self.sit_s) / 60)),
            events=[], debug=None, framing_ok=False, paused=self.paused,
            live="", live_ok=False, meter=None, pos=None, peak=self.peak_h,
            celebrate=self.t < self.locked_until,
            titles=[s["title"] for s in SETUP_STEPS],
        )
        d.update(kw)
        return d

    def update(self, pts, w, h, t, speaking=False):
        dt = 0.0 if self._last_t is None else max(0.0, min(t - self._last_t, 0.5))
        self._last_t = t
        self.t = t
        self.speaking = speaking
        events = []
        if self.paused:
            return self._out(headline="Paused", detail="Press p to resume.", events=events)

        ok, hint = tracking_check(pts, w, h)
        if ok:
            self.lost_for = 0.0
            raw, pos = compute_metrics(pts, w)
            self._smooth(raw, pos, dt)
        else:
            self.lost_for += dt
        self.tracking = ok or (self.m_s is not None and self.lost_for < LOST_TOLERANCE_S)

        if self.phase == "framing":
            return self._framing(pts, w, h, dt, events)
        if self.phase == "getready":
            return self._getready(ok, dt, events)
        if self.phase == "guide":
            return self._guide(ok, hint, dt, events)
        return self._coach(ok, hint, dt, events)

    def _framing(self, pts, w, h, dt, events):
        ok, hint = framing_check(pts, w, h)
        self.frame_ok_for = self.frame_ok_for + dt if ok else 0.0
        if self.frame_ok_for >= FRAMING_HOLD_S:
            self.phase = "getready"
            self.ready_t = self.ready_lost = 0.0
            events.append("step")
        return self._out(
            headline="Let's get you set up",
            detail=hint or "Perfect framing. Hold still for a moment...",
            tone="good" if ok else "warn",
            progress=min(1.0, self.frame_ok_for / FRAMING_HOLD_S),
            framing_ok=ok, events=events,
        )

    def _getready(self, ok, dt, events):
        if self._skip:
            self._skip = False
            self.ready_t = READY_S
        if ok:
            self.ready_lost = 0.0
            self.ready_t += dt
        else:
            self.ready_lost += dt
            if self.ready_lost > 6.0:
                self.phase = "framing"
                self.frame_ok_for = 0.0
        if self.phase == "getready" and self.ready_t >= READY_S and not self.speaking:
            self.phase = "guide"
            self.step, self.step_t, self.samples = 0, 0.0, []
            events.append("step")
        left = max(1, math.ceil(READY_S - self.ready_t))
        return self._out(
            headline="Get into position",
            detail=f"Sit the way you normally sit and look straight ahead. Starting in {left}...",
            detail2="You'll hear spoken instructions, so you don't need to look at the screen.",
            tone="info", progress=_clamp(self.ready_t / READY_S), events=events,
        )

    def _guide(self, ok, hint, dt, events):
        spec = SETUP_STEPS[self.step]
        last = self.step == len(SETUP_STEPS) - 1
        check_ok, live, meter = True, "", None
        if self.tracking:
            self.step_t += dt
            check_ok, live, meter = self._check(self.step)
            if check_ok and ok and self.step_t >= spec["min"]:
                self.ok_t += dt
            else:
                self.ok_t = 0.0
            if last:
                if check_ok and ok:
                    self.samples.append(dict(self.m_s))
                else:
                    self.samples = []
        need = 2.0 if last else 1.0
        finished = self.tracking and not self.speaking and (self.ok_t >= need or self.step_t >= spec["max"])
        if self._skip and self.m_s is not None:
            self._skip = False
            self.samples = [dict(self.m_s)] * 5
            self.extensions = 2
            self._finalize(events)
        elif finished:
            if last:
                if len(self.samples) < 5:
                    self.samples = [dict(self.m_s)] * 5
                    self.extensions = 2
                self._finalize(events)
            else:
                if self.step == 3:
                    self.peak_h = self.m_s["height"]
                self.step += 1
                self.step_t = self.ok_t = 0.0
                events.append("step")
        if self.phase == "coach":
            return self._coach(ok, hint, 0.0, events)
        spec = SETUP_STEPS[self.step]
        paused = not self.tracking
        return self._out(
            headline=f"Setup {self.step + 1}/{len(SETUP_STEPS)}: {spec['title']}",
            detail=hint if paused else spec["text"],
            detail2="Paused until I can see you again." if paused else spec["why"],
            tone="warn" if paused else "info",
            progress=_clamp(max(self.step_t / spec["max"], self.ok_t / need)),
            live=hint if paused else live, live_ok=check_ok and not paused, meter=meter,
            pos=dict(self.pos_s) if self.pos_s else None, events=events,
        )

    def _check(self, idx):
        """Live check for a setup step. Returns (ok, feedback, meter 0..1+)."""
        m = self.m_s
        if idx == 0:
            return True, "Take your time. I'm not measuring this one.", None
        if idx == 1:
            tilt = m["sh_tilt"]
            if abs(tilt) <= 3.0:
                return True, "Shoulders level. Hold it.", None
            low, high = ("right", "left") if tilt > 0 else ("left", "right")
            return False, f"Your {low} shoulder is lower. Lift it a little and relax your {high} one.", None
        if idx == 2:
            off, ht = m["offset"], m["head_tilt"]
            if abs(off) > 0.07:
                side = "right" if off > 0 else "left"
                return False, f"Your head is off to your {side}. Slide it back over your shoulders.", None
            if abs(ht) > 4.0:
                side = "right" if ht > 0 else "left"
                return False, f"Level your head, it is tilted to your {side}.", None
            return True, "Head stacked over your shoulders. Hold it.", None
        h = m["height"]
        if idx == 3:
            self.peak_h = max(self.peak_h, h)
            ratio = h / self.peak_h
            if self.step_t < 3.0:
                return False, "Lift through the crown of your head. Go taller...", ratio
            if ratio >= 0.97:
                return True, "That's your tallest. Hold it for a second.", ratio
            return False, f"You're at {ratio * 100:.0f}% of your tallest. Keep lifting, or hold what you've got.", ratio
        ratio = h / self.peak_h if self.peak_h else 1.0
        lo, hi = TALL_TARGET
        if lo <= ratio <= hi:
            return True, "Tall but relaxed. Hold still while I take the reading.", ratio
        if ratio > hi:
            return False, "Ease down a hair: relax about 10 percent, don't hold it stiff.", ratio
        return False, "You've dropped too far. Lift your chest back up a little.", ratio

    def _finalize(self, events):
        keys = self.samples[0].keys()
        med = {k: statistics.median(s[k] for s in self.samples) for k in keys}
        hs = [s["height"] for s in self.samples]
        spread = (max(hs) - min(hs)) / med["height"]
        if spread > 0.15 and self.extensions < 2:
            self.extensions += 1
            self.ok_t = 0.0
            self.samples = []
            self._say("Hold still a little longer...", 3.0)
            return
        very_off = abs(med["sh_tilt"]) > 15 or abs(med["offset"]) > 0.4
        if very_off and self.retries < 1:
            self.retries += 1
            self.step, self.step_t, self.samples, self.extensions = 1, 0.0, [], 0
            self.ok_t = 0.0
            self._say("That reading looks very lopsided. One more try: level your shoulders and centre your head.", 6.0)
            return
        slightly_off = abs(med["sh_tilt"]) > 6 or abs(med["offset"]) > 0.22
        self.baseline = med
        self.phase = "coach"
        self.good_streak = 0.0
        self.score_s, self.band = 100.0, "good"
        self.next_mark = 300.0
        self.locked_until = self.t + 4.0
        if slightly_off:
            self._say("Your shoulders or head looked a bit uneven. Press r to redo if that wasn't your best posture.", 8.0)
        events.append("ready")

    def _update_band(self, score):
        b = self.band
        if b == "good":
            if score < GOOD_EXIT:
                b = "bad" if score < BAD_ENTER else "adjust"
        elif b == "adjust":
            if score >= GOOD_ENTER:
                b = "good"
            elif score < BAD_ENTER:
                b = "bad"
        else:
            if score >= GOOD_ENTER:
                b = "good"
            elif score > BAD_EXIT:
                b = "adjust"
        self.band = b

    def _close_episode(self):
        if self._ep:
            issues = self._ep["issues"]
            top = max(issues, key=issues.get) if issues else None
            self.episodes.append((self._ep["start"], self.t, top))
            self._ep = None

    def _coach(self, ok, hint, dt, events):
        t = self.t
        dev = pen = contrib = None
        if ok and self.baseline:
            dev = deviations(self.m_s, self.baseline)
            pen = {k: penalty(dev[k], RULES[k]) for k in RULES}
            contrib = {k: RULES[k]["weight"] * pen[k] for k in RULES}
            raw_score = 100.0 * (1 - min(1.0, SCORE_GAIN * sum(contrib.values())))
            self.score_s += (raw_score - self.score_s) * (1 - math.exp(-dt / 1.0)) if dt > 0 else 0.0
            prev_band = self.band
            self._update_band(self.score_s)
            if prev_band != "good" and self.band == "good":
                self.recovered_until = t + 3.0

            self._pick_cue(contrib, t)

            self.time_in[self.band] += dt
            self.score_sum += self.score_s * dt
            self.score_time += dt
            if self.band != "good" and self.cue_key:
                self.issue_time[self.cue_key] = self.issue_time.get(self.cue_key, 0.0) + dt

            if self.band == "good":
                self.good_streak += dt
                self.best_streak = max(self.best_streak, self.good_streak)
                if self.good_streak >= self.next_mark:
                    events.append(f"streak:{int(self.next_mark // 60)}")
                    self.next_mark += 300.0
            else:
                self.good_streak = 0.0
                self.next_mark = 300.0

            if self.band == "bad":
                self.bad_for += dt
                if self.bad_for >= BAD_GRACE_S:
                    if self._ep is None:
                        self._ep = {"start": t - self.bad_for, "issues": {}}
                    if self.cue_key:
                        self._ep["issues"][self.cue_key] = self._ep["issues"].get(self.cue_key, 0.0) + dt
                    if t - self.last_nudge >= NUDGE_GAP_S:
                        events.append("nudge")
                        self.last_nudge = t
            else:
                self.bad_for = 0.0
                self._close_episode()

            self.sit_s += dt
            if self.sit_s >= BREAK_EVERY_MIN * 60 and not self.break_due:
                self.break_due = True
                events.append("break")

        elif not ok and self.lost_for >= AWAY_RESET_S:
            self.sit_s = 0.0
            self.break_due = False
            self.good_streak = 0.0
            self.bad_for = 0.0
            self._close_episode()

        return self._coach_out(ok, hint, dev, pen, contrib, events)

    def _pick_cue(self, contrib, t):
        top = max(contrib, key=contrib.get)
        cand = top if contrib[top] >= CUE_ON else None
        if self.cue_key and contrib[self.cue_key] >= CUE_OFF:
            switch = (cand and cand != self.cue_key
                      and contrib[cand] > 1.6 * contrib[self.cue_key] and t - self.cue_since > 2.5)
            if not switch:
                cand = self.cue_key
        if cand != self.cand_key:
            self.cand_key, self.cand_since = cand, t
        if cand is None:
            self.cue_key = None
        elif cand != self.cue_key and t - self.cand_since >= 1.5:
            self.cue_key, self.cue_since = cand, t

    def _coach_out(self, ok, hint, dev, pen, contrib, events):
        t = self.t
        if not ok and not self.tracking:
            return self._out(
                headline="I lost sight of you", detail=hint, detail2="Paused: this time isn't counted.",
                tone="warn", score=round(self.score_s), band=self.band, events=events)

        headline, tone, detail, detail2 = "", "good", "", ""
        cue = cue_text(self.cue_key, dev[self.cue_key]) if (self.cue_key and dev) else ""
        others = []
        if contrib:
            ranked = sorted((k for k in contrib if contrib[k] >= CUE_ON and k != self.cue_key),
                            key=lambda k: -contrib[k])
            others = [cue_text(k, dev[k]) for k in ranked[:1]]
        if self.band == "good":
            headline, tone = "Great posture", "good"
            if t < self.recovered_until:
                detail = "That's it. Hold that."
            elif cue:
                headline, detail = "Good, one small tweak", cue
            else:
                detail = "Nice and tall. Keep it up."
        elif self.band == "adjust":
            headline, tone = "Almost there", "warn"
            detail = cue or "Small drift. Line up with the green outline."
        else:
            headline, tone = "Let's reset your posture", "bad"
            detail = cue or "Line up with the green outline."
        if cue and others and self.band != "good":
            detail2 = "Then: " + others[0]

        ghost = live_head = None
        if ok and self.baseline and self.pos_s:
            p, b = self.pos_s, self.baseline
            ghost = {
                "head": (p["msx"] + b["offset"] * p["sw"], p["msy"] - b["height"] * p["sw"]),
                "sh_a": (p["msx"] - p["sw"] / 2, p["msy"]),
                "sh_b": (p["msx"] + p["sw"] / 2, p["msy"]),
                "sw": p["sw"],
            }
            live_head = (p["mex"], p["mey"])
        return self._out(
            headline=headline, detail=detail, detail2=detail2, tone=tone,
            score=round(self.score_s), band=self.band, ghost=ghost, live_head=live_head,
            events=events, debug={"dev": dev, "pen": pen, "m": self.m_s, "base": self.baseline},
        )
