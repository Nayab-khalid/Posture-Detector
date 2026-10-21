"""Shared test helpers: a synthetic 'person' and a simulated clock."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import posture_core as core

W, H = 1280, 720
FPS = 30


def make_pts(sw=300, cx=640, cy=540, height=0.41, pitch=0.12, scale=0.38, offset=0.0,
             sh_tilt=0.0, head_tilt=0.0, noise=0.6, vis=0.95):
    """Landmarks (pixels) for a front-facing person. All posture values are in shoulder widths."""
    r = lambda: random.gauss(0, noise)
    tilt, ht = math.radians(sh_tilt), math.radians(head_tilt)
    mex, mey, es = cx + offset * sw, cy - height * sw, scale * sw
    return {
        "sh_a": (cx - sw / 2 * math.cos(tilt) + r(), cy - sw / 2 * math.sin(tilt) + r(), vis),
        "sh_b": (cx + sw / 2 * math.cos(tilt) + r(), cy + sw / 2 * math.sin(tilt) + r(), vis),
        "ear_a": (mex - es / 2 * math.cos(ht) + r(), mey - es / 2 * math.sin(ht) + r(), vis),
        "ear_b": (mex + es / 2 * math.cos(ht) + r(), mey + es / 2 * math.sin(ht) + r(), vis),
        "nose": (mex + r(), mey + pitch * sw + r(), vis),
    }


class Sim:
    """Drives an Engine with a simulated clock."""

    def __init__(self, seed=1):
        random.seed(seed)
        self.engine = core.Engine()
        self.t = 0.0
        self.events = []
        self.speaking = False

    def run(self, seconds, pts_fn):
        out = None
        for _ in range(int(seconds * FPS)):
            self.t += 1 / FPS
            out = self.engine.update(pts_fn(), W, H, self.t, self.speaking)
            self.events += out["events"]
        return out


def complete_setup(sim, start=None, max_seconds=150):
    """Walk a cooperative person through setup; they fix each problem a moment after being asked."""
    state = dict(tilt=6.0, off=0.15, height=0.34)
    state.update(start or {})
    e = sim.engine
    for _ in range(int(max_seconds * FPS)):
        out = sim.run(1 / FPS, lambda: make_pts(sh_tilt=state["tilt"], offset=state["off"], height=state["height"]))
        if out["phase"] == "coach":
            return out
        if out["phase"] == "guide":
            if out["step"] == 1 and e.step_t > 2.5:
                state["tilt"] = 0.4
            if out["step"] == 2 and e.step_t > 2.5:
                state["off"] = 0.0
            if out["step"] == 3:
                state["height"] = min(0.44, state["height"] + 0.0008)
            if out["step"] == 4 and e.step_t > 1:
                state["height"] = 0.41
    raise AssertionError(f"setup did not finish; stuck in {e.phase} step {e.step}")


def good():
    return make_pts()
