import os
import unittest

import numpy as np

from helpers import Sim, complete_setup, good, make_pts, W, H, core

os.environ["POSTURE_VOICE_SILENT"] = "1"
import posture_detector as app


class FakeSpeaker:
    """Records speech and takes (simulated) time to talk, like the real one."""
    enabled = True

    def __init__(self):
        self.log, self.interrupts, self.t, self.busy_until = [], [], 0.0, 0.0

    def is_busy(self):
        return self.t < self.busy_until

    def say(self, text, interrupt=False):
        if interrupt:
            self.busy_until = self.t
        self.interrupts.append(interrupt)
        self.log.append(text)
        self.busy_until = max(self.busy_until, self.t) + len(text.split()) / 2.2 + 0.6

    def stop(self):
        self.busy_until = 0.0

    def texts(self):
        return " | ".join(self.log)


def drive(person, seconds, sim, speaker, voice):
    for _ in range(int(seconds * 30)):
        sim.t += 1 / 30
        speaker.t = sim.t
        st = sim.engine.update(person(), W, H, sim.t, speaker.is_busy())
        for ev in st["events"]:
            voice.on_event(ev, st, sim.t)
        voice.update(st, sim.t)
    return st


class VoiceTests(unittest.TestCase):
    def full_session(self):
        sim, sp = Sim(), FakeSpeaker()
        voice = app.Voice(sp)
        state = dict(tilt=6.0, off=0.15, height=0.34)
        for _ in range(120 * 30):
            st = drive(lambda: make_pts(sh_tilt=state["tilt"], offset=state["off"], height=state["height"]), 1 / 30, sim, sp, voice)
            if st["phase"] == "guide":
                if st["step"] == 1 and sim.engine.step_t > 2.5:
                    state["tilt"] = 0.4
                if st["step"] == 2 and sim.engine.step_t > 2.5:
                    state["off"] = 0.0
                if st["step"] == 3:
                    state["height"] = min(0.44, state["height"] + 0.0008)
                if st["step"] == 4 and sim.engine.step_t > 1:
                    state["height"] = 0.41
            if st["phase"] == "coach":
                break
        return sim, sp, voice

    def test_every_setup_step_is_spoken_in_order(self):
        _, sp, _ = self.full_session()
        text = sp.texts()
        positions = [text.index(f"Step {i} of 5") for i in range(1, 6)]
        self.assertEqual(positions, sorted(positions))

    def test_spoken_steps_never_mention_looking_at_the_screen(self):
        _, sp, _ = self.full_session()
        for line in sp.log:
            self.assertNotIn("green", line.lower())
            self.assertNotIn("watch", line.lower())

    def test_it_says_when_posture_is_locked_in(self):
        _, sp, _ = self.full_session()
        self.assertIn("Posture locked in", sp.texts())

    def test_setup_speech_is_never_interrupted(self):
        _, sp, _ = self.full_session()
        self.assertFalse(any(sp.interrupts), "setup should queue speech, never cut it off")

    def test_slump_is_spoken_and_recovery_is_praised(self):
        sim, sp, voice = self.full_session()
        drive(good, 12, sim, sp, voice)
        drive(lambda: make_pts(height=0.29), 20, sim, sp, voice)
        self.assertIn("Grow taller", sp.texts())
        drive(good, 15, sim, sp, voice)
        self.assertIn("good posture", sp.log[-1].lower())


class DrawTests(unittest.TestCase):
    def frame(self):
        return np.full((H, W, 3), 80, np.uint8)

    def test_every_screen_renders_without_error(self):
        sim = Sim()
        shots = {}
        for _ in range(20 * 30):
            sim.t += 1 / 30
            pts = good()
            st = sim.engine.update(pts, W, H, sim.t)
            shots.setdefault(st["phase"], (st, pts))
        complete_setup(sim)
        out = sim.run(1, lambda: make_pts(height=0.29))
        shots["slump"] = (out, make_pts(height=0.29))
        sim.engine.break_due = True
        shots["break"] = (sim.run(0.1, good), good())
        shots["lost"] = (sim.run(2, lambda: make_pts(vis=0.1)), make_pts(vis=0.1))
        shots["none"] = (Sim().run(0.2, lambda: None), None)
        self.assertTrue({"framing", "getready", "guide", "slump", "break", "lost", "none"} <= set(shots))
        for name, (st, pts) in shots.items():
            f = app.draw(self.frame(), st, pts, debug=True)
            self.assertEqual(f.shape, (H, W, 3), name)
            self.assertFalse(np.array_equal(f, self.frame()), f"{name} drew nothing")

    def test_all_on_screen_text_is_plain_ascii(self):
        texts = [s["text"] for s in core.SETUP_STEPS] + [s["why"] for s in core.SETUP_STEPS]
        texts += [core.cue_text(k, 1) for k in core.RULES]
        for t in texts:
            t.encode("ascii")

    def test_speech_engine_starts_muted(self):
        sp = app.Speaker(enabled=True, silent=True)
        try:
            import time
            time.sleep(1.5)
            self.assertTrue(sp.available)
            sp.say("test")
            self.assertTrue(sp.is_busy())
            sp.stop()
            self.assertFalse(sp.is_busy())
        finally:
            sp.close()


if __name__ == "__main__":
    unittest.main()
