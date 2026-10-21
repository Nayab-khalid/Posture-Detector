import unittest

from helpers import Sim, complete_setup, good, make_pts, W, H, core


class SetupTests(unittest.TestCase):
    def test_no_person_asks_to_sit_in_front(self):
        out = Sim().run(0.5, lambda: None)
        self.assertEqual(out["phase"], "framing")
        self.assertIn("can't see you", out["detail"])

    def test_hips_are_never_required(self):
        out = Sim().run(0.5, lambda: make_pts(vis=0.2))
        self.assertNotIn("hip", out["detail"].lower())
        self.assertIn("shoulders", out["detail"])

    def test_framing_then_countdown_then_guide(self):
        sim = Sim()
        phases = []
        for _ in range(20 * 30):
            out = sim.run(1 / 30, good)
            if not phases or phases[-1] != out["phase"]:
                phases.append(out["phase"])
        self.assertEqual(phases[:3], ["framing", "getready", "guide"])

    def test_cooperative_person_completes_setup_with_sane_baseline(self):
        sim = Sim()
        out = complete_setup(sim)
        self.assertEqual(out["phase"], "coach")
        b = sim.engine.baseline
        self.assertAlmostEqual(b["height"], 0.41, delta=0.02)
        self.assertAlmostEqual(b["offset"], 0.0, delta=0.03)
        self.assertLess(abs(b["sh_tilt"]), 2.0)
        self.assertIn("ready", sim.events)

    def test_setup_gives_specific_corrections(self):
        sim = Sim()
        seen = set()
        state = dict(tilt=7.0, off=0.18, height=0.34)
        for _ in range(40 * 30):
            out = sim.run(1 / 30, lambda: make_pts(sh_tilt=state["tilt"], offset=state["off"], height=state["height"]))
            seen.add(out["live"])
            if out["phase"] == "guide" and out["step"] == 1 and sim.engine.step_t > 5:
                state["tilt"] = 0
        self.assertTrue(any("right shoulder is lower" in s for s in seen), seen)

    def test_lopsided_setup_never_loops_forever(self):
        sim = Sim()
        out = complete_setup(sim, start=dict(tilt=20.0, off=0.0, height=0.41))
        self.assertEqual(out["phase"], "coach")
        self.assertLessEqual(sim.engine.retries, 1)

    def test_steps_wait_for_speech_to_finish(self):
        sim = Sim()
        sim.run(12, good)  # into the guide phase
        sim.speaking = True
        before = (sim.engine.phase, sim.engine.step)
        sim.run(40, good)
        self.assertEqual((sim.engine.phase, sim.engine.step), before)
        sim.speaking = False
        sim.run(8, good)
        self.assertNotEqual((sim.engine.phase, sim.engine.step), before)


class CoachingTests(unittest.TestCase):
    def setUp(self):
        self.sim = Sim()
        complete_setup(self.sim)
        self.sim.events.clear()

    def test_sitting_still_with_sensor_noise_stays_good_and_quiet(self):
        out = self.sim.run(60, good)
        self.assertEqual(out["band"], "good")
        self.assertGreaterEqual(out["score"], 95)
        self.assertNotIn("nudge", self.sim.events)

    def test_an_inch_of_movement_is_ignored(self):
        out = self.sim.run(10, lambda: make_pts(height=0.41 * 0.96, offset=0.03))
        self.assertEqual(out["band"], "good")

    def test_slump_gets_one_clear_instruction_then_a_nudge(self):
        out = self.sim.run(25, lambda: make_pts(height=0.29, pitch=0.2))
        self.assertEqual(out["band"], "bad")
        self.assertIn("Grow taller", out["detail"])
        self.assertIn("nudge", self.sim.events)
        self.assertTrue(self.sim.engine.episodes or self.sim.engine._ep)

    def test_recovery_returns_to_good(self):
        self.sim.run(15, lambda: make_pts(height=0.29))
        out = self.sim.run(10, good)
        self.assertEqual(out["band"], "good")

    def test_side_cues_use_your_left_and_right(self):
        out = self.sim.run(10, lambda: make_pts(offset=0.2))
        self.assertIn("right", out["detail"])
        out = self.sim.run(10, lambda: make_pts(offset=-0.2))
        self.assertIn("left", out["detail"])

    def test_losing_tracking_pauses_and_is_not_counted(self):
        good_before = self.sim.engine.time_in["good"]
        out = self.sim.run(3, lambda: make_pts(vis=0.1))
        self.assertEqual(out["headline"], "I lost sight of you")
        self.assertAlmostEqual(self.sim.engine.time_in["good"], good_before, delta=1.0)

    def test_stepping_away_resets_the_sit_timer(self):
        self.sim.run(30, good)
        self.assertGreater(self.sim.engine.sit_s, 20)
        self.sim.run(25, lambda: None)
        self.assertEqual(self.sim.engine.sit_s, 0.0)

    def test_break_reminder_after_30_minutes_and_snooze(self):
        for _ in range(31 * 60 * 10):
            self.sim.t += 0.1
            out = self.sim.engine.update(good(), W, H, self.sim.t)
            self.sim.events += out["events"]
        self.assertEqual(self.sim.events.count("break"), 1)
        self.assertTrue(self.sim.engine.break_due)
        self.sim.engine.dismiss_break()
        self.assertFalse(self.sim.engine.break_due)

    def test_five_minute_streak_milestone(self):
        for _ in range(int(5.2 * 60 * 10)):
            self.sim.t += 0.1
            out = self.sim.engine.update(good(), W, H, self.sim.t)
            self.sim.events += out["events"]
        self.assertIn("streak:5", self.sim.events)

    def test_locked_in_card_shows_briefly(self):
        sim = Sim()
        out = complete_setup(sim)
        self.assertTrue(out["celebrate"])
        out = sim.run(5, good)
        self.assertFalse(out["celebrate"])

    def test_summary_reports_time_and_top_issue(self):
        self.sim.run(20, good)
        self.sim.run(20, lambda: make_pts(height=0.29))
        s = self.sim.engine.summary()
        self.assertGreater(s["good_s"], 10)
        self.assertGreater(s["bad_s"], 5)
        self.assertEqual(s["top_issue"], "slumping")


class MetricTests(unittest.TestCase):
    def test_metrics_do_not_depend_on_distance_from_camera(self):
        near, far = make_pts(sw=400, noise=0), make_pts(sw=200, noise=0)
        mn, _ = core.compute_metrics(near, W)
        mf, _ = core.compute_metrics(far, W)
        for k in ("height", "pitch", "scale", "offset", "sh_tilt"):
            self.assertAlmostEqual(mn[k], mf[k], places=3, msg=k)

    def test_penalty_has_a_dead_zone(self):
        rule = core.RULES["height"]
        self.assertEqual(core.penalty(0.05, rule), 0.0)
        self.assertEqual(core.penalty(0.5, rule), 1.0)


if __name__ == "__main__":
    unittest.main()
