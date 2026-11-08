import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from astribot_image_service.server import AstribotControlBridge
from test_g1_style_service import FakeSDK
from astribot_image_service.server import AstribotActionServerThread, BufferedActionPlayerThread

class SafeServiceTest(unittest.TestCase):
    def setUp(self):
        self.sdk = FakeSDK()
        self.bridge = AstribotControlBridge(self.sdk, execute=True, command_frequency=50,
            control_way=getattr(self, "control_way", "filter"), use_wbc=False, add_default_torso=False,
            gripper_binary=False, gripper_threshold=50, control_mode="safe")
        self.now = 100.0
        self.clock = patch("astribot_image_service.server.time.monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.target = np.zeros(25)
        self.target[7] = .1
        self.target[14] = 40
        self.bridge.update_action(self.target)
        self.bridge._safe_tick()

    def test_tracking_diagnostic_matches_completed_tick(self):
        before = self.sdk.state.copy()
        self.now += .02
        self.bridge._safe_tick()
        status = self.bridge.snapshot()
        diag = status["tracking_diagnostic"]
        np.testing.assert_allclose(diag["feedback_before_command"], before)
        np.testing.assert_allclose(diag["sdk_command"], self.bridge._safe_command)
        np.testing.assert_allclose(diag["target"], self.target)
        self.assertTrue(diag["sdk_sent"])
        self.assertAlmostEqual(diag["cycle_elapsed_s"], .02)
        np.testing.assert_allclose(diag["command_velocity"], (self.bridge._safe_command-before)/.02)
        seq = diag["target_sequence"]
        self.bridge.update_action(self.target)
        # A newly accepted target does not relabel the previous SDK sample.
        self.assertEqual(self.bridge.snapshot()["tracking_diagnostic"]["target_sequence"], seq)
        self.assertGreater(self.bridge.snapshot()["actions"], seq)

    def test_combined_filter_commands_and_last_command_slew(self):
        self.now += .02
        self.bridge._safe_tick()
        self.assertAlmostEqual(self.sdk.state[7], .00002)
        self.assertAlmostEqual(self.sdk.state[14], .0008)
        names, opts = self.sdk.calls[-1]
        self.assertIn("astribot_arm_left", names)
        self.assertIn("astribot_gripper_left", names)
        self.assertEqual(opts["control_way"], "filter")
        # A small disturbance does not drag the next target with feedback.
        self.sdk.state[7] += .0001
        self.now += .02
        self.bridge._safe_tick()
        self.assertAlmostEqual(self.sdk.state[7], .00006)

    def test_command_acceleration_and_reversal(self):
        last_velocity = 0.
        for step in range(100):
            if step == 50:
                self.target[7] = -.1
                self.bridge.update_action(self.target)
            before = self.sdk.state[7]
            self.now += .02
            self.bridge._safe_tick()
            velocity = (self.sdk.state[7] - before) / .02
            self.assertLessEqual(abs(velocity), .020001)
            self.assertLessEqual(abs(velocity - last_velocity) / .02, .05001)
            last_velocity = velocity

    def test_measured_acceleration_rejected(self):
        self.sdk.state[7] = .009
        self.now += .11
        self.bridge._safe_tick()
        # Preserve a physically continuous feedback trajectory despite FakeSDK
        # instantly copying each command into its measured-state buffer.
        self.sdk.state[7] = 0
        self.now += .11
        count = len(self.sdk.calls)
        with self.assertRaisesRegex(RuntimeError, "acceleration.*joint=7"):
            self.bridge._safe_tick()
        self.assertEqual(len(self.sdk.calls), count)

    def test_sample_hold_noise_does_not_trigger_acceleration(self):
        for step in range(20):
            self.sdk.state[7] = .001 if step % 2 else 0
            self.now += .02
            self.bridge._safe_tick()

    def test_measured_runaway_rejected_before_next_sdk_call(self):
        count = len(self.sdk.calls)
        self.sdk.state[18] = 1.76
        self.now += .02
        with self.assertRaisesRegex(RuntimeError, r"speed.*joint=18 \(astribot_arm_right\[3\]\).*dt_s=.*grippers are excluded"):
            self.bridge._safe_tick()
        self.assertEqual(len(self.sdk.calls), count)

    def test_small_negative_gripper_feedback_at_startup(self):
        self.bridge._safe_origin = None
        self.sdk.state[22] = -.0869208723
        self.bridge._safe_tick()
        self.assertEqual(self.bridge._safe_command[22], 0)
        self.assertAlmostEqual(self.bridge._safe_feedback[22], -.0869208723)
        self.assertEqual(self.sdk.state[22], 0)

    def test_negative_gripper_tolerance_preserves_raw_feedback(self):
        for index in (14, 22):
            for value in (-1.22, -3.0, -5.0):
                with self.subTest(index=index, value=value):
                    self.bridge._safe_origin = None
                    self.sdk.state[index] = value
                    self.bridge._safe_tick()
                    self.assertGreaterEqual(self.bridge._safe_command[index], 0)
                    self.assertAlmostEqual(self.bridge._safe_feedback[index], value, places=5)

    def test_large_invalid_gripper_feedback_not_masked(self):
        self.bridge._safe_origin = None
        self.sdk.state[22] = -5.01
        count = len(self.sdk.calls)
        with self.assertRaisesRegex(RuntimeError, "initial gripper feedback"):
            self.bridge._safe_tick()
        self.assertEqual(len(self.sdk.calls), count)

    def test_target_envelope(self):
        self.target[7] = .2
        self.bridge.update_action(self.target)
        self.now += .02
        with self.assertRaisesRegex(RuntimeError, "safe envelope"):
            self.bridge._safe_tick()

    def test_stop_is_latched_and_hold_cannot_clear(self):
        self.bridge.disarm()
        with self.assertRaises(RuntimeError):
            self.bridge.update_action(self.target)
        with self.assertRaises(RuntimeError):
            self.bridge.hold_current()
        count = len(self.sdk.calls)
        self.bridge._safe_tick()
        self.assertEqual(len(self.sdk.calls), count)

    def test_stale_feedback_cycle(self):
        self.now += .3
        with self.assertRaisesRegex(RuntimeError, "cycle"):
            self.bridge._safe_tick()

    def test_worker_latches_feedback_fault(self):
        self.sdk.state[18] = 1.76
        self.now += .02
        count = len(self.sdk.calls)
        with patch.object(self.bridge._stop_event, "wait", side_effect=lambda _: self.bridge._stop_event.set()):
            self.bridge._safe_loop()
        self.assertIn("speed", self.bridge.snapshot()["fault"])
        self.assertEqual(len(self.sdk.calls), count)
        self.assertFalse(self.bridge.snapshot()["has_target"])

    def test_explicit_left_right_mapping(self):
        target = np.arange(25, dtype=float)
        names, commands = self.bridge._command_from_target(target)
        mapped = dict(zip(names, commands))
        self.assertEqual(mapped["astribot_arm_left"], list(range(7, 14)))
        self.assertEqual(mapped["astribot_arm_right"], list(range(15, 22)))
        self.assertEqual(mapped["astribot_gripper_left"], [14])
        self.assertEqual(mapped["astribot_gripper_right"], [22])

class SafeCombinedG1Test(SafeServiceTest):
    control_way = "direct"

    def test_post_retention_smoothing_preserves_goal_and_softens_reversal(self):
        self.bridge.safe_direct_gripper = True
        self.bridge.safe_gripper_feedback_tracking = True
        self.bridge.safe_gripper_limit_retention = .5
        self.bridge.safe_gripper_output_smoothing_s = .08
        self.bridge.gripper_velocity = 40
        self.bridge._safe_command[[14, 22]] = 35.2
        self.target[[14, 22]] = 40
        previous = 35.2
        for tick in range(40):
            self.bridge.update_action(self.target)
            self.sdk.state[[14, 22]] = 35.2
            self.now += .02
            self.bridge._safe_tick()
            actual = self.bridge._safe_command[14]
            self.assertGreaterEqual(actual, previous)
            self.assertLessEqual(actual, 38.000001)
            if tick == 0:
                self.assertAlmostEqual(actual, 35.2 + (1-np.exp(-.25))*2.8, places=6)
            previous = actual
        self.assertAlmostEqual(actual, 38, places=3)
        diag = self.bridge.snapshot()['tracking_diagnostic']
        np.testing.assert_allclose(diag['gripper_unsmoothed_sdk_command'], [38, 38])
        self.target[[14, 22]] = 10
        self.bridge.update_action(self.target)
        self.sdk.state[[14, 22]] = 35.2
        self.now += .02
        self.bridge._safe_tick()
        self.assertGreater(self.bridge._safe_command[14], 22.2)
        self.assertLess(self.bridge._safe_command[14], previous)
        self.bridge._safe_alignment = True
        self.target[[14, 22]] = 40
        self.bridge.update_action(self.target)
        self.sdk.state[[14, 22]] = 35.2
        self.now += .02
        self.bridge._safe_tick()
        np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], [36, 36])

    def test_gripper_half_clipping_both_directions_and_alignment(self):
        self.bridge.safe_direct_gripper = True
        self.bridge.safe_gripper_feedback_tracking = True
        self.bridge.safe_gripper_limit_retention = .5
        self.bridge.gripper_velocity = 40
        for goals, feedback, baseline, expected in [
            ([40, 40], [35.2, 35.2], [36, 36], [38, 38]),
            ([20, 10], [10, 20], [10.8, 19.2], [15.4, 14.6]),
            ([10.2, 19.8], [10, 20], [10.2, 19.8], [10.2, 19.8]),
            ([0, 100], [.2, 99.8], [0, 100], [0, 100]),
        ]:
            self.target[[14, 22]] = goals
            self.bridge.update_action(self.target)
            self.sdk.state[[14, 22]] = feedback
            self.now += .02
            self.bridge._safe_tick()
            np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], expected)
            diagnostic = self.bridge.snapshot()['tracking_diagnostic']
            np.testing.assert_allclose(diagnostic['gripper_baseline_sdk_command'], baseline)
            self.assertIsNone(diagnostic['gripper_velocity_limit'])
        self.bridge._safe_alignment = True
        self.target[[14, 22]] = [40, 40]
        self.bridge.update_action(self.target)
        self.sdk.state[[14, 22]] = [35.2, 35.2]
        self.now += .02
        self.bridge._safe_tick()
        np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], [36, 36])

    def test_feedback_grippers_preserve_body_and_combined_cadence(self):
        self.bridge.safe_direct_gripper = True
        self.bridge.safe_gripper_feedback_tracking = True
        self.bridge.gripper_velocity = 40
        self.target[[14, 22]] = [80, 19.8]
        self.bridge.update_action(self.target)
        # Far and near targets must move independently; stale feedback must
        # remain the reference, rather than accumulating command smoothing.
        for _ in range(3):
            self.sdk.state[[14, 22]] = [10, 20]
            self.now += .02
            self.bridge._safe_tick()
            np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], [10.8, 19.8])
        # The original body acceleration profile is unaffected by jaw motion.
        self.assertAlmostEqual(self.sdk.state[7], .00012)
        self.target[[14, 22]] = [0, 100]
        self.bridge.update_action(self.target)
        self.sdk.state[[14, 22]] = [.2, 99.8]
        self.now += .02
        self.bridge._safe_tick()
        np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], [0, 100])
        self.assertEqual(self.bridge.command_frequency, 50)
        self.assertIsNone(self.bridge._gripper_thread)
        names, _ = self.sdk.calls[-1]
        self.assertIn("astribot_arm_left", names)
        self.assertIn("astribot_gripper_left", names)

    def test_command_tracking_advances_with_stale_feedback_and_limits_reversal(self):
        self.bridge.safe_command_tracking = True
        self.bridge.safe_policy_velocity = .2
        self.bridge.safe_policy_acceleration = .5
        self.bridge.safe_tracking_error = .2
        self.target[:] = 0
        self.target[7] = .08
        self.target[15] = .04
        previous = self.bridge._safe_command.copy()
        previous_v = self.bridge._safe_velocity.copy()
        for step in range(80):
            if step == 40:
                self.target[7] = -.04
                self.target[15] = -.02
            self.bridge.update_action(self.target)
            self.sdk.state[:] = 0  # no measured motion: command must still progress
            self.now += .02
            self.bridge._safe_tick()
            command = self.bridge._safe_command.copy()
            v = (command - previous) / .02
            self.assertLessEqual(max(abs(v[[7, 15]])), .20001)
            self.assertLessEqual(max(abs(v[[7, 15]] - previous_v[[7, 15]]))/.02, .50001)
            if step == 30:
                self.assertGreater(command[7], .03)
                self.assertAlmostEqual(command[7], 2*command[15], places=4)
            previous, previous_v = command, v
        self.assertLess(command[7], .02)
        self.sdk.state[:] = 0
        self.bridge._safe_command[7] = .21
        self.now += .02
        with self.assertRaisesRegex(RuntimeError, "tracking error"):
            self.bridge._safe_tick()

    def test_direct_gripper_smoothing_reversal_and_deadband(self):
        self.bridge.safe_direct_gripper = True
        self.bridge.safe_gripper_smoothing_s = .12
        self.bridge.safe_gripper_velocity = 40
        self.bridge.safe_gripper_acceleration = 200
        previous = 0.
        old_velocity = 0.
        saw_reverse = False
        for tick in range(250):
            target = 12.82 if tick < 15 else (4.65 if tick < 50 else 32.9)
            self.target[14] = target
            self.target[22] = 0
            self.bridge.update_action(self.target)
            # Unchanging encoder feedback must not prevent command convergence.
            self.sdk.state[14] = .24
            self.now += .02
            self.bridge._safe_tick()
            command = self.bridge._safe_command[14]
            v = (command - previous) / .02
            self.assertLessEqual(abs(v), 40.00001)
            self.assertLessEqual(abs(v - old_velocity) / .02, 200.00001)
            if 15 <= tick < 50 and v < 0:
                saw_reverse = True
            self.assertEqual(self.bridge._safe_command[22], 0)
            previous, old_velocity = command, v
        self.assertTrue(saw_reverse)
        self.assertAlmostEqual(previous, 32.9, places=4)
        self.assertIsNone(self.bridge._gripper_thread)

    def test_g1_policy_group_speeds_with_acceleration_backstop(self):
        self.bridge.safe_policy_velocity = 1.0
        self.bridge.safe_policy_acceleration = 1.0
        self.bridge.safe_measured_velocity = 1.5
        self.bridge.safe_measured_acceleration = 5.0
        self.bridge.safe_joint_envelope = 1.0
        self.target[:] = 0
        self.target[[3, 7, 15, 24]] = [1, 1, .5, 1]
        self.bridge.update_action(self.target)
        previous = self.bridge._safe_command.copy()
        old_velocity = np.zeros(25)
        peak = 0
        for _ in range(50):
            self.now += .02
            self.bridge._safe_tick()
            v = (self.bridge._safe_command - previous) / .02
            self.assertLessEqual(abs(v[7]), 1.000001)
            self.assertLessEqual(abs(v[3]), .100001)
            self.assertLessEqual(abs(v[24]), .300001)
            self.assertLessEqual(np.max(np.abs(v - old_velocity)) / .02, 1.00001)
            self.assertAlmostEqual(v[7], 2 * v[15])
            peak = max(peak, v[7])
            previous = self.bridge._safe_command.copy()
            old_velocity = v
        self.assertGreater(peak, .5)
        self.assertEqual(self.bridge.snapshot()["safe_control_revision"], 6)

    def test_direct_gripper_passes_action_in_same_call_with_arm_limits(self):
        path = Path(__file__).resolve().parents[1] / "eval_s1.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_clip_s1_embedded_grippers")
        ns = dict(np=np, EvalRealConfig=object, ASTRIBOT_S1_GRIPPER_INDICES=[14, 22])
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        cfg = SimpleNamespace(astribot_direct_gripper_action=True, max_policy_ee_delta=1.)
        self.bridge.safe_direct_gripper = True
        for left, right in [(32.9, .5), (100, 100), (-1, 120)]:
            raw = self.target.copy()
            raw[[14, 22]] = [left, right]
            raw[[14, 22]] = ns["_clip_s1_embedded_grippers"](cfg, raw, self.sdk.state.copy(), None, 25)
            self.bridge.update_action(raw)
            previous = self.bridge._safe_command.copy()
            self.now += .02
            self.bridge._safe_tick()
            np.testing.assert_allclose(self.sdk.state[[14, 22]], np.clip([left, right], 0, 100))
            self.assertLessEqual(abs(self.bridge._safe_command[7] - previous[7]), .02 * .02 + 1e-9)
            self.assertIn("astribot_gripper_left", self.sdk.calls[-1][0])
            self.assertIn("astribot_arm_left", self.sdk.calls[-1][0])
        self.assertIsNone(self.bridge._gripper_thread)

    def test_gripper_pipeline_crosses_feedback_deadband(self):
        path = Path(__file__).resolve().parents[1] / "eval_s1.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_clip_s1_embedded_grippers")
        ns = dict(np=np, EvalRealConfig=object, ASTRIBOT_S1_GRIPPER_INDICES=[14, 22])
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        cfg = SimpleNamespace(astribot_safe_control=True, max_policy_ee_delta=1.,
                              limit_policy_delta_from_last_command=True)
        self.bridge.safe_gripper_velocity = 10
        self.bridge.safe_gripper_acceleration = 20
        raw = np.zeros(25)
        raw[14] = 31.52
        raw[22] = -.1
        previous = None
        measured = 0.24
        old_command = 0.
        old_velocity = 0.
        for tick in range(100):
            self.sdk.state[14] = measured
            if tick % 5 == 0:
                sent = raw.copy()
                sent[[14, 22]] = ns["_clip_s1_embedded_grippers"](
                    cfg, raw, self.sdk.state.copy(), previous, 25)
                self.assertLessEqual(sent[14] - (previous[14] if previous is not None else measured), 1.000001)
                self.bridge.update_action(sent)
                previous = sent.copy()
            self.now += .02
            self.bridge._safe_tick()
            command = self.bridge._safe_command[14]
            v = (command - old_command) / .02
            self.assertLessEqual(abs(v), 10.00001)
            self.assertLessEqual(abs(v - old_velocity) / .02, 20.00001)
            if abs(command - measured) > 2:  # Simulated SDK deadband.
                measured = command
            old_command, old_velocity = command, v
        self.assertGreater(measured, 10)
        self.assertEqual(self.bridge._safe_command[22], 0)

    def test_relaxed_policy_speed_and_acceleration(self):
        self.bridge.safe_policy_velocity = .05
        self.bridge.safe_policy_acceleration = .1
        previous = self.bridge._safe_command.copy()
        old_velocity = 0.
        peak = 0.
        for _ in range(50):
            self.now += .02
            self.bridge._safe_tick()
            v = (self.bridge._safe_command[7] - previous[7]) / .02
            self.assertLessEqual(abs(v), .050001)
            self.assertLessEqual(abs(v - old_velocity) / .02, .10001)
            peak = max(peak, abs(v))
            previous = self.bridge._safe_command.copy()
            old_velocity = v
        self.assertGreater(peak, .02)

    def test_gripper_endpoint_braking_saturates_without_body_fault(self):
        self.bridge.safe_gripper_velocity = 10
        self.bridge.safe_gripper_acceleration = 20
        self.bridge._safe_command[[14, 22]] = [99.99, .01]
        self.bridge._safe_velocity[[14, 22]] = [5, -5]
        self.sdk.state[[14, 22]] = [99.99, .01]
        self.target[[14, 22]] = [0, 100]  # Reverse while still moving outward.
        self.bridge.update_action(self.target)
        self.now += .02
        self.bridge._safe_tick()
        np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], [100, 0])
        np.testing.assert_allclose(self.bridge._safe_velocity[[14, 22]], [0, 0])
        self.assertIsNone(self.bridge.snapshot()["fault"])
        self.now += .02
        self.bridge._safe_tick()
        self.assertLess(self.bridge._safe_command[14], 100)
        self.assertGreater(self.bridge._safe_command[22], 0)

    def test_five_second_alignment_then_restore_policy_limits(self):
        self.bridge.safe_joint_envelope = 1.0
        self.bridge.safe_tracking_error = .1
        self.bridge.set_safe_alignment(True)
        goal = np.zeros(25)
        goal[[4, 7, 15, 24]] = [-.614, .3, -.2, .8447]
        previous = self.bridge._safe_command.copy()
        old_velocity = np.zeros(25)
        for step in range(1, 251):
            t = step / 250
            self.bridge.update_action(goal * (3 * t * t - 2 * t * t * t))
            self.now += .02
            self.bridge._safe_tick()
            v = (self.bridge._safe_command - previous) / .02
            self.assertLessEqual(np.max(np.abs(v)), .350001)
            self.assertLessEqual(np.max(np.abs(v - old_velocity)) / .02, .50001)
            previous = self.bridge._safe_command.copy()
            old_velocity = v
        np.testing.assert_allclose(self.sdk.state, goal, atol=.001)
        self.bridge.set_safe_alignment(False)
        self.assertFalse(self.bridge.snapshot()["safe_alignment"])
        self.bridge.update_action(goal + np.r_[np.zeros(24), .1])
        self.now += .02
        self.bridge._safe_tick()
        self.assertLessEqual(abs(self.bridge._safe_velocity[24]), .020001)

    def test_alignment_stalled_feedback_still_faults(self):
        self.bridge.set_safe_alignment(True)
        self.target[24] = .1
        self.bridge.update_action(self.target)
        with self.assertRaisesRegex(RuntimeError, "tracking error"):
            for _ in range(100):
                self.sdk.state[:] = 0
                self.now += .02
                self.bridge._safe_tick()

    def test_alignment_protocol_and_disarm(self):
        player = BufferedActionPlayerThread(self.bridge, action_frequency=10, max_buffer_steps=50)
        server = AstribotActionServerThread(self.bridge, player, host="127.0.0.1", port=0, replan_horizon=50)
        self.assertFalse(server._handle_request({"safe_alignment": True})["ok"])
        response = server._handle_request({"required_control_mode": "safe",
            "required_safe_execution_style": "g1_combined", "safe_alignment": True})
        self.assertTrue(response["control"]["safe_alignment"])
        self.bridge.disarm()
        self.assertFalse(self.bridge.snapshot()["safe_alignment"])
        with self.assertRaises(RuntimeError):
            self.bridge.set_safe_alignment(True)

    def test_idle_target_does_not_follow_opposite_arm_drift(self):
        target = np.zeros(25)
        self.bridge.update_action(target)
        for step in range(60):
            # Small opposite encoder disturbances while waiting for r. Keep
            # supplying raw feedback rather than FakeSDK's ideal command copy.
            self.sdk.state[7] = .001
            self.sdk.state[15] = -.001
            if step % 12 == 0:
                self.bridge.update_action(target)  # waiting heartbeat
            self.now += .02
            self.bridge._safe_tick()
            np.testing.assert_allclose(self.bridge._safe_command, target, atol=1e-12)
            np.testing.assert_allclose(self.bridge._safe_velocity, 0, atol=1e-12)

    def test_relaxed_body_thresholds_and_alignment_envelope(self):
        bridge = AstribotControlBridge(
            self.sdk, execute=True, command_frequency=50, control_way="direct",
            use_wbc=False, add_default_torso=False, gripper_binary=False,
            gripper_threshold=50, control_mode="safe", safe_joint_envelope=1.0,
            safe_measured_velocity=.2, safe_measured_acceleration=2.0,
            safe_tracking_error=.1)
        target = self.sdk.state.copy()
        target[24] = .8242
        bridge.update_action(target)
        bridge._safe_tick()
        self.now += .02
        self.sdk.state[24] += .003  # 0.15 rad/s feedback, between old/new limits.
        bridge._safe_tick()
        self.assertLessEqual(abs(bridge._safe_velocity[24]), .001001)
        self.assertEqual(bridge.snapshot()["safe_joint_envelope"], 1.0)
        target[24] = 1.1
        bridge.update_action(target)
        self.now += .02
        with self.assertRaisesRegex(RuntimeError, "safe envelope"):
            bridge._safe_tick()

    def test_relaxed_gripper_limits_leave_body_limits_unchanged(self):
        self.bridge.safe_gripper_velocity = 10
        self.bridge.safe_gripper_acceleration = 20
        previous = self.bridge._safe_command.copy()
        last_velocity = np.zeros(25)
        for _ in range(40):
            self.now += .02
            self.bridge._safe_tick()
            velocity = (self.bridge._safe_command - previous) / .02
            self.assertLessEqual(abs(velocity[7]), .020001)
            self.assertLessEqual(abs(velocity[7] - last_velocity[7]) / .02, .05001)
            self.assertLessEqual(abs(velocity[14]), 10.00001)
            self.assertLessEqual(abs(velocity[14] - last_velocity[14]) / .02, 20.00001)
            previous = self.bridge._safe_command.copy()
            last_velocity = velocity
        self.assertGreater(velocity[14], 1)

    def test_persistent_gripper_encoder_offset_at_both_limits(self):
        # Startup/hold sends raw encoder values, not pre-clipped policy targets.
        self.target[14], self.target[22] = 100.1, -.0594669804
        self.bridge.update_action(self.target)
        self.bridge._safe_origin = None
        for _ in range(20):
            # Unlike FakeSDK's ideal tracking, real encoders retain their offset.
            self.sdk.state[14], self.sdk.state[22] = 100.1, -.0960738584
            self.now += .02
            self.bridge._safe_tick()
            np.testing.assert_allclose(self.bridge._safe_command[[14, 22]], [100, 0])
            np.testing.assert_allclose(self.bridge._safe_velocity[[14, 22]], [0, 0])
            self.assertAlmostEqual(self.bridge._safe_feedback[22], -.0960738584)
        # Valid inward movement must still work from a slightly negative encoder.
        self.target[22] = 1
        self.bridge.update_action(self.target)
        self.sdk.state[22] = -.0960738584
        self.now += .02
        self.bridge._safe_tick()
        self.assertGreater(self.bridge._safe_command[22], 0)
        self.assertLessEqual(self.bridge._safe_velocity[22], .040001)

    def test_combined_filter_commands_and_last_command_slew(self):
        self.target[15] = .05
        self.bridge.update_action(self.target)
        for _ in range(20):
            self.now += .02
            self.bridge._safe_tick()
            self.assertAlmostEqual(self.sdk.state[7], 2 * self.sdk.state[15])
        names, opts = self.sdk.calls[-1]
        self.assertEqual(set(names), {"astribot_torso", "astribot_arm_left", "astribot_arm_right",
                                     "astribot_head", "astribot_gripper_left", "astribot_gripper_right"})
        self.assertEqual(opts["control_way"], "direct")
        self.assertGreater(self.sdk.state[14], 0)
        # Feedback lag must change the next command, while acceleration remains capped.
        before = self.bridge._safe_command.copy()
        old_velocity = self.bridge._safe_velocity.copy()
        self.sdk.state[7] -= .001
        self.now += .02
        self.bridge._safe_tick()
        velocity = (self.bridge._safe_command - before) / .02
        self.assertLess(velocity[7], old_velocity[7])
        self.assertLessEqual(abs(velocity[7] - old_velocity[7]), .00100001)

    def test_only_one_worker_is_started(self):
        with patch("astribot_image_service.server.threading.Thread") as worker:
            self.bridge.start()
        self.assertEqual(worker.call_count, 1)
        self.assertIsNone(self.bridge._gripper_thread)

    def test_execution_style_handshake_before_mutation(self):
        player = BufferedActionPlayerThread(self.bridge, action_frequency=10, max_buffer_steps=50)
        server = AstribotActionServerThread(self.bridge, player, host="127.0.0.1", port=0, replan_horizon=50)
        count = self.bridge.snapshot()["actions"]
        request = {"required_control_mode": "safe", "action": self.target.tolist()}
        self.assertFalse(server._handle_request(request)["ok"])
        self.assertEqual(self.bridge.snapshot()["actions"], count)
        request["required_safe_execution_style"] = "g1_combined"
        response = server._handle_request(request)
        self.assertTrue(response["ok"])
        self.assertEqual(response["control"]["safe_execution_style"], "g1_combined")
        request["chunk"] = [self.target.tolist()]
        self.assertFalse(server._handle_request(request)["ok"])


if __name__ == "__main__":
    unittest.main()
