import unittest
from unittest.mock import patch

import numpy as np

from astribot_image_service.server import (AstribotControlBridge, STATE_COMPONENTS, PROCESS_STOPPING,
    AstribotActionServerThread, BufferedActionPlayerThread)


class FakeSDK:
    def __init__(self):
        self.state = np.zeros(25)
        self.calls = []

    def get_current_joints_position(self, names):
        parts, offset = {}, 0
        for name, size in STATE_COMPONENTS:
            parts[name] = self.state[offset:offset + size].tolist()
            offset += size
        return [parts[name] for name in names]

    def set_joints_position(self, names, commands, **kwargs):
        self.calls.append((dict(zip(names, commands)), kwargs))
        offset = 0
        for name, size in STATE_COMPONENTS:
            if name in names:
                self.state[offset:offset + size] = commands[names.index(name)]
            offset += size


class TrackingTest(unittest.TestCase):
    def setUp(self):
        PROCESS_STOPPING.clear()
        self.sdk = FakeSDK()
        self.bridge = AstribotControlBridge(
            self.sdk, execute=True, command_frequency=250, control_way="direct",
            use_wbc=False, add_default_torso=False, gripper_binary=False,
            gripper_threshold=50, control_mode="g1", tracking_arm_velocity=1.0)

    def test_feedback_tracking_and_shared_arm_scaling(self):
        target = np.zeros(25)
        target[7], target[15] = 1.0, 0.5
        self.bridge.update_action(target)
        self.bridge._tracking_tick(grippers=False)
        self.assertAlmostEqual(self.sdk.state[7], .004)
        self.assertAlmostEqual(self.sdk.state[15], .002)
        self.bridge._tracking_tick(grippers=False)
        self.assertAlmostEqual(self.sdk.state[7], .008)
        self.assertAlmostEqual(self.sdk.state[15], .004)
        self.assertNotIn("astribot_gripper_left", self.sdk.calls[-1][0])
        self.assertEqual(self.sdk.calls[-1][1]["control_way"], "direct")

    def test_latest_target_replaces_old_direction(self):
        target = np.zeros(25)
        target[7] = 1
        self.bridge.update_action(target)
        self.bridge._tracking_tick(grippers=False)
        target[7] = -1
        self.bridge.update_action(target)
        self.bridge._tracking_tick(grippers=False)
        self.assertAlmostEqual(self.sdk.state[7], 0)

    def test_grippers_are_independent_continuous_targets(self):
        target = np.zeros(25)
        target[14], target[22] = 40, 20
        self.bridge.update_action(target)
        self.bridge._tracking_tick(grippers=False)
        self.assertEqual(self.sdk.state[14], 0)
        for _ in range(110):
            self.bridge._tracking_tick(grippers=True)
        np.testing.assert_allclose(self.sdk.state[[14, 22]], [40, 20])
        self.assertEqual(set(self.sdk.calls[-1][0]), {"astribot_gripper_left", "astribot_gripper_right"})

    def test_invalid_feedback_sends_nothing_and_stop_disarms(self):
        self.bridge.update_action(np.zeros(25))
        self.sdk.state[7] = np.nan
        with self.assertRaises(RuntimeError):
            self.bridge._tracking_tick(grippers=False)
        self.assertFalse(self.sdk.calls)
        self.sdk.state[:] = 0
        self.bridge.stop()
        self.bridge._tracking_tick(grippers=False)
        self.bridge._tracking_tick(grippers=True)
        self.assertFalse(self.sdk.calls)

    def test_protocol_rejects_old_clients_and_chunks_before_mutation(self):
        player = BufferedActionPlayerThread(self.bridge, action_frequency=120, max_buffer_steps=200)
        server = AstribotActionServerThread(self.bridge, player, host="127.0.0.1", port=0, replan_horizon=50)
        for request in [
            {"action": [0] * 25},
            {"required_control_mode": "legacy", "action": [0] * 25},
            {"required_control_mode": "g1", "chunk": [[0] * 25]},
        ]:
            self.assertFalse(server._handle_request(request)["ok"])
        self.assertEqual(self.bridge.snapshot()["actions"], 0)
        response = server._handle_request({"required_control_mode": "g1", "action": [0] * 25})
        self.assertTrue(response["ok"])
        self.assertEqual(response["control"]["control_mode"], "g1")

    def test_legacy_path_still_includes_grippers(self):
        bridge = AstribotControlBridge(self.sdk, execute=False, command_frequency=250,
            control_way="filter", use_wbc=False, add_default_torso=False,
            gripper_binary=False, gripper_threshold=50)
        names = bridge.update_action(np.zeros(25))
        self.assertIn("astribot_gripper_left", names)
        self.assertEqual(bridge.snapshot()["control_mode"], "legacy")

    def test_timeout_holds_feedback_and_new_target_resumes(self):
        target = np.zeros(25)
        target[7], target[14] = 1, 40
        self.bridge.update_action(target)
        self.bridge._tracking_tick(grippers=False)
        before = self.sdk.state.copy()
        self.bridge._last_target_monotonic -= 6
        self.bridge._tracking_tick(grippers=False)
        self.bridge._tracking_tick(grippers=True)
        np.testing.assert_allclose(self.sdk.state, before)
        self.assertTrue(self.bridge.snapshot()["stale_hold"])
        self.bridge.update_action(target)
        self.bridge._tracking_tick(grippers=False)
        self.assertGreater(self.sdk.state[7], before[7])
        self.assertFalse(self.bridge.snapshot()["stale_hold"])

    def test_sdk_fault_latches_until_explicit_hold(self):
        self.bridge.update_action(np.zeros(25))
        def fail(**kwargs):
            PROCESS_STOPPING.set()
            raise RuntimeError("simulated SDK failure")
        with patch.object(self.bridge, "_tracking_tick", side_effect=fail):
            self.bridge._tracking_loop(grippers=False)
        PROCESS_STOPPING.clear()
        with self.assertRaisesRegex(RuntimeError, "fault latched"):
            self.bridge.update_action(np.ones(25))
        self.assertFalse(self.sdk.calls)
        self.bridge.hold_current()
        self.assertIsNone(self.bridge.snapshot()["fault"])
        np.testing.assert_array_equal(self.bridge._latest_action, self.sdk.state)

    def test_blocking_move_cannot_bypass_tracking_limits(self):
        with self.assertRaisesRegex(RuntimeError, "blocking SDK move"):
            self.bridge.move_to_pose(np.ones(25), duration=0.2)
        self.assertFalse(self.sdk.calls)

    def test_dry_run_does_not_send(self):
        self.bridge.execute = False
        self.bridge.update_action(np.ones(25))
        self.bridge._tracking_tick(grippers=False)
        self.bridge._tracking_tick(grippers=True)
        self.assertFalse(self.sdk.calls)


if __name__ == "__main__":
    unittest.main()
