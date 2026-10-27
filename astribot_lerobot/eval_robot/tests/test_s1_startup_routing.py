"""Exercise the real slow-move function without importing CUDA or robot SDKs."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class StartupRoutingTest(unittest.TestCase):
    def run_move(self, *, g1, controller_g1=False, delayed=False):
        path = Path(__file__).resolve().parents[1] / 'eval_s1.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'move_arm_to_pose_slowly')
        clock = SimpleNamespace(now=0.0)
        def sleep(seconds):
            clock.now += seconds
        namespace = dict(np=np, logger_mp=logging.getLogger('test'),
            time=SimpleNamespace(perf_counter=lambda: clock.now, sleep=sleep),
            require_finite_vector=lambda name, value, *args: np.asarray(value, dtype=float),
            _tracking_arm_delta=lambda cfg, value: value,
            _embedded_gripper_delta=lambda cfg, value: np.array([]),
            _max_abs=lambda value: float(np.max(np.abs(value))) if len(value) else 0,
            _consume_manual_snapshot_requests=lambda *args: None,
            compute_slow_move_tau=lambda *args: np.zeros(25),
            _smoothstep=lambda value: value, ASTRIBOT_S1_FULL_DIM=25)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        class Controller:
            require_g1_control = controller_g1
            def __init__(self):
                self.q = np.zeros(25)
                self.streamed = []
                self.moves = 0
            def get_current_dual_arm_q(self):
                return self.q.copy()
            def ctrl_dual_arm(self, q, tau):
                self.streamed.append(clock.now)
                if not delayed or clock.now >= 6:
                    self.q = q.copy()
            def move_dual_arm_to_pose(self, q, **kwargs):
                self.moves += 1
                if g1 or controller_g1:
                    raise AssertionError('G1 startup must never send move')
                self.q = q.copy()
        ctrl = Controller()
        cfg = SimpleNamespace(arm='Astribot_S1', astribot_g1_control=g1,
            init_tolerance=.001, arm_velocity_limit=.25, slow_move_frequency=1,
            init_ee_velocity_limit=25, slow_move_tracking_pause_ratio=.98,
            arm_tracking_error_limit=.35, slow_move_tracking_wait_timeout_s=8,
            init_timeout_s=20, ee_tracking_tolerance=5, abort_on_ee_tracking_error=False)
        self.assertTrue(namespace['move_arm_to_pose_slowly'](cfg, ctrl, None, np.full(25,.1), 'test'))
        return ctrl

    def test_g1_real_entry_reaches_policy_setup(self):
        path = Path(__file__).resolve().parents[1] / 'eval_s1.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_main')
        function.decorator_list = []
        class SetupReached(Exception):
            pass
        def stop_before_setup(cfg):
            raise SetupReached()
        namespace = dict(np=np, logging=logging,
                         configure_policy_for_thermal_input_type=stop_before_setup)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        cfg = SimpleNamespace(astribot_adaptive_tracking=False, astribot_gripper_chunk_blend_steps=0, astribot_gripper_release_confirm_s=0.,
            astribot_direct_gripper_action=False, astribot_alignment_duration_s=0.,
            astribot_safe_control=False, astribot_g1_control=True, send_real_robot=True,
            arm='Astribot_S1', astribot_control_transport='remote',
            policy=SimpleNamespace(compile_model=False))
        with self.assertRaises(SetupReached):
            namespace['eval_main'](cfg)
        self.assertEqual(cfg.astribot_remote_interpolation_steps, 1)
        self.assertEqual(cfg.astribot_policy_smooth_alpha, 0.)

    def test_g1_startup_streams_instead_of_sdk_move(self):
        ctrl = self.run_move(g1=True)
        self.assertEqual(ctrl.moves, 0)
        self.assertTrue(ctrl.streamed)

    def test_controller_mode_also_prevents_move(self):
        self.assertEqual(self.run_move(g1=False, controller_g1=True).moves, 0)

    def test_legacy_keeps_sdk_move(self):
        self.assertEqual(self.run_move(g1=False).moves, 1)

    def test_slow_settling_refreshes_watchdog_target(self):
        ctrl = self.run_move(g1=True, delayed=True)
        self.assertGreater(ctrl.streamed[-1], 5)
        self.assertLess(max(np.diff(ctrl.streamed)), 1.1)


if __name__ == '__main__':
    unittest.main()
