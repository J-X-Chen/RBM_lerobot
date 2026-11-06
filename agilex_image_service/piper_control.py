"""Two independent Piper CAN interfaces, in the dataset's left-7/right-7 order."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

RAD_TO_MDEG = 180_000 / np.pi
JOINT_BOUNDARY_TOLERANCE = 0.05  # bounded startup/intermediate-command overshoot
POLICY_TARGET_TOLERANCE = 0.08  # prediction only; always clip to nominal limits before rate limiting
ARM_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


@dataclass
class ArmConfig:
    can: str
    joint_min: list[float]  # radians in the dataset/SDK joint convention
    joint_max: list[float]
    gripper_max_m: float
    # physical opening (m) = dataset gripper value * scale + offset
    gripper_scale: float = 1.0
    gripper_offset_m: float = 0.0

    def __post_init__(self):
        lo, hi = np.asarray(self.joint_min), np.asarray(self.joint_max)
        if lo.shape != (6,) or hi.shape != (6,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
            raise ValueError("joint_min/max must be six finite ordered limits in radians")
        if not self.can or not np.isfinite([self.gripper_max_m, self.gripper_scale, self.gripper_offset_m]).all():
            raise ValueError("Invalid CAN interface or gripper calibration")
        if self.gripper_max_m <= 0 or self.gripper_scale == 0:
            raise ValueError("gripper_max_m must be positive and gripper_scale nonzero")


def vector(action):
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (14,) or not np.isfinite(a).all():
        raise ValueError("Expected a finite 14D left-arm/gripper, right-arm/gripper vector")
    return a.copy()


class PiperBimanual:
    def __init__(
        self,
        left,
        right,
        *,
        execute=False,
        sdk_factory=None,
        feedback_timeout=1.0,
        joint_speed=0.3,
        gripper_speed=0.02,
        tracking_limit=0.35,
        speed_percent=20,
        gripper_effort=500,
        active_arm="both",
        command_timeout=5.0,
        smooth_control=False,
        control_frequency=30.0,
        joint_acceleration=0.6,
        gripper_acceleration=0.04,
        tracking_response_s=0.15,
    ):
        if left.can == right.can:
            raise ValueError("Left and right arms require distinct CAN interfaces")
        if active_arm not in {"left", "right", "both"}:
            raise ValueError("active_arm must be left, right or both")
        if (
            not np.isfinite([feedback_timeout, joint_speed, gripper_speed, tracking_limit]).all()
            or min(feedback_timeout, joint_speed, gripper_speed, tracking_limit) <= 0
        ):
            raise ValueError("Timeout, speeds and tracking limit must be positive")
        if not 1 <= speed_percent <= 100 or not 0 <= gripper_effort <= 5000:
            raise ValueError("Invalid SDK speed percentage or gripper effort")
        if not np.isfinite(command_timeout) or command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        self.command_timeout = command_timeout
        if not np.isfinite([control_frequency, joint_acceleration, gripper_acceleration]).all() or not (
            30 <= control_frequency <= 100 and joint_acceleration > 0 and gripper_acceleration > 0
        ):
            raise ValueError("Control frequency must be 30..100 Hz and accelerations positive")
        self.smooth_control = smooth_control
        self.control_frequency = control_frequency
        self.joint_acceleration = joint_acceleration
        self.gripper_acceleration = gripper_acceleration
        if not np.isfinite(tracking_response_s) or not 0.1 <= tracking_response_s <= 1.0:
            raise ValueError("tracking_response_s must be in [0.1, 1.0] seconds")
        self.tracking_response_s = tracking_response_s
        self.servo = None
        self.pending = None
        self.velocity = np.zeros(14)
        self.lock = threading.RLock()
        self.shutdown = threading.Event()
        self.watchdog = None
        self.heartbeat = time.monotonic()
        self.configs = [left, right]
        self.execute = execute
        self.factory = sdk_factory
        self.timeout = feedback_timeout
        self.joint_speed, self.gripper_speed = joint_speed, gripper_speed
        self.tracking_limit = tracking_limit
        self.speed_percent, self.effort = speed_percent, gripper_effort
        self.active_arm = active_arm
        self.arms = []
        self.enabled = False
        self.holding = False
        self.last = None
        self.anchor = None
        self.faulted = False

    def connect(self):
        if self.factory is None:
            try:
                from piper_sdk import C_PiperInterface_V2
            except ImportError as exc:
                raise ImportError("Install local SDK dependencies: python -m pip install -e ./piper_sdk") from exc
            self.factory = C_PiperInterface_V2
        try:
            for config in self.configs:
                arm = self.factory(can_name=config.can)
                self.arms.append(arm)
                # Read feedback without the SDK's optional initialization queries/configuration.
                arm.ConnectPort(piper_init=False)
            deadline = time.monotonic() + 5
            while True:
                try:
                    self.anchor = self.read_state()
                    # Feedback is an observation, not a requested target. Read-only connection must
                    # remain available for diagnosing a pose outside the configured command limits.
                    self.last = self.anchor.copy()
                    return
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
        except BaseException:
            self.close()
            raise

    def read_state(self):
        values = []
        for arm, config in zip(self.arms, self.configs):
            joint = arm.GetArmJointMsgs()
            grip = arm.GetArmGripperMsgs()
            status = arm.GetArmStatus()
            now = time.time()
            for msg in (joint, grip, status):
                age = now - msg.time_stamp
                if not np.isfinite(age) or not 0 <= age <= self.timeout:
                    raise RuntimeError(f"Missing/stale feedback on {config.can}: age={age}")
            if status.arm_status.err_code != 0 or status.arm_status.arm_status != 0:
                raise RuntimeError(f"Piper fault on {config.can}: {status.arm_status}")
            values.extend(getattr(joint.joint_state, f"joint_{i}") / RAD_TO_MDEG for i in range(1, 7))
            opening = grip.gripper_state.grippers_angle / 1_000_000
            values.append((opening - config.gripper_offset_m) / config.gripper_scale)
        return vector(values)

    def validate_target(self, action, tolerance=0.0):
        a = vector(action)
        for offset, config in zip((0, 7), self.configs):
            q = a[offset : offset + 6]
            if np.any(q < np.asarray(config.joint_min) - tolerance) or np.any(
                q > np.asarray(config.joint_max) + tolerance
            ):
                raise ValueError(f"Joint target outside configured hardware limits on {config.can}: {q}")
            opening = a[offset + 6] * config.gripper_scale + config.gripper_offset_m
            if not -min(tolerance, 0.002) <= opening <= config.gripper_max_m + min(tolerance, 0.002):
                raise ValueError(f"Gripper target {opening}m outside hardware range on {config.can}")
        return a

    def clip_target(self, action):
        target = vector(action)
        for offset, cfg in zip((0, 7), self.configs):
            target[offset : offset + 6] = np.clip(target[offset : offset + 6], cfg.joint_min, cfg.joint_max)
            opening = target[offset + 6] * cfg.gripper_scale + cfg.gripper_offset_m
            target[offset + 6] = (np.clip(opening, 0, cfg.gripper_max_m) - cfg.gripper_offset_m) / cfg.gripper_scale
        return target

    def prepare_target(self, action):
        target = vector(action)
        if self.active_arm == "left":
            target[7:] = self.anchor[7:]
        elif self.active_arm == "right":
            target[:7] = self.anchor[:7]
        self.validate_target(target, tolerance=JOINT_BOUNDARY_TOLERANCE)
        target = self.clip_target(target)
        if self.active_arm == "left":
            target[7:] = self.anchor[7:]
        elif self.active_arm == "right":
            target[:7] = self.anchor[:7]
        return target

    def enable(self):
        if not self.execute:
            return
        if self.faulted:
            raise RuntimeError("Controller fault is latched; inspect hardware before restarting")
        self.holding = False
        self.heartbeat = time.monotonic()
        self.last = self.read_state()
        self.anchor = self.last.copy()
        self.pending = self.last.copy()
        self.velocity.fill(0)
        self.validate_target(self.last, tolerance=JOINT_BOUNDARY_TOLERANCE)
        # Keep the measured anchor: clipping here would bypass the first-step speed limit.
        self.enabled = True  # partial enabling must also trigger stop on failure
        deadline = time.monotonic() + 5
        try:
            pending = list(self.arms)
            while pending:
                pending = [arm for arm in pending if not arm.EnablePiper()]
                if time.monotonic() > deadline:
                    raise RuntimeError("Piper enable timeout")
                time.sleep(0.02)
            self.heartbeat = time.monotonic()
            if self.watchdog is None or not self.watchdog.is_alive():
                self.watchdog = threading.Thread(target=self._monitor, daemon=True)
                self.watchdog.start()
            if self.smooth_control and (self.servo is None or not self.servo.is_alive()):
                self.servo = threading.Thread(target=self._servo_loop, daemon=True)
                self.servo.start()
        except BaseException:
            self.stop()
            raise

    def _monitor(self):
        while not self.shutdown.wait(0.05):
            try:
                with self.lock:
                    state = self.read_state()
                    if self.holding:
                        if np.max(np.abs(state[ARM_INDICES] - self.last[ARM_INDICES])) > self.tracking_limit:
                            raise RuntimeError("Holding feedback deviated from target")
                    elif time.monotonic() - self.heartbeat > self.command_timeout:
                        raise RuntimeError("No fresh control command")
            except Exception:
                logger.exception("Piper watchdog stopped control")
                self.stop()
                return

    def send(self, action, dt):
        with self.lock:
            return self._send(action, dt)

    def _servo_loop(self):
        period = 1 / self.control_frequency
        previous = time.monotonic()
        while not self.shutdown.wait(period):
            try:
                with self.lock:
                    now = time.monotonic()
                    dt = min(period, now - previous)
                    previous = now
                    if self.faulted or not self.enabled:
                        return
                    if self.holding:
                        continue
                    if now - self.heartbeat > self.command_timeout:
                        raise RuntimeError("No fresh client target")
                    self._send(self.pending, dt, servo_tick=True)
            except Exception:
                logger.exception("Piper smooth control stopped")
                self.stop()
                return

    def finish_hold(self):
        """After verified normal return, retain the final powered position target."""
        with self.lock:
            if not self.enabled or self.faulted:
                raise RuntimeError("Cannot finish an inactive/faulted controller")
            state = self.read_state()
            if self.smooth_control and np.max(np.abs(self.pending - self.last)) > 1e-4:
                raise RuntimeError("Final smoothed target has not settled")
            if (np.max(np.abs(state[ARM_INDICES] - self.last[ARM_INDICES])) > 0.03
                    or np.max(np.abs(state[[6, 13]] - self.last[[6, 13]])) > 0.003):
                raise RuntimeError("Final pose has not settled")
            self.holding = True
            self.velocity.fill(0)

    def _send(self, action, dt, servo_tick=False):
        if not np.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("dt must be a control period in (0, 1] seconds")
        if self.faulted or self.holding or (self.execute and not self.enabled):
            raise RuntimeError("Controller not enabled or faulted")
        try:
            target = vector(action)
            if self.active_arm == "left":
                target[7:] = self.anchor[7:]
            elif self.active_arm == "right":
                target[:7] = self.anchor[:7]
            # Allow minor model overshoot at limits; reject gross unit/calibration errors.
            self.validate_target(target, tolerance=POLICY_TARGET_TOLERANCE)
            target = self.clip_target(target)
            # The inactive arm holds its measured anchor, including a small boundary offset.
            if self.active_arm == "left":
                target[7:] = self.anchor[7:]
            elif self.active_arm == "right":
                target[:7] = self.anchor[:7]
            state = self.read_state()
            if self.execute and np.max(np.abs(state[ARM_INDICES] - self.last[ARM_INDICES])) > self.tracking_limit:
                raise RuntimeError("Arm feedback is too far from the previous command")
            if self.smooth_control and self.execute and not servo_tick:
                self.pending = target.copy()
                self.heartbeat = time.monotonic()  # Only external targets renew the watchdog.
                return self.last.copy()
            limits = np.full(14, self.joint_speed * dt)
            for i, cfg in zip((6, 13), self.configs):
                limits[i] = self.gripper_speed * dt / abs(cfg.gripper_scale)
            base = self.last if self.execute else state
            command = base + np.clip(target - base, -limits, limits)
            if servo_tick:
                acceleration = np.full(14, self.joint_acceleration)
                for i, cfg in zip((6, 13), self.configs):
                    acceleration[i] = self.gripper_acceleration / abs(cfg.gripper_scale)
                error = target - base
                vmax = limits / dt
                # Ease toward target, brake near it, and bound changes in commanded velocity.
                desired = np.sign(error) * np.minimum.reduce([
                    vmax, np.abs(error) / self.tracking_response_s, np.sqrt(2 * acceleration * np.abs(error))])
                velocity = self.velocity + np.clip(desired - self.velocity, -acceleration * dt, acceleration * dt)
                delta = velocity * dt
                # Never overshoot or keep moving away after a target reversal. Position safety
                # wins over acceleration continuity at these boundaries.
                delta = np.where(delta * error > 0, np.sign(error) * np.minimum(np.abs(delta), np.abs(error)), 0)
                command = base + delta
                self.velocity = delta / dt
            if self.execute:
                self.validate_target(command, tolerance=JOINT_BOUNDARY_TOLERANCE)
                previous_excess = np.abs(base - self.clip_target(base))
                next_excess = np.abs(command - self.clip_target(command))
                if np.any(next_excess > previous_excess + 1e-10):
                    raise RuntimeError("Command would move farther outside hardware limits")
            if self.execute:
                # Validate both arms before issuing any CAN commands.
                for offset, arm, cfg in zip((0, 7), self.arms, self.configs):
                    joints = np.rint(command[offset : offset + 6] * RAD_TO_MDEG).astype(int)
                    grip = round((command[offset + 6] * cfg.gripper_scale + cfg.gripper_offset_m) * 1_000_000)
                    arm.MotionCtrl_2(0x01, 0x01, self.speed_percent, 0x00)
                    arm.JointCtrl(*joints.tolist())
                    arm.GripperCtrl(grip, self.effort, 0x01, 0)
                if not servo_tick:
                    self.heartbeat = time.monotonic()
                self.last = command
            else:
                self.last = state  # read-only previews never integrate fictitious robot motion
            return command
        except BaseException:
            self.stop()
            raise

    def stop(self):
        with self.lock:
            self._stop()

    def _stop(self):
        self.pending = None
        self.velocity.fill(0)
        self.holding = False
        self.faulted = True
        if self.execute and self.enabled:
            for arm in self.arms:
                try:
                    arm.MotionCtrl_1(0x01, 0, 0)
                except Exception:
                    logger.exception("Piper emergency stop send failed; attempting the other arm")
        self.enabled = False

    def close(self):
        self.shutdown.set()
        self.stop()
        if self.watchdog is not None:
            self.watchdog.join(timeout=1)
        if self.servo is not None:
            self.servo.join(timeout=1)
        for arm in self.arms:
            try:
                arm.DisconnectPort()
            except Exception:
                logger.exception("Piper disconnect failed")
        self.arms.clear()
