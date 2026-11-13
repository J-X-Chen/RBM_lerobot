"""Local latest-target loop, exclusive client ownership, bounded motion and latched stops."""
import threading
import time
import numpy as np
from .contract import BODY, GRIPPERS, PROTOCOL, vector


class H1Control:
    def __init__(self, sdk, limits, *, execute=False, frequency=100., timeout=.5,
                 joint_velocity=.15, lift_velocity=.02, gripper_velocity=.5,
                 envelope=.5, lift_envelope=.15, clock=time.monotonic):
        values = (frequency, timeout, joint_velocity, lift_velocity, gripper_velocity, envelope, lift_envelope)
        if not all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError('Control limits must be finite and positive')
        self.sdk, self.low, self.high = sdk, *limits
        self.execute, self.frequency, self.timeout = execute, frequency, timeout
        self.clock = clock
        self.lock = threading.RLock()
        self.velocity_limit = np.full(21, joint_velocity)
        self.velocity_limit[16] = lift_velocity
        self.velocity_limit[GRIPPERS] = gripper_velocity
        self.acceleration_limit = self.velocity_limit * 4
        self.envelope = np.full(21, envelope)
        self.envelope[16] = lift_envelope
        self.envelope[GRIPPERS] = 1.5
        self.owner = None
        self.fault = None
        self.target = self.command = self.origin = None
        self.feedback = self.sdk.read()
        self.velocity = np.zeros(21)
        self.last_tick = self.last_contact = self.clock()
        self.sent = 0
        self.diagnostic = None

    def _read(self):
        q = vector(self.sdk.read(), 23)
        # Small encoder offset at an open gripper is accepted without masking its feedback.
        tolerance = np.full(21, .005)
        tolerance[GRIPPERS] = .03
        if np.any(q[:21] < self.low - tolerance) or np.any(q[:21] > self.high + tolerance):
            raise RuntimeError('Feedback outside joint limits')
        return q

    def _latch(self, reason):
        if self.fault is None:
            self.fault = str(reason)
        self.target = None
        self.velocity[:] = 0

    def disconnect(self, owner):
        with self.lock:
            if owner == self.owner:
                self._latch('control client disconnected')

    def handle(self, request, owner):
        with self.lock:
            if request.get('protocol') != PROTOCOL:
                raise ValueError('H1 protocol mismatch')
            commands = [k for k in ('get_state', 'arm', 'heartbeat', 'hold_current', 'disarm', 'action') if k in request]
            if len(commands) != 1:
                raise ValueError('Exactly one control operation required')
            op = commands[0]
            if op != 'action' and request[op] is not True:
                raise ValueError('Control operation must be true')
            if op == 'get_state':
                return self.snapshot()
            if op == 'disarm':
                # Any operator may stop, but only the owner may command movement.
                self._latch('disarmed by client')
                return self.snapshot()
            if self.fault:
                raise RuntimeError(f'Latched fault: {self.fault}; restart service after inspection')
            if op == 'arm':
                if self.owner is not None:
                    raise RuntimeError('Control already owned')
                if self.execute and not self.sdk.ready():
                    raise RuntimeError('H1 must be initialized in LOW_LEVEL before arming')
                self.feedback = self._read()
                self.origin = self.feedback.copy()
                self.command = self.origin.copy()
                self.command[:21] = np.clip(self.command[:21], self.low, self.high)
                self.command[21:] = 0
                self.target = self.command.copy()
                self.owner = owner
                self.last_tick = self.clock()
            elif self.owner != owner:
                raise RuntimeError('Only the armed client can update control')
            elif op == 'action':
                target = vector(request['action'], 19)
                # Constant-column unnormalization can leave sub-microunit roundoff.
                if np.all(target[GRIPPERS] >= -1e-5) and np.all(target[GRIPPERS] <= 1.5+1e-5):
                    target[GRIPPERS] = np.clip(target[GRIPPERS], 0., 1.5)
                if np.any(target < self.low[:19]) or np.any(target > self.high[:19]):
                    raise ValueError('Action outside URDF/SDK limits')
                if np.any(abs(target - self.origin[:19]) > self.envelope[:19]):
                    raise ValueError('Action outside arming envelope')
                self.target[:19] = target
            elif op == 'hold_current':
                self.feedback = self._read()
                self.target[:19] = np.clip(self.feedback[:19], self.low[:19], self.high[:19])
            self.last_contact = self.clock()
            return self.snapshot()

    def snapshot(self):
        now = self.clock()
        return dict(ok=True, protocol=PROTOCOL, state=self.feedback.tolist(),
                    control=dict(server_time=now, execute=self.execute, armed=self.owner is not None and not self.fault,
                                 fault=self.fault, state_age_s=now-self.last_tick, sent=self.sent,
                                 target=None if self.target is None else self.target.tolist(),
                                 tracking_diagnostic=self.diagnostic))

    def tick(self):
        with self.lock:
            now = self.clock()
            dt = now - self.last_tick
            if self.fault:
                return
            try:
                q = self._read()
                completed = self.clock()
                if completed-now > self.timeout:
                    raise RuntimeError('SDK feedback polling timeout')
                now, dt = completed, completed-self.last_tick
                if self.owner is not None:
                    if now - self.last_contact > self.timeout:
                        raise RuntimeError('Control watchdog expired')
                    if not 0 < dt <= self.timeout:
                        raise RuntimeError('Control cycle timeout')
                    if self.execute and not self.sdk.ready():
                        raise RuntimeError('H1 control mode or initialization changed')
                    if np.any(abs(q[:21] - self.origin[:21]) > self.envelope + .02):
                        raise RuntimeError('Measured motion outside arming envelope')
                    if np.max(abs(q[21:])) > .03:
                        raise RuntimeError('Unexpected chassis movement')
                    if np.max(abs(q[19:21] - self.origin[19:21])) > .05:
                        raise RuntimeError('Unexpected head movement')
                    # Measured motion is checked independently of policy targets.
                    if np.any(abs(q[BODY] - self.feedback[BODY]) / dt > self.velocity_limit[BODY] * 5):
                        raise RuntimeError('Measured velocity exceeded limit')
                    if self.execute and np.any(abs(q[BODY] - self.command[BODY]) > np.maximum(self.velocity_limit[BODY], .03)):
                        raise RuntimeError('Tracking error exceeded limit')
                    error = self.target[:21] - self.command[:21]
                    # Include one discrete tick in the stopping distance; sqrt(2*a*d)
                    # alone can cross a hard boundary at the final gripper step.
                    braking = np.sqrt((self.acceleration_limit*dt)**2 + 2*self.acceleration_limit*abs(error)) - self.acceleration_limit*dt
                    desired = np.sign(error) * np.minimum(self.velocity_limit, braking)
                    velocity = self.velocity + np.clip(desired-self.velocity, -self.acceleration_limit*dt, self.acceleration_limit*dt)
                    command = self.command.copy()
                    command[:21] += velocity * dt
                    if np.any(command[:21] < self.low) or np.any(command[:21] > self.high):
                        raise RuntimeError('Limited command would exceed joint limits')
                    command[21:] = 0
                    if self.execute:
                        self.sdk.write(command)
                        self.sent += 1
                        if self.clock()-now > self.timeout:
                            raise RuntimeError("SDK command timeout")
                    self.diagnostic = dict(feedback_before_command=q.tolist(), sdk_command=command.tolist(),
                                           target=self.target.tolist(), command_velocity=velocity.tolist(),
                                           sdk_sent=self.execute, cycle_elapsed_s=dt)
                    self.command, self.velocity = command, velocity
                self.feedback, self.last_tick = q, now
            except Exception as exc:
                self._latch(exc)

    def run(self, stop):
        while not stop.is_set():
            started = self.clock()
            self.tick()
            stop.wait(max(0., 1/self.frequency - (self.clock()-started)))
