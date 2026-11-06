import argparse
import collections
import ctypes
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path.home() / 'code_file'
MASTER = 'http://127.0.0.1:11311'
CAN = {'can_left': (1000000, '1-13:1.0'),
       'can_right': (1000000, '1-12:1.0'), 'can0': (500000, '1-4:1.0')}

def say(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)

def group_alive(pgid):
    # Ignore zombies: they cannot execute or keep devices open.
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                return True
        except (OSError, ValueError, IndexError):
            pass
    return False

def descendants():
    table = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            table[int(p.parent.name)] = (int(fields[1]), fields[0], fields[19])
        except (OSError, ValueError, IndexError):
            pass
    owned = {os.getpid()}
    while True:
        added = {pid for pid, (parent, _, _) in table.items() if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    return {pid: table[pid] for pid in owned if pid != os.getpid() and pid in table}

def live_descendants():
    return {pid: info for pid, info in descendants().items() if info[1] not in ('Z', 'X')}

def signal_descendants(sig):
    for pid, (_, _, birth) in reversed(list(live_descendants().items())):
        try:
            fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()
            if fields[19] == birth:
                os.kill(pid, sig)
        except (ProcessLookupError, FileNotFoundError):
            pass

class Supervisor:
    def __init__(self, directory):
        # roslaunch nodes may create their own sessions. Adopt orphaned descendants
        # so a crashed launcher cannot hide them from cleanup.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), 'Cannot enable child-subreaper cleanup')
        self.directory = directory
        self.children = []
        self.stop_signal = None
        self.closed = False
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self.on_signal)

    def on_signal(self, sig, frame):
        self.stop_signal = sig

    def start(self, name, command):
        if self.stop_signal:
            raise InterruptedError('收到停止信号')
        path = self.directory / (name + '.log')
        with path.open('w') as output:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
        self.children.append((name, proc, path))
        say('启动 {}，PID/进程组={}，日志={}'.format(name, proc.pid, path))
        return proc

    def check(self):
        if self.stop_signal:
            raise InterruptedError('收到停止信号')
        for name, proc, path in self.children:
            status = proc.poll()
            if status is not None:
                raise RuntimeError('{} 已退出，退出码 {}，日志 {}'.format(name, status, path))

    def pause(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.1, max(0, end - time.monotonic())))

    def cleanup(self):
        if self.closed:
            return
        self.closed = True
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if not self.children:
            return
        say('正在关闭本次启动的所有进程及其子进程……')
        # Signal cameras/arms before the ROS master; wait for every group.
        for sig, grace in ((signal.SIGINT, 12), (signal.SIGTERM, 4), (signal.SIGKILL, 3)):
            active = [(n, p) for n, p, _ in reversed(self.children) if group_alive(p.pid)]
            if not active and not live_descendants():
                break
            for name, proc in active:
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    pass
            signal_descendants(sig)
            end = time.monotonic() + grace
            while time.monotonic() < end:
                for _, proc, _ in self.children:
                    proc.poll()
                if not any(group_alive(p.pid) for _, p, _ in self.children) and not live_descendants():
                    break
                time.sleep(0.1)
        for _, proc, _ in self.children:
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        remaining = [(n, p.pid) for n, p, _ in self.children if group_alive(p.pid)]
        remaining.extend(('descendant', pid) for pid in live_descendants())
        if remaining:
            raise RuntimeError('清理后仍存在进程组：{}'.format(remaining))
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        say('本次启动的进程已全部退出。CAN 接口保持启用，日志已保留。')

    def show_errors(self):
        for name, _, path in self.children:
            try:
                tail = collections.deque(path.open(errors='replace'), maxlen=12)
                print('\n--- {} 最后日志 ---\n{}'.format(name, ''.join(tail)), flush=True)
            except OSError:
                pass

def existing_ros():
    conflicts = []
    tokens = {'roscore', 'rosmaster', 'roslaunch', 'nodelet', 'astra_camera_node',
              'piper_start_ms_node.py', 'piper_start_slave_node.py',
              'piper_read_master_node.py'}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
            if any(os.path.basename(a.decode(errors='replace')) in tokens for a in args):
                conflicts.append('{}: {}'.format(p.parent.name,
                    ' '.join(a.decode(errors='replace') for a in args)[:240]))
        except OSError:
            pass
    with socket.socket() as sock:
        sock.settimeout(0.3)
        if sock.connect_ex(('127.0.0.1', 11311)) == 0:
            conflicts.append('本机 11311 端口已被占用')
    return conflicts

def can_problems():
    data = json.loads(subprocess.check_output(['ip', '-j', '-details', 'link', 'show'], text=True))
    links = {entry['ifname']: entry for entry in data}
    failures = []
    for name, (bitrate, usb) in CAN.items():
        link = links.get(name, {})
        info = link.get('linkinfo', {}).get('info_data', {})
        rate = info.get('bittiming', {}).get('bitrate', info.get('bitrate'))
        if 'UP' not in link.get('flags', []) or rate != bitrate:
            failures.append('{} 未启用或波特率不是 {}'.format(name, bitrate))
            continue
        driver = subprocess.run(['ethtool', '-i', name], capture_output=True, text=True)
        if 'bus-info: ' + usb not in driver.stdout:
            failures.append('{} USB 映射不符合预设 {}'.format(name, usb))
    return failures

class TopicHealth:
    def __init__(self):
        import rospy
        from sensor_msgs.msg import Image, JointState
        self.lock = threading.Lock()
        self.data = {}
        self.subs = []
        self.started = time.monotonic()
        for topic in ('/master/joint_left', '/master/joint_right',
                      '/puppet/joint_left', '/puppet/joint_right'):
            self.add(rospy, topic, JointState, 'joint')
        for side in ('l', 'r', 'f'):
            for kind in ('color', 'depth'):
                self.add(rospy, '/camera_{}/{}/image_raw'.format(side, kind), Image, 'image')
        self.can_last = {}
        for name in ('can_left', 'can_right'):
            count = int((Path('/sys/class/net') / name / 'statistics/rx_packets').read_text())
            self.can_last[name] = (count, self.started, False)

    def add(self, rospy, topic, message_type, kind):
        self.data[topic] = {'count': 0, 'first': None, 'last': None, 'detail': ''}
        def callback(msg):
            if kind == 'joint':
                valid = len(msg.position) == 7 and all(math.isfinite(v) for v in msg.position)
                detail = '7 维关节/夹爪'
            else:
                valid = msg.width > 0 and msg.height > 0 and len(msg.data) >= msg.height * msg.step > 0
                detail = '{}x{} {}'.format(msg.width, msg.height, msg.encoding)
            if valid:
                now = time.monotonic()
                with self.lock:
                    d = self.data[topic]
                    d['count'] += 1
                    d['first'] = now if d['first'] is None else d['first']
                    d['last'] = now
                    d['detail'] = detail
        self.subs.append(rospy.Subscriber(topic, message_type, callback,
                         queue_size=1, buff_size=4 * 1024 * 1024))

    def problems(self, stale_seconds):
        now = time.monotonic()
        with self.lock:
            problems = [t for t, d in self.data.items()
                        if d['count'] < 2 or now - (d['last'] or self.started) > stale_seconds]
        for name, (old, last, seen) in list(self.can_last.items()):
            try:
                count = int((Path('/sys/class/net') / name / 'statistics/rx_packets').read_text())
                if count != old:
                    last, seen = now, True
                self.can_last[name] = (count, last, seen)
                if not seen or now - last > stale_seconds:
                    problems.append(name + ' 没有持续接收 CAN 数据')
            except OSError:
                problems.append(name + ' 接口消失')
        return problems

    def report(self):
        with self.lock:
            for topic, d in self.data.items():
                elapsed = (d['last'] or 0) - (d['first'] or 0)
                hz = (d['count'] - 1) / elapsed if elapsed > 0 else 0
                say('[通过] {} | {} | 已收 {} 条 | 观测约 {:.1f} Hz'.format(
                    topic, d['detail'], d['count'], hz))

def self_test():
    # Lifecycle regression tests use fake processes, never ROS or CAN.
    for scenario in ('ctrl_c', 'child_failure', 'startup_failure', 'orphaned_node'):
        with tempfile.TemporaryDirectory(prefix='aloha-supervisor-test-') as tmp:
            folder = Path(tmp)
            manager = Supervisor(folder)
            child_code = ('import subprocess,sys,time; '
                          'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"],start_new_session=True); '
                          'open(sys.argv[1],"w").write(str(p.pid)); time.sleep(120)')
            if scenario == 'orphaned_node':
                child_code = child_code.rsplit('time.sleep(120)', 1)[0] + 'time.sleep(.3); sys.exit(9)'
            try:
                manager.start('fake_tree', [sys.executable, '-c', child_code, str(folder/'grandchild.pid')])
                until = time.monotonic() + 5
                while not (folder/'grandchild.pid').exists():
                    if time.monotonic() > until:
                        raise RuntimeError('fake process did not start')
                    time.sleep(.05)
                if scenario == 'ctrl_c':
                    os.kill(os.getpid(), signal.SIGINT)
                    try:
                        manager.check()
                    except InterruptedError:
                        pass
                    else:
                        raise AssertionError('SIGINT was not handled')
                elif scenario in ('child_failure', 'orphaned_node'):
                    if scenario == 'child_failure':
                        manager.start('fake_failure', [sys.executable, '-c', 'raise SystemExit(7)'])
                    time.sleep(.5)
                    try:
                        manager.check()
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('child failure was not detected')
                else:
                    try:
                        manager.start('bad_start', ['/this/executable/does/not/exist'])
                    except FileNotFoundError:
                        pass
                    else:
                        raise AssertionError('startup should have failed')
            finally:
                manager.cleanup()
            assert not any(group_alive(p.pid) for _, p, _ in manager.children)
            assert not live_descendants()
            say('[自测通过] {}：父进程和孙进程均无残留'.format(scenario))
    return 0

def main():
    parser = argparse.ArgumentParser(description='前台启动 ALOHA 数采准备；Ctrl+C 统一关闭。')
    parser.add_argument('--startup-timeout', type=float, default=60, help='启动检查超时秒数，默认 60')
    parser.add_argument('--stale-timeout', type=float, default=8, help='运行中连续无消息的容忍秒数，默认 8')
    parser.add_argument('--check-only', action='store_true', help='只检查已有进程、CAN，不启动或修改设备')
    parser.add_argument('--self-test', action='store_true', help='用假进程测试 Ctrl+C/失败清理，不访问设备')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.startup_timeout <= 0 or args.stale_timeout <= 0:
        parser.error('超时时间必须大于 0')
    if args.check_only:
        conflicts = existing_ros()
        issues = can_problems()
        say('已有 ROS 会话：' + ('\n' + '\n'.join(conflicts) if conflicts else '无'))
        say('CAN 检查：' + ('；'.join(issues) if issues else '三路接口名称、速率及 USB 映射通过'))
        return 1 if conflicts or issues else 0
    log_root = ROOT / 'cobot_magic/logs/collection_prepare'
    log_root.mkdir(parents=True, exist_ok=True)
    lock = (log_root / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        say('已有一份准备脚本正在运行，请先在原终端 Ctrl+C。')
        return 1
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    logs = log_root / stamp
    logs.mkdir()
    manager = Supervisor(logs)
    rospy = None
    result = 1
    try:
        conflicts = existing_ros()
        if conflicts:
            raise RuntimeError('发现已有 ROS 会话；请先在原终端 Ctrl+C，本脚本不会接管或关闭它们：\n' + '\n'.join(conflicts))
        config_needed = bool(can_problems())
        if config_needed:
            say('CAN 需要配置，请按提示输入 sudo 密码（仅用于 CAN 接口配置）。')
            # Read password from the actual terminal, not this embedded program's stdin.
            with open('/dev/tty', 'r') as tty:
                auth = subprocess.run(['sudo', '-v'], stdin=tty)
            if auth.returncode:
                raise RuntimeError('sudo 验证失败，尚未启动硬件程序')
        os.environ['ROS_MASTER_URI'] = MASTER
        os.environ['ROS_IP'] = '127.0.0.1'
        os.environ.pop('ROS_HOSTNAME', None)
        manager.start('roscore', ['roscore', '-p', '11311'])
        import xmlrpc.client
        class Transport(xmlrpc.client.Transport):
            def make_connection(self, host):
                connection = super().make_connection(host)
                connection.timeout = 1
                return connection
        until = time.monotonic() + 20
        while True:
            manager.check()
            try:
                reply = xmlrpc.client.ServerProxy(MASTER, transport=Transport()).getPid('/collection_prepare')
                if reply[0] == 1:
                    break
            except (OSError, xmlrpc.client.Error):
                pass
            if time.monotonic() > until:
                raise RuntimeError('roscore 启动超时')
            manager.pause(.2)
        say('[通过] roscore 就绪')
        if config_needed:
            manager.check()
            config = subprocess.run(['sudo', '-n', 'bash', str(ROOT/'cobot_magic/Piper_ros_private-ros-noetic/can_config.sh')], timeout=30)
            if config.returncode:
                raise RuntimeError('CAN 配置脚本失败')
        issues = can_problems()
        if issues:
            raise RuntimeError('CAN 配置核验失败：' + '；'.join(issues))
        say('[通过] CAN 配置；机械臂模式固定为 mode=0、auto_enable=false')
        manager.start('arms', ['roslaunch', 'piper', 'start_ms_piper.launch', 'mode:=0', 'auto_enable:=false'])
        manager.start('cameras', ['roslaunch', 'astra_camera', 'multi_camera.launch'])
        import rospy as ros
        rospy = ros
        rospy.init_node('collection_prepare_monitor', anonymous=True, disable_signals=True)
        health = TopicHealth()
        until = time.monotonic() + args.startup_timeout
        next_notice = 0
        while True:
            manager.check()
            missing = health.problems(args.stale_timeout)
            if not missing:
                break
            if time.monotonic() > until:
                raise RuntimeError('启动检查超时，未就绪：' + '；'.join(missing))
            if time.monotonic() >= next_notice:
                say('等待数据：' + '；'.join(missing))
                next_notice = time.monotonic() + 5
            manager.pause(.25)
        listed = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=5)
        if listed.returncode:
            raise RuntimeError('rostopic list 失败：' + listed.stderr)
        print('\n--- rostopic list ---\n' + listed.stdout, flush=True)
        (logs/'topics.txt').write_text(listed.stdout)
        health.report()
        say('【准备成功】机械臂反馈和三路 RGB/深度均已收到。可在另一终端开始采集。')
        say('本终端将持续监控；Ctrl+C 统一关闭本次启动的 ROS/机械臂节点/相机。')
        say('完整子进程日志：' + str(logs))
        next_notice = time.monotonic() + 15
        while True:
            manager.pause(.5)
            if rospy.is_shutdown():
                raise RuntimeError('ROS 监控节点被关闭')
            missing = health.problems(args.stale_timeout)
            if missing:
                raise RuntimeError('运行中数据中断：' + '；'.join(missing))
            if time.monotonic() >= next_notice:
                say('运行正常：3 个托管进程、10 个反馈话题和两路机械臂 CAN 持续有数据。Ctrl+C 停止。')
                next_notice = time.monotonic() + 15
    except InterruptedError:
        say('收到停止信号，开始统一退出。')
        result = 0 if manager.stop_signal == signal.SIGINT else 128 + (manager.stop_signal or 15)
    except Exception as error:
        say('【失败】' + str(error))
        manager.show_errors()
    finally:
        try:
            manager.cleanup()
        except Exception as error:
            say('【清理失败】' + str(error))
            result = 1
        if rospy is not None:
            rospy.signal_shutdown('collection preparation stopped')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return result

if __name__ == '__main__':
    sys.exit(main())