"""Actual H1 evaluation loop over mock TCP/ZMQ, controlled through a pseudo-terminal."""
import json
import os
from pathlib import Path
import pty
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from zerith_lerobot.eval_robot import eval_h1
from zerith_lerobot.eval_robot import h1_runtime
from zerith_image_service.contract import limits_from_urdf, default_urdf
from zerith_image_service.control import H1Control
from zerith_image_service.sdk import MockSDK
from zerith_image_service.server import ActionServer
from zerith_image_service.cameras import CameraPublisher


class LoopTests(unittest.TestCase):
    def run_loop(self, asynchronous):
        import torch
        stop = threading.Event()
        sdk = MockSDK()
        pose = sdk.state.copy()
        control = H1Control(sdk, limits_from_urdf(default_urdf(eval_h1.ROOT / 'zerith_H1_SDK1.3.6')), execute=True)
        server = ActionServer(('127.0.0.1',0),control,stop)
        workers = [threading.Thread(target=server.serve_forever,daemon=True),
                   threading.Thread(target=control.run,args=(stop,),daemon=True)]
        reserve = [socket.socket() for _ in range(3)]
        for sock in reserve: sock.bind(('127.0.0.1',0))
        ports = [sock.getsockname()[1] for sock in reserve]
        for sock in reserve: sock.close()
        publisher = CameraPublisher(None, ['a','b','c'],'127.0.0.1',ports,stop,mock=True)
        workers.append(publisher)
        for worker in workers: worker.start()
        self.assertTrue(publisher.ready.wait(3))
        master, slave = pty.openpty()
        terminal = os.fdopen(slave, 'r')
        errors = []
        class FakePolicy:
            n_action_steps = 4
            def __init__(self, *args): pass
            def predict(self, observation, reset=False):
                time.sleep(.08)
                return np.repeat(pose[None,:19],10,axis=0)
        try:
            with tempfile.TemporaryDirectory() as folder:
                log = Path(folder) / 'eval.jsonl'
                argv = ['eval_h1','--mode=live','--execute','--checkpoint=unused','--device=cpu',
                        '--host=127.0.0.1',f'--control-port={server.server_address[1]}',
                        '--camera-ports', *map(str,ports),f'--log={log}']
                if asynchronous: argv.append('--async-prefetch')
                def events():
                    if not log.exists(): return []
                    return [json.loads(line) for line in log.read_text().splitlines() if line]
                def wait_for(condition):
                    end = time.monotonic()+5
                    while time.monotonic() < end:
                        if errors: raise errors[0]
                        if condition(): return
                        time.sleep(.01)
                    self.fail(f'Loop condition timeout: {events()}')
                def run():
                    try: eval_h1.main()
                    except BaseException as exc: errors.append(exc)
                with patch.object(sys,'argv',argv), patch.object(sys,'stdin',terminal), \
                     patch.object(h1_runtime,'H1Policy',FakePolicy), \
                     patch.object(h1_runtime,'load_start',return_value={'observation.state':torch.tensor(pose),'task':'hold'}):
                    thread = threading.Thread(target=run,daemon=True)
                    thread.start()
                    wait_for(lambda: log.exists())
                    time.sleep(.05)
                    os.write(master,b'r')
                    wait_for(lambda: control.owner is not None)
                    time.sleep(.1)
                    os.write(master,b's')
                    wait_for(lambda: sum(e.get('event')=='action' for e in events()) >= 3)
                    os.write(master,b'p')
                    wait_for(lambda: any(e.get('key')=='p' for e in events()))
                    count = sum(e.get('event')=='action' for e in events())
                    time.sleep(.2)
                    self.assertEqual(count,sum(e.get('event')=='action' for e in events()))
                    os.write(master,b'b')
                    wait_for(lambda: any(e.get('key')=='b' for e in events()))
                    time.sleep(.1)
                    os.write(master,b's')
                    wait_for(lambda: sum(e.get('event')=='action' for e in events()) > count)
                    os.write(master,b'q')
                    thread.join(3)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors,[])
                    self.assertIsNotNone(control.fault)
        finally:
            stop.set()
            terminal.close(); os.close(master)
            server.shutdown(); server.server_close()
            for worker in workers: worker.join(2)

    def test_sync(self): self.run_loop(False)
    def test_async(self): self.run_loop(True)


if __name__ == '__main__': unittest.main()
