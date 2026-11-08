"""Loopback-only tests of the actual TCP and ZMQ paths, always with MockSDK."""
import socket
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from zerith_image_service.contract import ControlClient, default_urdf, limits_from_urdf
from zerith_image_service.control import H1Control
from zerith_image_service.sdk import MockSDK
from zerith_image_service.server import ActionServer
from zerith_image_service.cameras import CameraPublisher, CameraSubscriber


class TransportTests(unittest.TestCase):
    def test_mock_tcp_camera_and_disconnect(self):
        root = Path(__file__).resolve().parents[2]
        sdk = MockSDK()
        stop = threading.Event()
        control = H1Control(sdk, limits_from_urdf(default_urdf(root / 'zerith_H1_SDK1.3.6')), execute=True)
        server = ActionServer(('127.0.0.1', 0), control, stop)
        network = threading.Thread(target=server.serve_forever, daemon=True)
        worker = threading.Thread(target=control.run, args=(stop,), daemon=True)
        reservations = [socket.socket() for _ in range(3)]
        for sock in reservations:
            sock.bind(('127.0.0.1', 0))
        ports = [sock.getsockname()[1] for sock in reservations]
        for sock in reservations:
            sock.close()
        publisher = CameraPublisher(None, ['a','b','c'], '127.0.0.1', ports, stop, mock=True)
        client = cameras = spectator = None
        network.start(); worker.start(); publisher.start()
        try:
            self.assertTrue(publisher.ready.wait(3))
            self.assertIsNone(publisher.error)
            client = ControlClient('127.0.0.1', server.server_address[1])
            cameras = CameraSubscriber('127.0.0.1', ports, max_age=.15)
            frames = cameras.read(timeout=3, server_time=client.request(get_state=True)["control"]["server_time"])
            self.assertEqual(set(frames), {'cam_high','cam_left_wrist','cam_right_wrist'})
            self.assertTrue(all(f.shape == (480,640,3) for f in frames.values()))
            self.assertEqual(sdk.calls, [])
            time.sleep(2.1)  # Read-only clients survive long model inference.
            state = np.array(client.request(get_state=True)['state'])
            client.request(arm=True)
            state[0] = .01
            client.request(action=state[:19].tolist())
            spectator = ControlClient('127.0.0.1', server.server_address[1])
            spectator.request(get_state=True)
            spectator.close(); spectator = None
            time.sleep(.08)
            reply = client.request(get_state=True)
            self.assertIsNone(reply['control']['fault'])
            self.assertGreater(reply['state'][0], 0)
            client.close(); client = None
            time.sleep(.05)
            self.assertIn('disconnected', control.fault)
            count = len(sdk.calls)
            time.sleep(.04)
            self.assertEqual(len(sdk.calls), count)
            stop.set()
            publisher.join(2)
            time.sleep(.2)
            # Drain any final frames once, then require the cache to expire.
            try:
                cameras.read(timeout=.01)
            except TimeoutError:
                pass
            time.sleep(.2)
            with self.assertRaises(TimeoutError):
                cameras.read(timeout=.05)
        finally:
            stop.set()
            if client: client.close()
            if spectator: spectator.close()
            if cameras: cameras.close()
            server.shutdown(); server.server_close()
            worker.join(2); publisher.join(2); network.join(2)


if __name__ == '__main__':
    unittest.main()
