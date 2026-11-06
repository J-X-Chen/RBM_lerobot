"""Port routing checks without ROS, CAN, or network sockets."""

import contextlib
import io
import threading
import unittest
from types import SimpleNamespace

from server import Bridge, build_parser, camera_routes, validate_ports


class PortTests(unittest.TestCase):
    def test_custom_ports_reach_publisher_bind(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--head-port=6100", "--left-wrist-port=6101", "--right-wrist-port=6102", "--control-port=6201"]
        )
        validate_ports(parser, args)
        bound, closed = [], []
        bridge = Bridge.__new__(Bridge)
        bridge.args = args
        bridge.robot = None
        bridge.stop = threading.Event()
        bridge.stop.set()  # initialize/close sockets only; never enter image loop
        bridge.context = SimpleNamespace(
            socket=lambda *a: SimpleNamespace(
                setsockopt=lambda *a: None, bind=bound.append, close=lambda: closed.append(True)
            )
        )
        bridge.loop()
        self.assertEqual(bound, ["tcp://127.0.0.1:6100", "tcp://127.0.0.1:6101", "tcp://127.0.0.1:6102"])
        self.assertEqual(len(closed), 3)

    def test_defaults_and_invalid_ports(self):
        parser = build_parser()
        self.assertEqual([p for _, p in camera_routes(parser.parse_args([]))], [55555, 55556, 55557])
        for flags in [["--head-port=6200"], ["--head-port=0"], ["--right-wrist-port=65536"]]:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                validate_ports(parser, parser.parse_args(flags))


if __name__ == "__main__":
    unittest.main()
