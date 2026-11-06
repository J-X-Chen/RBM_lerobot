import json
import unittest
from unittest.mock import patch

from can_setup import configure_can


class SetupTests(unittest.TestCase):
    hardware = {"left": {"can": "can_left"}, "right": {"can": "can_right"}}

    def test_healthy_interfaces_are_untouched(self):
        links = [{"ifname": name, "flags": ["UP"], "linkinfo": {
            "info_data": {"bittiming": {"bitrate": 1000000}}}}
            for name in ("can_left", "can_right")]
        with patch("can_setup.subprocess.check_output", return_value=json.dumps(links)), patch("can_setup.subprocess.run") as run:
            configure_can(self.hardware)
            run.assert_not_called()

    def test_recover_by_usb_without_touching_third_interface(self):
        links = [{"ifname": name, "flags": []} for name in ("can0", "can1", "can2")]

        def read(command, **kwargs):
            if command[0] == "ip":
                return json.dumps(links)
            return "bus-info: " + {"can0": "1-4:1.0", "can1": "1-12:1.0", "can2": "1-13:1.0"}[command[-1]]

        with patch("can_setup.subprocess.check_output", side_effect=read), patch("can_setup.subprocess.run") as run:
            configure_can(self.hardware)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(["sudo", "ip", "link", "set", "dev", "can2", "name", "can_left"], commands)
            self.assertIn(["sudo", "ip", "link", "set", "dev", "can1", "name", "can_right"], commands)
            self.assertFalse(any("can0" in command for command in commands))

    def test_missing_usb_fails_before_any_changes(self):
        with patch("can_setup.subprocess.check_output", return_value="[]"), patch("can_setup.subprocess.run") as run:
            with self.assertRaises(ValueError):
                configure_can(self.hardware)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
