"""CAN startup decisions, using recorded frame classes and fake listeners only."""

import unittest
from passive_can import REQUIRED_FEEDBACK, check_feedback


def link(state):
    return {"flags": ["UP"], "linkinfo": {"info_data": {"state": state, "bittiming": {"bitrate": 1000000}}}}


def good(name, seconds):
    return {"valid_frames": {key: 200 for key in REQUIRED_FEEDBACK}, "errors": {}}


class PreflightTests(unittest.TestCase):
    def test_fresh_feedback_passes_even_with_passive_or_warning_label(self):
        for state in ("ERROR-ACTIVE", "ERROR-PASSIVE", "ERROR-WARNING"):
            with self.subTest(state=state):
                self.assertTrue(check_feedback("can_left", link(state), listener=good)["valid_frames"])

    def test_active_errors_are_not_ignored_when_feedback_arrives(self):
        def errors(name, seconds):
            report = good(name, seconds)
            report["errors"] = {"no_ack": 10}
            return report

        with self.assertRaisesRegex(ValueError, "current CAN errors"):
            check_feedback("can_left", link("ERROR-PASSIVE"), listener=errors)

    def test_no_feedback_or_master_targets_are_not_sufficient(self):
        for frames in (
            {},
            {"0x155": 200, "0x156": 200, "0x157": 200},
            {key: 1 for key in REQUIRED_FEEDBACK},
            {"0x2a1": 200},
        ):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "missing repeated"):
                check_feedback(
                    "can_left", link("ERROR-ACTIVE"), listener=lambda *args: {"valid_frames": frames, "errors": {}}
                )

    def test_bus_off_still_blocks_before_listening(self):
        def unexpected(*args):
            raise AssertionError("must not listen to bus-off interface")

        with self.assertRaisesRegex(ValueError, "cannot communicate"):
            check_feedback("can_left", link("BUS-OFF"), listener=unexpected)

    def test_wrong_bitrate_and_down_still_block(self):
        for data in ({}, {"flags": ["UP"], "linkinfo": {"info_data": {"bittiming": {"bitrate": 500000}}}}):
            with self.assertRaisesRegex(ValueError, "1000000"):
                check_feedback("can_left", data, listener=good)


if __name__ == "__main__":
    unittest.main()
