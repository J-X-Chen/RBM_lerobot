import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aloha_tracking import TrackingPacer


def test_hysteresis_and_stall_timeout():
    p = TrackingPacer()
    assert not p.wait(0.07, 0)
    assert p.wait(0.08, 1)
    assert p.wait(0.06, 2)
    assert not p.wait(0.05, 2.5)
    assert p.wait(0.08, 3)
    with pytest.raises(RuntimeError, match="stalled"):
        p.wait(0.08, 5)
    p.reset()
    assert not p.wait(0.01, 6)
