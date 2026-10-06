#!/usr/bin/env python

import pytest
import torch

from lerobot.datasets import video_utils


def test_decode_video_frames_torchcodec_falls_back_to_pyav(monkeypatch):
    def fail_torchcodec(*args, **kwargs):
        raise RuntimeError("invalid packet")

    def decode_pyav(video_path, timestamps, tolerance_s, *, return_uint8=False, is_depth=False):
        assert video_path == "bad.mp4"
        assert timestamps == [0.0]
        assert tolerance_s == 1e-4
        assert return_uint8
        assert not is_depth
        return torch.zeros(1, 3, 4, 4, dtype=torch.uint8)

    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", fail_torchcodec)
    monkeypatch.setattr(video_utils, "decode_video_frames_pyav", decode_pyav)

    frames = video_utils.decode_video_frames(
        "bad.mp4",
        [0.0],
        1e-4,
        backend="torchcodec",
        return_uint8=True,
    )

    assert frames.shape == (1, 3, 4, 4)
    assert frames.dtype == torch.uint8


def test_decode_video_frames_reports_path_when_fallback_also_fails(monkeypatch):
    monkeypatch.setattr(
        video_utils,
        "decode_video_frames_torchcodec",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invalid packet")),
    )
    monkeypatch.setattr(
        video_utils,
        "decode_video_frames_pyav",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pyav failed")),
    )

    with pytest.raises(RuntimeError, match="video: bad.mp4"):
        video_utils.decode_video_frames(
            "bad.mp4",
            [0.0],
            1e-4,
            backend="torchcodec",
        )
