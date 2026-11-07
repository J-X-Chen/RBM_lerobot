import numpy as np
import pytest
import torch
from cloak_mask.runtime import RobotMask


class Provider:
    def render(self, state, cameras):
        return {c: np.array([[[0, 1], [1, 0]]]) for c in cameras}
    def close(self):
        pass


@pytest.mark.parametrize('layout', ['CHW', 'HWC'])
@pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
def test_fill_preserves_background_and_input(layout, dtype):
    mask = RobotMask('aloha', assets_root='/tmp', fill_rgb=[17, 128, 255])
    mask.provider = Provider()
    shape = (3, 2, 2) if layout == 'CHW' else (2, 2, 3)
    source = torch.zeros(shape, dtype=dtype)
    obs = {'observation.images.head': source}
    try:
        result = mask.apply(obs, np.zeros(14), layout=layout)['observation.images.head']
        hwc = result.permute(1, 2, 0) if layout == 'CHW' else result
        color = torch.tensor([17, 128, 255], dtype=dtype)
        if dtype == torch.float32:
            color = color / 255
        assert torch.equal(hwc[0, 1], color)
        assert not hwc[0, 0].any()
        assert not source.any()
    finally:
        mask.close()


@pytest.mark.parametrize('color', [[-1, 0, 0], [0, 256, 0], [0, 0], [0., .5, 1.]])
def test_invalid_color(color):
    with pytest.raises(ValueError, match='three integers'):
        RobotMask('aloha', assets_root='/tmp', fill_rgb=color)


def test_resolution_mismatch():
    mask = RobotMask('aloha', assets_root='/tmp')
    mask.provider = Provider()
    try:
        with pytest.raises(ValueError, match='resolution mismatch'):
            mask.apply({'observation.images.head': torch.zeros(3, 4, 4)}, np.zeros(14), layout='CHW')
    finally:
        mask.close()
