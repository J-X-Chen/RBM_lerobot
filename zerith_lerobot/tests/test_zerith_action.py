"""Run with: python -m unittest discover -s zerith_lerobot/tests -v."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lerobot/src'))
from lerobot.utils.zerith_action import NAMES, ZerithActionLayout, prepare_zerith_datasets


class Table:
    def __init__(self, actions):
        self.data = {'action': actions, 'observation.state': actions + 0.1}

    def with_format(self, format):
        return self.data


class Dataset:
    def __init__(self, actions):
        self.hf_dataset = Table(actions)
        features = {key: {'shape': [23], 'names': NAMES.copy()} for key in self.hf_dataset.data}
        self.meta = SimpleNamespace(robot_type='zerith_h1_pro', features=features,
                                    info=SimpleNamespace(features=features), stats={})

    def __getitem__(self, i):
        return {key: torch.tensor(value[i]) for key, value in self.hf_dataset.data.items()}

    def __len__(self):
        return len(self.hf_dataset.data['action'])


class ZerithTests(unittest.TestCase):
    def setUp(self):
        self.actions = np.zeros((4, 23), dtype=np.float32)
        self.actions[:, :7] = np.arange(4)[:, None]
        self.actions[:, 8:19] = np.arange(4)[:, None]
        self.actions[:, 19] = 0.25  # Nonzero constants must also survive reconstruction.

    def test_roundtrip_numpy_and_torch_chunks(self):
        layout = ZerithActionLayout.fit(self.actions)
        self.assertEqual(len(layout.indices), 19)
        for value in (self.actions[None], torch.tensor(self.actions[None])):
            np.testing.assert_array_equal(layout.expand(layout.select(value)), value)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'layout.json'
            layout.save(p)
            self.assertEqual(layout, ZerithActionLayout.load(p))

    def test_metadata_stats_state_and_source_preserved(self):
        raw = Dataset(self.actions)
        view, _, layout = prepare_zerith_datasets(raw)
        self.assertEqual(raw.meta.features['action']['shape'], [23])
        self.assertEqual(view.meta.features['action']['shape'], [19])
        self.assertEqual(view[0]['observation.state'].shape, (23,))
        self.assertEqual(view[0]['action'].shape, (19,))
        np.testing.assert_allclose(view.meta.stats['action']['q99'],
                                   np.quantile(self.actions[:, layout.indices], .99, axis=0))

    def test_eval_uses_training_stats_and_checks_constants(self):
        evaluation = self.actions.copy()
        evaluation[:, 0] += 10
        train, val, _ = prepare_zerith_datasets(Dataset(self.actions), Dataset(evaluation))
        np.testing.assert_array_equal(train.meta.stats['action']['mean'], val.meta.stats['action']['mean'])
        evaluation[0, 20] = 1
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            prepare_zerith_datasets(Dataset(self.actions), Dataset(evaluation))

    def test_constant_left_gripper_is_retained(self):
        layout = ZerithActionLayout.fit(self.actions)
        self.assertIn(7, layout.indices)
        self.assertEqual(layout.indices, list(range(19)))
        evaluation = self.actions.copy()
        evaluation[0, 7] = 0.5
        _, val, _ = prepare_zerith_datasets(Dataset(self.actions), Dataset(evaluation))
        self.assertEqual(val[0]['action'][7].item(), 0.5)

    def test_invalid_data(self):
        for bad in (np.zeros((4, 23)), np.full((4, 23), np.nan), np.zeros((4, 25))):
            with self.assertRaises(ValueError):
                ZerithActionLayout.fit(bad)

    def test_tiny_variation_is_not_discarded(self):
        self.actions[0, 22] = 1e-9
        self.assertIn(22, ZerithActionLayout.fit(self.actions).indices)


if __name__ == '__main__':
    unittest.main()
