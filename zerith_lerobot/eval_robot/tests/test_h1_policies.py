"""Small real PI05/DM05 checkpoint -> H1 evaluation runtime tests, no native SDK."""
import numpy as np
import pytest
import torch
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import apply_robot_dataset_views
from lerobot.policies.dm05.configuration_dm05 import DM05Config
from tests.policies.dm05.test_dm05_integration import _save_tiny_real_processor, _tiny_core_config
from tests.policies.dm05.test_zerith_dm05 import _record_dataset
from tests.policies.test_cobot_magic_training import tiny_pi05
from zerith_lerobot.eval_robot.h1_runtime import H1Policy, live_observation


@pytest.mark.parametrize('kind', ['pi05', 'dm05'])
def test_h1_runtime_checkpoint(tmp_path, monkeypatch, kind):
    processor = tmp_path / 'processor'
    _save_tiny_real_processor(processor)
    if kind == 'pi05':
        cfg = tiny_pi05(monkeypatch, processor)
    else:
        core = _tiny_core_config()
        core.action_dim = 32
        cfg = DM05Config(core_config=core.to_dict(), processor_name_or_path=str(processor), device='cpu',
                         dtype=torch.float32, chunk_size=2, n_action_steps=2, diffusion_steps=1,
                         vlm_gradient_checkpointing=False, ae_gradient_checkpointing=False)
    root = tmp_path / 'data'
    _record_dataset(root, 23, 'zerith_h1_pro')
    train = TrainPipelineConfig(dataset=DatasetConfig(repo_id='local/test',root=str(root),video_backend='pyav',zerith_action_view='auto'),policy=cfg)
    ds, _ = apply_robot_dataset_views(train, make_dataset(train))
    policy = make_policy(cfg, ds_meta=ds.meta)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=ds.meta.stats)
    sample = ds[0]
    # Initialize tokenizer/stateful processor assets before saving.
    pre(sample)
    checkpoint = tmp_path / 'checkpoint'
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename='policy_preprocessor.json')
    post.save_pretrained(checkpoint, config_filename='policy_postprocessor.json')
    ds.layout.save(checkpoint / 'zerith_action_layout.json')
    runtime = H1Policy(checkpoint, 'cpu')
    frames = {key: np.full((16,24,3), [12,34,56], dtype=np.uint8)
              for key in ('cam_high','cam_left_wrist','cam_right_wrist')}
    obs = live_observation(np.zeros(23), frames, 'pick')
    np.testing.assert_allclose(obs['observation.images.cam_high'][:,0,0], np.array([56,34,12])/255, atol=1e-7)
    actions = runtime.predict(obs, reset=True)
    assert actions.shape == (2,19) and np.isfinite(actions).all()
    assert runtime.layout.expand(actions).shape == (2,23)
