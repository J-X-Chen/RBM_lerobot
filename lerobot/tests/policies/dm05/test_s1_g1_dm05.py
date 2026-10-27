"""Real miniature DM05 training/checkpoint round trips for the S1/G1 contracts."""

import ast
import logging
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dm05.configuration_dm05 import DM05Config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import apply_robot_dataset_views
from lerobot.utils.astribot_action import expand_astribot_s1_arm_gripper
from lerobot.utils.dex3_action import expand_dex3_action
from tests.policies.dm05.test_dm05_integration import _save_tiny_real_processor, _tiny_core_config

PROJECT = Path(__file__).resolve().parents[4]
CAMERAS = [f"observation.images.{camera}" for camera in ("head", "left", "right")]


def _record_dataset(root, dim, robot):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": [f"joint_{i}" for i in range(dim)],
        },
        "action": {"dtype": "float32", "shape": (dim,), "names": [f"joint_{i}" for i in range(dim)]},
        **{
            key: {"dtype": "image", "shape": (16, 24, 3), "names": ["height", "width", "channels"]}
            for key in CAMERAS
        },
    }
    ds = LeRobotDataset.create(
        "local/test", fps=10, features=features, root=root, robot_type=robot, use_videos=False
    )
    for frame in range(4):
        values = np.arange(dim, dtype=np.float32) * 0.01 + frame * 0.001
        ds.add_frame(
            {
                "observation.state": values,
                "action": values.copy(),
                "task": "pick",
                **{key: np.full((16, 24, 3), frame * 20, dtype=np.uint8) for key in CAMERAS},
            }
        )
    ds.save_episode()
    ds.finalize()


@pytest.mark.parametrize(
    ("robot", "dim", "view", "dex3", "state_dim", "action_dim"),
    [
        ("astribot_s1", 25, "full", "auto", 25, 25),
        ("astribot_s1", 25, "arms_grippers", "auto", 16, 16),
        ("unitree_g1_dex3", 28, "full", "auto", 28, 16),
        ("unitree_g1_dex3", 28, "full", False, 28, 28),
        ("unitree_g1_dex3", 28, "full", "ignore", 28, 16),
    ],
)
def test_dm05_robot_training_and_eval_roundtrip(tmp_path, robot, dim, view, dex3, state_dim, action_dim):
    root = tmp_path / "data"
    _record_dataset(root, dim, robot)
    processor_path = tmp_path / "processor"
    _save_tiny_real_processor(processor_path)
    core = _tiny_core_config()
    core.action_dim = 32
    config = DM05Config(
        core_config=core.to_dict(),
        processor_name_or_path=str(processor_path),
        device="cpu",
        dtype=torch.float32,
        chunk_size=2,
        n_action_steps=2,
        diffusion_steps=1,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        freeze_vlm_embedding=False,
        llm_attn_implementation="eager",
        vision_attn_implementation="eager",
        action_attn_implementation="eager",
    )
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/test",
            root=str(root),
            video_backend="pyav",
            use_imagenet_stats=False,
            astribot_action_view=view,
            compress_dex3_actions=dex3,
        ),
        policy=config,
    )
    raw = make_dataset(train_cfg)
    dataset, held_out = apply_robot_dataset_views(train_cfg, raw, make_dataset(train_cfg))
    assert held_out.meta.features["action"]["shape"] == dataset.meta.features["action"]["shape"]
    policy = make_policy(config, ds_meta=dataset.meta)
    assert tuple(policy.config.input_features["observation.state"].shape) == (state_dim,)
    assert tuple(policy.config.output_features["action"].shape) == (action_dim,)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    processed = pre(batch)
    loss, _ = policy(processed)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    optimizer.step()
    policy.eval()
    noise = torch.zeros(2, 2, 32)
    with torch.no_grad():
        expected = policy.predict_action_chunk(processed, noise=noise)
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    saved = PreTrainedConfig.from_pretrained(checkpoint)
    saved.pretrained_path = checkpoint
    # eval_robot reads the full-body reference recording, not the transformed training view.
    reloaded = make_policy(saved, ds_meta=raw.meta, preserve_checkpoint_features=True)
    assert tuple(reloaded.config.input_features["observation.state"].shape) == (state_dim,)
    assert tuple(reloaded.config.output_features["action"].shape) == (action_dim,)
    eval_pre, eval_post = make_pre_post_processors(reloaded.config, pretrained_path=checkpoint)
    with torch.no_grad():
        actual = reloaded.predict_action_chunk(eval_pre(batch), noise=noise)
        decoded = eval_post(actual)
    torch.testing.assert_close(actual, expected)
    assert decoded.shape == (2, 2, action_dim) and torch.isfinite(decoded).all()
    # Run the actual synchronous and asynchronous client inference functions without SDK imports.
    client_robot = "astribot" if robot == "astribot_s1" else "unitree"
    observation = {key: value[0] for key, value in batch.items() if key.startswith("observation.")}
    sync = _load_eval_functions(client_robot)
    action = sync["predict_action"](
        observation, reloaded, torch.device("cpu"), eval_pre, eval_post, False, task="pick", use_dataset=True
    )
    assert action.shape == (action_dim,) and torch.isfinite(action).all()
    async_functions = _load_eval_functions(client_robot, asynchronous=True)
    actions = async_functions["predict_action_chunk_async"](
        observation,
        reloaded,
        torch.device("cpu"),
        eval_pre,
        eval_post,
        False,
        task="pick",
        actions_per_chunk=2,
        use_dataset=True,
    )
    assert len(actions) == 2 and all(action.shape == (action_dim,) for action in actions)
    if robot == "astribot_s1" and action_dim == 16:
        assert expand_astribot_s1_arm_gripper(decoded, torch.zeros(2, 2, 25)).shape[-1] == 25
    if robot == "unitree_g1_dex3" and action_dim == 16:
        assert expand_dex3_action(decoded).shape[-1] == 28


def _load_eval_functions(robot, asynchronous=False):
    """Exercise the real client functions without importing robot SDKs or opening hardware."""
    stem = "s1" if robot == "astribot" else "g1"
    path = (
        PROJECT
        / f"{robot}_lerobot/eval_robot"
        / (f"eval_{stem}_async.py" if asynchronous else "utils/utils.py")
    )
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {
        "_uses_relative_action_chunks",
        "predict_action",
        "clear_cached_eval_actions",
        "_feature_dim",
        "_postprocessor_action_dim",
        "_maybe_trim_action_for_postprocessor",
        "_maybe_trim_dex3_action_for_postprocessor",
        "prepare_observation_for_policy",
        "_normalise_action_chunk",
        "_select_action_chunk_fallback",
        "predict_action_chunk_async",
    }
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "ACTION": "action",
        "_predict_chunk_fallback_warned": False,
        "torch": torch,
        "np": np,
        "copy": copy,
        "nullcontext": nullcontext,
        "logger_mp": logging.getLogger(__name__),
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("robot", ["astribot", "unitree"])
def test_eval_relative_dm05_chunk_keeps_generation_anchor(robot):
    functions = _load_eval_functions(robot)
    policy = SimpleNamespace(config=SimpleNamespace(type="dm05", use_relative_actions=True, n_action_steps=2))
    calls = []

    def predict(batch):
        calls.append(batch["observation.state"].clone())
        return torch.ones(1, 2, 16)

    policy.predict_action_chunk = predict
    anchor = {}

    def pre(batch):
        anchor["state"] = batch["observation.state"]
        return batch

    def post(actions):
        return actions + anchor["state"][:, None]

    for value in (10.0, 100.0):
        action = functions["predict_action"](
            {"observation.state": torch.full((16,), value)},
            policy,
            torch.device("cpu"),
            pre,
            post,
            False,
            use_dataset=False,
        )
        torch.testing.assert_close(action, torch.full((16,), 11.0))
    assert len(calls) == 1
    functions["clear_cached_eval_actions"](policy)
    assert not getattr(policy, "_eval_decoded_action_queue", [])
    assert not getattr(policy, "_unitree_eval_decoded_action_queue", [])


@pytest.mark.parametrize(("robot", "stem"), [("astribot", "s1"), ("unitree", "g1")])
@pytest.mark.parametrize("suffix", ["", "_async"])
def test_robot_entrypoints_prioritize_shared_library(robot, stem, suffix):
    path = PROJECT / f"{robot}_lerobot/eval_robot/eval_{stem}{suffix}.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in {"_PROJECT_ROOT", "_LEROBOT_SRC"} for t in node.targets
            )
        )
        or (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_import_root"
        )
    ]
    fake_sys = SimpleNamespace(path=[str(PROJECT / f"{robot}_lerobot/lerobot/src")])
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
        {"__file__": str(path), "Path": Path, "sys": fake_sys},
    )
    assert fake_sys.path[0] == str(PROJECT / "lerobot/src")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "make_policy"
    ]
    assert calls and all(
        any(k.arg == "preserve_checkpoint_features" for k in call.keywords) for call in calls
    )
