from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig
except ImportError:
    from .crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig


def _load_mjx_train_module() -> Any:
    module_path = Path(__file__).with_name("train.mjx.py")
    spec = importlib.util.spec_from_file_location("train_mjx", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def latest_checkpoint(log_root: str) -> Path:
    root = Path(log_root)
    candidates = sorted(
        (path for path in root.glob("**/checkpoints/*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No Brax checkpoint directories found under {root}.")
    return candidates[-1]


def resolve_checkpoint(path: str | None, log_root: str) -> Path:
    if path is None:
        return latest_checkpoint(log_root)

    checkpoint = Path(path)
    if (checkpoint / "ppo_network_config.json").exists():
        return checkpoint.resolve()

    child_candidates = sorted(
        (child for child in checkpoint.glob("*") if child.is_dir() and child.name.isdigit()),
        key=lambda child: (child.stat().st_mtime, child.name),
    )
    if child_candidates:
        return child_candidates[-1].resolve()

    nested_candidates = sorted(
        (child for child in checkpoint.glob("**/checkpoints/*") if child.is_dir() and child.name.isdigit()),
        key=lambda child: (child.stat().st_mtime, child.name),
    )
    if nested_candidates:
        return nested_candidates[-1].resolve()

    raise FileNotFoundError(f"Could not find a Brax checkpoint under {checkpoint}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a Brax/MJX PPO checkpoint in the CPU MuJoCo viewer.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint step dir, checkpoints dir, or run dir.")
    parser.add_argument("--log-root", default="logs/crankbot_walk_mjx")
    parser.add_argument("--xml", default="scene.xml", help="MuJoCo XML to visualize.")
    parser.add_argument("--device", default="cpu", help="JAX inference platform: cpu or gpu.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_policy(checkpoint: Path, deterministic: bool) -> Any:
    from flax import linen
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    params = ppo_checkpoint.load(checkpoint)
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size={"state": (CrankBotWalkEnv.num_obs,), "privileged_state": (CrankBotWalkEnv.num_privileged_obs,)},
        action_size=CrankBotWalkEnv.num_actions,
        policy_hidden_layer_sizes=(256, 256, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        activation=linen.elu,
        policy_obs_key="state",
        value_obs_key="privileged_state",
        init_noise_std=0.5,
    )
    return ppo_networks.make_inference_fn(ppo_network)(params, deterministic=deterministic)


def main() -> None:
    args = parse_args()
    train_mjx = _load_mjx_train_module()
    train_mjx.configure_jax_platform(args.device)

    import jax
    from jax import numpy as jp

    checkpoint = resolve_checkpoint(args.checkpoint, args.log_root)
    policy = load_policy(checkpoint, args.deterministic)

    env_cfg = CrankBotWalkEnvConfig(xml_path=args.xml, num_envs=args.num_envs, seed=args.seed)
    env = CrankBotWalkEnv(env_cfg, device="cpu")

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data[0])

    rng = jax.random.PRNGKey(args.seed)
    obs = env.get_observations()
    print(f"Loaded Brax checkpoint: {checkpoint}")
    try:
        for _ in range(args.steps):
            actor_obs = np.asarray(obs["actor"], dtype=np.float32)
            critic_obs = np.asarray(obs["critic"], dtype=np.float32)
            rng, action_rng = jax.random.split(rng)
            action, _ = policy(
                {
                    "state": jp.asarray(actor_obs),
                    "privileged_state": jp.asarray(critic_obs),
                },
                action_rng,
            )
            actions = torch.as_tensor(np.asarray(action).copy(), dtype=torch.float32, device=env.device)
            obs, _, _, _ = env.step(actions)

            if viewer is not None:
                viewer.sync()
            if args.realtime:
                time.sleep(env.cfg.control_dt)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
