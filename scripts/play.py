from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

try:
    from crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig
    from train import build_train_cfg
except ImportError:
    from .crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig
    from .train import build_train_cfg


def latest_checkpoint(log_root: str) -> Path:
    candidates = sorted(Path(log_root).glob("**/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoints found under {log_root}.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained CrankBot walking policy.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to latest under --log-root.")
    parser.add_argument("--log-root", default="logs/crankbot_walk")
    parser.add_argument("--xml", default="scene.xml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Run exactly this many control steps. Overrides --episodes when set.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of completed episodes to play.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep env.cfg.control_dt seconds after each action.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Custom sleep in seconds after each action. Overrides --realtime.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--num-steps-per-env",
        type=int,
        default=80,
        help="rsl_rl rollout length used to rebuild the runner config; it does not control playback length.",
    )
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--logger", choices=("wandb", "tensorboard", "none"), default="none")
    parser.add_argument("--wandb-project", default="crankbot-walk")
    parser.add_argument("--run-name", default="play")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.log_root)
    if args.sleep is not None and args.sleep < 0.0:
        raise ValueError("--sleep must be non-negative.")
    if args.steps is not None and args.steps < 0:
        raise ValueError("--steps must be non-negative.")
    if args.episodes < 0:
        raise ValueError("--episodes must be non-negative.")
    sleep_s = args.sleep

    env_cfg = CrankBotWalkEnvConfig(xml_path=args.xml, num_envs=args.num_envs, seed=args.seed)
    env = CrankBotWalkEnv(env_cfg, device="cpu")
    if args.realtime and args.sleep is None:
        sleep_s = env.cfg.control_dt

    runner = OnPolicyRunner(env, build_train_cfg(args), log_dir=None, device=args.device)
    runner.load(str(checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data[0])

    obs = env.get_observations().to(args.device)
    try:
        completed_episodes = 0
        steps = 0
        while (args.steps is not None and steps < args.steps) or (
            args.steps is None and completed_episodes < args.episodes
        ):
            with torch.inference_mode():
                actions = policy(obs).to(env.device)
            obs, _, dones, _ = env.step(actions)
            obs = obs.to(args.device)
            steps += 1
            completed_episodes += int(dones.sum().item())

            if viewer is not None:
                viewer.sync()
            if sleep_s is not None:
                time.sleep(sleep_s)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
