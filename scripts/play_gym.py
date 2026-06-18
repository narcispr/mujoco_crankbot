from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import SAC

try:
    from crankbot_walk_env import CrankBotWalkEnvConfig
    from crankbot_walk_gym_env import CrankBotWalkGymEnv
except ImportError:
    from .crankbot_walk_env import CrankBotWalkEnvConfig
    from .crankbot_walk_gym_env import CrankBotWalkGymEnv


def latest_model(log_root: str) -> Path:
    root = Path(log_root)
    candidates = sorted(
        list(root.glob("**/final_model.zip")) + list(root.glob("**/sac_crankbot_walk_*_steps.zip")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No SB3 .zip models found under {root}.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a Gymnasium/SB3 SAC CrankBot walking policy.")
    parser.add_argument("--model", default=None, help="SB3 .zip model path. Defaults to latest under --log-root.")
    parser.add_argument("--log-root", default="logs/crankbot_walk_gym")
    parser.add_argument("--xml", default="scene.xml")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model) if args.model else latest_model(args.log_root)

    env = CrankBotWalkGymEnv(CrankBotWalkEnvConfig(xml_path=args.xml, num_envs=1, seed=args.seed))
    model = SAC.load(str(model_path), device=args.device)

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data[0])

    obs, _ = env.reset(seed=args.seed)
    print(f"Loaded SB3 SAC model: {model_path}")
    try:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
            if viewer is not None:
                viewer.sync()
            if args.realtime:
                time.sleep(env.cfg.control_dt)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
