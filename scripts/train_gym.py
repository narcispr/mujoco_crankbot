from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

try:
    from crankbot_walk_env import CrankBotWalkEnvConfig
    from crankbot_walk_gym_env import CrankBotWalkGymEnv
except ImportError:
    from .crankbot_walk_env import CrankBotWalkEnvConfig
    from .crankbot_walk_gym_env import CrankBotWalkGymEnv


class GoalMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        logs = [info["log"] for info in infos if "log" in info]
        if not logs:
            return True

        def mean_log(key: str) -> float:
            return float(np.mean([log[key] for log in logs]))

        goal_forward = mean_log("/curriculum/goal_forward")
        goal_lateral = mean_log("/curriculum/goal_lateral")
        self.logger.record("goal/range", mean_log("/env/mean_goal_range"))
        self.logger.record("goal/reached_rate", mean_log("/env/goal_reached"))
        self.logger.record("goal/success_rate", mean_log("/env/success_rate"))
        self.logger.record("goal/body_contact", mean_log("/env/body_contact"))
        self.logger.record("curriculum/goal_forward", goal_forward)
        self.logger.record("curriculum/goal_lateral", goal_lateral)
        self.logger.record("curriculum/goal_max_distance", math.hypot(goal_forward, goal_lateral))
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrankBot walking with Gymnasium and SB3 SAC.")
    parser.add_argument("--xml", default="scene.xml", help="MuJoCo XML path relative to the repo root.")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-root", default="logs/crankbot_walk_gym")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--logger", choices=("wandb", "tensorboard", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="crankbot-walk-gym")
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--save-vecnormalize", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="Warmup transitions before SAC updates. Defaults to max(20000, num_envs * max_episode_steps * 2).",
    )
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--ent-coef", default="auto_0.1")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.learning_starts is None:
        args.learning_starts = max(20_000, args.num_envs * CrankBotWalkEnvConfig.max_episode_steps * 2)

    log_dir = (Path(args.log_root) / args.run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    monitor_dir = log_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def make_env(rank: int):
        def _init() -> Monitor:
            cfg = CrankBotWalkEnvConfig(xml_path=args.xml, num_envs=1, seed=args.seed + rank)
            return Monitor(CrankBotWalkGymEnv(cfg), filename=str(monitor_dir / f"{rank}"))

        return _init

    vec_env_cls = DummyVecEnv if args.vec_env == "dummy" else SubprocVecEnv
    env = vec_env_cls([make_env(rank) for rank in range(args.num_envs)])

    checkpoint_callback: BaseCallback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac_crankbot_walk",
        save_replay_buffer=args.save_replay_buffer,
        save_vecnormalize=args.save_vecnormalize,
    )
    callbacks: list[BaseCallback] = [checkpoint_callback, GoalMetricsCallback()]
    wandb_run = None

    tensorboard_log = str(log_dir) if args.logger in ("wandb", "tensorboard") else None
    if args.logger == "wandb":
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                dir=str(log_dir),
                config=vars(args),
                sync_tensorboard=True,
            )
            callbacks.append(
                WandbCallback(
                    gradient_save_freq=0,
                    model_save_path=str(log_dir / "wandb_model"),
                    verbose=1,
                )
            )
        except ImportError:
            print("wandb is not installed; continuing with TensorBoard logging.")
            tensorboard_log = str(log_dir)
    callback: BaseCallback = CallbackList(callbacks)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": args.net_arch},
        seed=args.seed,
        device=args.device,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(log_dir / "final_model"))
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
