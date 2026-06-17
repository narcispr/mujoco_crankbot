from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

try:
    from crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig
except ImportError:
    from .crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig


def build_train_cfg(args: argparse.Namespace) -> dict:
    logger_cfg = None
    if args.logger == "wandb":
        logger_cfg = {"class_name": "WandbLogWriter", "project_name": args.wandb_project}
    elif args.logger == "tensorboard":
        logger_cfg = "tensorboard"

    return {
        "seed": args.seed,
        "run_name": args.run_name,
        "num_steps_per_env": args.num_steps_per_env,
        "save_interval": args.save_interval,
        "check_for_nan": True,
        "logger": logger_cfg,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_range": [0.05, 1.0],
                "std_type": "scalar",
                "learn_std": True,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
        },
    }


def build_env_cfg(args: argparse.Namespace) -> CrankBotWalkEnvConfig:
    return CrankBotWalkEnvConfig(xml_path=args.xml, num_envs=args.num_envs, seed=args.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrankBot walking with rsl_rl PPO.")
    parser.add_argument("--xml", default="scene.xml", help="MuJoCo XML path relative to the repo root.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-steps-per-env", type=int, default=80)
    parser.add_argument("--device", default="cpu", help="Torch training device, for example cpu or cuda:0.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--logger", choices=("wandb", "tensorboard", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="crankbot-walk")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-root", default="logs/crankbot_walk")
    parser.add_argument("--init-at-random-ep-len", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_dir = None
    if args.logger != "none":
        log_dir = str(Path(args.log_root) / args.run_name)

    env = CrankBotWalkEnv(build_env_cfg(args), device="cpu")
    runner = OnPolicyRunner(env, build_train_cfg(args), log_dir=log_dir, device=args.device)
    runner.add_git_repo_to_log(str(Path(__file__).resolve().parents[1]))
    runner.learn(args.iterations, init_at_random_ep_len=args.init_at_random_ep_len)


if __name__ == "__main__":
    main()
