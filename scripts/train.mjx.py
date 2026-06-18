from __future__ import annotations

import argparse
from datetime import datetime
import functools
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


def _load_mjx_env_module() -> Any:
    module_path = Path(__file__).with_name("crankbot_walk_env.mjx.py")
    spec = importlib.util.spec_from_file_location("crankbot_walk_env_mjx", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrankBot walking with MJX and Brax PPO.")
    parser.add_argument("--xml", default="scene.mjx.xml", help="MJX-compatible MuJoCo XML path relative to repo root.")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--num-steps-per-env", type=int, default=80)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "gpu", "tpu"), default="gpu")
    parser.add_argument("--max-devices-per-host", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-cost", type=float, default=0.01)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--num-updates-per-batch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-evals", type=int, default=20)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=50, help="Kept for CLI parity; Brax checkpoints per eval.")
    parser.add_argument("--logger", choices=("wandb", "tensorboard", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="crankbot-walk-mjx")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-root", default="logs/crankbot_walk_mjx")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--use-target-curriculum-values", action="store_true")
    return parser.parse_args()


def configure_jax_platform(platform: str) -> None:
    if platform != "auto":
        os.environ.setdefault("JAX_PLATFORMS", "cuda" if platform == "gpu" else platform)


def build_env_cfg(args: argparse.Namespace, env_module: Any) -> Any:
    return env_module.CrankBotWalkMJXEnvConfig(
        xml_path=args.xml,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_steps=args.episode_length,
        use_target_curriculum_values=args.use_target_curriculum_values,
    )


def make_progress_fn(args: argparse.Namespace, log_dir: Path | None):
    wandb_run = None
    summary_writer = None

    if args.logger == "wandb":
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, dir=str(log_dir) if log_dir else None)
        except ImportError:
            print("wandb is not installed; continuing with stdout logging.")
    elif args.logger == "tensorboard" and log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(str(log_dir))
        except ImportError:
            print("TensorBoard writer is not installed; continuing with stdout logging.")

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        scalar_metrics = {key: float(value) for key, value in metrics.items() if getattr(value, "shape", ()) == ()}
        reward = scalar_metrics.get("eval/episode_reward", scalar_metrics.get("eval/episode_reward_std"))
        line = f"steps={num_steps}"
        if reward is not None:
            line += f" eval_reward={reward:.4f}"
        if "eval/episode_length" in scalar_metrics:
            line += f" eval_len={scalar_metrics['eval/episode_length']:.1f}"
        print(line, flush=True)

        if wandb_run is not None:
            wandb_run.log(scalar_metrics, step=num_steps)
        if summary_writer is not None:
            for key, value in scalar_metrics.items():
                summary_writer.add_scalar(key, value, num_steps)
            summary_writer.flush()

    return progress


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    configure_jax_platform(args.jax_platform)

    import jax
    from flax import linen
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    env_module = _load_mjx_env_module()
    cfg = build_env_cfg(args, env_module)
    env = env_module.CrankBotWalkMJXEnv(cfg)

    total_timesteps = args.num_timesteps
    if total_timesteps is None:
        total_timesteps = args.iterations * args.num_envs * args.num_steps_per_env

    log_dir = None
    checkpoint_dir = None
    if args.logger != "none" or not args.no_checkpoints:
        log_dir = (Path(args.log_root) / args.run_name).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_checkpoints and log_dir is not None:
        checkpoint_dir = str(log_dir / "checkpoints")

    if (args.batch_size * args.num_minibatches) % args.num_envs != 0:
        raise ValueError("--batch-size * --num-minibatches must be divisible by --num-envs for Brax PPO.")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(256, 256, 128),
        value_hidden_layer_sizes=(256, 256, 128),
        activation=linen.elu,
        policy_obs_key="state",
        value_obs_key="privileged_state",
        init_noise_std=0.5,
    )

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"Training {args.num_envs} MJX envs for {total_timesteps} environment steps.", flush=True)

    make_policy, params, metrics = ppo.train(
        environment=env,
        num_timesteps=total_timesteps,
        max_devices_per_host=args.max_devices_per_host,
        wrap_env=True,
        num_envs=args.num_envs,
        episode_length=cfg.max_episode_steps,
        action_repeat=1,
        wrap_env_fn=env_module.wrap_for_training,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.num_steps_per_env,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=True,
        reward_scaling=args.reward_scaling,
        clipping_epsilon=args.clip_epsilon,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_loss_coefficient=args.value_loss_coef,
        desired_kl=0.01,
        learning_rate_schedule="ADAPTIVE_KL",
        network_factory=network_factory,
        seed=args.seed,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        progress_fn=make_progress_fn(args, log_dir),
        save_checkpoint_path=checkpoint_dir,
    )

    del make_policy
    del params
    print("Final metrics:")
    for key, value in sorted(metrics.items()):
        if getattr(value, "shape", ()) == ():
            print(f"  {key}: {float(value):.6g}")


if __name__ == "__main__":
    main()
