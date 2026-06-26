from __future__ import annotations

import argparse
from datetime import datetime
import functools
import importlib.util
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
    parser = argparse.ArgumentParser(description="Train CrankBot with Brax PPO/SAC on the MJX-style env.")
    parser.add_argument("--xml", default="scene.mjx.xml", help="MuJoCo XML path relative to the repo root.")
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--backend", choices=("mjx", "warp"), default="mjx")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-eval-envs", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--unroll-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-devices-per-host", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--num-evals", type=int, default=20)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256, 128])

    ppo = parser.add_argument_group("PPO")
    ppo.add_argument("--entropy-cost", type=float, default=0.01)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-epsilon", type=float, default=0.2)
    ppo.add_argument("--value-loss-coef", type=float, default=1.0)
    ppo.add_argument("--max-grad-norm", type=float, default=1.0)
    ppo.add_argument("--num-minibatches", type=int, default=8)
    ppo.add_argument("--num-updates-per-batch", type=int, default=4)
    ppo.add_argument("--batch-size", type=int, default=8192)

    sac = parser.add_argument_group("SAC")
    sac.add_argument("--min-replay-size", type=int, default=50_000)
    sac.add_argument("--max-replay-size", type=int, default=2_000_000)
    sac.add_argument("--sac-batch-size", type=int, default=2048)
    sac.add_argument("--grad-updates-per-step", type=int, default=1)
    sac.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--logger", choices=("wandb", "tensorboard", "none"), default="wandb")
    parser.add_argument("--wandb-project", default="crankbot-walk-mjx")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-root", default="logs/crankbot_walk_mjx")
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser.parse_args()


def build_env_cfg(args: argparse.Namespace, env_module: Any) -> Any:
    return env_module.CrankBotWalkMJXEnvConfig(
        xml_path=args.xml,
        num_envs=args.num_envs,
        seed=args.seed,
        max_episode_steps=args.episode_length,
    )


def make_progress_fn(args: argparse.Namespace, log_dir: Path | None):
    wandb_run = None
    summary_writer = None

    if args.logger == "wandb":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                dir=str(log_dir) if log_dir else None,
                config=vars(args),
            )
        except ImportError:
            print("wandb is not installed; continuing with stdout logging.", flush=True)
    elif args.logger == "tensorboard" and log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(str(log_dir))
        except ImportError:
            print("TensorBoard writer is not installed; continuing with stdout logging.", flush=True)

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        scalar_metrics = {}
        for key, value in metrics.items():
            shape = getattr(value, "shape", ())
            if shape == ():
                scalar_metrics[key] = float(value)

        line = f"steps={num_steps}"
        for key in ("eval/episode_reward", "eval/avg_episode_length", "eval/mean_goal_range", "eval/goal_reached"):
            if key in scalar_metrics:
                line += f" {key}={scalar_metrics[key]:.4f}"
        print(line, flush=True)

        if wandb_run is not None:
            wandb_run.log(scalar_metrics, step=num_steps)
        if summary_writer is not None:
            for key, value in scalar_metrics.items():
                summary_writer.add_scalar(key, value, num_steps)
            summary_writer.flush()

    return progress


def _ppo_train(args: argparse.Namespace, env: Any, env_module: Any, log_dir: Path | None, checkpoint_dir: str | None):
    from flax import linen
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    if (args.batch_size * args.num_minibatches) % args.num_envs != 0:
        raise ValueError("--batch-size * --num-minibatches must be divisible by --num-envs for Brax PPO.")

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(args.net_arch),
        value_hidden_layer_sizes=tuple(args.net_arch),
        activation=linen.elu,
        init_noise_std=0.5,
    )
    return ppo.train(
        environment=env,
        num_timesteps=args.timesteps,
        max_devices_per_host=args.max_devices_per_host,
        wrap_env=True,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=1,
        wrap_env_fn=env_module.wrap_for_training,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
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


def _sac_train(args: argparse.Namespace, env: Any, env_module: Any, log_dir: Path | None, checkpoint_dir: str | None):
    from flax import linen
    from brax.training.agents.sac import networks as sac_networks
    from brax.training.agents.sac import train as sac

    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=tuple(args.net_arch),
        activation=linen.elu,
        init_noise_std=0.5,
    )
    return sac.train(
        environment=env,
        num_timesteps=args.timesteps,
        episode_length=args.episode_length,
        wrap_env=True,
        wrap_env_fn=env_module.wrap_for_training,
        action_repeat=1,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        learning_rate=args.learning_rate,
        discounting=args.discounting,
        seed=args.seed,
        batch_size=args.sac_batch_size,
        num_evals=args.num_evals,
        normalize_observations=True,
        max_devices_per_host=args.max_devices_per_host,
        reward_scaling=args.reward_scaling,
        tau=args.tau,
        min_replay_size=args.min_replay_size,
        max_replay_size=args.max_replay_size,
        grad_updates_per_step=args.grad_updates_per_step,
        network_factory=network_factory,
        progress_fn=make_progress_fn(args, log_dir),
        checkpoint_logdir=checkpoint_dir,
    )


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = datetime.now().strftime(f"{args.algorithm}_%Y%m%d_%H%M%S")

    env_module = _load_mjx_env_module()
    cfg = build_env_cfg(args, env_module)
    try:
        env = env_module.CrankBotWalkMJXEnv(cfg, backend=args.backend)
    except RuntimeError as exc:
        print(f"Cannot create backend={args.backend}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc

    log_dir = None
    checkpoint_dir = None
    if args.logger != "none" or not args.no_checkpoints:
        log_dir = (Path(args.log_root) / args.run_name).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_checkpoints and log_dir is not None:
        checkpoint_dir = str(log_dir / "checkpoints")

    import jax

    print(f"JAX devices: {jax.devices()}", flush=True)
    print(
        f"Training {args.algorithm.upper()} on backend={args.backend} with "
        f"{args.num_envs} envs for {args.timesteps} env steps.",
        flush=True,
    )

    train_fn = _ppo_train if args.algorithm == "ppo" else _sac_train
    make_policy, params, metrics = train_fn(args, env, env_module, log_dir, checkpoint_dir)
    del make_policy
    del params

    print("Final metrics:", flush=True)
    for key, value in sorted(metrics.items()):
        if getattr(value, "shape", ()) == ():
            print(f"  {key}: {float(value):.6g}", flush=True)


if __name__ == "__main__":
    main()
