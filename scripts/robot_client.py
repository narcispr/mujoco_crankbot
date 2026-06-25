from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


JOINT_ORDER = (
    "front_left_shoulder",
    "front_left_elbow",
    "front_right_shoulder",
    "front_right_elbow",
    "back_left_shoulder",
    "back_left_elbow",
    "back_right_shoulder",
    "back_right_elbow",
)
MUJOCO_JOINT_NAMES = tuple(f"{joint}_joint" for joint in JOINT_ORDER)

Q_STAND = np.array([-1.15, -2.34, 1.15, 2.34, 2.0, -2.34, -2.0, 2.34], dtype=np.float64)
ACTION_SCALE = np.array([0.07, 0.10, 0.07, 0.10, 0.07, 0.10, 0.07, 0.10], dtype=np.float64)
COMMAND_DELTA_LIMIT = 0.75
JOINT_RANGE_MIN = np.array([-math.pi, -math.pi, 0.0, 0.0, 0.0, -math.pi, -math.pi, 0.0], dtype=np.float64)
JOINT_RANGE_MAX = np.array([0.0, 0.0, math.pi, math.pi, math.pi, 0.0, 0.0, math.pi], dtype=np.float64)
Q_CMD_MIN = np.maximum(JOINT_RANGE_MIN, Q_STAND - COMMAND_DELTA_LIMIT)
Q_CMD_MAX = np.minimum(JOINT_RANGE_MAX, Q_STAND + COMMAND_DELTA_LIMIT)

CONTROL_DT = 0.1
GAIT_FREQUENCY_HZ = 1.0
HISTORY_LEN = 4
TARGET_GOAL_RANGE = max(math.hypot(0.50, 0.21), 1e-6)
DEFAULT_PRESETS = Path(__file__).resolve().parent / "config" / "robot_presets.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
GOAL_SITE_NAME = "goal_site"
GOAL_SITE_Z = 0.02


@dataclass
class RobotUdpClient:
    host: str
    port: int
    timeout: float
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)

    def close(self) -> None:
        self.sock.close()

    def send_setpoints(self, values: np.ndarray, wait_ack: bool = False) -> str | None:
        values = validated_setpoints(values)
        message = "SET " + " ".join(f"{value:.4f}" for value in values)
        return self.send_command(message, wait_ack=wait_ack)

    def send_command(self, command: str, wait_ack: bool = True) -> str | None:
        if self.dry_run:
            print(command)
            return None

        self.sock.sendto(command.encode("ascii"), (self.host, self.port))
        if not wait_ack:
            return None

        try:
            data, _ = self.sock.recvfrom(1024)
            return data.decode("ascii", errors="replace").strip()
        except socket.timeout:
            return None


class MujocoMirror:
    def __init__(self, xml_path: Path, open_viewer: bool) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("mujoco is required unless --disable-mujoco is used. Install requirements.txt.") from exc

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(resolve_repo_path(xml_path)))
        self.data = mujoco.MjData(self.model)
        self.joint_qposadr = self._joint_qpos_addresses()
        self.base_body_id = self._required_id(self.mujoco.mjtObj.mjOBJ_BODY, "base")
        self.goal_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, GOAL_SITE_NAME)
        self.goal_pos_world: np.ndarray | None = None
        self.viewer: Any | None = None
        self.reset(Q_STAND)

        if open_viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.sync()

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self, q_cmd: np.ndarray) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.joint_qposadr] = np.asarray(q_cmd, dtype=np.float64).reshape(8)
        self.data.ctrl[:] = np.asarray(q_cmd, dtype=np.float64).reshape(8)
        self.mujoco.mj_forward(self.model, self.data)
        self._update_goal_site()

    def command_setpoints(self, setpoints: np.ndarray) -> None:
        self.data.ctrl[:] = setpoints_to_q_cmd(setpoints)

    def current_setpoints(self) -> np.ndarray:
        return q_cmd_to_setpoints(self.data.qpos[self.joint_qposadr])

    def set_goal_from_relative(self, goal_range_m: float, goal_bearing_rad: float) -> None:
        base_xy = self.data.xpos[self.base_body_id, :2]
        goal_angle = self.base_yaw() + goal_bearing_rad
        self.goal_pos_world = base_xy + goal_range_m * np.array(
            [math.cos(goal_angle), math.sin(goal_angle)],
            dtype=np.float64,
        )
        self._update_goal_site()

    def goal_feedback(self) -> tuple[float, float]:
        if self.goal_pos_world is None:
            raise RuntimeError("MuJoCo goal has not been initialized.")
        delta = self.goal_pos_world - self.data.xpos[self.base_body_id, :2]
        goal_range_m = float(np.linalg.norm(delta))
        goal_angle = math.atan2(delta[1], delta[0])
        goal_bearing_rad = wrap_to_pi(goal_angle - self.base_yaw())
        return goal_range_m, goal_bearing_rad

    def base_yaw(self) -> float:
        mat = self.data.xmat[self.base_body_id].reshape(3, 3)
        return math.atan2(-mat[1, 1], -mat[0, 1])

    def step_for(self, duration_s: float) -> None:
        steps = max(1, int(round(duration_s / self.timestep)))
        for _ in range(steps):
            self.mujoco.mj_step(self.model, self.data)
        self.sync()

    def sync(self) -> None:
        if self.viewer is None:
            return
        if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed.")
        self.viewer.sync()

    def _joint_qpos_addresses(self) -> np.ndarray:
        addresses: list[int] = []
        for name in MUJOCO_JOINT_NAMES:
            joint_id = self._required_id(self.mujoco.mjtObj.mjOBJ_JOINT, name)
            addresses.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(addresses, dtype=np.int32)

    def _update_goal_site(self) -> None:
        if self.goal_site_id < 0 or self.goal_pos_world is None:
            return
        self.model.site_pos[self.goal_site_id] = (self.goal_pos_world[0], self.goal_pos_world[1], GOAL_SITE_Z)
        self.mujoco.mj_forward(self.model, self.data)

    def _required_id(self, obj_type: Any, name: str) -> int:
        obj_id = self.mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Could not find MuJoCo object '{name}'.")
        return obj_id


@dataclass
class OutputBridge:
    client: RobotUdpClient
    mujoco: MujocoMirror | None
    wait_ack: bool

    def close(self) -> None:
        if self.mujoco is not None:
            self.mujoco.close()
        self.client.close()

    def send_target(
        self,
        setpoints: np.ndarray,
        *,
        period_s: float,
        esp_from_mujoco: bool = False,
    ) -> np.ndarray:
        setpoints = validated_setpoints(setpoints)
        if self.mujoco is not None:
            self.mujoco.command_setpoints(setpoints)
            self.mujoco.step_for(period_s)

        esp_setpoints = self.mujoco.current_setpoints() if esp_from_mujoco and self.mujoco is not None else setpoints
        ack = self.client.send_setpoints(esp_setpoints, wait_ack=self.wait_ack)
        if ack:
            print(ack)
        return esp_setpoints


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validated_setpoints(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(8)
    return np.clip(values, -1.0, 1.0)


def setpoints_to_q_cmd(setpoints: np.ndarray) -> np.ndarray:
    setpoints = validated_setpoints(setpoints)
    return Q_CMD_MIN + 0.5 * (setpoints + 1.0) * (Q_CMD_MAX - Q_CMD_MIN)


def q_cmd_to_setpoints(q_cmd: np.ndarray) -> np.ndarray:
    q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(8)
    return np.clip(2.0 * (q_cmd - Q_CMD_MIN) / (Q_CMD_MAX - Q_CMD_MIN) - 1.0, -1.0, 1.0)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def latest_model(log_root: str) -> Path:
    root = Path(log_root)
    candidates = sorted(
        list(root.glob("**/final_model.zip")) + list(root.glob("**/sac_crankbot_walk_*_steps.zip")),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No SB3 .zip models found under {root}.")
    return candidates[-1]


def load_presets(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as file:
        presets = json.load(file)
    for name, preset in presets.items():
        values = np.asarray(preset.get("setpoints"), dtype=np.float64)
        if values.shape != (8,):
            raise ValueError(f"Preset '{name}' must contain exactly 8 setpoints.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError(f"Preset '{name}' contains values outside [-1, 1].")
    return presets


def stream_constant_setpoints(
    bridge: OutputBridge,
    values: np.ndarray,
    rate_hz: float,
    duration_s: float,
    *,
    esp_from_mujoco: bool,
) -> None:
    if rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive.")
    if duration_s < 0.0:
        raise ValueError("--duration must be non-negative.")

    period_s = 1.0 / rate_hz
    start = time.monotonic()
    step_idx = 0
    while True:
        bridge.send_target(values, period_s=period_s, esp_from_mujoco=esp_from_mujoco)
        if time.monotonic() - start >= duration_s:
            break

        step_idx += 1
        next_time = start + step_idx * period_s
        time.sleep(max(0.0, next_time - time.monotonic()))


def prompt_float(value: float | None, label: str) -> float:
    if value is not None:
        return value
    while True:
        raw = input(f"{label}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print("Enter a number.")


def build_actor_observation(
    q_cmd_history: np.ndarray,
    action_history: np.ndarray,
    goal_range_m: float,
    goal_bearing_rad: float,
    phase: float,
) -> np.ndarray:
    return np.concatenate(
        (
            q_cmd_history.reshape(-1),
            action_history.reshape(-1),
            np.array(
                [
                    goal_bearing_rad / math.pi,
                    goal_range_m / TARGET_GOAL_RANGE,
                    math.sin(phase),
                    math.cos(phase),
                ],
                dtype=np.float64,
            ),
        )
    ).astype(np.float32)


def adapt_observation_for_model(actor_obs: np.ndarray, expected_dim: int, allow_zero_padding: bool) -> np.ndarray:
    if expected_dim == actor_obs.shape[0]:
        return actor_obs
    if expected_dim == 80 and allow_zero_padding:
        return np.concatenate((actor_obs, np.zeros(12, dtype=np.float32))).astype(np.float32)
    raise ValueError(
        f"Model expects observation size {expected_dim}, but the real robot client builds "
        f"{actor_obs.shape[0]}. Train/export the policy with --disable-privileged, or use "
        "--allow-privileged-zero-padding only for bench tests."
    )


def run_policy(args: argparse.Namespace, bridge: OutputBridge) -> None:
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for policy mode. Install requirements.txt.") from exc

    model_path = Path(args.model) if args.model else latest_model(args.log_root)
    goal_range_m = prompt_float(args.goal_range, "Goal range in meters")
    goal_bearing_rad = math.radians(prompt_float(args.goal_bearing_deg, "Goal bearing in degrees"))
    max_time_s = prompt_float(args.max_time, "Maximum execution time in seconds")

    model = SAC.load(str(model_path), device=args.device)
    expected_dim = int(model.observation_space.shape[0])

    q_cmd = Q_STAND.copy()
    q_cmd_history = np.zeros((HISTORY_LEN, 8), dtype=np.float64)
    action_history = np.zeros((HISTORY_LEN, 8), dtype=np.float64)
    phase = 0.0
    q_stand_setpoints = q_cmd_to_setpoints(Q_STAND)

    if bridge.mujoco is not None:
        bridge.mujoco.reset(Q_STAND)
        bridge.mujoco.set_goal_from_relative(goal_range_m, goal_bearing_rad)

    print(f"Loaded policy: {model_path}")
    print("Sending q_stand before policy execution.")
    stream_constant_setpoints(
        bridge,
        q_stand_setpoints,
        args.rate_hz,
        args.prime_duration,
        esp_from_mujoco=False,
    )

    start = time.monotonic()
    step_idx = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= max_time_s:
            break

        if bridge.mujoco is not None:
            goal_range_m, goal_bearing_rad = bridge.mujoco.goal_feedback()

        actor_obs = build_actor_observation(q_cmd_history, action_history, goal_range_m, goal_bearing_rad, phase)
        obs = adapt_observation_for_model(actor_obs, expected_dim, args.allow_privileged_zero_padding)
        action, _ = model.predict(obs, deterministic=args.deterministic)
        action = np.asarray(action, dtype=np.float64).reshape(8)
        action = np.clip(action, -1.0, 1.0)

        q_cmd = np.clip(q_cmd + action * ACTION_SCALE, Q_CMD_MIN, Q_CMD_MAX)
        setpoints = q_cmd_to_setpoints(q_cmd)
        bridge.send_target(setpoints, period_s=CONTROL_DT, esp_from_mujoco=False)

        q_cmd_history[1:] = q_cmd_history[:-1].copy()
        q_cmd_history[0] = q_cmd - Q_STAND
        action_history[1:] = action_history[:-1].copy()
        action_history[0] = action
        phase = (phase + 2.0 * math.pi * GAIT_FREQUENCY_HZ * CONTROL_DT) % (2.0 * math.pi)
        if bridge.mujoco is None:
            goal_range_m = max(0.0, goal_range_m - args.range_rate * CONTROL_DT)

        step_idx += 1
        next_time = start + step_idx * CONTROL_DT
        time.sleep(max(0.0, next_time - time.monotonic()))


def make_mujoco(args: argparse.Namespace) -> MujocoMirror | None:
    if args.command not in {"preset", "manual", "policy"}:
        return None
    if args.disable_mujoco:
        return None
    return MujocoMirror(Path(args.xml), open_viewer=not args.mujoco_headless)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send CrankBot servo setpoints to the ESP32-C3 bridge.")
    parser.add_argument("--host", default="192.168.4.1", help="ESP32-C3 IP address.")
    parser.add_argument("--port", type=int, default=4210, help="ESP32-C3 UDP port.")
    parser.add_argument("--timeout", type=float, default=0.25, help="UDP ACK timeout in seconds.")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="Command send rate for presets/manual modes.")
    parser.add_argument("--presets", type=Path, default=DEFAULT_PRESETS)
    parser.add_argument("--wait-ack", action="store_true", help="Print ESP acknowledgements.")
    parser.add_argument("--dry-run", action="store_true", help="Print UDP commands without sending them.")
    parser.add_argument(
        "--disable-mujoco",
        action="store_true",
        help="Do not open or step MuJoCo for preset/manual/policy commands.",
    )
    parser.add_argument(
        "--mujoco-headless",
        action="store_true",
        help="Step the MuJoCo mirror without opening a viewer. Useful for tests.",
    )
    parser.add_argument("--xml", default="scene.xml", help="MuJoCo XML used by the simulator mirror.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-presets", help="Show available preset poses.")

    preset_parser = subparsers.add_parser("preset", help="Send a named preset pose.")
    preset_parser.add_argument("name")
    preset_parser.add_argument("--duration", type=float, default=2.0)

    manual_parser = subparsers.add_parser("manual", help="Send 8 explicit target setpoints in [-1, 1].")
    manual_parser.add_argument("values", type=float, nargs=8)
    manual_parser.add_argument("--duration", type=float, default=1.0)

    command_parser = subparsers.add_parser("command", help="Send a raw ESP command such as PING, STATUS, STOP.")
    command_parser.add_argument("message")

    policy_parser = subparsers.add_parser("policy", help="Run an SB3 SAC policy and stream its actions.")
    policy_parser.add_argument("--model", default=None, help="SB3 .zip model path. Defaults to latest under --log-root.")
    policy_parser.add_argument("--log-root", default="logs/crankbot_walk_gym")
    policy_parser.add_argument("--device", default="auto")
    policy_parser.add_argument("--goal-range", type=float, default=None, help="Initial goal range in meters.")
    policy_parser.add_argument("--goal-bearing-deg", type=float, default=None, help="Initial relative goal bearing.")
    policy_parser.add_argument("--max-time", type=float, default=None, help="Execution time limit in seconds.")
    policy_parser.add_argument(
        "--range-rate",
        type=float,
        default=0.0,
        help="Open-loop goal range decrease in m/s, used only with --disable-mujoco.",
    )
    policy_parser.add_argument("--prime-duration", type=float, default=1.0)
    policy_parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    policy_parser.add_argument(
        "--allow-privileged-zero-padding",
        action="store_true",
        help="Pad the 68 real-observable inputs to 80 with zeros. Only for bench tests.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list-presets":
        presets = load_presets(args.presets)
        for name, preset in presets.items():
            print(f"{name}: {preset.get('description', '')}")
        return

    bridge = OutputBridge(
        client=RobotUdpClient(args.host, args.port, args.timeout, dry_run=args.dry_run),
        mujoco=make_mujoco(args),
        wait_ack=args.wait_ack,
    )
    try:
        if args.command == "preset":
            presets = load_presets(args.presets)
            if args.name not in presets:
                raise KeyError(f"Unknown preset '{args.name}'. Use list-presets.")
            values = np.asarray(presets[args.name]["setpoints"], dtype=np.float64)
            stream_constant_setpoints(bridge, values, args.rate_hz, args.duration, esp_from_mujoco=False)
            return

        if args.command == "manual":
            values = np.asarray(args.values, dtype=np.float64)
            if np.any(values < -1.0) or np.any(values > 1.0):
                raise ValueError("Manual setpoints must be in [-1, 1].")
            stream_constant_setpoints(
                bridge,
                values,
                args.rate_hz,
                args.duration,
                esp_from_mujoco=bridge.mujoco is not None,
            )
            return

        if args.command == "command":
            ack = bridge.client.send_command(args.message)
            print(ack if ack else "No response")
            return

        if args.command == "policy":
            run_policy(args, bridge)
            return

        raise RuntimeError(f"Unhandled command: {args.command}")
    finally:
        bridge.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
