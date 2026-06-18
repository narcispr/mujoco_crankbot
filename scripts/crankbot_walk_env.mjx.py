from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brax import math as brax_math
from brax.envs.base import PipelineEnv, State, Wrapper
from brax.envs.wrappers import training as brax_training
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco


JOINT_NAMES = (
    "front_left_shoulder_joint",
    "front_left_elbow_joint",
    "front_right_shoulder_joint",
    "front_right_elbow_joint",
    "back_left_shoulder_joint",
    "back_left_elbow_joint",
    "back_right_shoulder_joint",
    "back_right_elbow_joint",
)

FOOT_GEOM_NAMES = (
    "front_left_lower_leg_foot",
    "front_right_lower_leg_foot",
    "back_left_lower_leg_foot",
    "back_right_lower_leg_foot",
)

CRANKBOT_INFO_KEYS = (
    "rng",
    "q_cmd",
    "last_action",
    "q_cmd_history",
    "action_history",
    "commands",
    "phase",
    "vx_error_history",
    "wz_error_history",
    "error_history_pos",
    "episode_body_contact",
)


@dataclass
class CrankBotWalkMJXEnvConfig:
    xml_path: str = "scene.mjx.xml"
    num_envs: int = 1024
    seed: int = 1
    simulation_dt: float = 0.002
    control_dt: float = 0.1
    decimation: int = 50
    max_episode_steps: int = 80
    history_len: int = 4
    gait_frequency: float = 1.0
    global_max_vx: float = 0.15
    global_max_wz: float = 0.50
    initial_max_vx: float = 0.05
    initial_max_wz: float = 0.0
    target_max_vx: float = 0.15
    target_max_wz: float = 0.50
    initial_pose_noise: float = 0.10
    target_initial_pose_noise: float = 0.50
    servo_position_error: float = 0.0
    command_delta_limit: float = 0.75
    k_vx: float = 4.0
    k_wz: float = 2.0
    smoothness_coef: float = 0.002
    target_smoothness_coef: float = 0.005
    action_coef: float = 0.0005
    target_action_coef: float = 0.001
    body_contact_penalty: float = 0.5
    target_body_contact_penalty: float = 1.0
    fall_penalty: float = 2.0
    fall_height: float = 0.025
    vx_success_threshold: float = 0.025
    wz_success_threshold: float = 0.05
    use_target_curriculum_values: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrankBotRandomResetWrapper(Wrapper):
    """Brax training wrapper that randomizes CrankBot episode state on reset."""

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        if "steps" in info:
            info["steps"] = jp.where(state.done, jp.zeros_like(info["steps"]), info["steps"])
            state = state.replace(info=info)

        state = state.replace(done=jp.zeros_like(state.done))
        stepped = self.env.step(state, action)
        reset_state = self.env.reset(stepped.info["rng"])

        def where_done(reset_value: jax.Array, value: jax.Array) -> jax.Array:
            if not hasattr(value, "shape") or len(value.shape) == 0:
                return value
            done = stepped.done
            if done.shape and done.shape[0] != value.shape[0]:
                return value
            if done.shape:
                done = jp.reshape(done, [value.shape[0]] + [1] * (len(value.shape) - 1))
            return jp.where(done, reset_value, value)

        pipeline_state = jax.tree_util.tree_map(where_done, reset_state.pipeline_state, stepped.pipeline_state)
        obs = jax.tree_util.tree_map(where_done, reset_state.obs, stepped.obs)

        info = dict(stepped.info)
        for key in CRANKBOT_INFO_KEYS:
            if key in info and key in reset_state.info:
                info[key] = jax.tree_util.tree_map(where_done, reset_state.info[key], info[key])
        return stepped.replace(pipeline_state=pipeline_state, obs=obs, info=info)


def wrap_for_training(
    env: Any,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Any | None = None,
) -> Wrapper:
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = brax_training.DomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    return CrankBotRandomResetWrapper(env)


class CrankBotWalkMJXEnv(PipelineEnv):
    num_actions = 8
    num_obs = 68
    num_privileged_obs = 79

    def __init__(self, cfg: CrankBotWalkMJXEnvConfig, backend: str = "mjx") -> None:
        self.cfg = cfg

        xml_path = Path(cfg.xml_path).expanduser()
        if not xml_path.is_absolute():
            xml_path = Path(__file__).resolve().parents[1] / xml_path
        self.xml_path = xml_path

        mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        mj_model.opt.timestep = cfg.simulation_dt
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace(
            {
                "opt.timestep": cfg.simulation_dt,
                "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                "opt.iterations": 1,
                "opt.ls_iterations": 4,
            }
        )
        super().__init__(sys=sys, backend=backend, n_frames=cfg.decimation)

        self._model = mj_model
        self._root_joint_id = self._joint_id("root_freejoint")
        self._root_qposadr = int(mj_model.jnt_qposadr[self._root_joint_id])
        self._root_dofadr = int(mj_model.jnt_dofadr[self._root_joint_id])
        self._base_body_id = self._body_id("base")
        self._base_geom_id = self._geom_id("base_collision")
        self._floor_geom_id = self._optional_geom_id("floor")
        self._foot_geom_ids = jp.array([self._geom_id(name) for name in FOOT_GEOM_NAMES], dtype=jp.int32)

        joint_ids = [self._joint_id(name) for name in JOINT_NAMES]
        self._joint_qposadr = jp.array(mj_model.jnt_qposadr[joint_ids], dtype=jp.int32)
        self._joint_dofadr = jp.array(mj_model.jnt_dofadr[joint_ids], dtype=jp.int32)
        joint_ranges = jp.array(mj_model.jnt_range[joint_ids], dtype=jp.float32)

        self._qpos0 = jp.array(mj_model.qpos0, dtype=jp.float32)
        self._q_stand = jp.array([-1.15, -2.34, 1.15, 2.34, 2.0, -2.34, -2.0, 2.34], dtype=jp.float32)
        self._action_scale = jp.array([0.07, 0.10, 0.07, 0.10, 0.07, 0.10, 0.07, 0.10], dtype=jp.float32)
        self._q_cmd_min = jp.maximum(joint_ranges[:, 0], self._q_stand - cfg.command_delta_limit)
        self._q_cmd_max = jp.minimum(joint_ranges[:, 1], self._q_stand + cfg.command_delta_limit)
        self._joint_min = joint_ranges[:, 0]
        self._joint_max = joint_ranges[:, 1]

        if cfg.use_target_curriculum_values:
            self._max_vx = cfg.target_max_vx
            self._max_wz = cfg.target_max_wz
            self._initial_pose_noise = cfg.target_initial_pose_noise
            self._smoothness_coef = cfg.target_smoothness_coef
            self._action_coef = cfg.target_action_coef
            self._body_contact_penalty = cfg.target_body_contact_penalty
        else:
            self._max_vx = cfg.initial_max_vx
            self._max_wz = cfg.initial_max_wz
            self._initial_pose_noise = cfg.initial_pose_noise
            self._smoothness_coef = cfg.smoothness_coef
            self._action_coef = cfg.action_coef
            self._body_contact_penalty = cfg.body_contact_penalty

    def reset(self, rng: jax.Array) -> State:
        (
            rng,
            pose_rng,
            servo_rng,
            qvel_rng,
            vx_rng,
            wz_rng,
            phase_rng,
        ) = jax.random.split(rng, 7)

        q_cmd = self._q_stand + jax.random.uniform(
            pose_rng,
            (self.num_actions,),
            minval=-self._initial_pose_noise,
            maxval=self._initial_pose_noise,
        )
        q_cmd = jp.clip(q_cmd, self._q_cmd_min, self._q_cmd_max)
        q_real = q_cmd + jax.random.uniform(
            servo_rng,
            (self.num_actions,),
            minval=-self.cfg.servo_position_error,
            maxval=self.cfg.servo_position_error,
        )
        q_real = jp.clip(q_real, self._joint_min, self._joint_max)

        q = self._qpos0.at[self._joint_qposadr].set(q_real)
        qd = jax.random.uniform(qvel_rng, (self.sys.qd_size(),), minval=-0.05, maxval=0.05)
        qd = qd.at[self._root_dofadr : self._root_dofadr + 6].set(0.0)
        pipeline_state = self.pipeline_init(q, qd, ctrl=q_cmd)

        q_cmd_history = jp.repeat((q_cmd - self._q_stand)[None, :], self.cfg.history_len, axis=0)
        action_history = jp.zeros((self.cfg.history_len, self.num_actions), dtype=jp.float32)
        commands = jp.array(
            [
                jax.random.uniform(vx_rng, (), minval=0.0, maxval=self._max_vx),
                jax.random.uniform(wz_rng, (), minval=-self._max_wz, maxval=self._max_wz),
            ],
            dtype=jp.float32,
        )
        phase = jax.random.uniform(phase_rng, (), minval=0.0, maxval=2.0 * math.pi)
        zero = jp.array(0.0, dtype=jp.float32)
        info = {
            "rng": rng,
            "q_cmd": q_cmd,
            "last_action": jp.zeros(self.num_actions, dtype=jp.float32),
            "q_cmd_history": q_cmd_history,
            "action_history": action_history,
            "commands": commands,
            "phase": phase,
            "vx_error_history": jp.zeros(20, dtype=jp.float32),
            "wz_error_history": jp.zeros(20, dtype=jp.float32),
            "error_history_pos": jp.array(0, dtype=jp.int32),
            "episode_body_contact": zero,
        }
        measurements = self._measurements(pipeline_state, info["episode_body_contact"])
        obs = self._get_obs(info, measurements)
        metrics = self._metrics(measurements, zero, zero, zero, zero)
        metrics["reward"] = zero
        return State(pipeline_state, obs, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        info = dict(state.info)
        previous_action = info["last_action"]
        q_cmd = jp.clip(info["q_cmd"] + action * self._action_scale, self._q_cmd_min, self._q_cmd_max)
        pipeline_state = self.pipeline_step(state.pipeline_state, q_cmd)

        phase = (info["phase"] + 2.0 * math.pi * self.cfg.gait_frequency * self.cfg.control_dt) % (
            2.0 * math.pi
        )
        measurements = self._measurements(pipeline_state, info["episode_body_contact"])
        reward, reward_terms = self._reward(action, previous_action, info["commands"], measurements)
        fall = measurements["fall"]
        done = fall

        pos = info["error_history_pos"]
        vx_error = jp.abs(info["commands"][0] - measurements["base_velocity_body"][0])
        wz_error = jp.abs(info["commands"][1] - measurements["base_velocity_body"][2])
        vx_error_history = info["vx_error_history"].at[pos].set(vx_error)
        wz_error_history = info["wz_error_history"].at[pos].set(wz_error)
        error_history_pos = (pos + 1) % vx_error_history.shape[0]

        q_cmd_history = jp.concatenate([(q_cmd - self._q_stand)[None, :], info["q_cmd_history"][:-1]], axis=0)
        action_history = jp.concatenate([action[None, :], info["action_history"][:-1]], axis=0)

        rng, _ = jax.random.split(info["rng"])
        info.update(
            {
                "rng": rng,
                "q_cmd": q_cmd,
                "last_action": action,
                "q_cmd_history": q_cmd_history,
                "action_history": action_history,
                "phase": phase,
                "vx_error_history": vx_error_history,
                "wz_error_history": wz_error_history,
                "error_history_pos": error_history_pos,
                "episode_body_contact": measurements["episode_body_contact"],
            }
        )
        obs = self._get_obs(info, measurements)
        metrics = self._metrics(
            measurements,
            reward_terms["r_vx"],
            reward_terms["r_wz"],
            reward_terms["smoothness"],
            reward_terms["action_mag"],
        )
        metrics["reward"] = reward
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def _get_obs(self, info: dict[str, jax.Array], measurements: dict[str, jax.Array]) -> dict[str, jax.Array]:
        commands = info["commands"]
        actor_obs = jp.concatenate(
            [
                info["q_cmd_history"].reshape(-1),
                info["action_history"].reshape(-1),
                commands[0:1] / self.cfg.global_max_vx,
                commands[1:2] / self.cfg.global_max_wz,
                jp.sin(info["phase"])[None],
                jp.cos(info["phase"])[None],
            ]
        )
        critic_obs = jp.concatenate(
            [
                actor_obs,
                measurements["base_velocity_body"],
                measurements["base_z"][None],
                measurements["roll_pitch"],
                measurements["foot_contacts"],
                measurements["body_contact"][None],
            ]
        )
        return {"state": actor_obs, "privileged_state": critic_obs}

    def _measurements(self, pipeline_state: Any, episode_body_contact: jax.Array) -> dict[str, jax.Array]:
        quat = pipeline_state.xquat[self._base_body_id]
        qd = pipeline_state.qd
        linear_world = qd[self._root_dofadr : self._root_dofadr + 3]
        angular_world = qd[self._root_dofadr + 3 : self._root_dofadr + 6]
        linear_body = brax_math.rotate(linear_world, brax_math.quat_inv(quat))
        angular_body = brax_math.rotate(angular_world, brax_math.quat_inv(quat))
        base_velocity_body = jp.array([linear_body[0], linear_body[1], angular_body[2]], dtype=jp.float32)

        base_z = pipeline_state.xpos[self._base_body_id, 2]
        roll_pitch = self._roll_pitch_from_xmat(pipeline_state.xmat[self._base_body_id])
        foot_contacts, body_contact = self._contact_flags(pipeline_state)
        episode_body_contact = jp.maximum(episode_body_contact, body_contact)
        fall = jp.where(base_z < self.cfg.fall_height, 1.0, 0.0)
        return {
            "base_velocity_body": base_velocity_body,
            "base_z": base_z,
            "roll_pitch": roll_pitch,
            "foot_contacts": foot_contacts,
            "body_contact": body_contact,
            "episode_body_contact": episode_body_contact,
            "fall": fall,
        }

    def _reward(
        self,
        action: jax.Array,
        previous_action: jax.Array,
        commands: jax.Array,
        measurements: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        evx = (commands[0] - measurements["base_velocity_body"][0]) / self.cfg.global_max_vx
        ewz = (commands[1] - measurements["base_velocity_body"][2]) / self.cfg.global_max_wz
        r_vx = jp.exp(-self.cfg.k_vx * evx**2)
        r_wz = jp.exp(-self.cfg.k_wz * ewz**2)
        smoothness = jp.sum((action - previous_action) ** 2)
        action_mag = jp.sum(action**2)
        reward = (
            r_vx
            + 0.5 * r_wz
            - self._smoothness_coef * smoothness
            - self._action_coef * action_mag
            - self._body_contact_penalty * measurements["body_contact"]
            - self.cfg.fall_penalty * measurements["fall"]
        )
        terms = {
            "r_vx": r_vx,
            "r_wz": r_wz,
            "smoothness": smoothness,
            "action_mag": action_mag,
        }
        return reward, terms

    def _metrics(
        self,
        measurements: dict[str, jax.Array],
        r_vx: jax.Array,
        r_wz: jax.Array,
        smoothness: jax.Array,
        action_mag: jax.Array,
    ) -> dict[str, jax.Array]:
        return {
            "mean_vx": measurements["base_velocity_body"][0],
            "mean_wz": measurements["base_velocity_body"][2],
            "body_contact": measurements["body_contact"],
            "fall": measurements["fall"],
            "reward_vx": r_vx,
            "reward_wz": r_wz,
            "smoothness": smoothness,
            "action_mag": action_mag,
            "curriculum_max_vx": jp.array(self._max_vx, dtype=jp.float32),
            "curriculum_max_wz": jp.array(self._max_wz, dtype=jp.float32),
        }

    def _contact_flags(self, pipeline_state: Any) -> tuple[jax.Array, jax.Array]:
        contact = pipeline_state.contact
        if contact is None:
            return jp.zeros(4, dtype=jp.float32), jp.array(0.0, dtype=jp.float32)

        geom1 = contact.geom1
        geom2 = contact.geom2
        active = contact.dist <= 0.0
        if self._floor_geom_id >= 0:
            active = active & ((geom1 == self._floor_geom_id) | (geom2 == self._floor_geom_id))

        foot_contacts = []
        for foot_geom_id in self._foot_geom_ids:
            foot_contacts.append(jp.any(active & ((geom1 == foot_geom_id) | (geom2 == foot_geom_id))).astype(jp.float32))
        body_contact = jp.any(active & ((geom1 == self._base_geom_id) | (geom2 == self._base_geom_id))).astype(jp.float32)
        return jp.array(foot_contacts, dtype=jp.float32), body_contact

    @staticmethod
    def _roll_pitch_from_xmat(xmat: jax.Array) -> jax.Array:
        mat = xmat.reshape(3, 3)
        roll = jp.arctan2(mat[2, 1], mat[2, 2])
        pitch = jp.arctan2(-mat[2, 0], jp.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        return jp.array([roll, pitch], dtype=jp.float32)

    def _joint_id(self, name: str) -> int:
        return self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def _body_id(self, name: str) -> int:
        return self._required_id(mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_id(self, name: str) -> int:
        return self._required_id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def _optional_geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _required_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self._model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Could not find MuJoCo object '{name}' in {self.xml_path}.")
        return obj_id
