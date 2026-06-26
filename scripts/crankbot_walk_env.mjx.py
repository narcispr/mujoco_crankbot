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

LEG_CONTACT_GEOM_NAMES = (
    "front_left_leg_collision",
    "front_right_leg_collision",
    "back_left_leg_collision",
    "back_right_leg_collision",
)

LOWER_LEG_CONTACT_GEOM_NAMES = (
    "front_left_lower_leg_collision",
    "front_right_lower_leg_collision",
    "back_left_lower_leg_collision",
    "back_right_lower_leg_collision",
)

CURRICULUM_KEYS = (
    "goal_forward",
    "goal_lateral",
    "goal_reach_radius",
    "initial_pose_noise",
    "smoothness_coef",
    "action_coef",
    "body_contact_penalty",
    "success_rate",
)

RESET_INFO_KEYS = (
    "rng",
    "q_cmd",
    "last_action",
    "q_cmd_history",
    "action_history",
    "goal_pos_world",
    "goal_bearing",
    "goal_range",
    "goal_reached",
    "phase",
    *CURRICULUM_KEYS,
)


@dataclass
class CrankBotWalkMJXEnvConfig:
    xml_path: str = "scene.mjx.xml"
    num_envs: int = 4096
    seed: int = 1
    simulation_dt: float = 0.002
    control_dt: float = 0.1
    decimation: int = 50
    max_episode_steps: int = 80
    history_len: int = 4
    gait_frequency: float = 1.0
    initial_goal_forward: float = 0.25
    target_goal_forward: float = 0.50
    initial_goal_lateral: float = 0.0
    target_goal_lateral: float = 0.21
    goal_reach_radius: float = 0.20
    target_goal_reach_radius: float = 0.13
    goal_reward_scale: float = 0.25
    initial_pose_noise: float = 0.10
    target_initial_pose_noise: float = 0.25
    servo_position_error: float = 0.0
    command_delta_limit: float = 0.75
    smoothness_coef: float = 0.002
    target_smoothness_coef: float = 0.005
    action_coef: float = 0.0005
    target_action_coef: float = 0.001
    idle_action_penalty: float = 0.02
    idle_action_tolerance: float = 1e-4
    leg_contact_penalty: float = 0.15
    lower_leg_contact_penalty: float = 0.0
    body_contact_penalty: float = 0.5
    target_body_contact_penalty: float = 1.0
    fall_penalty: float = 2.0
    fall_height: float = 0.025
    curriculum_success_threshold: float = 0.75
    curriculum_ema_alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrankBotTrainingWrapper(Wrapper):
    """Vectorizes the env, keeps random resets, and advances per-env curriculum."""

    def __init__(
        self,
        env: PipelineEnv,
        episode_length: int,
        action_repeat: int = 1,
        randomization_fn: Any | None = None,
    ) -> None:
        if randomization_fn is not None:
            raise NotImplementedError("Domain randomization is not wired for CrankBot MJX yet.")
        super().__init__(brax_training.VmapWrapper(env))
        self._base_env = env
        self._episode_length = episode_length
        self._action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        steps = jp.zeros(rng.shape[:-1], dtype=jp.int32)
        state.info["steps"] = steps
        state.info["truncation"] = jp.zeros_like(state.done)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = state.replace(done=jp.zeros_like(state.done))

        def frame_step(carry: State, _: Any) -> tuple[State, jax.Array]:
            next_state = self.env.step(carry, action)
            return next_state, next_state.reward

        stepped, rewards = jax.lax.scan(frame_step, state, (), length=self._action_repeat)
        reward = jp.sum(rewards, axis=0)

        steps = state.info["steps"] + self._action_repeat
        time_out = steps >= self._episode_length
        env_done = stepped.done > 0.0
        done = jp.logical_or(env_done, time_out).astype(jp.float32)
        truncation = jp.logical_and(time_out, jp.logical_not(env_done)).astype(jp.float32)

        curriculum = self._next_curriculum(stepped.info, done)
        reset_rng = stepped.info["rng"]
        reset_state = jax.vmap(self._base_env.reset_with_curriculum)(
            reset_rng,
            curriculum["goal_forward"],
            curriculum["goal_lateral"],
            curriculum["goal_reach_radius"],
            curriculum["initial_pose_noise"],
            curriculum["smoothness_coef"],
            curriculum["action_coef"],
            curriculum["body_contact_penalty"],
            curriculum["success_rate"],
        )

        def where_done(reset_value: jax.Array, value: jax.Array) -> jax.Array:
            if not hasattr(value, "shape") or value.shape[:1] != done.shape[:1]:
                return value
            mask = jp.reshape(done, done.shape + (1,) * (len(value.shape) - 1))
            return jp.where(mask, reset_value, value)

        pipeline_state = jax.tree_util.tree_map(where_done, reset_state.pipeline_state, stepped.pipeline_state)
        obs = where_done(reset_state.obs, stepped.obs)

        info = dict(stepped.info)
        for key in RESET_INFO_KEYS:
            info[key] = where_done(reset_state.info[key], stepped.info[key])
        info["steps"] = jp.where(done > 0.0, jp.zeros_like(steps), steps)
        info["truncation"] = truncation

        return stepped.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def _next_curriculum(self, info: dict[str, jax.Array], done: jax.Array) -> dict[str, jax.Array]:
        success = info["goal_reached"].astype(jp.float32)
        success_sample = jp.where(done > 0.0, success, info["success_rate"])
        success_rate = jp.where(
            done > 0.0,
            (1.0 - self._base_env.cfg.curriculum_ema_alpha) * info["success_rate"]
            + self._base_env.cfg.curriculum_ema_alpha * success_sample,
            info["success_rate"],
        )
        promote = jp.logical_and(done > 0.0, success_rate > self._base_env.cfg.curriculum_success_threshold)
        cfg = self._base_env.cfg
        return {
            "goal_forward": jp.where(promote, jp.minimum(cfg.target_goal_forward, info["goal_forward"] * 1.15), info["goal_forward"]),
            "goal_lateral": jp.where(promote, jp.minimum(cfg.target_goal_lateral, info["goal_lateral"] + 0.03), info["goal_lateral"]),
            "goal_reach_radius": jp.where(
                promote,
                jp.maximum(cfg.target_goal_reach_radius, info["goal_reach_radius"] - 0.01),
                info["goal_reach_radius"],
            ),
            "initial_pose_noise": jp.where(
                promote,
                jp.minimum(cfg.target_initial_pose_noise, info["initial_pose_noise"] + 0.03),
                info["initial_pose_noise"],
            ),
            "smoothness_coef": jp.where(
                promote,
                jp.minimum(cfg.target_smoothness_coef, info["smoothness_coef"] + 0.00025),
                info["smoothness_coef"],
            ),
            "action_coef": jp.where(
                promote,
                jp.minimum(cfg.target_action_coef, info["action_coef"] + 0.00005),
                info["action_coef"],
            ),
            "body_contact_penalty": jp.where(
                promote,
                jp.minimum(cfg.target_body_contact_penalty, info["body_contact_penalty"] + 0.05),
                info["body_contact_penalty"],
            ),
            "success_rate": jp.where(promote, jp.zeros_like(success_rate), success_rate),
        }


def wrap_for_training(
    env: PipelineEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Any | None = None,
) -> Wrapper:
    return CrankBotTrainingWrapper(env, episode_length, action_repeat, randomization_fn)


class CrankBotWalkMJXEnv(PipelineEnv):
    num_actions = 8
    num_obs = 68

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
        try:
            super().__init__(sys=sys, backend=backend, n_frames=cfg.decimation)
        except ValueError as exc:
            if backend == "warp":
                raise RuntimeError(
                    "This Brax PipelineEnv build does not expose backend='warp'. "
                    "MuJoCo Playground currently requires Python >=3.11 and depends on "
                    "brax>=0.14.2 plus warp-lang>=1.11."
                ) from exc
            raise

        self._model = mj_model
        self._root_joint_id = self._joint_id("root_freejoint")
        self._root_qposadr = int(mj_model.jnt_qposadr[self._root_joint_id])
        self._root_dofadr = int(mj_model.jnt_dofadr[self._root_joint_id])
        self._base_body_id = self._body_id("base")
        self._base_geom_id = self._geom_id("base_collision")
        self._floor_geom_id = self._optional_geom_id("floor")
        self._foot_geom_ids = jp.array([self._geom_id(name) for name in FOOT_GEOM_NAMES], dtype=jp.int32)
        self._leg_contact_geom_ids = jp.array([self._geom_id(name) for name in LEG_CONTACT_GEOM_NAMES], dtype=jp.int32)
        self._lower_leg_contact_geom_ids = jp.array(
            [self._geom_id(name) for name in LOWER_LEG_CONTACT_GEOM_NAMES],
            dtype=jp.int32,
        )

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
        self._target_goal_range = max(math.hypot(cfg.target_goal_forward, cfg.target_goal_lateral), 1e-6)

    def reset(self, rng: jax.Array) -> State:
        return self.reset_with_curriculum(
            rng,
            jp.array(self.cfg.initial_goal_forward, dtype=jp.float32),
            jp.array(self.cfg.initial_goal_lateral, dtype=jp.float32),
            jp.array(self.cfg.goal_reach_radius, dtype=jp.float32),
            jp.array(self.cfg.initial_pose_noise, dtype=jp.float32),
            jp.array(self.cfg.smoothness_coef, dtype=jp.float32),
            jp.array(self.cfg.action_coef, dtype=jp.float32),
            jp.array(self.cfg.body_contact_penalty, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
        )

    def reset_with_curriculum(
        self,
        rng: jax.Array,
        goal_forward: jax.Array,
        goal_lateral: jax.Array,
        goal_reach_radius: jax.Array,
        initial_pose_noise: jax.Array,
        smoothness_coef: jax.Array,
        action_coef: jax.Array,
        body_contact_penalty: jax.Array,
        success_rate: jax.Array,
    ) -> State:
        rng, pose_rng, servo_rng, qvel_rng, goal_rng, phase_rng = jax.random.split(rng, 6)
        q_cmd = self._q_stand + jax.random.uniform(
            pose_rng,
            (self.num_actions,),
            minval=-initial_pose_noise,
            maxval=initial_pose_noise,
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

        goal_pos_world, goal_bearing, goal_range, goal_reached = self._sample_goal(
            goal_rng,
            pipeline_state,
            goal_forward,
            goal_lateral,
            goal_reach_radius,
        )
        q_cmd_history = jp.repeat((q_cmd - self._q_stand)[None, :], self.cfg.history_len, axis=0)
        action_history = jp.zeros((self.cfg.history_len, self.num_actions), dtype=jp.float32)
        phase = jax.random.uniform(phase_rng, (), minval=0.0, maxval=2.0 * math.pi)
        zero = jp.array(0.0, dtype=jp.float32)
        info = {
            "rng": rng,
            "q_cmd": q_cmd,
            "last_action": jp.zeros(self.num_actions, dtype=jp.float32),
            "q_cmd_history": q_cmd_history,
            "action_history": action_history,
            "goal_pos_world": goal_pos_world,
            "goal_bearing": goal_bearing,
            "goal_range": goal_range,
            "goal_reached": goal_reached,
            "phase": phase,
            "goal_forward": goal_forward,
            "goal_lateral": goal_lateral,
            "goal_reach_radius": goal_reach_radius,
            "initial_pose_noise": initial_pose_noise,
            "smoothness_coef": smoothness_coef,
            "action_coef": action_coef,
            "body_contact_penalty": body_contact_penalty,
            "success_rate": success_rate,
        }
        measurements = self._measurements(pipeline_state, goal_pos_world, goal_reach_radius)
        obs = self._get_obs(info)
        metrics = self._metrics(info, measurements, zero, zero, zero, zero, zero)
        metrics["reward"] = zero
        return State(pipeline_state, obs, zero, zero, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0)
        info = dict(state.info)
        previous_action = info["last_action"]
        q_cmd = jp.clip(info["q_cmd"] + action * self._action_scale, self._q_cmd_min, self._q_cmd_max)
        pipeline_state = self.pipeline_step(state.pipeline_state, q_cmd)
        phase = (info["phase"] + 2.0 * math.pi * self.cfg.gait_frequency * self.cfg.control_dt) % (2.0 * math.pi)

        measurements = self._measurements(pipeline_state, info["goal_pos_world"], info["goal_reach_radius"])
        reward, reward_terms = self._reward(action, previous_action, info, measurements)
        done = jp.maximum(measurements["fall"], measurements["goal_reached"])

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
                "goal_bearing": measurements["goal_bearing"],
                "goal_range": measurements["goal_range"],
                "goal_reached": measurements["goal_reached"],
                "phase": phase,
            }
        )
        obs = self._get_obs(info)
        metrics = self._metrics(
            info,
            measurements,
            reward_terms["goal_reward"],
            reward_terms["smoothness"],
            reward_terms["action_mag"],
            reward_terms["idle_penalty"],
            reward_terms["contact_penalty"],
        )
        metrics["reward"] = reward
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def _get_obs(self, info: dict[str, jax.Array]) -> jax.Array:
        return jp.concatenate(
            [
                info["q_cmd_history"].reshape(-1),
                info["action_history"].reshape(-1),
                (info["goal_bearing"] / math.pi)[None],
                (info["goal_range"] / self._target_goal_range)[None],
                jp.sin(info["phase"])[None],
                jp.cos(info["phase"])[None],
            ]
        )

    def _measurements(
        self,
        pipeline_state: Any,
        goal_pos_world: jax.Array,
        goal_reach_radius: jax.Array,
    ) -> dict[str, jax.Array]:
        quat = pipeline_state.xquat[self._base_body_id]
        qd = pipeline_state.qd
        linear_world = qd[self._root_dofadr : self._root_dofadr + 3]
        angular_world = qd[self._root_dofadr + 3 : self._root_dofadr + 6]
        linear_body = brax_math.rotate(linear_world, brax_math.quat_inv(quat))
        angular_body = brax_math.rotate(angular_world, brax_math.quat_inv(quat))
        base_velocity_body = jp.array([linear_body[0], linear_body[1], angular_body[2]], dtype=jp.float32)

        base_xmat = pipeline_state.xmat[self._base_body_id]
        base_z = pipeline_state.xpos[self._base_body_id, 2]
        roll_pitch = self._roll_pitch_from_xmat(base_xmat)
        foot_contacts, leg_contact, lower_leg_contact, body_contact = self._contact_flags(pipeline_state)
        goal_bearing, goal_range, goal_reached = self._goal_signals(pipeline_state, goal_pos_world, goal_reach_radius)
        fall = jp.where(base_z < self.cfg.fall_height, 1.0, 0.0)
        return {
            "base_velocity_body": base_velocity_body,
            "base_z": base_z,
            "roll_pitch": roll_pitch,
            "foot_contacts": foot_contacts,
            "leg_contact": leg_contact,
            "lower_leg_contact": lower_leg_contact,
            "body_contact": body_contact,
            "goal_bearing": goal_bearing,
            "goal_range": goal_range,
            "goal_reached": goal_reached,
            "fall": fall,
        }

    def _reward(
        self,
        action: jax.Array,
        previous_action: jax.Array,
        info: dict[str, jax.Array],
        measurements: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        goal_reward = -jp.tanh(measurements["goal_range"] / self.cfg.goal_reward_scale)
        smoothness = jp.sum((action - previous_action) ** 2)
        action_mag = jp.sum(action**2)
        idle_action = jp.all(jp.abs(action) <= self.cfg.idle_action_tolerance)
        idle_penalty = self.cfg.idle_action_penalty * idle_action.astype(jp.float32) * (1.0 - measurements["goal_reached"])
        contact_penalty = (
            self.cfg.leg_contact_penalty * measurements["leg_contact"]
            + self.cfg.lower_leg_contact_penalty * measurements["lower_leg_contact"]
            + info["body_contact_penalty"] * measurements["body_contact"]
        )
        reward = (
            goal_reward
            - info["smoothness_coef"] * smoothness
            - info["action_coef"] * action_mag
            - idle_penalty
            - contact_penalty
            - self.cfg.fall_penalty * measurements["fall"]
        )
        return reward, {
            "goal_reward": goal_reward,
            "smoothness": smoothness,
            "action_mag": action_mag,
            "idle_penalty": idle_penalty,
            "contact_penalty": contact_penalty,
        }

    def _metrics(
        self,
        info: dict[str, jax.Array],
        measurements: dict[str, jax.Array],
        goal_reward: jax.Array,
        smoothness: jax.Array,
        action_mag: jax.Array,
        idle_penalty: jax.Array,
        contact_penalty: jax.Array,
    ) -> dict[str, jax.Array]:
        return {
            "mean_goal_range": measurements["goal_range"],
            "goal_reached": measurements["goal_reached"],
            "leg_contact": measurements["leg_contact"],
            "lower_leg_contact": measurements["lower_leg_contact"],
            "body_contact": measurements["body_contact"],
            "fall": measurements["fall"],
            "reward_goal": goal_reward,
            "smoothness": smoothness,
            "action_mag": action_mag,
            "idle_penalty": idle_penalty,
            "contact_penalty": contact_penalty,
            "success_rate": info["success_rate"],
            "curriculum_goal_forward": info["goal_forward"],
            "curriculum_goal_lateral": info["goal_lateral"],
            "curriculum_goal_reach_radius": info["goal_reach_radius"],
        }

    def _sample_goal(
        self,
        rng: jax.Array,
        pipeline_state: Any,
        goal_forward: jax.Array,
        goal_lateral: jax.Array,
        goal_reach_radius: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        forward_rng, lateral_rng = jax.random.split(rng)
        forward = jax.random.uniform(forward_rng, (), minval=self.cfg.initial_goal_forward, maxval=goal_forward)
        lateral = jax.random.uniform(lateral_rng, (), minval=-goal_lateral, maxval=goal_lateral)
        mat = pipeline_state.xmat[self._base_body_id].reshape(3, 3)
        base_xy = pipeline_state.xpos[self._base_body_id, :2]
        forward_axis = -mat[:2, 1]
        left_axis = mat[:2, 0]
        goal_pos_world = base_xy + forward * forward_axis + lateral * left_axis
        goal_bearing, goal_range, goal_reached = self._goal_signals(pipeline_state, goal_pos_world, goal_reach_radius)
        return goal_pos_world, goal_bearing, goal_range, goal_reached

    def _goal_signals(
        self,
        pipeline_state: Any,
        goal_pos_world: jax.Array,
        goal_reach_radius: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        delta = goal_pos_world - pipeline_state.xpos[self._base_body_id, :2]
        goal_angle = jp.arctan2(delta[1], delta[0])
        bearing = self._wrap_to_pi(goal_angle - self._yaw_from_xmat(pipeline_state.xmat[self._base_body_id]))
        goal_range = jp.linalg.norm(delta)
        goal_reached = (goal_range < goal_reach_radius).astype(jp.float32)
        return bearing, goal_range, goal_reached

    def _contact_flags(self, pipeline_state: Any) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        contact = pipeline_state.contact
        if contact is None:
            zero = jp.array(0.0, dtype=jp.float32)
            return jp.zeros(4, dtype=jp.float32), zero, zero, zero

        geom1 = contact.geom1
        geom2 = contact.geom2
        active = contact.dist <= 0.0
        if self._floor_geom_id >= 0:
            active = active & ((geom1 == self._floor_geom_id) | (geom2 == self._floor_geom_id))

        def touches(geom_id: jax.Array) -> jax.Array:
            return jp.any(active & ((geom1 == geom_id) | (geom2 == geom_id))).astype(jp.float32)

        foot_contacts = jax.vmap(touches)(self._foot_geom_ids)
        leg_contact = jp.any(jax.vmap(touches)(self._leg_contact_geom_ids)).astype(jp.float32)
        lower_leg_contact = jp.any(jax.vmap(touches)(self._lower_leg_contact_geom_ids)).astype(jp.float32)
        body_contact = touches(jp.array(self._base_geom_id, dtype=jp.int32))
        return foot_contacts, leg_contact, lower_leg_contact, body_contact

    @staticmethod
    def _roll_pitch_from_xmat(xmat: jax.Array) -> jax.Array:
        mat = xmat.reshape(3, 3)
        roll = jp.arctan2(mat[2, 1], mat[2, 2])
        pitch = jp.arctan2(-mat[2, 0], jp.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        return jp.array([roll, pitch], dtype=jp.float32)

    @staticmethod
    def _yaw_from_xmat(xmat: jax.Array) -> jax.Array:
        mat = xmat.reshape(3, 3)
        return jp.arctan2(-mat[1, 1], -mat[0, 1])

    @staticmethod
    def _wrap_to_pi(angle: jax.Array) -> jax.Array:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

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
