from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


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


@dataclass
class CrankBotWalkEnvConfig:
    xml_path: str = "scene.xml"
    num_envs: int = 32
    seed: int = 1
    simulation_dt: float = 0.002
    control_dt: float = 0.1
    decimation: int = 50
    max_episode_steps: int = 80
    history_len: int = 4
    gait_frequency: float = 1.0
    initial_goal_forward: float = 0.25
    target_goal_forward: float = 0.60
    initial_goal_lateral: float = 0.0
    target_goal_lateral: float = 0.25
    goal_reach_radius: float = 0.20
    target_goal_reach_radius: float = 0.10
    goal_reward_scale: float = 0.25
    goal_site_name: str = "goal_site"
    goal_site_z: float = 0.02
    initial_pose_noise: float = 0.10
    target_initial_pose_noise: float = 0.35
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

    def to_dict(self) -> dict:
        return asdict(self)


class CrankBotWalkEnv(VecEnv):
    num_actions = 8
    num_obs = 68
    num_privileged_obs = 80

    def __init__(self, cfg: CrankBotWalkEnvConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("CrankBotWalkEnv runs MuJoCo on CPU; keep env device='cpu'.")

        self.num_envs = cfg.num_envs
        self.max_episode_length = cfg.max_episode_steps
        self.rng = np.random.default_rng(cfg.seed)

        xml_path = Path(cfg.xml_path).expanduser()
        if not xml_path.is_absolute():
            xml_path = Path(__file__).resolve().parents[1] / xml_path
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = cfg.simulation_dt
        self.data = [mujoco.MjData(self.model) for _ in range(self.num_envs)]

        self.root_joint_id = self._joint_id("root_freejoint")
        self.root_qposadr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_dofadr = int(self.model.jnt_dofadr[self.root_joint_id])
        self.base_body_id = self._body_id("base")
        self.base_geom_id = self._geom_id("base_collision")
        self.floor_geom_id = self._optional_geom_id("floor")
        self.goal_site_id = self._optional_site_id(cfg.goal_site_name)
        self.foot_geom_ids = np.array([self._geom_id(name) for name in FOOT_GEOM_NAMES], dtype=np.int32)
        self.leg_contact_geom_ids = np.array(
            [self._geom_id(name) for name in LEG_CONTACT_GEOM_NAMES],
            dtype=np.int32,
        )
        self.lower_leg_contact_geom_ids = np.array(
            [self._geom_id(name) for name in LOWER_LEG_CONTACT_GEOM_NAMES],
            dtype=np.int32,
        )

        self.joint_ids = np.array([self._joint_id(name) for name in JOINT_NAMES], dtype=np.int32)
        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.joint_ranges = self.model.jnt_range[self.joint_ids].astype(np.float64)

        self.q_stand = np.array([-1.15, -2.34, 1.15, 2.34, 2.0, -2.34, -2.0, 2.34], dtype=np.float64)
        self.action_scale = np.array([0.07, 0.10, 0.07, 0.10, 0.07, 0.10, 0.07, 0.10], dtype=np.float64)
        self.q_cmd_min = np.maximum(self.joint_ranges[:, 0], self.q_stand - cfg.command_delta_limit)
        self.q_cmd_max = np.minimum(self.joint_ranges[:, 1], self.q_stand + cfg.command_delta_limit)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.num_privileged_obs), dtype=torch.float32, device=self.device
        )
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.q_cmd = np.repeat(self.q_stand[None, :], self.num_envs, axis=0)
        self.last_action = np.zeros((self.num_envs, self.num_actions), dtype=np.float64)
        self.q_cmd_history = np.zeros((self.num_envs, cfg.history_len, self.num_actions), dtype=np.float64)
        self.action_history = np.zeros_like(self.q_cmd_history)
        self.goal_pos_world = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.goal_bearing = np.zeros(self.num_envs, dtype=np.float64)
        self.goal_range = np.zeros(self.num_envs, dtype=np.float64)
        self.goal_reached = np.zeros(self.num_envs, dtype=bool)
        self.phase = np.zeros(self.num_envs, dtype=np.float64)
        self.base_velocity_body = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.base_yaw = np.zeros(self.num_envs, dtype=np.float64)
        self.base_z = np.zeros(self.num_envs, dtype=np.float64)
        self.roll_pitch = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.foot_contacts = np.zeros((self.num_envs, 4), dtype=np.float64)
        self.leg_contact = np.zeros(self.num_envs, dtype=np.float64)
        self.lower_leg_contact = np.zeros(self.num_envs, dtype=np.float64)
        self.body_contact = np.zeros(self.num_envs, dtype=np.float64)
        self.fall = np.zeros(self.num_envs, dtype=np.float64)
        self.success_history: deque[bool] = deque(maxlen=100)
        self.target_goal_range = max(math.hypot(cfg.target_goal_forward, cfg.target_goal_lateral), 1e-6)

        self.current_goal_forward = cfg.initial_goal_forward
        self.current_goal_lateral = cfg.initial_goal_lateral
        self.current_goal_reach_radius = cfg.goal_reach_radius
        self.current_initial_pose_noise = cfg.initial_pose_noise
        self.current_smoothness_coef = cfg.smoothness_coef
        self.current_action_coef = cfg.action_coef
        self.current_body_contact_penalty = cfg.body_contact_penalty

        self.reset()

    def reset(self) -> TensorDict:
        self.reset_idx(np.arange(self.num_envs, dtype=np.int32))
        self._fill_observation_buffers()
        return self.get_observations()

    def reset_idx(self, env_ids: np.ndarray) -> None:
        if env_ids.size == 0:
            return

        for env_id in env_ids:
            data = self.data[int(env_id)]
            mujoco.mj_resetData(self.model, data)

            q_cmd = self.q_stand + self.rng.uniform(
                -self.current_initial_pose_noise, self.current_initial_pose_noise, self.num_actions
            )
            q_cmd = np.clip(q_cmd, self.q_cmd_min, self.q_cmd_max)
            q_real = q_cmd + self.rng.uniform(-self.cfg.servo_position_error, self.cfg.servo_position_error, 8)
            q_real = np.clip(q_real, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

            data.qpos[:] = self.model.qpos0
            data.qpos[self.root_qposadr + 2] = self.model.qpos0[self.root_qposadr + 2]
            data.qpos[self.joint_qposadr] = q_real
            data.qvel[:] = self.rng.uniform(-0.05, 0.05, self.model.nv)
            data.qvel[self.root_dofadr : self.root_dofadr + 6] = 0.0
            data.ctrl[:] = q_cmd
            mujoco.mj_forward(self.model, data)
            self.base_yaw[env_id] = self._yaw_from_xmat(data.xmat[self.base_body_id])

            self.q_cmd[env_id] = q_cmd
            self.last_action[env_id] = 0.0
            self.q_cmd_history[env_id] = q_cmd - self.q_stand
            self.action_history[env_id] = 0.0
            self._sample_goal(int(env_id), data)
            self.phase[env_id] = self.rng.uniform(0.0, 2.0 * math.pi)
            self.goal_reached[env_id] = False

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.episode_length_buf[ids] = 0
        self.reset_buf[ids] = False
        self._update_goal_signals(env_ids)
        self._update_goal_site()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = actions.detach().cpu().numpy().astype(np.float64)
        actions_np = np.clip(actions_np, -1.0, 1.0)

        previous_action = self.last_action.copy()
        self.q_cmd = np.clip(self.q_cmd + actions_np * self.action_scale, self.q_cmd_min, self.q_cmd_max)
        for env_id, data in enumerate(self.data):
            data.ctrl[:] = self.q_cmd[env_id]
            for _ in range(self.cfg.decimation):
                mujoco.mj_step(self.model, data)

        self.phase = (self.phase + 2.0 * math.pi * self.cfg.gait_frequency * self.cfg.control_dt) % (2.0 * math.pi)
        self.episode_length_buf += 1

        self._update_measurements()
        self._compute_rewards(actions_np, previous_action)
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = torch.logical_or(
            torch.logical_or(time_outs, torch.as_tensor(self.fall > 0.0, device=self.device)),
            torch.as_tensor(self.goal_reached, device=self.device),
        )
        self.reset_buf[:] = dones

        self._update_histories(actions_np)
        self._update_success_and_curriculum(dones.cpu().numpy().astype(bool))
        extras = self._make_extras(time_outs)

        done_ids = dones.nonzero(as_tuple=False).flatten().cpu().numpy().astype(np.int32)
        if done_ids.size:
            self.reset_idx(done_ids)

        self._fill_observation_buffers()
        self.last_action = actions_np
        self.last_action[done_ids] = 0.0
        return self.get_observations(), self.rew_buf.clone(), dones.clone(), extras

    def get_observations(self) -> TensorDict:
        return TensorDict(
            {"actor": self.obs_buf.clone(), "critic": self.privileged_obs_buf.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf.clone()

    def _update_measurements(self) -> None:
        velocity = np.zeros(6, dtype=np.float64)
        for env_id, data in enumerate(self.data):
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, velocity, True
            )
            base_xmat = data.xmat[self.base_body_id]
            self.base_velocity_body[env_id] = (velocity[3], velocity[4], velocity[2])
            self.base_yaw[env_id] = self._yaw_from_xmat(base_xmat)
            self.base_z[env_id] = data.xpos[self.base_body_id, 2]
            self.roll_pitch[env_id] = self._roll_pitch_from_xmat(base_xmat)
            foot_contacts, leg_contact, lower_leg_contact, body_contact = self._contact_flags(data)
            self.foot_contacts[env_id] = foot_contacts
            self.leg_contact[env_id] = float(leg_contact)
            self.lower_leg_contact[env_id] = float(lower_leg_contact)
            self.body_contact[env_id] = float(body_contact)
            self.fall[env_id] = float(self.base_z[env_id] < self.cfg.fall_height)
        self._update_goal_signals(np.arange(self.num_envs, dtype=np.int32))

    def _compute_rewards(self, actions: np.ndarray, previous_action: np.ndarray) -> None:
        goal_reward = -np.tanh(self.goal_range / self.cfg.goal_reward_scale)
        smoothness = np.sum((actions - previous_action) ** 2, axis=1)
        action_mag = np.sum(actions**2, axis=1)
        idle_action = np.all(np.abs(actions) <= self.cfg.idle_action_tolerance, axis=1)
        idle_penalty = self.cfg.idle_action_penalty * idle_action * ~self.goal_reached
        reward = (
            goal_reward
            - self.current_smoothness_coef * smoothness
            - self.current_action_coef * action_mag
            - idle_penalty
            - self.cfg.leg_contact_penalty * self.leg_contact
            - self.cfg.lower_leg_contact_penalty * self.lower_leg_contact
            - self.current_body_contact_penalty * self.body_contact
            - self.cfg.fall_penalty * self.fall
        )
        self.rew_buf[:] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)

    def _update_histories(self, actions: np.ndarray) -> None:
        self.q_cmd_history[:, 1:] = self.q_cmd_history[:, :-1]
        self.q_cmd_history[:, 0] = self.q_cmd - self.q_stand
        self.action_history[:, 1:] = self.action_history[:, :-1]
        self.action_history[:, 0] = actions

    def _update_success_and_curriculum(self, dones: np.ndarray) -> None:
        for env_id in np.nonzero(dones)[0]:
            self.success_history.append(bool(self.goal_reached[env_id]))

        if len(self.success_history) == self.success_history.maxlen and np.mean(self.success_history) > 0.75:
            self.current_goal_forward = min(self.cfg.target_goal_forward, self.current_goal_forward * 1.15)
            self.current_goal_lateral = min(self.cfg.target_goal_lateral, self.current_goal_lateral + 0.03)
            self.current_goal_reach_radius = max(
                self.cfg.target_goal_reach_radius, self.current_goal_reach_radius - 0.01
            )
            self.current_initial_pose_noise = min(
                self.cfg.target_initial_pose_noise, self.current_initial_pose_noise + 0.03
            )
            self.current_smoothness_coef = min(self.cfg.target_smoothness_coef, self.current_smoothness_coef + 0.00025)
            self.current_action_coef = min(self.cfg.target_action_coef, self.current_action_coef + 0.00005)
            self.current_body_contact_penalty = min(
                self.cfg.target_body_contact_penalty, self.current_body_contact_penalty + 0.05
            )
            self.success_history.clear()

    def _fill_observation_buffers(self) -> None:
        actor_obs = np.concatenate(
            (
                self.q_cmd_history.reshape(self.num_envs, -1),
                self.action_history.reshape(self.num_envs, -1),
                (self.goal_bearing[:, None] / math.pi),
                (self.goal_range[:, None] / self.target_goal_range),
                np.sin(self.phase)[:, None],
                np.cos(self.phase)[:, None],
            ),
            axis=1,
        )
        critic_obs = np.concatenate(
            (
                actor_obs,
                self.base_velocity_body,
                self.base_z[:, None],
                self.roll_pitch,
                self.foot_contacts,
                self.leg_contact[:, None],
                self.body_contact[:, None],
            ),
            axis=1,
        )
        self.obs_buf[:] = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        self.privileged_obs_buf[:] = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)

    def _make_extras(self, time_outs: torch.Tensor) -> dict:
        success_rate = float(np.mean(self.success_history)) if self.success_history else 0.0
        return {
            "time_outs": time_outs.to(dtype=torch.float32, device=self.device),
            "log": {
                "/env/mean_goal_range": torch.as_tensor(self.goal_range.mean(), device=self.device),
                "/env/goal_reached": torch.as_tensor(self.goal_reached.mean(), device=self.device),
                "/env/leg_contact": torch.as_tensor(self.leg_contact.mean(), device=self.device),
                "/env/lower_leg_contact": torch.as_tensor(self.lower_leg_contact.mean(), device=self.device),
                "/env/body_contact": torch.as_tensor(self.body_contact.mean(), device=self.device),
                "/env/success_rate": torch.as_tensor(success_rate, device=self.device),
                "/curriculum/goal_forward": torch.as_tensor(self.current_goal_forward, device=self.device),
                "/curriculum/goal_lateral": torch.as_tensor(self.current_goal_lateral, device=self.device),
                "/curriculum/goal_reach_radius": torch.as_tensor(
                    self.current_goal_reach_radius, device=self.device
                ),
            },
        }

    def _sample_goal(self, env_id: int, data: mujoco.MjData) -> None:
        forward = self.rng.uniform(self.cfg.initial_goal_forward, self.current_goal_forward)
        lateral = 0.0
        if self.current_goal_lateral > 0.0:
            lateral = self.rng.uniform(-self.current_goal_lateral, self.current_goal_lateral)

        base_xmat = data.xmat[self.base_body_id].reshape(3, 3)
        base_xy = data.xpos[self.base_body_id, :2]
        forward_axis = -base_xmat[:2, 1]
        left_axis = base_xmat[:2, 0]
        self.goal_pos_world[env_id] = base_xy + forward * forward_axis + lateral * left_axis

    def _update_goal_signals(self, env_ids: np.ndarray) -> None:
        for env_id in env_ids:
            data = self.data[int(env_id)]
            delta = self.goal_pos_world[env_id] - data.xpos[self.base_body_id, :2]
            goal_angle = math.atan2(delta[1], delta[0])
            self.goal_bearing[env_id] = self._wrap_to_pi(goal_angle - self.base_yaw[env_id])
            self.goal_range[env_id] = float(np.linalg.norm(delta))
            self.goal_reached[env_id] = self.goal_range[env_id] < self.current_goal_reach_radius

    def _update_goal_site(self) -> None:
        if self.goal_site_id < 0 or self.num_envs == 0:
            return

        self.model.site_pos[self.goal_site_id] = (
            self.goal_pos_world[0, 0],
            self.goal_pos_world[0, 1],
            self.cfg.goal_site_z,
        )
        mujoco.mj_forward(self.model, self.data[0])

    def _contact_flags(self, data: mujoco.MjData) -> tuple[np.ndarray, bool, bool, bool]:
        foot_contacts = np.zeros(4, dtype=np.float64)
        leg_contact = False
        lower_leg_contact = False
        body_contact = False
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            geom_pair = (contact.geom1, contact.geom2)
            if self.floor_geom_id >= 0 and self.floor_geom_id not in geom_pair:
                continue
            for foot_idx, foot_geom_id in enumerate(self.foot_geom_ids):
                if foot_geom_id in geom_pair:
                    foot_contacts[foot_idx] = 1.0
            leg_contact = leg_contact or any(leg_geom_id in geom_pair for leg_geom_id in self.leg_contact_geom_ids)
            lower_leg_contact = lower_leg_contact or any(
                leg_geom_id in geom_pair for leg_geom_id in self.lower_leg_contact_geom_ids
            )
            body_contact = body_contact or self.base_geom_id in geom_pair
        return foot_contacts, leg_contact, lower_leg_contact, body_contact

    @staticmethod
    def _roll_pitch_from_xmat(xmat: np.ndarray) -> np.ndarray:
        mat = xmat.reshape(3, 3)
        roll = math.atan2(mat[2, 1], mat[2, 2])
        pitch = math.atan2(-mat[2, 0], math.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
        return np.array([roll, pitch], dtype=np.float64)

    def _joint_id(self, name: str) -> int:
        return self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name)

    def _body_id(self, name: str) -> int:
        return self._required_id(mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom_id(self, name: str) -> int:
        return self._required_id(mujoco.mjtObj.mjOBJ_GEOM, name)

    def _optional_geom_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _optional_site_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _required_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Could not find MuJoCo object '{name}' in {self.xml_path}.")
        return obj_id

    @staticmethod
    def _yaw_from_xmat(xmat: np.ndarray) -> float:
        mat = xmat.reshape(3, 3)
        return math.atan2(-mat[1, 1], -mat[0, 1])

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
