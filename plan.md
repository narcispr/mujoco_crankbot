# mplementation plan for rsl_rl + PPO + privileged_obs

## 1. Goal

Implement a MuJoCo CPU vectorized environment for the 8-servo quadruped robot using:

- rsl_rl
- PPO
- asymmetric actor-critic
- actor observations based only on information available on the real robot
- privileged critic observations using simulator-only state

The robot already exists as a MuJoCo XML model at file crank_bot.xml with:

- free root body
- 8 position-controlled joints
- visualization meshed geoms
- 2 x contact geoms per leg + 1 for the body

The environment will train a policy to track commanded forward velocity and yaw velocity.

## 2. Simulation timing

Use MuJoCo at:

simulation_dt = 0.002 s
simulation_frequency = 500 Hz

Use policy/control at:

dt_control = 0.1 s
control_frequency = 10 Hz
decimation = 50

That means:

1 policy step = 50 MuJoCo mj_step calls

Episode length:

max_episode_steps = 80
episode_time = 8.0 s

Success is defined if during the last 20 steps the average speed matches (with some error) the desired speed.

Do not terminate on success. Use success only as a curriculum signal.

## 3. Action definition

The action is continuous:

action ∈ [-1, 1]^8

Each action component represents an increment in commanded joint target.

Use different scales for shoulders and elbows:

shoulder_action_scale = 0.07 rad/step
elbow_action_scale    = 0.10 rad/step

Assume a fixed joint ordering, for example:

0 front_left_shoulder
1 front_left_elbow
2 front_right_shoulder
3 front_right_elbow
4 back_left_shoulder
5 back_left_elbow
6 back_right_shoulder
7 back_right_elbow

Define:

action_scale = [
    0.07, 0.10,
    0.07, 0.10,
    0.07, 0.10,
    0.07, 0.10,
]

Update commanded positions:

q_cmd_next = q_cmd_current + action_scale * action

Articulation limits and standing pose (q_stand) are defined as:

- front_left_shoulder => limits: [-pi, 0], standing -1.15
- front_left_elbow => limits: [-pi, 0], standing -2.34
- front_right_shoulder => limits: [0, pi], standing 1.15
- front_right_elbow => limits: [0, pi], standing 2.34
- back_left_shoulder => limits: [0, pi], standing 2
- back_left_elbow => limits: [-pi, 0], standing -2.34
- back_right_shoulder => limits: [-pi, 0], standing -2
- back_right_elbow => limits: [0, pi], standing 2.34

Clip around the standing pose:

q_cmd_next = clip(q_cmd_next, q_stand - q_limit, q_stand + q_limit)

Start with conservative limits:


The MuJoCo position actuators receive:

data.ctrl[:] = q_cmd_next

assuming each actuator ctrlrange is in radians and matches the joint target range.

## 4. Desired commands

At each episode reset, sample one command and keep it constant for the full episode.

Actor receives normalized commands:

- command_vx = vx_des / GLOBAL_MAX_VX
- command_wz = wz_des / GLOBAL_MAX_WZ

where

- vx_des = uniform(-current_max_vx, current_max_vx)
- wz_des = uniform(-current_max_wz, current_max_wz)

Start with small velocities:

current_max_vx = 0.05 m/s
current_max_wz = 0.20 rad/s

Later curriculum may increase them to:

current_max_vx = 0.10 m/s
current_max_wz = 0.35 rad/s

Then:

current_max_vx = 0.15 m/s
current_max_wz = 0.50 rad/s

Define 

GLOBAL_MAX_VX  = 0.15 m/s
GLOBAL_MAX_WZ = 0.50 rad/s


## 5. Gait phase

Maintain a phase variable per environment:

phase += 2.0 * pi * gait_frequency * dt_control
phase = phase % (2.0 * pi)

Initial value at reset:

phase = uniform(0, 2*pi)

Start with:

gait_frequency = 1.0 Hz

Observation uses:

sin_phase = sin(phase)
cos_phase = cos(phase)


## 6. Actor observation

The actor should not receive MuJoCo qpos or qvel directly, because the real MG996/MG90 servos do not provide real feedback.

Use commanded joint history instead:

q_cmd_history, not q_real

Use relative commanded positions:

q_cmd_rel = q_cmd - q_stand

Use history length:

N = 4

Actor observation:

actor_obs =
[
  q_cmd_rel[t],
  q_cmd_rel[t-1],
  q_cmd_rel[t-2],
  q_cmd_rel[t-3],

  action[t-1],
  action[t-2],
  action[t-3],
  action[t-4],

  command_vx, 
  command_wz,

  sin(phase),
  cos(phase)
]

Dimension:

4 * 8 + 4 * 8 + 2 + 2 = 68

Rationale:

q_cmd_history:
  approximates what the real robot knows: what positions were commanded.

action_history:
  gives the policy recent control intent, especially useful when clipping,
  saturation, delays, and different shoulder/elbow scales are present.

command_vx_norm and command_wz_norm:
  tell the policy what the user wants.

phase:
  provides an internal periodic clock that helps discover rhythmic gaits.


## 7. Critic observation

Use privileged information for the critic.

Critic observation:

critic_obs =
[
  actor_obs,

  measured_vx_body,
  measured_vy_body,
  measured_wz_body,

  base_z,
  roll,
  pitch,

  foot_contact_FL,
  foot_contact_FR,
  foot_contact_BL,
  foot_contact_BR,

  body_contact
]

Dimension:

68 + 3 + 3 + 4 + 1 = 79

Where:

measured_vx_body:
  robot forward velocity expressed in body frame

measured_vy_body:
  lateral velocity in body frame

measured_wz_body:
  yaw rate in body frame or world z yaw rate

base_z:
  root/base height

roll, pitch:
  body orientation, excluding yaw

foot contacts:
  booleans or floats from MuJoCo contact detection

body_contact:
  1 if torso/base touches the floor, else 0


Use the body named "base" for the spoeed height and orientation and the spheres named "[front/back]_[left/right]_lower_leg_foot" for the feet contacts.

Rationale:

The critic is allowed to see simulator-only information.
This improves value estimation during PPO training.
The actor will not receive these quantities at deployment time.



## 8. Reward

Use:

evx = (vx_des - vx_measured) / GLOBAL_MAX_VX
ewz = (wz_des - wz_measured) / GLOBAL_MAX_WZ

r_vx = exp(-k_vx * evx**2)
r_wz = exp(-k_wz * ewz**2)


reward = (
    1.0 * r_vx
    + 0.5 * r_wz
    - 0.005 * np.sum((action - last_action)**2)
    - 0.001 * np.sum(action**2)
    - 1.0 * body_contact
    - 2.0 * fall
)

Initial constants:

k_vx = 4.0
k_wz = 2.0



## 9. Body contact penalty


Body contact:

body_contact = base collision geom touches floor

use body_contact as a penalty but not termination.



## 10. Success metric for curriculum

Do not terminate on success initially.

Define success at episode end:

success = (
  not body_contact
  and mean_abs_vx_error_last_20_steps < vx_threshold
  and mean_abs_wz_error_last_20_steps < wz_threshold
)

Initial thresholds:

vx_threshold = 0.025 m/s
wz_threshold = 0.05 rad/s

Later tighten:

vx_threshold = 0.02 m/s
wz_threshold = 0.04 rad/s

Then:

vx_threshold = 0.015 m/s
wz_threshold = 0.035 rad/s

Track:

success_rate = successes_last_100_episodes / 100. Ensure that 100 episodes have already completed.

Use success metric to update all curriculum variables (see 11)

## 11. Adaptive curriculum

Maintain curriculum variables:

current_max_vx
current_max_wz
initial_pose_noise
regularization_scale
success_thresholds

Initial values:

current_max_vx = 0.05
current_max_wz = 0.00
initial_pose_noise = 0.10 rad
smoothness_coef = 0.002
action_coef = 0.0005
body_contact_penalty = 0.5

Target final values:

max_vx = 0.15
max_wz = 0.50
initial_pose_noise = 0.50 rad
smoothness_coef = 0.005
action_coef = 0.001
body_contact_penalty = 1.0

Every time:

success_rate_last_100 > 0.75

increase difficulty slightly:

max_vx *= 1.15
max_wz += 0.05
initial_pose_noise += 0.03
smoothness_coef += small_increment
tighten success thresholds slightly

Clamp to maximum values.


I would not start with backward walking. Start forward only.

## 12. Reset logic

At reset:

q_cmd = q_stand + uniform(-initial_pose_noise, +initial_pose_noise)
q_cmd = clip(q_cmd, q_stand - q_limit, q_stand + q_limit)

Set MuJoCo joint positions close to q_cmd, but optionally add small physical mismatch:

q_real = q_cmd + uniform(-servo_position_error, servo_position_error)

Start with:

servo_position_error = 0.0

Later:

servo_position_error = 0.03 rad

Set velocities to small random values:

qvel = uniform(-0.05, 0.05)

Set root pose:

base at nominal height
small roll/pitch noise
yaw random optional

Initialize histories:

q_cmd_history[:] = q_cmd - q_stand
action_history[:] = 0
phase = uniform(0, 2*pi)



## 13. Environment API for rsl_rl

Implement a vectorized environment class with buffers:

self.obs_buf              # actor obs, shape [num_envs, 68]
self.privileged_obs_buf   # critic obs, shape [num_envs, 79]
self.rew_buf              # shape [num_envs]
self.reset_buf            # shape [num_envs]
self.episode_length_buf   # shape [num_envs]

Required methods conceptually:

reset()
step(actions)
get_observations()
get_privileged_observations()

Depending on exact rsl_rl version, match the expected VecEnv interface.

The step(actions) function:

1. clip actions to [-1, 1]
2. update q_cmd with shoulder/elbow scales
3. clip q_cmd around q_stand
4. write q_cmd to MuJoCo actuator controls
5. run decimation MuJoCo steps
6. compute contacts
7. compute base velocities/orientation
8. compute reward
9. compute done/reset
10. update histories
11. fill obs_buf and privileged_obs_buf
12. reset environments that are done
15. Training configuration

Start with:

num_envs = 32
control_frequency = 10 Hz
episode_steps = 80

PPO rough starting values:

num_steps_per_env = 80
num_learning_epochs = 5
num_mini_batches = 4
gamma = 0.99
lam = 0.95
learning_rate = 3e-4
entropy_coef = 0.01
clip_param = 0.2
desired_kl = 0.01

Actor-critic network:

actor hidden dims:  [256, 256, 128]
critic hidden dims: [256, 256, 128]
activation: ELU or Tanh

Observation normalization is strongly recommended.