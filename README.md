# sharework-mujoco

MuJoCo model of the sharework cell (UR10e + Robotiq 2F-85 + 3 RealSense:
wrist, rs1, rs2), usable in three ways:

1. **Standalone MuJoCo** — script/notebook, manual teleop, visual debugging.
2. **Gymnasium** — for RL (SAC and other stable-baselines3/other libraries algorithms).
3. **LeRobot** — as an evaluation front-end for already-trained ACT/VLA policies.

## Installation

```bash
pip install -e .                    # base (mujoco, gymnasium, numpy)
pip install -e ".[sac]"             # + stable-baselines3
pip install -e ".[lerobot]"         # + lerobot
pip install -e ".[live-view]"       # + opencv-python (for the live wrist camera window)
```

## 1. Standalone usage

```bash
python3 examples/standalone_view.py                    # interactive viewer
python3 examples/standalone_view.py --live              # 3D viewer + live wrist camera window
python3 examples/standalone_view.py --wrist-cam out.png # snapshot from the wrist camera
```

Or in Python:

```python
from sharework_mujoco.sim import SharedworkCellSim

sim = SharedworkCellSim()
sim.step(sim.data.ctrl)          # one physics step
img = sim.render(camera="rs1_d435")
state = sim.get_state_vector()   # 6 arm joints + gripper opening
```

For a standalone MJCF to use with `simulate` or other tools (not Python):

```python
from sharework_mujoco.scene_builder import build_and_save_xml
build_and_save_xml("sharework_cell_mujoco.xml")
```

## 2. Usage with Gymnasium (RL)

Two registered environments:

- `SharedworkCell-v0` — observation = state only (7-dim: 6 joints + gripper).
- `SharedworkCellVision-v0` — observation = `{"pixels": {cam: image}, "agent_pos": state}`.

```python
import gymnasium as gym
import sharework_mujoco  # registers the envs

env = gym.make("SharedworkCell-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Action: `Box(-1, 1, shape=(7,))`, normalized; internally remapped to each
actuator's actual `ctrlrange` (read from the model, not hardcoded — the
gripper, for instance, is `[0, 255]`).

### Reward

By default the reward is always `0.0` — there is no predefined task, it
must be defined by the environment user by passing a `reward_fn`:

```python
Callable[[SharedworkCellEnv], float]
```

It is called on every `step()`, AFTER physics has advanced, with the
environment itself as the only argument. Inside it you can read any state
via `env.sim` (an instance of `SharedworkCellSim`, see [sim.py](src/sharework_mujoco/sim.py)):

- `env.sim.data` / `env.sim.model` — raw MuJoCo mjData/mjModel (site/body
  positions, sensors, contacts, etc.)
- `env.sim.arm_qpos`, `env.sim.arm_qvel`, `env.sim.gripper_qpos` — already
  extracted proprioceptive state
- `env._elapsed_steps` — current step in the episode, useful for
  time-dependent rewards

Two ways to hook it up:

```python
# 1) at the constructor (recommended — gym.make() forwards extra kwargs
#    to the entry_point registered in __init__.py)
env = gym.make("SharedworkCell-v0", reward_fn=your_reward_fn)

# 2) on an already-created env (note: .unwrapped, because gym.make() wraps
#    the env in TimeLimit/OrderEnforcing and the attribute must go on the
#    actual instance)
env.unwrapped._reward_fn = your_reward_fn
```

Minimal example (reaching: bring the TCP close to a fixed point), the
same one used in `examples/train_sac.py`:

```python
import mujoco
import numpy as np

def reaching_reward(env) -> float:
    model, data = env.sim.model, env.sim.data
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur10e_attachment_site")
    tcp_pos = data.site_xpos[tcp_id]
    target = np.array([0.3, 0.0, 0.875 + 0.3])  # above the table, see fixed_parts.xml
    return -np.linalg.norm(tcp_pos - target)
```

**Limitations to keep in mind**: `reward_fn` only controls the reward, not
`terminated`/`truncated` — in the current code `terminated` is always
`False` and `truncated` depends only on the number of steps
(`max_episode_steps`). If you need to end the episode on success/failure
(e.g. task solved, object dropped), you must subclass `SharedworkCellEnv`
and override `step()`.

See `examples/train_sac.py` for a complete example (purely demonstrative
reaching reward, to be replaced with your actual task).

Validated with `gymnasium.utils.env_checker.check_env` (no errors/warnings)
on both variants.

## 3. Usage with LeRobot (ACT / VLA)

The observations of `SharedworkCellVision-v0` follow the same
`pixels`/`agent_pos` convention that `lerobot.envs.utils.preprocess_observation`
expects (verified against the lerobot 0.6.1 source), so no dedicated
adapter is needed. See `examples/lerobot_eval.py`.

**Not tested end-to-end** in this development environment (lerobot/torch
require more disk space than was available in the sandbox it was written
in) — read the comments at the top of the script before using it, in
particular about the difference between the minimal inference pipeline
shown there and the more complete one (`PolicyProcessorPipeline`) that
LeRobot uses for real deployment.

## Structure

```
src/sharework_mujoco/
├── scene_builder.py   # builds the MjSpec (fixed_parts + ur10e + 2f85 + cameras)
├── sim.py             # standalone wrapper (reset/step/render)
├── gym_env.py          # SharedworkCellEnv(gymnasium.Env)
└── assets/             # fixed_parts.xml, menagerie source xml, meshes
```

## Technical notes / assumptions to verify

- **Camera orientation** (wrist/rs1/rs2): mount(REP-103)->optical rotation
  computed from the confirmed ROS2 convention (`wrist_link -> color_optical_frame`,
  rpy `-pi/2,0,-pi/2`), also applied to rs1/rs2 assuming the same standard
  REP-103 convention (reasonable for RealSense, but not verified against
  an actual photo of rs1/rs2 specifically).
- **open_tip/closed_tip**: offsets (0.149/0.163 m) calibrated on the real
  Robotiq mesh + custom coupler, not on the menagerie 2f85 mesh (slightly
  different stack-up, ~0.156m). Good as a first reference, recalibrate by
  eye if you need precise values for grasping.
- **RL reward**: no real task/reward defined, needs to be added.
