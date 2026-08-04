from gymnasium.envs.registration import register

from .sim import SharedworkCellSim
from .gym_env import SharedworkCellEnv
from . import scene_builder

__all__ = ["SharedworkCellSim", "SharedworkCellEnv", "scene_builder"]

register(
    id="SharedworkCell-v0",
    entry_point="sharework_mujoco.gym_env:SharedworkCellEnv",
    max_episode_steps=500,
    kwargs={"image_obs": False},
)

register(
    id="SharedworkCellVision-v0",
    entry_point="sharework_mujoco.gym_env:SharedworkCellEnv",
    max_episode_steps=500,
    kwargs={"image_obs": True},
)
