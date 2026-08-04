"""
SharedworkCellEnv: ambiente Gymnasium sopra SharedworkCellSim.

Le osservazioni seguono la convenzione "pixels" (dict camera->immagine) +
"agent_pos" (stato proprioceptivo) usata da molti ambienti manipolazione
(es. gym-pusht, gym-aloha) e riconosciuta direttamente da
`lerobot.envs.utils.preprocess_observation` -- quindi questo stesso ambiente
funziona sia per RL "puro" (SAC ecc., tipicamente in modalita' image_obs=False,
solo stato) sia come front-end per valutare/deployare policy LeRobot
(ACT, VLA, ...) in modalita' image_obs=True.

La reward di default e' 0.0 sempre: il task/reward reale dipende da cosa
Manuel vuole allenare (reach? pick&place?) e va definito a parte, per
esempio passando `reward_fn` al costruttore o sottoclassando `compute_reward`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .sim import SharedworkCellSim, ALL_CAMERAS


class SharedworkCellEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        image_obs: bool = False,
        cameras: Sequence[str] = ALL_CAMERAS,
        camera_width: int = 640,
        camera_height: int = 480,
        n_substeps: int = 5,
        max_episode_steps: int = 500,
        reward_fn: Optional[Callable[["SharedworkCellEnv"], float]] = None,
        render_camera: str = "wrist_d435",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.sim = SharedworkCellSim(camera_width=camera_width, camera_height=camera_height)
        self.image_obs = image_obs
        self.cameras = list(cameras)
        self.n_substeps = n_substeps
        self.max_episode_steps = max_episode_steps
        self._reward_fn = reward_fn
        self._render_camera = render_camera
        self._elapsed_steps = 0

        # ctrlrange reale del modello: uso per rescalare l'azione, ma lo
        # spazio ESPOSTO all'agente e' normalizzato in [-1,1] su ogni
        # dimensione, come raccomandato per SAC/algoritmi SB3 (vedi
        # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
        self._ctrl_low = self.sim.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self._ctrl_high = self.sim.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # bound reali per lo stato (6 joint range braccio + range gripper),
        # invece di -inf/+inf: piu' informativo per l'agente e niente
        # warning dal checker gymnasium
        arm_low = self.sim.model.jnt_range[self.sim._arm_jids, 0]
        arm_high = self.sim.model.jnt_range[self.sim._arm_jids, 1]
        grip_low = self.sim.model.jnt_range[self.sim._gripper_jids[0], 0]
        grip_high = self.sim.model.jnt_range[self.sim._gripper_jids[0], 1]
        state_low = np.concatenate([arm_low, [grip_low]]).astype(np.float32)
        state_high = np.concatenate([arm_high, [grip_high]]).astype(np.float32)
        agent_pos_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        if self.image_obs:
            img_space = spaces.Box(low=0, high=255, shape=(camera_height, camera_width, 3), dtype=np.uint8)
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({cam: img_space for cam in self.cameras}),
                "agent_pos": agent_pos_space,
            })
        else:
            self.observation_space = agent_pos_space

    # ------------------------------------------------------------------
    def _get_obs(self):
        state = self.sim.get_state_vector()
        if not self.image_obs:
            return state
        pixels = {cam: self.sim.render(camera=cam) for cam in self.cameras}
        return {"pixels": pixels, "agent_pos": state}

    def _get_info(self):
        return {"elapsed_steps": self._elapsed_steps}

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        qpos = (options or {}).get("qpos")
        self.sim.reset(qpos=qpos, seed=seed)
        self._elapsed_steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == self.action_space.shape, (
            f"azione di shape {action.shape}, attesa {self.action_space.shape}"
        )
        action = np.clip(action, -1.0, 1.0)
        # da [-1,1] normalizzato al ctrlrange reale di ciascun attuatore
        ctrl = self._ctrl_low + (action + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)
        self.sim.step(ctrl, n_substeps=self.n_substeps)
        self._elapsed_steps += 1

        reward = self._reward_fn(self) if self._reward_fn is not None else 0.0
        terminated = False  # nessun criterio di successo/fallimento definito di default
        truncated = self._elapsed_steps >= self.max_episode_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.sim.render(camera=self._render_camera)

    def close(self):
        pass
