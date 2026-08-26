"""
SharedworkCellSim: wrapper leggero attorno al modello MuJoCo della cella.

Non dipende da gymnasium: e' pensato per uso "standalone" (script, notebook,
debug, teleop manuale) oltre che come base per SharedworkCellEnv (gym_env.py).
"""
from __future__ import annotations

import numpy as np
import mujoco

from .scene_builder import build_model

ARM_JOINTS = [
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
]
GRIPPER_JOINTS = ["2f85_right_driver_joint", "2f85_left_driver_joint"]

# Posa iniziale reale del sistema ur_robotiq ros2_control (fake hardware).
DEFAULT_INITIAL_QPOS = {
    "ur10e_shoulder_pan_joint": 1.570796327,
    "ur10e_shoulder_lift_joint": -2.443460953,
    "ur10e_elbow_joint": 2.443460953,
    "ur10e_wrist_1_joint": -1.570796327,
    "ur10e_wrist_2_joint": 1.570796327,
    "ur10e_wrist_3_joint": 0.0,
    "2f85_right_driver_joint": 0.0,
    "2f85_left_driver_joint": 0.0,
}

ALL_CAMERAS = ["wrist_d435", "rs1_d435", "rs2_d435"]


class SharedworkCellSim:
    """Simulazione standalone della cella: costruisce il modello una volta
    (in memoria, nessun file scritto su disco), poi espone step/reset/render.
    """

    def __init__(self, initial_qpos: dict[str, float] | None = None,
                 camera_width: int = 640, camera_height: int = 480):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.initial_qpos = dict(DEFAULT_INITIAL_QPOS)
        if initial_qpos:
            self.initial_qpos.update(initial_qpos)

        self._renderer = mujoco.Renderer(self.model, height=camera_height, width=camera_width)

        # id cache
        self._arm_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self._gripper_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in GRIPPER_JOINTS]

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, qpos: dict[str, float] | None = None, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        pose = dict(self.initial_qpos)
        if qpos:
            pose.update(qpos)
        for name, val in pose.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            self.data.qpos[self.model.jnt_qposadr[jid]] = val

        # allineo ctrl al qpos per ogni attuatore di POSIZIONE (biastype
        # affine, es. il gripper), altrimenti al primo step il giunto viene
        # tirato verso ctrl=0. Gli attuatori di COPPIA del braccio (biastype
        # none) restano invece a ctrl=0: coppia nulla = nessuna azione, la
        # gravita' e' gia' compensata via gravcomp nel modello.
        for aid in range(self.model.nu):
            if (self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                    and self.model.actuator_biastype[aid] == mujoco.mjtBias.mjBIAS_AFFINE):
                jid = self.model.actuator_trnid[aid, 0]
                self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]

        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    def step(self, ctrl: np.ndarray, n_substeps: int = 1):
        """ctrl: vettore lungo model.nu (di solito 6 giunti braccio + 1 gripper)."""
        self.data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    def render(self, camera: str = "wrist_d435") -> np.ndarray:
        """Ritorna un'immagine HWC uint8 dalla camera indicata (o dalla
        camera libera di default se camera=None)."""
        self._renderer.update_scene(self.data, camera=camera if camera else -1)
        return self._renderer.render()

    # ------------------------------------------------------------------
    @property
    def arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in self._arm_jids])

    @property
    def arm_qvel(self) -> np.ndarray:
        return np.array([self.data.qvel[self.model.jnt_dofadr[j]] for j in self._arm_jids])

    @property
    def gripper_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self.model.jnt_qposadr[j]] for j in self._gripper_jids])

    def get_state_vector(self) -> np.ndarray:
        """Stato proprioceptivo: 6 posizioni braccio + 1 apertura gripper
        (media dei due driver, che si muovono in modo speculare)."""
        return np.concatenate([self.arm_qpos, [self.gripper_qpos.mean()]]).astype(np.float32)
