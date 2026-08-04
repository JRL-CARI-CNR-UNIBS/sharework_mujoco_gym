# sharework-mujoco

Modello MuJoCo della cella sharework (UR10e + Robotiq 2F-85 + 3 RealSense:
wrist, rs1, rs2), utilizzabile in tre modi:

1. **Standalone MuJoCo** — script/notebook, teleop manuale, debug visivo.
2. **Gymnasium** — per RL (SAC e altri algoritmi di stable-baselines3/altre librerie).
3. **LeRobot** — come front-end di valutazione per policy ACT/VLA gia' addestrate.

## Installazione

```bash
pip install -e .                    # base (mujoco, gymnasium, numpy)
pip install -e ".[sac]"             # + stable-baselines3
pip install -e ".[lerobot]"         # + lerobot
pip install -e ".[live-view]"       # + opencv-python (per la finestra live wrist camera)
```

## 1. Uso standalone

```bash
python3 examples/standalone_view.py                    # viewer interattivo
python3 examples/standalone_view.py --live              # viewer 3D + finestra live wrist camera
python3 examples/standalone_view.py --wrist-cam out.png # snapshot dalla wrist camera
```

Oppure in Python:

```python
from sharework_mujoco.sim import SharedworkCellSim

sim = SharedworkCellSim()
sim.step(sim.data.ctrl)          # un passo di fisica
img = sim.render(camera="rs1_d435")
state = sim.get_state_vector()   # 6 giunti braccio + apertura gripper
```

Per un MJCF standalone da usare con `simulate` o altri tool (non Python):

```python
from sharework_mujoco.scene_builder import build_and_save_xml
build_and_save_xml("sharework_cell_mujoco.xml")
```

## 2. Uso con Gymnasium (RL)

Due ambienti registrati:

- `SharedworkCell-v0` — osservazione = solo stato (7-dim: 6 giunti + gripper).
- `SharedworkCellVision-v0` — osservazione = `{"pixels": {cam: immagine}, "agent_pos": stato}`.

```python
import gymnasium as gym
import sharework_mujoco  # registra gli env

env = gym.make("SharedworkCell-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Azione: `Box(-1, 1, shape=(7,))`, normalizzata; internamente viene
rimappata al `ctrlrange` reale di ciascun attuatore (letto dal modello, non
hardcoded — il gripper ad es. e' `[0, 255]`).

**Reward**: di default sempre `0.0` — non esiste un task predefinito.
Passa una funzione tua:

```python
env.unwrapped._reward_fn = la_tua_funzione_di_reward
```

Vedi `examples/train_sac.py` per un esempio completo (con un reward di
reaching puramente dimostrativo, da sostituire).

Validato con `gymnasium.utils.env_checker.check_env` (nessun errore/warning)
su entrambe le varianti.

## 3. Uso con LeRobot (ACT / VLA)

Le osservazioni di `SharedworkCellVision-v0` seguono la stessa convenzione
`pixels`/`agent_pos` che `lerobot.envs.utils.preprocess_observation` si
aspetta (verificato contro il sorgente di lerobot 0.6.1), quindi non serve
un adapter dedicato. Vedi `examples/lerobot_eval.py`.

**Non testato end-to-end** in questo ambiente di sviluppo (lerobot/torch
richiedono piu' spazio disco di quanto disponibile nel sandbox in cui e'
stato scritto) — leggi i commenti in cima allo script prima di usarlo,
in particolare sulla differenza tra la pipeline di inferenza minimale
mostrata li' e quella piu' completa (`PolicyProcessorPipeline`) che LeRobot
usa per il deployment reale.

## Struttura

```
src/sharework_mujoco/
├── scene_builder.py   # costruisce lo MjSpec (fixed_parts + ur10e + 2f85 + camere)
├── sim.py             # wrapper standalone (reset/step/render)
├── gym_env.py          # SharedworkCellEnv(gymnasium.Env)
└── assets/             # fixed_parts.xml, xml sorgente menagerie, mesh
```

## Note tecniche / assunzioni da verificare

- **Orientazione camere** (wrist/rs1/rs2): rotazione mount(REP-103)->optical
  calcolata dalla convenzione ROS2 confermata (`wrist_link -> color_optical_frame`,
  rpy `-pi/2,0,-pi/2`), applicata anche a rs1/rs2 assumendo la stessa
  convenzione standard REP-103 (ragionevole per RealSense, ma non verificata
  su una foto reale di rs1/rs2 specificamente).
- **open_tip/closed_tip**: offset (0.149/0.163 m) calibrati sulla mesh
  Robotiq reale + coupler custom, non sulla mesh 2f85 di menagerie (stack-up
  leggermente diverso, ~0.156m). Buoni come primo riferimento, ricalibrali
  a vista se ti servono precisi per grasping.
- **Reward RL**: nessun task/reward reale definito, va aggiunto.
