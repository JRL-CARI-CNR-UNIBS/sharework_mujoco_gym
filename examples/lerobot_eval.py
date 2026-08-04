"""
Come collegare SharedworkCellVision-v0 a una policy LeRobot (ACT, VLA, ...)
gia' addestrata, per valutarla in simulazione invece che sul robot vero.

STATO DI VERIFICA (leggilo prima di usarlo):
- La parte "env -> preprocess_observation()" e' verificata contro il codice
  sorgente reale di lerobot 0.6.1 (ho scaricato il wheel e letto
  lerobot/envs/utils.py): si aspetta esattamente le chiavi "pixels"
  (dict camera->immagine HWC uint8) e "agent_pos" che SharedworkCellEnv
  produce gia'.
- La parte "policy.select_action(batch)" e' verificata per firma contro
  lerobot/policies/act/modeling_act.py (stessa versione).
- NON ho potuto eseguire questo script end-to-end: stable-baselines3/lerobot
  richiedono torch, che in questo sandbox non e' installabile per spazio
  disco esaurito. Provalo tu con --timesteps piccoli prima di fidarti.
- La pipeline di inferenza "vera" di lerobot 0.6.1 e' piu' elaborata di
  quella qui sotto (usa anche PolicyProcessorPipeline per pre/post-processing
  con le statistiche di normalizzazione del dataset, vedi
  lerobot/rollout/inference/sync.py) -- se il TUO checkpoint ACT e' stato
  salvato/allenato con quella pipeline, riusa il tuo codice di deploy
  esistente per il pre/post-processing (quello che gia' usi su UR10e reale)
  e limitati a cambiare la SORGENTE delle osservazioni (da ROS a questo
  env), invece di fidarti ciecamente del frammento minimale qui sotto.

    pip install "sharework-mujoco[lerobot]"
    python3 lerobot_eval.py --checkpoint /path/al/tuo/act_checkpoint
"""
import argparse

import gymnasium as gym
import numpy as np

import sharework_mujoco  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path o hub id del checkpoint ACT")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.envs.utils import preprocess_observation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # Le chiavi delle immagini devono combaciare con quelle usate in
    # training (policy.config.input_features / image_features). Se il tuo
    # dataset usava nomi diversi da wrist_d435/rs1_d435/rs2_d435, rinomina
    # qui (es. {"observation.images.wrist_d435": "observation.images.top"}).
    env = gym.make("SharedworkCellVision-v0", render_mode="rgb_array")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        policy.reset()
        for t in range(args.max_steps):
            # preprocess_observation vuole batch (numero campioni, non
            # singolo), quindi aggiungo la dim batch=1 con [None, ...]
            gym_obs = {
                "pixels": {k: v[None, ...] for k, v in obs["pixels"].items()},
                "agent_pos": obs["agent_pos"][None, ...],
            }
            batch = preprocess_observation(gym_obs)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                action = policy.select_action(batch)
            action_np = action.squeeze(0).cpu().numpy()

            # NOTA: policy.select_action ritorna l'azione nello spazio in
            # cui e' stata allenata (probabilmente ctrl "fisico", non
            # normalizzato [-1,1] come si aspetta questo env -- vedi
            # SharedworkCellEnv.step). Va rimappata secondo il ctrlrange
            # reale (env.unwrapped._ctrl_low/_ctrl_high) prima di passarla
            # a step(), a meno che tu non stia gia' allenando su azioni
            # normalizzate.
            low = env.unwrapped._ctrl_low
            high = env.unwrapped._ctrl_high
            action_norm = 2 * (action_np - low) / (high - low) - 1
            action_norm = np.clip(action_norm, -1, 1)

            obs, reward, terminated, truncated, info = env.step(action_norm)
            if terminated or truncated:
                break
        print(f"episodio {ep}: terminato dopo {t + 1} step")

    env.close()


if __name__ == "__main__":
    main()
