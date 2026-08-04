"""
Esempio di training SAC (stable-baselines3) sull'ambiente di stato
(SharedworkCell-v0, no immagini -- SAC su pixel grezzi con Dict obs
richiederebbe un feature extractor CNN dedicato, vedi nota in fondo).

    pip install "sharework-mujoco[sac]"
    python3 train_sac.py

NON TESTATO END-TO-END in sandbox (stable-baselines3 richiede torch, non
installabile qui per limiti di spazio disco). La struttura segue l'uso
documentato standard di SB3 (https://stable-baselines3.readthedocs.io/),
ma vale la pena un primo giro breve (--timesteps 1000) sulla tua macchina
prima di lanciare un training lungo, per scovare eventuali intoppi che non
ho potuto vedere da qui.

IMPORTANTE -- reward: SharedworkCellEnv ha reward=0.0 di default (nessun
task definito). Qui sotto uso un esempio minimo di reaching (avvicinare il
TCP a un punto fisso sopra il tavolo) solo per mostrare come si collega un
reward_fn; sostituiscilo con il task che ti interessa davvero.
"""
import argparse

import gymnasium as gym
import numpy as np

import sharework_mujoco  # noqa: F401  (registra gli env)


def reaching_reward(env) -> float:
    """Esempio minimo: -distanza tra il TCP (site 'ur10e_attachment_site')
    e un target fisso 0.3m sopra il centro del tavolo. Sostituiscila con la
    tua reward reale."""
    import mujoco
    model, data = env.sim.model, env.sim.data
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur10e_attachment_site")
    tcp_pos = data.site_xpos[tcp_id]
    target = np.array([0.3, 0.0, 0.875 + 0.3])  # sopra il tavolo, vedi fixed_parts.xml
    dist = np.linalg.norm(tcp_pos - target)
    return -dist


def make_env():
    env = gym.make("SharedworkCell-v0", render_mode="rgb_array")
    env.unwrapped._reward_fn = reaching_reward  # vedi nota sopra
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", default="sac_sharework.zip")
    args = parser.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env

    env = make_env()
    check_env(env.unwrapped)  # validazione SB3-specifica, oltre a quella gymnasium gia' fatta

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"Salvato modello in {args.out}")


if __name__ == "__main__":
    main()
