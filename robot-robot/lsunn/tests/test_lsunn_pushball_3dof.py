"""
Test LS-UNN PPO policy trained on PushBall 3-DoF.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE
from lsunn.lsunn_policy import LatentEnv


# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATA_DIR = Path("./data/LSUNN")

NUM_EPISODES = 2000
MAX_STEPS = 150
DEVICE = "cpu"

N_JOINTS = 3  # nombre d'articulations du bras (indépendant du state_dim du VAE)


# -------------------------------------------------
# Charger le VAE
# -------------------------------------------------
# NB: on n'impose plus state_dim/latent_dim en dur (DEFAULT_LATENT_DIM / arm_obs=8).
# Ce checkpoint "pushball_bases" peut avoir été entraîné avec une observation
# étendue (arm_obs + infos tâche) et/ou une autre dimension latente. On lit
# donc directement la config depuis les tenseurs du checkpoint pour éviter
# tout size mismatch, quel que soit le run qui l'a produit.

ckpt_path = DATA_DIR / "base_3dof.pt"
state_dict = torch.load(ckpt_path, map_location=DEVICE)

inferred_state_dim = state_dict["encoder.net.0.weight"].shape[1]
inferred_hidden_dim = state_dict["encoder.net.0.weight"].shape[0]
inferred_latent_dim = state_dict["encoder.mu_head.weight"].shape[0]

print(
    f"  Checkpoint inféré: state_dim={inferred_state_dim}, "
    f"latent_dim={inferred_latent_dim}, hidden_dim={inferred_hidden_dim}"
)

base_vae = BaseVAE(
    state_dim=inferred_state_dim,
    latent_dim=inferred_latent_dim,
    hidden_dim=inferred_hidden_dim,
).to(DEVICE)

base_vae.load_state_dict(state_dict)

base_vae.eval()

for p in base_vae.parameters():
    p.requires_grad = False


# -------------------------------------------------
# Créer l'environnement latent
# -------------------------------------------------

def make_env():
    env = PushBallEnv_3dof(
        render_mode=None,
        max_steps=MAX_STEPS,
    )

    env = LatentEnv(
        env,
        base_vae,
        DEVICE,
        n_joints=N_JOINTS,
    )

    return Monitor(env)


env = DummyVecEnv([make_env])

env = VecNormalize.load(
    str(DATA_DIR / "lsunn_3dof" / "vec_normalize.pkl"),
    env,
)

env.training = False
env.norm_reward = False


# -------------------------------------------------
# Charger le PPO
# -------------------------------------------------

model = PPO.load(
    DATA_DIR / "lsunn_3dof" / "policy",
    env=env,
    device=DEVICE,
)

print("LS-UNN PPO 3DoF loaded.")


# -------------------------------------------------
# Évaluation
# -------------------------------------------------

successes = 0
steps_on_success = []
final_dist_failure = []

print(f"\nTesting {NUM_EPISODES} episodes...\n")

for ep in range(NUM_EPISODES):

    obs = env.reset()

    done = False
    step_count = 0
    info_last = {}

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, done, infos = env.step(action)

        done = done[0]
        info_last = infos[0]

        step_count += 1

    if info_last.get("target_reached", False):

        successes += 1
        steps_on_success.append(step_count)

    else:

        final_dist_failure.append(
            info_last.get("dist_ball_target", np.nan)
        )

    if (ep + 1) % 500 == 0:

        print(
            f"{ep+1}/{NUM_EPISODES} "
            f"- success rate = {100*successes/(ep+1):.1f}%"
        )


# -------------------------------------------------
# Résultats
# -------------------------------------------------

print("\n" + "=" * 60)

print(f"Episodes          : {NUM_EPISODES}")
print(f"Successes         : {successes}")
print(f"Success rate      : {100*successes/NUM_EPISODES:.2f}%")

print("-" * 60)

if steps_on_success:

    print(f"Mean steps        : {np.mean(steps_on_success):.2f}")
    print(f"Min / Max steps   : {np.min(steps_on_success)} / {np.max(steps_on_success)}")

if final_dist_failure:

    print(f"Mean failure dist : {np.mean(final_dist_failure):.4f}")
    print(f"Min / Max dist    : {np.min(final_dist_failure):.4f} / {np.max(final_dist_failure):.4f}")

print("=" * 60)

env.close()