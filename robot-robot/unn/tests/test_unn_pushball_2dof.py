"""
Test UNN policy trained on PushBall 2-DoF (no transfer).
Mirrors pushball_2dof.py but uses the UNNPolicy pipeline:
    Cartesian encoder → PPO latent policy → IK → joint action.
"""

import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from unn.bases_unn import CartesianStateEncoder
from unn.unn_policy import UNNPolicy

# --- Configuration ---
POLICY_DIR   = "./data/UNN/unn_pushball_2dof"
POLICY_NAME  = "final"
NUM_EPISODES = 20000
MAX_STEPS    = 150
DEVICE       = "cpu"

# --- Charger la politique UNN ---
encoder = CartesianStateEncoder(max_reach=3.0)
policy = UNNPolicy.load(POLICY_DIR, name=POLICY_NAME, encoder=encoder, device=DEVICE)
print(f"UNN 2-DoF policy loaded from '{POLICY_DIR}/{POLICY_NAME}_*'")

# --- Créer l'environnement ---
env = DummyVecEnv([lambda: Monitor(PushBallEnv_2dof(render_mode=None, max_steps=MAX_STEPS))])

# --- Statistiques ---
successes          = 0
steps_on_success   = []
final_dist_failure = []

print(f"\nLancement de {NUM_EPISODES} épisodes de test (UNN 2-DoF)...\n")

for ep in range(NUM_EPISODES):
    obs      = env.reset()[0]   # shape (obs_size,)
    done     = False
    step_count = 0
    info_last  = {}

    while not done:
        action = policy.predict(obs, deterministic=True)   # (n_joints,)
        obs, _, dones, infos = env.step(action.reshape(1, -1))
        obs = obs[0]           # unwrap VecEnv batch dim
        step_count += 1
        done      = dones[0]
        info_last = infos[0]

    if info_last.get("target_reached", False):
        successes += 1
        steps_on_success.append(step_count)
    else:
        final_dist_failure.append(info_last.get("dist_ball_target", float("nan")))

    if (ep + 1) % 500 == 0:
        print(f"  {ep+1}/{NUM_EPISODES} – taux de réussite courant : "
              f"{100 * successes / (ep + 1):.1f}%")

# --- Résultats ---
success_rate = 100 * successes / NUM_EPISODES
print("\n" + "=" * 50)
print(f"  Épisodes testés       : {NUM_EPISODES}")
print(f"  Réussites             : {successes}")
print(f"  Taux de réussite      : {success_rate:.1f}%")
print("-" * 50)
if steps_on_success:
    print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
    print(f"  Steps min / max       : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
if final_dist_failure:
    print(f"  Distance moy (échec)  : {np.mean(final_dist_failure):.4f} m")
    print(f"  Distance min / max    : {np.min(final_dist_failure):.4f} / {np.max(final_dist_failure):.4f} m")
print("=" * 50)

env.close()
