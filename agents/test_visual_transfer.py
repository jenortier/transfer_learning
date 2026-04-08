import time
import numpy as np
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.arm3dof_env import Arm3DoFEnv

# Import de la classe TransferPolicy
from agents.transfer_reaching import TransferPolicy

# Chemin vers le modèle de transfer learning sauvegardé
TRANSFER_MODEL_DIR = "models/ppo_transfer_2to3_1"

num_tests = 5
max_steps = 100

# --- Charger le modèle de transfer learning ---
print(f"Loading transfer model from: {TRANSFER_MODEL_DIR}")
policy = TransferPolicy.load_from_config(TRANSFER_MODEL_DIR, device="cpu")

# --- Créer l'environnement 3-DoF avec rendu visuel ---
def make_env():
    return Monitor(Arm3DoFEnv(render_mode="human"))

env = DummyVecEnv([make_env])

# --- Accès à l'env interne pour render() et infos ---
inner_env = env.envs[0].unwrapped

for test_ep in range(num_tests):
    obs = env.reset()  # obs shape: (1, 10), raw
    total_reward = 0.0
    print(f"\nTest épisode {test_ep+1} : cible = {inner_env.target}")

    for step in range(max_steps):
        # Utiliser la politique de transfer learning
        action = policy.predict(obs[0])  # obs[0] pour avoir shape (10,)
        obs, reward, dones, infos = env.step(action)
        total_reward += float(reward[0])

        inner_env.render()
        time.sleep(0.02)

        if dones[0]:
            break

    info = infos[0]
    if info.get("target_reached", False):
        print(f"  ✓ Cible atteinte en {step+1} steps ! Récompense = {total_reward:.2f}")
    else:
        print(f"  ✗ Échec après {step+1} steps. Dist finale = {info.get('dist', '?'):.4f} m")

env.close()
