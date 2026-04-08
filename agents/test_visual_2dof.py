import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.arm2dof_env import Arm2DoFEnv

run_id = 1
MODEL_PATH   = f"models/ppo_reach_2dof_{run_id}/best_model.zip"
VECNORM_PATH = f"models/ppo_reach_2dof_{run_id}/vec_normalize.pkl"

num_tests = 5
max_steps = 100

# --- Charger modèle + VecNormalize (indispensable) ---
model = PPO.load(MODEL_PATH)

def make_env():
    return Monitor(Arm2DoFEnv(render_mode="human"))

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training   = False
env.norm_reward = False

# --- Accès à l'env interne pour render() et infos ---
inner_env = env.envs[0].unwrapped

for test_ep in range(num_tests):
    obs = env.reset()
    total_reward = 0.0
    print(f"\nTest épisode {test_ep+1} : cible = {inner_env.target}")

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        total_reward += float(reward[0])

        inner_env.render()
        time.sleep(0.01)

        if dones[0]:
            break

    info = infos[0]
    if info.get("target_reached", False):
        print(f"  ✓ Cible atteinte en {step+1} steps ! Récompense = {total_reward:.2f}")
    else:
        print(f"  ✗ Échec après {step+1} steps. Dist finale = {info.get('dist', '?'):.4f} m")

env.close()
