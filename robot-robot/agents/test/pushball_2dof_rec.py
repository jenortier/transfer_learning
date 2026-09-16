import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.env_pushball_2dof import PushBallEnv_2dof

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
MODEL_ROOT = ROOT / "data" / "models"

# --- Configuration ---
run_id = 3 ##2
MODEL_PATH  = MODEL_ROOT / f"ppo_pushball_2dof_rec_{run_id}" / "ppo_pushball_final.zip" #"best_model.zip" 
VECNORM_PATH = MODEL_ROOT / f"ppo_pushball_2dof_rec_{run_id}" / "vec_normalize.pkl"
NUM_EPISODES = 500
MAX_STEPS    = 150

# --- Charger le modèle ---
model = PPO.load(MODEL_PATH, custom_objects={"learning_rate": 0.0003, "lr_schedule": lambda _: 0.0003, "clip_range": lambda _: 0.2})

# --- Créer l'environnement AVEC VecNormalize (indispensable !) ---
def make_env():
    return Monitor(PushBallEnv_2dof(render_mode=None))

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
env.training = False      # ne pas mettre à jour les stats
env.norm_reward = False   # on veut la reward brute pour les stats

# --- Statistiques ---
successes = 0
steps_on_success = []
final_dist_on_failure = []

print(f"Lancement de {NUM_EPISODES} épisodes de test...\n")

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = False
    step_count = 0
    info_last = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        step_count += 1
        done = dones[0]
        info_last = infos[0]

    if info_last.get("target_reached", False):
        successes += 1
        steps_on_success.append(step_count)
    else:
        final_dist_on_failure.append(info_last.get("dist_ball_target", float("nan")))

# --- Résultats ---
success_rate = 100 * successes / NUM_EPISODES
print("=" * 45)
print(f"  Épisodes testés       : {NUM_EPISODES}")
print(f"  Réussites             : {successes}")
print(f"  Taux de réussite      : {success_rate:.1f}%")
print("-" * 45)
if steps_on_success:
    print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
    print(f"  Steps min / max       : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
if final_dist_on_failure:
    print(f"  Distance moy (échec)  : {np.mean(final_dist_on_failure):.4f} m")
    print(f"  Distance min / max    : {np.min(final_dist_on_failure):.4f} / {np.max(final_dist_on_failure):.4f} m")
print("=" * 45)

env.close()
