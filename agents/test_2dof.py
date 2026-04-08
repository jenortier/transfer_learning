import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from envs.arm2dof_env import Arm2DoFEnv

# --- Configuration ---
run_id = 1
MODEL_PATH  = f"models/ppo_reach_2dof_{run_id}/best_model.zip"   # best_model plutôt que final
VECNORM_PATH = f"models/ppo_reach_2dof_{run_id}/vec_normalize.pkl"
NUM_EPISODES = 2000
MAX_STEPS    = 100

# --- Charger le modèle ---
model = PPO.load(MODEL_PATH)

# --- Créer l'environnement AVEC VecNormalize (indispensable !) ---
def make_env():
    return Monitor(Arm2DoFEnv(render_mode=None))

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
        final_dist_on_failure.append(info_last.get("dist", float("nan")))

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
