import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from torch.utils.tensorboard import SummaryWriter

from envs.cartpole_env import CartPoleEnv

# --- TensorBoard Writer ---
writer = SummaryWriter(log_dir="logs/rl_experiment") 

# ======================================================
# 🔧 CONFIG CPU (VM FRIENDLY)
# ======================================================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# ======================================================
# 📁 DOSSIERS
# ======================================================
LOG_DIR = "models/tensorboard"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# 🌍 ENVIRONNEMENTS
# ======================================================
def make_env():
    env = CartPoleEnv()
    env = Monitor(env)
    env = TimeLimit(env, max_episode_steps=500)
    return env

train_env = make_vec_env(make_env, n_envs=1) ###################
eval_env  = make_vec_env(make_env, n_envs=1)

# ======================================================
# 🧠 AGENT PPO
# ======================================================
policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=torch.nn.Tanh,
)

model = PPO(
    "MlpPolicy",
    train_env,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=LOG_DIR,   # 🔥 IMPORTANT
    policy_kwargs=policy_kwargs,
)

# ======================================================
# 📊 CALLBACK EVAL
# ======================================================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=MODEL_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# ======================================================
# 🚀 ENTRAÎNEMENT
# ======================================================
model.learn(
    total_timesteps=200_000,
    callback=eval_callback,
    tb_log_name="PPO_CartPole"  # 🔥 NOM DE L’EXPÉRIENCE
)

# --- Fermeture propre du writer ---
writer.close()

# ======================================================
# 💾 SAUVEGARDE
# ======================================================
model.save(os.path.join(MODEL_DIR, "ppo_cartpole"))
print("✅ Entraînement terminé et logs TensorBoard écrits")

