import os
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.arm3dof_env import Arm3DoFEnv
import time


def linear_schedule(initial_value: float):
    """
    Décroissance linéaire du LR : 100% → 10% sur toute la durée.
    Évite les grands sauts de gradient en début d'entraînement.
    """
    def func(progress_remaining: float) -> float:
        return max(initial_value * (0.1 + 0.9 * progress_remaining), 1e-5)
    return func


def sync_envs_normalization(train_env, eval_env):
    """Copy VecNormalize running stats from train_env to eval_env."""
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

TOTAL_TIMESTEPS = 5_000_000


# ==============================
# Directories
# ==============================
run_id              = int(time.time())
run_name            = "ppo_reach_3dof"
tensorboard_log_dir = f"./logs/{run_name}_{run_id}/"
model_dir           = f"./models/{run_name}_{run_id}/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

print(f"Logs  : {tensorboard_log_dir}")
print(f"Models: {model_dir}")


# ==============================
# Environment factory
# ==============================
def make_env():
    return Monitor(Arm3DoFEnv(render_mode=None))


# ==============================
# Training environment
# ==============================
train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
)

# ==============================
# Evaluation environment
# ==============================
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    training=False,
    clip_obs=10.0,
)


# ==============================
# Evaluation callback with VecNormalize sync + vec_normalize save
# ==============================
class SyncedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        sync_envs_normalization(self.training_env, self.eval_env)

        result = super()._on_step()

        if self.last_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.last_mean_reward
            vec_path = os.path.join(self.best_model_save_path, "vec_normalize.pkl")
            self.training_env.save(vec_path)

        return result


eval_callback = SyncedEvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=model_dir,
    eval_freq=5_000,
    n_eval_episodes=50,          # plus d'épisodes → estimation plus fiable
    deterministic=True,
    render=False,
)

# ==============================
# Policy — réseau un peu plus grand
# ==============================
policy_kwargs = dict(
    activation_fn=nn.Tanh,
    net_arch=[256, 256],
    log_std_init=-1.0,   # std initiale ~0.37 → politique plus précise dès le départ
)

# ==============================
# Device setup (GPU or CPU)
# ==============================
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")

# ==============================
# PPO model — hyperparamètres ajustés
# ==============================
model = PPO(
    "MlpPolicy",
    train_env,
    #device=device,
    learning_rate=linear_schedule(3e-4),  # 3e-4 → 3e-5 linéairement
    n_steps=8192,        # ↑ 4096→8192 : plus de diversité par rollout pour le 3dof
    batch_size=1024,     # ↑ 512→1024 : cohérent avec n_steps=8192
    n_epochs=3,          # ↓ 5→3 : 3dof génère des gradients plus grands → moins de passes
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    ent_coef=0.001,  # légère entropie pour éviter convergence prématurée
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.01,      # plus strict pour le 3dof : espace d'action plus grand
    verbose=1,
    tensorboard_log=tensorboard_log_dir,
    policy_kwargs=policy_kwargs,
)

# ==============================
# Training
# ==============================
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
)

# ==============================
# Save final
# ==============================
model_path = os.path.join(model_dir, "ppo_arm2d_final")
vec_path   = os.path.join(model_dir, "vec_normalize.pkl")

model.save(model_path)
train_env.save(vec_path)

print("Training finished.")
print(f"Model  : {model_path}")
print(f"VecNorm: {vec_path}")
