"""
Phase 3: Entraînement LSUNN (PPO) dans l'espace latent partagé.
Bases VAE gelées pendant l'entraînement.
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

torch.set_num_threads(16)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from lsunn.bases_vae import BaseVAE, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM
from lsunn.lsunn_policy import LatentEnv

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data/LSUNN")
LATENT_DIM = DEFAULT_LATENT_DIM
HIDDEN_DIM = DEFAULT_HIDDEN_DIM

# Hyperparamètres PPO (choix propres)
PPO_CONFIG = {
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 5,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [256, 256]},
}


def make_latent_env(env_cls, base_vae, device, seed=0):
    def _init():
        raw = env_cls(render_mode=None, max_steps=150)
        env = LatentEnv(raw, base_vae, device)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_latent_policy(
    env_cls,
    base_vae: BaseVAE,
    save_dir: Path,
    device: str = "cpu",
    n_envs: int = 8,
    total_timesteps: int = 16_000_000, ##5_000_000,
    run_id: str = "lsunn_2dof",
) -> tuple[PPO, VecNormalize]:
    """
    Entraîne une politique PPO sur l'environnement latent.
    Base VAE est gelée (requires_grad=False, eval()).
    """
    base_vae.eval()
    for param in base_vae.parameters():
        param.requires_grad = False
    
    def make_env(rank):
        return make_latent_env(env_cls, base_vae, device, seed=rank)
    
    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0
    )
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ppo = PPO(
        "MlpPolicy",
        train_env,
        **PPO_CONFIG,
        device=device,
        verbose=1,
    )
    
    ppo.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    ppo.save(save_dir / "policy")
    train_env.save(save_dir / "vec_normalize.pkl")
    
    return ppo, train_env


def train_lsunn_policies(
    base_2dof: BaseVAE,
    base_3dof: BaseVAE,
    device: str = "cpu",
    total_timesteps: int = 5_000_000,
    n_envs: int = 8,
) -> tuple:
    """
    Entraîne les politiques LSUNN pour 2DoF et 3DoF.
    Les bases VAE sont gelées.
    """
    print("\n" + "=" * 60)
    print("Phase 3: Training LSUNN policies (VAE bases frozen)")
    print("=" * 60)
    
    print("\n  Training 2DoF LSUNN policy...")
    ppo_2dof, vec_norm_2dof = train_latent_policy(
        PushBallEnv_2dof,
        base_2dof,
        DATA_DIR / "lsunn_2dof",
        device,
        n_envs,
        total_timesteps // 2,
        "lsunn_2dof",
    )
    
    print("\n  Training 3DoF LSUNN policy...")
    ppo_3dof, vec_norm_3dof = train_latent_policy(
        PushBallEnv_3dof,
        base_3dof,
        DATA_DIR / "lsunn_3dof",
        device,
        n_envs,
        total_timesteps // 2,
        "lsunn_3dof",
    )
    
    return ppo_2dof, vec_norm_2dof, ppo_3dof, vec_norm_3dof


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Charger les bases VAE
    print(f"  Loading VAE weights from {DATA_DIR} ...")
    base_2dof = BaseVAE(6, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    base_3dof = BaseVAE(8, LATENT_DIM, HIDDEN_DIM).to(DEVICE)
    base_2dof.load_state_dict(torch.load(DATA_DIR / "base_2dof.pt", map_location=DEVICE))
    base_3dof.load_state_dict(torch.load(DATA_DIR / "base_3dof.pt", map_location=DEVICE))
    
    # Entraînement
    train_lsunn_policies(base_2dof, base_3dof, device=DEVICE)
    
    print(f"\n  LSUNN policies saved → {DATA_DIR}/lsunn_{{2,3}}dof/")


if __name__ == "__main__":
    main()