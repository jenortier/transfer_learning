"""
LS-UNN Policy — PPO agent operating in the shared latent space.

Architecture finale:
  Bi(obs) → z → UNN(z, oτ) → zd → Bo(zd) → x_rd → action (vitesse)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from lsunn.bases_vae import BaseVAE


class UNNPolicy:
    """
    Policy complète LS-UNN:
      1. Encode raw observation → latent z via Bi (stochastic ou déterministe)
      2. PPO policy → latent désiré zd
      3. Decode zd → état reconstruit x_rd via Bo
      4. Extract vitesse de x_rd → action
    """
    
    def __init__(
        self,
        base_vae: BaseVAE,
        ppo_policy: PPO,
        vec_normalize: Optional[VecNormalize] = None,
        device: str = "cpu",
        stochastic_encoding: bool = False,  # False = use μ (déterministe)
        n_joints: Optional[int] = None,  # nb d'articulations (2 ou 3) — explicite, ne dépend pas de state_dim
    ):
        self.device = device
        self.base_vae = base_vae
        self.ppo_policy = ppo_policy
        self.vec_normalize = vec_normalize
        self.stochastic_encoding = stochastic_encoding
        
        self.base_vae.eval()
        self.ppo_policy.policy.eval()
        
        # Dimensions de l'observation arm_obs
        self.arm_dim = self.base_vae.state_dim
        self.latent_dim = self.base_vae.latent_dim

        # Nombre d'articulations : ne PAS déduire de state_dim (fragile si le
        # VAE a été entraîné sur une observation étendue, ex. arm_obs + task_obs).
        # Par défaut on retombe sur l'ancienne heuristique pour rester
        # rétro-compatible, mais il est recommandé de le passer explicitement.
        if n_joints is None:
            n_joints = 2 if self.arm_dim == 6 else 3
        self.n_joints = n_joints

    @torch.no_grad()
    def predict(
        self,
        raw_obs: np.ndarray,
        deterministic: bool = True,
        return_components: bool = False
    ) -> np.ndarray:
        """
        Full LS-UNN forward pass:
          1. Encode raw_obs → latent z (Bi)
          2. PPO → latent désiré zd
          3. Decode zd → x_rd (Bo)
          4. Extract velocity component → action
        
        Args:
            raw_obs: observation complète (arm_obs + task_obs)
            deterministic: si True, PPO déterministe
            return_components: si True, retourne (action, z, zd, x_rd)
        """
        # 1. Extraire arm_obs (les premières dimensions)
        if raw_obs.ndim == 1:
            arm_obs = raw_obs[:self.arm_dim]
        else:
            arm_obs = raw_obs[:, :self.arm_dim]
        
        # 2. Encoder arm_obs → z
        z = self.base_vae.encode_np(arm_obs, device=self.device, stochastic=self.stochastic_encoding)
        
        # 3. Normaliser z (si VecNormalize disponible)
        if self.vec_normalize is not None:
            z_norm = self.vec_normalize.normalize_obs(z)
        else:
            z_norm = z
        
        # 4. PPO → latent désiré zd
        zd, _ = self.ppo_policy.predict(z_norm, deterministic=deterministic)
        
        # 5. Selon l'architecture (voir LatentEnv.step pour l'explication) :
        #    zd peut déjà être l'action brute (dim n_joints), ou un latent
        #    désiré (dim latent_dim) à décoder via Bo.
        n = self.n_joints
        zd_arr = np.asarray(zd)
        last_dim = zd_arr.shape[-1]

        x_rd = None
        if last_dim == n:
            action = zd_arr
        else:
            x_rd = self.base_vae.decode_np(zd, device=self.device)
            omega_max = 2.0  # correspond à OMEGA_MAX dans les environnements
            action = x_rd[n:2*n] * omega_max if x_rd.ndim == 1 else x_rd[:, n:2*n] * omega_max
        
        # Clipper l'action dans [-1, 1] (les environnements attendent [-1, 1])
        action = np.clip(action, -1.0, 1.0)
        
        if return_components:
            # x_rd vaut None si l'architecture "directe" a été détectée
            # (pas de décodage Bo pour cette action).
            return action, z, zd, x_rd
        return action

    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.ppo_policy.save(save_path / "unn_ppo_policy.zip")
        if self.vec_normalize is not None:
            self.vec_normalize.save(save_path / "unn_vec_normalize.pkl")
        print(f"  UNN Policy saved → {save_path}")

    @classmethod
    def load(
        cls,
        save_dir: str,
        base_vae: BaseVAE,
        vec_normalize: Optional[VecNormalize] = None,
        device: str = "cpu",
        stochastic_encoding: bool = False,
        n_joints: Optional[int] = None,
    ):
        save_path = Path(save_dir)
        ppo = PPO.load(save_path / "unn_ppo_policy.zip", device=device)
        return cls(base_vae, ppo, vec_normalize, device, stochastic_encoding, n_joints=n_joints)


class LatentEnv(gym.Wrapper):
    """
    Wrapper Gymnasium qui transforme l'observation en [z, oτ] pour l'entrée PPO.
    """
    def __init__(self, env: gym.Env, base_vae: BaseVAE, device: str = "cpu",
                 n_joints: Optional[int] = None):
        super().__init__(env)
        self.base_vae = base_vae
        self.device = device
        self.base_vae.eval()

        # Nombre d'articulations, explicite — ne pas déduire de state_dim
        # (fragile si le VAE encode plus que le seul arm_obs cinématique).
        if n_joints is None:
            n_joints = 2 if base_vae.state_dim == 6 else 3
        self.n_joints = n_joints
        
        # Dimensions de l'entrée PPO: latent_dim + task_obs_dim
        task_dim = env.observation_space.shape[0] - base_vae.state_dim
        obs_dim = base_vae.latent_dim + task_dim
        
        self.observation_space = spaces.Box(-10.0, 10.0, (obs_dim,), dtype=np.float32)
        
        # Stockage pour oτ
        self.last_raw_obs: Optional[np.ndarray] = None

    @torch.no_grad()
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_raw_obs = obs.copy()
        return self._build_ppo_obs(obs), info

    @torch.no_grad()
    def step(self, zd):
        """
        zd: sortie du PPO. Deux architectures possibles selon comment la
        politique a été entraînée:
          - "décodée": zd est un latent désiré de dimension latent_dim,
            qu'on décode via Bo pour en extraire une vitesse (architecture
            Bo∘U∘Bi décrite dans le docstring du module).
          - "directe": zd EST déjà l'action de vitesse, de dimension
            n_joints (le PPO a été entraîné avec l'action_space brut de
            l'environnement, sans décodage).
        On détecte laquelle des deux s'applique à partir de la dimension
        réelle reçue, pour rester compatible avec les deux types de
        checkpoints sans devoir le savoir à l'avance.
        """
        zd = np.asarray(zd)
        n = self.n_joints
        omega_max = 2.0

        if zd.shape[-1] == n:
            # Architecture directe : le PPO produit déjà l'action (vitesse).
            action = zd
        else:
            # Architecture décodée : zd est un latent désiré → Bo(zd) → vitesse.
            x_rd = self.base_vae.decode_np(zd, device=self.device)
            action = x_rd[n:2*n]
            action = action * omega_max

        action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_raw_obs = obs.copy()
        
        return self._build_ppo_obs(obs), reward, terminated, truncated, info

    def _build_ppo_obs(self, obs: np.ndarray) -> np.ndarray:
        """Construit [z, oτ] pour l'entrée PPO."""
        arm_obs = obs[:self.base_vae.state_dim]
        task_obs = obs[self.base_vae.state_dim:]
        
        # Encoder en déterministe (μ) pendant l'entraînement UNN
        z = self.base_vae.encode_np(arm_obs, device=self.device, stochastic=False)
        
        return np.concatenate([z.flatten(), task_obs], axis=0)