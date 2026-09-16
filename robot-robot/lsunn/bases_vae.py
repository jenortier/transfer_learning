"""
LS-UNN Bases VAE — Variational Autoencoders for shared latent space alignment.
Based on: "Towards Zero-Shot Cross-Agent Transfer Learning via
Latent-Space Universal Notice Network" (Beaussant et al., 2024)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_num_threads(8)

# ============================================================================
# Dimensions — uniquement arm_obs (pas l'état complet)
# ============================================================================
ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8

# À fixer après latent_dim_search.py
DEFAULT_LATENT_DIM = 4 ###2 ##5  # À remplacer par la valeur trouvée
DEFAULT_HIDDEN_DIM = 256


# ============================================================================
# VAE Encoder Base
# ============================================================================
class VAEEncoder(nn.Module):
    """Maps state → (μ, log_σ²) in latent space."""
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


# ============================================================================
# VAE Decoder Base
# ============================================================================
class VAEDecoder(nn.Module):
    """Maps latent z → reconstructed state."""
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================================
# Full VAE (Encoder + Decoder)
# ============================================================================
class BaseVAE(nn.Module):
    """Full VAE: encodes arm_obs → latent z, decodes z → arm_obs."""
    def __init__(self, state_dim: int, latent_dim: int = DEFAULT_LATENT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(state_dim, latent_dim, hidden_dim)
        self.decoder = VAEDecoder(state_dim, latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def encode(self, x: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        """Encode avec ou sans sampling stochastique."""
        mu, logvar = self.encoder(x)[1], self.encoder(x)[2]
        if stochastic:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    @torch.no_grad()
    def encode_np(self, x: np.ndarray, device: str = "cuda", stochastic: bool = True) -> np.ndarray:
        t = torch.tensor(x, dtype=torch.float32, device=device)
        return self.encode(t, stochastic=stochastic).cpu().numpy()

    @torch.no_grad()
    def decode_np(self, z: np.ndarray, device: str = "cuda") -> np.ndarray:
        t = torch.tensor(z, dtype=torch.float32, device=device)
        return self.decoder(t).cpu().numpy()


# ============================================================================
# VAE Trainer
# ============================================================================
class BasesVAETrainer:
    """
    Hyperparamètres corrigés:
    - alpha = 1.0 (reconstruction)
    - beta = 0.00015 (KL)
    - gamma = 2/3 (similarité latente)
    - lmbda = 1/3 (reconstruction croisée)
    """
    
    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 device: str = "cpu"):
        self.device = device
        self.latent_dim = latent_dim
        
        self.base_2dof = BaseVAE(ARM_OBS_2DOF, latent_dim, hidden_dim).to(device)
        self.base_3dof = BaseVAE(ARM_OBS_3DOF, latent_dim, hidden_dim).to(device)
        
        self.reconstruction_crit = nn.MSELoss()
        self.similarity_crit = nn.MSELoss()
        
        # Hyperparamètres corrigés
        self.alpha = 1.0
        self.beta = 0.00015
        self.gamma = 2.0 / 3.0
        self.lmbda = 1.0 / 3.0

    def train(self, trajectories: Dict, epochs: int = 100,
              batch_size: int = 100, lr: float = 5e-4,
              log_dir: Optional[str] = None):
        """
        Entraîne les deux VAE sur les arm_states uniquement.
        """
        print("\n" + "="*60)
        print("LS-UNN: Joint VAE Bases Training (arm_obs only)")
        print("="*60)
        print(f"  Device: {self.device}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  α(recon)={self.alpha}  β(KL)={self.beta}  γ(sim)={self.gamma}  λ(cross)={self.lmbda}")

        # Utiliser uniquement arm_states
        s2 = torch.tensor(trajectories['arm_states_2dof'], dtype=torch.float32).to(self.device)
        s3 = torch.tensor(trajectories['arm_states_3dof'], dtype=torch.float32).to(self.device)

        print(f"  Dataset: {len(s2):,} samples\n")

        optimizer = optim.Adam(
            list(self.base_2dof.parameters()) + list(self.base_3dof.parameters()),
            lr=lr, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        writer = SummaryWriter(log_dir) if log_dir is not None else None

        pbar = tqdm(range(epochs), desc="Training VAE Bases")

        for epoch in pbar:
            self.base_2dof.train()
            self.base_3dof.train()

            perm = torch.randperm(len(s2))
            loss_sums = {"total": 0.0, "reconstruction": 0.0,
                         "kl": 0.0, "similarity": 0.0, "cross": 0.0}
            n_b = 0

            for i in range(0, len(s2), batch_size):
                idx = perm[i:i + batch_size]
                b2, b3 = s2[idx], s3[idx]

                # Forward
                z2, mu2, logvar2 = self.base_2dof.encoder(b2)
                z3, mu3, logvar3 = self.base_3dof.encoder(b3)

                y2 = self.base_2dof.decoder(z2)
                y3 = self.base_3dof.decoder(z3)
                y_cross_2 = self.base_2dof.decoder(z3)
                y_cross_3 = self.base_3dof.decoder(z2)

                # Losses
                loss_recon = self.alpha * (
                    self.reconstruction_crit(y2, b2) +
                    self.reconstruction_crit(y3, b3)
                )

                loss_kl = self.beta * (
                    -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / b2.shape[0] +
                    -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp()) / b3.shape[0]
                )

                loss_sim = self.gamma * self.similarity_crit(z2, z3)

                loss_cross = self.lmbda * (
                    self.reconstruction_crit(y_cross_2, b2) +
                    self.reconstruction_crit(y_cross_3, b3)
                )

                loss = loss_recon + loss_kl + loss_sim + loss_cross

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sums["total"] += loss.item()
                loss_sums["reconstruction"] += loss_recon.item()
                loss_sums["kl"] += loss_kl.item()
                loss_sums["similarity"] += loss_sim.item()
                loss_sums["cross"] += loss_cross.item()
                n_b += 1

            scheduler.step()
            epoch_losses = {name: value / n_b for name, value in loss_sums.items()}
            pbar.set_postfix({'loss': f'{epoch_losses["total"]:.4f}'})

            if writer is not None:
                for name, value in epoch_losses.items():
                    writer.add_scalar(f"loss/{name}", value, epoch)
                writer.add_scalar("training/learning_rate", scheduler.get_last_lr()[0], epoch)
                writer.flush()

        if writer is not None:
            writer.close()

        print("\n  Training complete")

    def save(self, save_dir: str, run_id: str = "bases"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.base_2dof.state_dict(), save_path / f"{run_id}_2dof.pt")
        torch.save(self.base_3dof.state_dict(), save_path / f"{run_id}_3dof.pt")

        config = {
            'latent_dim': self.latent_dim,
            'state_dim_2dof': ARM_OBS_2DOF,
            'state_dim_3dof': ARM_OBS_3DOF,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'lmbda': self.lmbda,
        }
        with open(save_path / f"{run_id}_config.pkl", 'wb') as f:
            pickle.dump(config, f)

        print(f"  Bases saved → {save_path}")

    def load(self, save_dir: str, run_id: str = "bases"):
        save_path = Path(save_dir)
        self.base_2dof.load_state_dict(
            torch.load(save_path / f"{run_id}_2dof.pt", map_location=self.device)
        )
        self.base_3dof.load_state_dict(
            torch.load(save_path / f"{run_id}_3dof.pt", map_location=self.device)
        )
        print(f"  Bases loaded ← {save_path}")