"""
Recherche de la dimension latente optimale (Fig. A.10 du papier).
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.bases_vae import BaseVAE

# Dimensions
ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8


def load_trajectories(traj_path: str) -> dict:
    with open(traj_path, 'rb') as f:
        return pickle.load(f)


def train_single_vae_pair(
    s2: torch.Tensor,
    s3: torch.Tensor,
    latent_dim: int,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 100,
    lr: float = 5e-4,
    device: str = "cpu",
) -> float:
    """Entraîne une paire de VAE avec les hparams corrigés et retourne l'erreur de reconstruction."""
    
    base_2dof = BaseVAE(ARM_OBS_2DOF, latent_dim, hidden_dim).to(device)
    base_3dof = BaseVAE(ARM_OBS_3DOF, latent_dim, hidden_dim).to(device)
    
    optimizer = optim.Adam(
        list(base_2dof.parameters()) + list(base_3dof.parameters()),
        lr=lr
    )
    recon_crit = nn.MSELoss()
    sim_crit = nn.MSELoss()
    
    # Split train/val
    n = len(s2)
    perm = torch.randperm(n)
    train_idx = perm[:int(0.8 * n)]
    val_idx = perm[int(0.8 * n):]
    
    s2_train, s3_train = s2[train_idx], s3[train_idx]
    s2_val, s3_val = s2[val_idx], s3[val_idx]
    
    # Hyperparamètres corrigés
    alpha = 1.0
    beta = 0.00015
    gamma = 2.0 / 3.0
    lmbda = 1.0 / 3.0
    
    for epoch in tqdm(range(epochs), unit=" epoch", desc=f"dim={latent_dim}"): 
        perm_train = torch.randperm(len(s2_train))

        #for i in range(0, len(s2_train), batch_size):
        for i in tqdm(range(0, len(s2_train), batch_size), unit=" batch", desc=f"dim={latent_dim}", leave=False):
            idx = perm_train[i:i+batch_size]
            b2, b3 = s2_train[idx], s3_train[idx]
            
            z2, mu2, logvar2 = base_2dof.encoder(b2)
            z3, mu3, logvar3 = base_3dof.encoder(b3)
            
            recon2 = base_2dof.decoder(z2)
            recon3 = base_3dof.decoder(z3)
            cross2 = base_2dof.decoder(z3)
            cross3 = base_3dof.decoder(z2)
            
            loss_recon = recon_crit(recon2, b2) + recon_crit(recon3, b3)
            loss_kl = (-0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()) / b2.shape[0] +
                       -0.5 * torch.sum(1 + logvar3 - mu3.pow(2) - logvar3.exp()) / b3.shape[0])
            loss_sim = sim_crit(z2, z3)
            loss_cross = recon_crit(cross2, b2) + recon_crit(cross3, b3)
            
            loss = alpha * loss_recon + beta * loss_kl + gamma * loss_sim + lmbda * loss_cross
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Évaluation: MSE de reconstruction pure sur validation
    with torch.no_grad():
        z2_val, _, _ = base_2dof.encoder(s2_val)
        z3_val, _, _ = base_3dof.encoder(s3_val)
        recon2_val = base_2dof.decoder(z2_val)
        recon3_val = base_3dof.decoder(z3_val)
        mse2 = recon_crit(recon2_val, s2_val).item()
        mse3 = recon_crit(recon3_val, s3_val).item()
    
    return (mse2 + mse3) / 2


def main(traj_path: str = "./data/LSUNN/trajectories.pkl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data = load_trajectories(traj_path)
    s2 = torch.tensor(data['arm_states_2dof'], dtype=torch.float32, device=device)
    s3 = torch.tensor(data['arm_states_3dof'], dtype=torch.float32, device=device)
    
    latent_dims = [2, 3, 4, 5, 6, 8]
    results = []
    
    for dim in latent_dims:
        print(f"\nTesting latent_dim = {dim}")
        mse = train_single_vae_pair(s2, s3, dim, device=device)
        results.append((dim, mse))
        print(f"  Reconstruction MSE: {mse:.6f}")
    
    # Plot
    dims, ms = zip(*results)
    plt.figure(figsize=(8, 5))
    plt.plot(dims, ms, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Reconstruction MSE (validation)')
    plt.title('VAE Reconstruction Error vs Latent Dimension')
    plt.grid(True, alpha=0.3)
    plt.savefig('latent_dim_search.png', dpi=150)
    print("\nFigure saved to latent_dim_search.png")
    
    # Recommendation: find elbow point
    print("\n=== Recommendation ===")
    print(f"Results: {results}")
    print(f"Elbow around dim={dims[np.argmin(np.diff(ms)) + 1]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(traj_path=sys.argv[1])
    else:
        main()