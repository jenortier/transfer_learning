"""
Phase 2: Entraînement des BaseVAE sur arm_states uniquement.
Charge les trajectoires depuis DATA_DIR/trajectories.pkl.
Sauvegarde les poids dans DATA_DIR/{base_2dof,base_3dof}.pt.
"""

import sys
import numpy as np
import pickle
from pathlib import Path

import torch

torch.set_num_threads(16)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.bases_vae import BasesVAETrainer, DEFAULT_LATENT_DIM, DEFAULT_HIDDEN_DIM

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data/LSUNN")
TRAJ_PATH = DATA_DIR / "trajectories.pkl"


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"  Loading trajectories from {TRAJ_PATH} …")
    
    with open(TRAJ_PATH, "rb") as f:
        data = pickle.load(f)
    
    print(f"  arm_states_2dof: {data['arm_states_2dof'].shape}")
    print(f"  arm_states_3dof: {data['arm_states_3dof'].shape}")

    trainer = BasesVAETrainer(
        latent_dim=DEFAULT_LATENT_DIM,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        device=DEVICE,
    )

    trainer.train(
        data,
        epochs=100,
        batch_size=100,
        lr=5e-4,
        log_dir=str(DATA_DIR / "tensorboard"),
    )
    trainer.save(str(DATA_DIR), run_id="base")

    print(f"\n  VAE weights saved → {DATA_DIR}/base_{{2,3}}dof.pt")


if __name__ == "__main__":
    main()