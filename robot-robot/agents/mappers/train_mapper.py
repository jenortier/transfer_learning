# agents/mappers/train_mapper.py
import os
import time
import numpy as np
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import argparse

from agents.mappers.mapper_direct import DirectMapper
from agents.mappers.mapper_residual import ResidualMapper


# ==========================================================
# CPU
# ==========================================================
torch.set_num_threads(16)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# ARGPARSE
# ==========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--residual",
    action="store_true",
    help="Use residual mapping instead of direct mapping"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=300,
    help="Number of training epochs"
)

args = parser.parse_args()

use_residual = args.residual
epochs = args.epochs


# ==========================================================
# LOSS FUNCTIONS (corrigées pour les bonnes dimensions)
# ==========================================================
def compute_loss(pred, target, w_pos=1.0, w_vel=0.5):
    """
    pred: (batch, 8) - prédiction de s3 (3DoF)
    target: (batch, 8) - s3 réel
    
    Index 0-2: θ1/π, θ2/π, θ3/π
    Index 3-5: dθ1/ω_max, dθ2/ω_max, dθ3/ω_max
    Index 6-7: x/r, y/r (position EE)
    """
    
    # Position des articulations (3 premières dimensions)
    pos_loss = nn.MSELoss()(
        pred[:, :3],
        target[:, :3]
    )
    
    # Vitesses des articulations (3 dimensions suivantes)
    vel_loss = nn.MSELoss()(
        pred[:, 3:6],
        target[:, 3:6]
    )
    
    # Position de l'effecteur (2 dernières dimensions)
    ee_loss = nn.MSELoss()(
        pred[:, 6:8],
        target[:, 6:8]
    )
    
    return w_pos * pos_loss + w_vel * vel_loss + 0.1 * ee_loss


def residual_loss(pred, target, w_pos=1.0, w_vel=0.5):
    """
    pred   = s2 + delta (8 dims)
    target = s3 (8 dims)
    
    Index 0-2: θ1/π, θ2/π, θ3/π
    Index 3-5: dθ1/ω_max, dθ2/ω_max, dθ3/ω_max
    Index 6-7: x/r, y/r (position EE)
    """
    
    pos_loss = nn.MSELoss()(
        pred[:, :3],
        target[:, :3]
    )
    
    vel_loss = nn.MSELoss()(
        pred[:, 3:6],
        target[:, 3:6]
    )
    
    ee_loss = nn.MSELoss()(
        pred[:, 6:8],
        target[:, 6:8]
    )
    
    return w_pos * pos_loss + w_vel * vel_loss + 0.1 * ee_loss


# ==========================================================
# PATHS
# ==========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "DIRECT_PPO")


# ==========================================================
# LOAD DATA
# ==========================================================
def load_aligned_data():
    data_path = os.path.join(DATA_DIR, "aligned_traj_2to3.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Aligned trajectory data not found at {data_path}. "
            "Run the trajectory alignment step first."
        )
    data = np.load(data_path)
    # data: [s2 (6 dims) | s3 (8 dims)]
    s2 = data[:, :6]   # 6 dims for 2DoF
    s3 = data[:, 6:14] # 8 dims for 3DoF
    return s2, s3


# ==========================================================
# MAIN
# ==========================================================
if multiprocessing.current_process().name == "MainProcess":

    s2, s3 = load_aligned_data()
    
    print(f"Loaded data: s2 shape {s2.shape}, s3 shape {s3.shape}")

    dataset = TensorDataset(
        torch.tensor(s2, dtype=torch.float32),
        torch.tensor(s3, dtype=torch.float32)
    )

    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # ==========================================================
    # MODEL SELECTION
    # ==========================================================
    if use_residual:
        print("Using RESIDUAL mapper")
        model = ResidualMapper()
        loss_fn = residual_loss
        save_dir = os.path.join(CHECKPOINT_DIR, "residual", "checkpoints")
        save_name = "mapper_residual.pt"
    else:
        print("Using DIRECT mapper")
        model = DirectMapper()
        loss_fn = compute_loss
        save_dir = os.path.join(CHECKPOINT_DIR, "complete", "checkpoints")
        save_name = "mapper_direct.pt"

    # Les mappers doivent être modifiés pour prendre input_dim=6, output_dim=8
    # Nous allons les adapter dynamiquement
    if hasattr(model, 'mlp'):
        # DirectMapper
        model.mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 8)
        )
    elif hasattr(model, 'delta_net'):
        # ResidualMapper
        model.delta_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 8)
        )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    # ==========================================================
    # OPTIMIZER
    # ==========================================================
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ==========================================================
    # TRAIN LOOP
    # ==========================================================
    print("\nUsing device:", DEVICE, "\n")
    
    start_time = time.time()
    pbar = tqdm(total=epochs, desc="Epochs", unit="ep", device=DEVICE)
    
    ##for epoch in range(epochs):
    for epoch in pbar:

        total_loss = 0.0

        for s2_batch, s3_batch in loader:

            pred = model(s2_batch)

            loss = loss_fn(pred, s3_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        ##pbar.add(1)
        elapsed = time.time() - start_time

        #print(f"Epoch {epoch:03d} | loss = {total_loss:.6f}")
        pbar.set_postfix({"loss": f"{total_loss:.6f}"})

    ##pbar.close()
    
    # ==========================================================
    # SAVE MODEL
    # ==========================================================
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
