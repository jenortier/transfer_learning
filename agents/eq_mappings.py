"""
Transfer Learning via Trajectory Correspondence
================================================
Part 2: Train State Mapper MLP  (3-DoF obs  → 2-DoF obs  equivalent)
Part 3: Train Action Mapper MLP (3-DoF obs + 2-DoF action → 3-DoF action)

Key improvements over v1:
  • Action mapper is CONTEXT-AWARE: input = (obs_3dof, action_2dof).
    The same 2-DoF action implies different 3-DoF actions depending on the
    current joint configuration (Jacobian varies with state). Adding the 3-DoF
    state as context lets the mapper condition on the configuration.
  • Larger networks (256 units + LayerNorm).
  • More epochs + cosine-annealing LR schedule.
  • Train/val split for early stopping on best val loss.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Tuple


# ============================================================================
# PART 2: STATE MAPPER  3-DoF obs (10) → 2-DoF obs (9)
# ============================================================================

class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int = 10, output_dim: int = 9, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StateMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4):
        self.device = device
        self.model = StateMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def _split(self, x, y, val_frac=0.1):
        n = len(x)
        idx = torch.randperm(n)
        cut = int(n * (1 - val_frac))
        tr, vl = idx[:cut], idx[cut:]
        return x[tr], y[tr], x[vl], y[vl]

    def train(self, trajectories: Dict, epochs: int = 300, batch_size: int = 512):
        print("\n[Part 2] Training State Mapper (3DoF → 2DoF)")

        s3 = torch.tensor(
            np.concatenate(trajectories['states_3dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        s2 = torch.tensor(
            np.concatenate(trajectories['states_2dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        print(f"  Dataset: {len(s3):,} samples")

        x_tr, y_tr, x_vl, y_vl = self._split(s3, s2)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        best_val = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(x_tr))
            total_loss, n_b = 0.0, 0
            for i in range(0, len(x_tr), batch_size):
                idx = perm[i:i + batch_size]
                loss = self.criterion(self.model(x_tr[idx]), y_tr[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); n_b += 1
            scheduler.step()

            if (epoch + 1) % 50 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.criterion(self.model(x_vl), y_vl).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  Epoch {epoch+1:3d}/{epochs}  train={total_loss/n_b:.6f}  val={val_loss:.6f}"
                      f"  {'← best' if val_loss == best_val else ''}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  State Mapper training complete (best val loss: {best_val:.6f})")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Part 2] State Mapper saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[Part 2] State Mapper loaded ← {path}")


# ============================================================================
# PART 3: ACTION MAPPER  (3-DoF obs, 2-DoF action) → 3-DoF action
# ============================================================================

class ActionMapperMLP(nn.Module):
    """
    Context-aware action mapper.

    Input:  concat(obs_3dof [10], action_2dof [2]) = 12 dimensions
    Output: action_3dof [3]

    The 3-DoF state provides the Jacobian context so the mapper can decide
    how to distribute the 2-DoF joint motion across the 3 DoFs.
    """
    def __init__(self, state_dim: int = 10, action_2dof_dim: int = 2,
                 output_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_2dof_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim),
            nn.Tanh(),   # actions bounded to [-1, 1]
        )

    def forward(self, state_3dof: torch.Tensor, action_2dof: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_3dof, action_2dof], dim=-1)
        return self.net(x)


class ActionMapperTrainer:
    def __init__(self, device: str = "cpu", lr: float = 3e-4):
        self.device = device
        self.model = ActionMapperMLP().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def _split(self, *tensors, val_frac=0.1):
        n = len(tensors[0])
        idx = torch.randperm(n)
        cut = int(n * (1 - val_frac))
        tr, vl = idx[:cut], idx[cut:]
        return tuple(t[tr] for t in tensors) + tuple(t[vl] for t in tensors)

    def train(self, trajectories: Dict, epochs: int = 300, batch_size: int = 512):
        print("\n[Part 3] Training Action Mapper (obs_3dof + action_2dof → action_3dof)")

        s3 = torch.tensor(
            np.concatenate(trajectories['states_3dof'],  axis=0), dtype=torch.float32
        ).to(self.device)
        a2 = torch.tensor(
            np.concatenate(trajectories['actions_2dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        a3 = torch.tensor(
            np.concatenate(trajectories['actions_3dof'], axis=0), dtype=torch.float32
        ).to(self.device)
        print(f"  Dataset: {len(s3):,} samples")

        splits = self._split(s3, a2, a3)
        s3_tr, a2_tr, a3_tr = splits[0], splits[1], splits[2]
        s3_vl, a2_vl, a3_vl = splits[3], splits[4], splits[5]

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        best_val = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(s3_tr))
            total_loss, n_b = 0.0, 0
            for i in range(0, len(s3_tr), batch_size):
                idx = perm[i:i + batch_size]
                pred = self.model(s3_tr[idx], a2_tr[idx])
                loss = self.criterion(pred, a3_tr[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item(); n_b += 1
            scheduler.step()

            if (epoch + 1) % 50 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.criterion(self.model(s3_vl, a2_vl), a3_vl).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                print(f"  Epoch {epoch+1:3d}/{epochs}  train={total_loss/n_b:.6f}  val={val_loss:.6f}"
                      f"  {'← best' if val_loss == best_val else ''}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"  Action Mapper training complete (best val loss: {best_val:.6f})")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Part 3] Action Mapper saved → {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"[Part 3] Action Mapper loaded ← {path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir          = Path("./data/transfer_learning")
    traj_path         = data_dir / "trajectories.pkl"
    state_mapper_path = data_dir / "state_mapper.pt"
    action_mapper_path = data_dir / "action_mapper.pt"

    print("\n" + "="*70)
    print("LOADING TRAJECTORIES")
    print("="*70)
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)
    meta = trajectories['metadata']
    print(f"  {meta['n_pairs']} pairs  |  source: {meta.get('source', '?')}"
          f"  |  seed: {meta['seed']}")
    if meta.get('source') != 'ppo_policy':
        print("\n  ⚠  WARNING: trajectories were NOT generated with PPO policies.")
        print("     Re-run eq_trajectories.py first for best results.\n")

    # ---- State Mapper ----
    print("\n" + "="*70)
    print("PART 2: STATE MAPPER  (3-DoF obs → 2-DoF obs)")
    print("="*70)
    st = StateMapperTrainer(device=device)
    st.train(trajectories, epochs=300, batch_size=512)
    st.save(str(state_mapper_path))

    # ---- Action Mapper ----
    print("\n" + "="*70)
    print("PART 3: ACTION MAPPER  (obs_3dof + action_2dof → action_3dof)")
    print("="*70)
    at = ActionMapperTrainer(device=device)
    at.train(trajectories, epochs=300, batch_size=512)
    at.save(str(action_mapper_path))

    print("\n" + "="*70)
    print("MAPPER TRAINING COMPLETE")
    print("="*70)
    print(f"  State Mapper  → {state_mapper_path}")
    print(f"  Action Mapper → {action_mapper_path}")
    print("\nTransfer pipeline at inference:")
    print("  1. obs_2dof_equiv = state_mapper(obs_3dof)")
    print("  2. obs_2dof_norm  = vec_norm_2dof.normalize_obs(obs_2dof_equiv)")
    print("  3. action_2dof    = policy_2dof(obs_2dof_norm)")
    print("  4. action_3dof    = action_mapper(obs_3dof, action_2dof)   ← context-aware")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
