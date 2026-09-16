"""
Train mappers using trajectories from genetic algorithm.
Uses ONLY joint positions and velocities (no end-effector in input).

Key insight from Antonio:
- State mapper: use joint_positions + joint_velocities only
- Action mapper: use joint_velocities as target actions

This makes the mapper purely kinematic and more generalizable.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import sys
from .mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    fk_from_arm_state_torch,
    angle_slice_for_arm_dim,
    velocity_slice_for_arm_dim,
    eff_slice_for_arm_dim,
)

# Dimensions
ARM_OBS_2DOF = 6   # Full arm obs (θ1,θ2, dθ1,dθ2, eff_x,eff_y)
ARM_OBS_3DOF = 8   # Full arm obs (θ1,θ2,θ3, dθ1,dθ2,dθ3, eff_x,eff_y)
STATE_DIM_2DOF_KIN = 4   # Only joint positions + velocities (θ1,θ2, dθ1,dθ2)
STATE_DIM_3DOF_KIN = 6   # Only joint positions + velocities (θ1,θ2,θ3, dθ1,dθ2,dθ3)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
EPOCHS = 500
LR = 1e-3

# Loss weights (tuneable)
# Increase angle supervision and reduce FK dominance to prefer matching ground-truth joints
ANGLE_WEIGHT = 8.0
VEL_WEIGHT = 1.0
EFF_WEIGHT = 0.5
FK_WEIGHT = 0.2
ANGLE_SINCOS_WEIGHT = 4.0

DATA_PATH = Path("./data/DIRECT_GEN_ALGO/trajectories_aligned_ss.pkl")
SAVE_DIR = Path("./data/DIRECT_GEN_ALGO")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Cycle-consistency weights
CYCLE_STATE_WEIGHT = 1.0
CYCLE_ACTION_WEIGHT = 1.0






def extract_joint_state(arm_obs: np.ndarray, dof: int) -> np.ndarray:
    """
    Extract only joint positions and velocities from full arm observation.
    
    For 2-DoF: indices 0-3 (θ1,θ2, dθ1,dθ2)
    For 3-DoF: indices 0-5 (θ1,θ2,θ3, dθ1,dθ2,dθ3)
    """
    if dof == 2:
        return arm_obs[:, :4]  # (N, 4)
    else:
        return arm_obs[:, :6]  # (N, 6)


def load_trajectories():
    """Load trajectories and extract joint states only."""
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    
    # Get full arm observations
    s2_full = data['segments_2dof']        # (N, L, 6)
    s3_full = data['segments_3dof']        # (N, L, 8)
    s2_hist = data['segments_2dof_history']
    s3_hist = data['segments_3dof_history']
    a2 = data['segments_actions_2dof']     # (N, L, 2)
    a3 = data['segments_actions_3dof']     # (N, L, 3)
    
    # Extract only joint positions + velocities (kinematic state)
    N, L, _ = s2_full.shape
    s2_joint = s2_full[:, :, :4]           # (N, L, 4)
    s3_joint = s3_full[:, :, :6]           # (N, L, 6)

    print(f"Loaded {N} segments of length {L}")
    print(f"  2-DoF joint state shape: {s2_joint.shape} (theta1,theta2, dtheta1,dtheta2)")
    print(f"  3-DoF joint state shape: {s3_joint.shape} (theta1,theta2,theta3, dtheta1,dtheta2,dtheta3)")

    return {
        's2_full': s2_full,
        's3_full': s3_full,
        's2_hist': s2_hist,
        's3_hist': s3_hist,
        's2_joint': s2_joint,
        's3_joint': s3_joint,
        'a2': a2,
        'a3': a3,
        'n_segments': N,
        'seq_len': L,
    }


def create_weighted_sampler(labels, spatial_weight=0.5, temporal_weight=0.5):
    """Create indices for balanced sampling between spatial and temporal."""
    n_spatial = np.sum(labels == 0)
    n_temporal = np.sum(labels == 1)
    
    spatial_indices = np.where(labels == 0)[0]
    temporal_indices = np.where(labels == 1)[0]
    
    np.random.shuffle(spatial_indices)
    np.random.shuffle(temporal_indices)
    
    return spatial_indices, temporal_indices, n_spatial, n_temporal


def angular_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    """Mean squared angular error taking wrapping into account.

    Inputs are normalized angles in [-1, 1] representing radians/pi.
    """
    pa = pred_norm * torch.pi
    ta = target_norm * torch.pi
    diff = torch.atan2(torch.sin(pa - ta), torch.cos(pa - ta))
    return torch.mean(diff * diff)


def train_mapper_2to3(data):
    """
    Train mapper: (3-DoF joint_state, 2-DoF action) → 3-DoF action
    
    The PPO outputs 2-DoF actions (joint velocity commands).
    We need to map them to 3-DoF actions using 3-DoF joint state.
    """
    print("\n" + "="*60)
    print("Training Action Mapper: 2-DoF -> 3-DoF (Kinematic only)")
    print("="*60)
    print(f"  Input: 3-DoF joint state (6) + 2-DoF action (2)")
    print(f"  Output: 3-DoF action (3)")
    
    # Prepare combined spatial+history inputs for state mapper (3-DoF)
    s3_sp = data['s3_full']      # (N, L, 8)
    s3_hist = data['s3_hist']    # (N, L, 32)
    s2_sp = data['s2_full']      # (N, L, 6)
    a2 = data['a2']              # (N, L, 2)
    a3 = data['a3']              # (N, L, 3)

    N, L, _ = s3_sp.shape
    # Flatten time across segments
    X_state = np.concatenate([
        s3_sp.reshape(-1, 8),
        s3_hist.reshape(-1, 32)
    ], axis=1)   # (N*L, 40)
    y_state = s2_sp.reshape(-1, 6)  # target mapped 2-DoF full obs

    print(f"Training StateMapper 3->2 on {X_state.shape[0]} samples")

    # Convert to tensors
    Xs_t = torch.tensor(X_state, dtype=torch.float32)
    ys_t = torch.tensor(y_state, dtype=torch.float32)

    # Train/val split
    n_samples = len(Xs_t)
    indices = np.random.RandomState(42).permutation(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = torch.utils.data.TensorDataset(Xs_t[train_idx], ys_t[train_idx])
    val_dataset = torch.utils.data.TensorDataset(Xs_t[val_idx], ys_t[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=(DEVICE == 'cuda'))

    state_mapper = StateMapperMLP(input_dim=40, output_dim=6).to(DEVICE)
    opt_s = optim.Adam(state_mapper.parameters(), lr=LR, weight_decay=1e-5)
    sched_s = optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=EPOCHS)
    crit = nn.MSELoss()

    best_val = float('inf')
    best_state_s = None
    pbar = tqdm(range(EPOCHS), desc="StateMapper 3->2")
    for epoch in pbar:
        state_mapper.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = state_mapper(xb)

            angle_sl = angle_slice_for_arm_dim(yb.shape[-1])
            vel_sl = velocity_slice_for_arm_dim(yb.shape[-1])
            eff_sl = eff_slice_for_arm_dim(yb.shape[-1])

            angle_loss = angular_mse(pred[..., angle_sl], yb[..., angle_sl])
            pa = pred[..., angle_sl] * torch.pi
            ta = yb[..., angle_sl] * torch.pi
            sincos_loss = torch.mean((torch.sin(pa) - torch.sin(ta)) ** 2 + (torch.cos(pa) - torch.cos(ta)) ** 2)
            pa = pred[..., angle_sl] * torch.pi
            ta = yb[..., angle_sl] * torch.pi
            sincos_loss = torch.mean((torch.sin(pa) - torch.sin(ta)) ** 2 + (torch.cos(pa) - torch.cos(ta)) ** 2)
            vel_loss = torch.mean((pred[..., vel_sl] - yb[..., vel_sl]) ** 2)
            eff_loss = torch.mean((pred[..., eff_sl] - yb[..., eff_sl]) ** 2)
            fk_loss = torch.mean((fk_from_arm_state_torch(pred) - fk_from_arm_state_torch(yb)) ** 2)

            loss = ANGLE_WEIGHT * angle_loss + ANGLE_SINCOS_WEIGHT * sincos_loss + VEL_WEIGHT * vel_loss + EFF_WEIGHT * eff_loss + FK_WEIGHT * fk_loss

            opt_s.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state_mapper.parameters(), 1.0)
            opt_s.step()
            tr_loss += loss.item()
        sched_s.step()

        # val
        state_mapper.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = state_mapper(xb)

                angle_sl = angle_slice_for_arm_dim(yb.shape[-1])
                vel_sl = velocity_slice_for_arm_dim(yb.shape[-1])
                eff_sl = eff_slice_for_arm_dim(yb.shape[-1])

                angle_loss = angular_mse(pred[..., angle_sl], yb[..., angle_sl])
                pa = pred[..., angle_sl] * torch.pi
                ta = yb[..., angle_sl] * torch.pi
                sincos_loss = torch.mean((torch.sin(pa) - torch.sin(ta)) ** 2 + (torch.cos(pa) - torch.cos(ta)) ** 2)
                vel_loss = torch.mean((pred[..., vel_sl] - yb[..., vel_sl]) ** 2)
                eff_loss = torch.mean((pred[..., eff_sl] - yb[..., eff_sl]) ** 2)
                fk_loss = torch.mean((fk_from_arm_state_torch(pred) - fk_from_arm_state_torch(yb)) ** 2)

                loss = ANGLE_WEIGHT * angle_loss + ANGLE_SINCOS_WEIGHT * sincos_loss + VEL_WEIGHT * vel_loss + EFF_WEIGHT * eff_loss + FK_WEIGHT * fk_loss
                val_loss += loss.item()

        avg_tr = tr_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val:
            best_val = avg_val
            best_state_s = {k: v.clone() for k, v in state_mapper.state_dict().items()}
        pbar.set_postfix({'train': f'{avg_tr:.6f}', 'val': f'{avg_val:.6f}', 'best': f'{best_val:.6f}'})

    if best_state_s is not None:
        state_mapper.load_state_dict(best_state_s)

    # --- Train action mapper (2->3) using combined 3-DoF input (40) and padded 2-DoF actions (4)
    X_act_state = X_state  # (N*L, 40)
    X_act_in = a2.reshape(-1, 2)
    # pad to 4 dims
    pad = np.zeros((X_act_in.shape[0], max(0, 4 - X_act_in.shape[1])), dtype=X_act_in.dtype)
    X_act_in_p = np.concatenate([X_act_in, pad], axis=1)
    y_act = a3.reshape(-1, 3)

    Xs_t = torch.tensor(X_act_state, dtype=torch.float32)
    Xa_t = torch.tensor(X_act_in_p, dtype=torch.float32)
    ya_t = torch.tensor(y_act, dtype=torch.float32)

    n_samples = len(Xs_t)
    indices = np.random.RandomState(42).permutation(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = torch.utils.data.TensorDataset(Xs_t[train_idx], Xa_t[train_idx], ya_t[train_idx])
    val_dataset = torch.utils.data.TensorDataset(Xs_t[val_idx], Xa_t[val_idx], ya_t[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == 'cuda'))

    action_mapper = ActionMapperMLP(state_dim=40, act_in_dim=4, act_out_dim=3).to(DEVICE)
    opt_a = optim.Adam(action_mapper.parameters(), lr=LR, weight_decay=1e-5)
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=EPOCHS)

    best_val = float('inf')
    best_state_a = None
    pbar = tqdm(range(EPOCHS), desc="ActionMapper 2->3")
    for epoch in pbar:
        action_mapper.train()
        tr_loss = 0.0
        for s, a_in, a_tgt in train_loader:
            s = s.to(DEVICE); a_in = a_in.to(DEVICE); a_tgt = a_tgt.to(DEVICE)
            pred = action_mapper(s, a_in)
            loss = crit(pred, a_tgt)
            opt_a.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(action_mapper.parameters(), 1.0); opt_a.step()
            tr_loss += loss.item()
        sched_a.step()

        action_mapper.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, a_in, a_tgt in val_loader:
                s = s.to(DEVICE); a_in = a_in.to(DEVICE); a_tgt = a_tgt.to(DEVICE)
                val_loss += crit(action_mapper(s, a_in), a_tgt).item()

        avg_tr = tr_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        if avg_val < best_val:
            best_val = avg_val
            best_state_a = {k: v.clone() for k, v in action_mapper.state_dict().items()}
        pbar.set_postfix({'train': f'{avg_tr:.6f}', 'val': f'{avg_val:.6f}', 'best': f'{best_val:.6f}'})

    if best_state_a is not None:
        action_mapper.load_state_dict(best_state_a)

    # Save both mappers in single checkpoint (state_mapper + action_mapper)
    save_path = SAVE_DIR / "transfer_2to3_seq.pt"
    torch.save({
        'state_mapper': state_mapper.state_dict(),
        'action_mapper': action_mapper.state_dict(),
    }, save_path)

    print(f"  Saved -> {save_path} (state best val: {best_val:.6f})")
    return state_mapper, action_mapper


def train_mapper_3to2(data):
    """
    Train mapper: (2-DoF joint_state, 3-DoF action) → 2-DoF action
    """
    print("\n" + "="*60)
    print("Training Action Mapper: 3-DoF -> 2-DoF (Kinematic only)")
    print("="*60)
    print(f"  Input: 2-DoF joint state (4) + 3-DoF action (3)")
    print(f"  Output: 2-DoF action (2)")
    
    # For 3->2 training we build combined 2-DoF inputs (spatial+history)
    s2_sp = data['s2_full']
    s2_hist = data['s2_hist']
    s3_sp = data['s3_full']
    a2 = data['a2']
    a3 = data['a3']

    # Flatten
    X_state = np.concatenate([s2_sp.reshape(-1, 6), s2_hist.reshape(-1, 24)], axis=1)  # (N*L, 30)
    y_state = s3_sp.reshape(-1, 8)  # target 3-DoF full obs

    print(f"Training StateMapper 2->3 on {X_state.shape[0]} samples")

    Xs_t = torch.tensor(X_state, dtype=torch.float32)
    ys_t = torch.tensor(y_state, dtype=torch.float32)

    n_samples = len(Xs_t)
    indices = np.random.RandomState(42).permutation(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = torch.utils.data.TensorDataset(Xs_t[train_idx], ys_t[train_idx])
    val_dataset = torch.utils.data.TensorDataset(Xs_t[val_idx], ys_t[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == 'cuda'))

    state_mapper = StateMapperMLP(input_dim=30, output_dim=8).to(DEVICE)
    opt_s = optim.Adam(state_mapper.parameters(), lr=LR, weight_decay=1e-5)
    sched_s = optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=EPOCHS)
    crit = nn.MSELoss()

    best_val = float('inf')
    best_state_s = None
    pbar = tqdm(range(EPOCHS), desc="StateMapper 2->3")
    for epoch in pbar:
        state_mapper.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = state_mapper(xb)

            angle_sl = angle_slice_for_arm_dim(yb.shape[-1])
            vel_sl = velocity_slice_for_arm_dim(yb.shape[-1])
            eff_sl = eff_slice_for_arm_dim(yb.shape[-1])

            angle_loss = angular_mse(pred[..., angle_sl], yb[..., angle_sl])
            pa = pred[..., angle_sl] * torch.pi
            ta = yb[..., angle_sl] * torch.pi
            sincos_loss = torch.mean((torch.sin(pa) - torch.sin(ta)) ** 2 + (torch.cos(pa) - torch.cos(ta)) ** 2)
            vel_loss = torch.mean((pred[..., vel_sl] - yb[..., vel_sl]) ** 2)
            eff_loss = torch.mean((pred[..., eff_sl] - yb[..., eff_sl]) ** 2)
            fk_loss = torch.mean((fk_from_arm_state_torch(pred) - fk_from_arm_state_torch(yb)) ** 2)

            loss = ANGLE_WEIGHT * angle_loss + ANGLE_SINCOS_WEIGHT * sincos_loss + VEL_WEIGHT * vel_loss + EFF_WEIGHT * eff_loss + FK_WEIGHT * fk_loss

            opt_s.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(state_mapper.parameters(), 1.0); opt_s.step()
            tr_loss += loss.item()
        sched_s.step()
        state_mapper.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = state_mapper(xb)

                angle_sl = angle_slice_for_arm_dim(yb.shape[-1])
                vel_sl = velocity_slice_for_arm_dim(yb.shape[-1])
                eff_sl = eff_slice_for_arm_dim(yb.shape[-1])

                angle_loss = angular_mse(pred[..., angle_sl], yb[..., angle_sl])
                pa = pred[..., angle_sl] * torch.pi
                ta = yb[..., angle_sl] * torch.pi
                sincos_loss = torch.mean((torch.sin(pa) - torch.sin(ta)) ** 2 + (torch.cos(pa) - torch.cos(ta)) ** 2)
                vel_loss = torch.mean((pred[..., vel_sl] - yb[..., vel_sl]) ** 2)
                eff_loss = torch.mean((pred[..., eff_sl] - yb[..., eff_sl]) ** 2)
                fk_loss = torch.mean((fk_from_arm_state_torch(pred) - fk_from_arm_state_torch(yb)) ** 2)

                loss = ANGLE_WEIGHT * angle_loss + ANGLE_SINCOS_WEIGHT * sincos_loss + VEL_WEIGHT * vel_loss + EFF_WEIGHT * eff_loss + FK_WEIGHT * fk_loss
                val_loss += loss.item()
        avg_tr = tr_loss / len(train_loader); avg_val = val_loss / len(val_loader)
        if avg_val < best_val:
            best_val = avg_val; best_state_s = {k: v.clone() for k, v in state_mapper.state_dict().items()}
        pbar.set_postfix({'train': f'{avg_tr:.6f}', 'val': f'{avg_val:.6f}', 'best': f'{best_val:.6f}'})

    if best_state_s is not None:
        state_mapper.load_state_dict(best_state_s)

    # --- Train action mapper (3->2) using combined 2-DoF input (30) and 3-DoF actions (3)
    Xs = X_state
    Xa = a3.reshape(-1, 3)
    ya = a2.reshape(-1, 2)

    Xs_t = torch.tensor(Xs, dtype=torch.float32)
    Xa_t = torch.tensor(Xa, dtype=torch.float32)
    ya_t = torch.tensor(ya, dtype=torch.float32)

    n_samples = len(Xs_t)
    indices = np.random.RandomState(42).permutation(n_samples)
    split = int(0.9 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_dataset = torch.utils.data.TensorDataset(Xs_t[train_idx], Xa_t[train_idx], ya_t[train_idx])
    val_dataset = torch.utils.data.TensorDataset(Xs_t[val_idx], Xa_t[val_idx], ya_t[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE == 'cuda'))

    action_mapper = ActionMapperMLP(state_dim=30, act_in_dim=3, act_out_dim=2).to(DEVICE)
    opt_a = optim.Adam(action_mapper.parameters(), lr=LR, weight_decay=1e-5)
    sched_a = optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=EPOCHS)

    best_val = float('inf')
    best_state_a = None
    pbar = tqdm(range(EPOCHS), desc="ActionMapper 3->2")
    for epoch in pbar:
        action_mapper.train()
        tr_loss = 0.0
        for s, a_in, a_tgt in train_loader:
            s = s.to(DEVICE); a_in = a_in.to(DEVICE); a_tgt = a_tgt.to(DEVICE)
            pred = action_mapper(s, a_in)
            loss = crit(pred, a_tgt)
            opt_a.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(action_mapper.parameters(), 1.0); opt_a.step()
            tr_loss += loss.item()
        sched_a.step()
        action_mapper.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, a_in, a_tgt in val_loader:
                s = s.to(DEVICE); a_in = a_in.to(DEVICE); a_tgt = a_tgt.to(DEVICE)
                val_loss += crit(action_mapper(s, a_in), a_tgt).item()
        avg_tr = tr_loss / len(train_loader); avg_val = val_loss / len(val_loader)
        if avg_val < best_val:
            best_val = avg_val; best_state_a = {k: v.clone() for k, v in action_mapper.state_dict().items()}
        pbar.set_postfix({'train': f'{avg_tr:.6f}', 'val': f'{avg_val:.6f}', 'best': f'{best_val:.6f}'})

    if best_state_a is not None:
        action_mapper.load_state_dict(best_state_a)

    save_path = SAVE_DIR / "transfer_3to2_seq.pt"
    torch.save({
        'state_mapper': state_mapper.state_dict(),
        'action_mapper': action_mapper.state_dict(),
    }, save_path)

    print(f"  Saved -> {save_path} (state best val: {best_val:.6f})")
    return state_mapper, action_mapper


def main():
    # Re-evaluate device at runtime (ensures correct interpreter / env)
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*60)
    print("Training Mappers with Kinematic-Only Inputs")
    print("="*60)
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  torch: {torch.__version__}, cuda_available={torch.cuda.is_available()}, cuda={torch.version.cuda}")
    print(f"  python: {sys.executable}")
    
    # Load trajectories
    print("\nLoading trajectories...")
    data = load_trajectories()
    
    # Train mappers
    train_mapper_2to3(data)
    train_mapper_3to2(data)
    
    print("\nMappers trained successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    
    main()
