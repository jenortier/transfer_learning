import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from agents.transfer_gen_algo.mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    fk_from_arm_state_torch,
    angle_slice_for_arm_dim,
    velocity_slice_for_arm_dim,
    eff_slice_for_arm_dim,
)

# Small helpers

def angular_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    pa = pred_norm * torch.pi
    ta = target_norm * torch.pi
    diff = torch.atan2(torch.sin(pa - ta), torch.cos(pa - ta))
    return torch.mean(diff * diff)


def build_dataset(data_path: Path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    s2_sp = data['segments_2dof'].reshape(-1, 6)
    s3_sp = data['segments_3dof'].reshape(-1, 8)
    s2_hist = data['segments_2dof_history'].reshape(-1, 24)
    s3_hist = data['segments_3dof_history'].reshape(-1, 32)

    a2 = data['segments_actions_2dof'].reshape(-1, 2)
    a3 = data['segments_actions_3dof'].reshape(-1, 3)

    X3_in = np.concatenate([s3_sp, s3_hist], axis=1)  # 40
    X2_in = np.concatenate([s2_sp, s2_hist], axis=1)  # 30

    return {
        'X3_in': X3_in.astype(np.float32),
        'X2_in': X2_in.astype(np.float32),
        's3_sp': s3_sp.astype(np.float32),
        's2_sp': s2_sp.astype(np.float32),
        's3_hist': s3_hist.astype(np.float32),
        's2_hist': s2_hist.astype(np.float32),
        'a2': a2.astype(np.float32),
        'a3': a3.astype(np.float32),
    }


def train_joint(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PATH = Path('./data/DIRECT_GEN_ALGO/trajectories_aligned_ss.pkl')
    SAVE_DIR = Path('./data/DIRECT_GEN_ALGO')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = build_dataset(PATH)
    n = cfg['X3_in'].shape[0]
    indices = np.random.RandomState(42).permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    # tensors
    X3 = torch.tensor(cfg['X3_in'], dtype=torch.float32)
    X2 = torch.tensor(cfg['X2_in'], dtype=torch.float32)
    s3_sp = torch.tensor(cfg['s3_sp'], dtype=torch.float32)
    s2_sp = torch.tensor(cfg['s2_sp'], dtype=torch.float32)
    s3_hist = torch.tensor(cfg['s3_hist'], dtype=torch.float32)
    s2_hist = torch.tensor(cfg['s2_hist'], dtype=torch.float32)
    a2 = torch.tensor(cfg['a2'], dtype=torch.float32)
    a3 = torch.tensor(cfg['a3'], dtype=torch.float32)

    train_ds = TensorDataset(
        X3[train_idx], X2[train_idx], s3_sp[train_idx], s2_sp[train_idx], s3_hist[train_idx], s2_hist[train_idx], a2[train_idx], a3[train_idx]
    )
    val_ds = TensorDataset(
        X3[val_idx], X2[val_idx], s3_sp[val_idx], s2_sp[val_idx], s3_hist[val_idx], s2_hist[val_idx], a2[val_idx], a3[val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # models
    state_3to2 = StateMapperMLP(40, 6).to(device)
    state_2to3 = StateMapperMLP(30, 8).to(device)
    action_2to3 = ActionMapperMLP(40, 4, 3).to(device)
    action_3to2 = ActionMapperMLP(30, 3, 2).to(device)

    # opts
    state_opt = optim.Adam(list(state_3to2.parameters()) + list(state_2to3.parameters()), lr=args.lr, weight_decay=1e-5)
    action_opt = optim.Adam(list(action_2to3.parameters()) + list(action_3to2.parameters()), lr=args.lr, weight_decay=1e-5)

    # loss weights (tunable) - increased reconstruction and angular supervision
    ANGLE_WEIGHT = 12.0
    ANGLE_SINCOS_WEIGHT = 6.0
    VEL_WEIGHT = 1.0
    EFF_WEIGHT = 0.5
    FK_WEIGHT = 0.1
    RECON_WEIGHT = 4.0
    CYCLE_STATE_WEIGHT = 2.0
    CYCLE_ACTION_WEIGHT = 1.0

    mse = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(args.epochs):
        state_3to2.train(); state_2to3.train(); action_2to3.train(); action_3to2.train()
        tr_state_loss = 0.0
        tr_action_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for X3_b, X2_b, s3_b, s2_b, s3hist_b, s2hist_b, a2_b, a3_b in pbar:
            X3_b = X3_b.to(device); X2_b = X2_b.to(device)
            s3_b = s3_b.to(device); s2_b = s2_b.to(device)
            s3hist_b = s3hist_b.to(device); s2hist_b = s2hist_b.to(device)
            a2_b = a2_b.to(device); a3_b = a3_b.to(device)

            # ------- State forward losses -------
            pred2_from3 = state_3to2(X3_b)
            pred3_from2 = state_2to3(X2_b)

            # slices
            angle_sl_2 = angle_slice_for_arm_dim(6)
            vel_sl_2 = velocity_slice_for_arm_dim(6)
            eff_sl_2 = eff_slice_for_arm_dim(6)

            angle_sl_3 = angle_slice_for_arm_dim(8)
            vel_sl_3 = velocity_slice_for_arm_dim(8)
            eff_sl_3 = eff_slice_for_arm_dim(8)

            # 3->2 losses (pred2_from3 vs s2_b)
            angle_loss_3to2 = angular_mse(pred2_from3[..., angle_sl_2], s2_b[..., angle_sl_2])
            pa = pred2_from3[..., angle_sl_2] * torch.pi
            ta = s2_b[..., angle_sl_2] * torch.pi
            sincos_loss_3to2 = torch.mean((torch.sin(pa) - torch.sin(ta)) ** 2 + (torch.cos(pa) - torch.cos(ta)) ** 2)
            vel_loss_3to2 = mse(pred2_from3[..., vel_sl_2], s2_b[..., vel_sl_2])
            eff_loss_3to2 = mse(pred2_from3[..., eff_sl_2], s2_b[..., eff_sl_2])
            fk_loss_3to2 = mse(fk_from_arm_state_torch(pred2_from3), fk_from_arm_state_torch(s2_b))
            recon_loss_3to2 = mse(pred2_from3, s2_b)

            loss_3to2 = ANGLE_WEIGHT * angle_loss_3to2 + ANGLE_SINCOS_WEIGHT * sincos_loss_3to2 + VEL_WEIGHT * vel_loss_3to2 + EFF_WEIGHT * eff_loss_3to2 + FK_WEIGHT * fk_loss_3to2 + RECON_WEIGHT * recon_loss_3to2

            # 2->3 losses (pred3_from2 vs s3_b)
            angle_loss_2to3 = angular_mse(pred3_from2[..., angle_sl_3], s3_b[..., angle_sl_3])
            pa3 = pred3_from2[..., angle_sl_3] * torch.pi
            ta3 = s3_b[..., angle_sl_3] * torch.pi
            sincos_loss_2to3 = torch.mean((torch.sin(pa3) - torch.sin(ta3)) ** 2 + (torch.cos(pa3) - torch.cos(ta3)) ** 2)
            vel_loss_2to3 = mse(pred3_from2[..., vel_sl_3], s3_b[..., vel_sl_3])
            eff_loss_2to3 = mse(pred3_from2[..., eff_sl_3], s3_b[..., eff_sl_3])
            fk_loss_2to3 = mse(fk_from_arm_state_torch(pred3_from2), fk_from_arm_state_torch(s3_b))
            recon_loss_2to3 = mse(pred3_from2, s3_b)

            loss_2to3 = ANGLE_WEIGHT * angle_loss_2to3 + ANGLE_SINCOS_WEIGHT * sincos_loss_2to3 + VEL_WEIGHT * vel_loss_2to3 + EFF_WEIGHT * eff_loss_2to3 + FK_WEIGHT * fk_loss_2to3 + RECON_WEIGHT * recon_loss_2to3

            # cycle state losses
            # 3 -> 2 -> 3
            # to feed 2->3 we need pred2_from3 + 2-DoF history
            s2hist_part = X2_b[:, -24:]
            pred3_back = state_2to3(torch.cat([pred2_from3, s2hist_part], dim=1))
            cycle_loss_3 = mse(pred3_back, s3_b)

            # 2 -> 3 -> 2
            s3hist_part = X3_b[:, -32:]
            pred2_back = state_3to2(torch.cat([pred3_from2, s3hist_part], dim=1))
            cycle_loss_2 = mse(pred2_back, s2_b)

            total_state_loss = loss_3to2 + loss_2to3 + CYCLE_STATE_WEIGHT * (cycle_loss_2 + cycle_loss_3)

            # step state optimizers
            state_opt.zero_grad()
            total_state_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(state_3to2.parameters()) + list(state_2to3.parameters()), 1.0)
            state_opt.step()

            tr_state_loss += float(total_state_loss.detach().cpu().numpy())

            # ------- Action forward + cycle losses -------
            # prepare inputs
            s3_comb = X3_b  # (batch,40)
            s2_comb = X2_b  # (batch,30)
            a2_pad = torch.cat([a2_b, torch.zeros((a2_b.shape[0], 2), device=device)], dim=1)

            pred_a3 = action_2to3(s3_comb, a2_pad)
            loss_a2to3 = mse(pred_a3, a3_b)

            pred_a2 = action_3to2(s2_comb, a3_b)
            loss_a3to2 = mse(pred_a2, a2_b)

            # action cycle 2->3->2
            pred_a2_back = action_3to2(s2_comb, pred_a3)
            cycle_a2 = mse(pred_a2_back, a2_b)

            # action cycle 3->2->3
            pad_pred_a2 = torch.cat([pred_a2, torch.zeros((pred_a2.shape[0], 2), device=device)], dim=1)
            pred_a3_back = action_2to3(s3_comb, pad_pred_a2)
            cycle_a3 = mse(pred_a3_back, a3_b)

            total_action_loss = loss_a2to3 + loss_a3to2 + CYCLE_ACTION_WEIGHT * (cycle_a2 + cycle_a3)

            action_opt.zero_grad()
            total_action_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(action_2to3.parameters()) + list(action_3to2.parameters()), 1.0)
            action_opt.step()

            tr_action_loss += float(total_action_loss.detach().cpu().numpy())

            pbar.set_postfix({
                's_loss': f'{tr_state_loss/ (pbar.n+1):.6f}',
                'a_loss': f'{tr_action_loss/ (pbar.n+1):.6f}'
            })

        # Validation
        state_3to2.eval(); state_2to3.eval(); action_2to3.eval(); action_3to2.eval()
        val_state = 0.0
        val_action = 0.0
        with torch.no_grad():
            for X3_b, X2_b, s3_b, s2_b, s3hist_b, s2hist_b, a2_b, a3_b in val_loader:
                X3_b = X3_b.to(device); X2_b = X2_b.to(device)
                s3_b = s3_b.to(device); s2_b = s2_b.to(device)
                s3hist_b = s3hist_b.to(device); s2hist_b = s2hist_b.to(device)
                a2_b = a2_b.to(device); a3_b = a3_b.to(device)

                pred2_from3 = state_3to2(X3_b)
                pred3_from2 = state_2to3(X2_b)

                # simple recon mse
                val_state += float(mse(pred2_from3, s2_b).cpu().numpy()) + float(mse(pred3_from2, s3_b).cpu().numpy())

                a2_pad = torch.cat([a2_b, torch.zeros((a2_b.shape[0], 2), device=device)], dim=1)
                pred_a3 = action_2to3(X3_b, a2_pad)
                pred_a2 = action_3to2(X2_b, a3_b)
                val_action += float(mse(pred_a3, a3_b).cpu().numpy()) + float(mse(pred_a2, a2_b).cpu().numpy())

        avg_val = val_state / (2 * len(val_loader)) + val_action / (2 * len(val_loader))
        print(f'Epoch {epoch+1}/{args.epochs}  train_state_loss: {tr_state_loss/len(train_loader):.6f}  train_action_loss: {tr_action_loss/len(train_loader):.6f}  val_loss: {avg_val:.6f}')

        # optionally save best
        if avg_val < best_val:
            best_val = avg_val
            save_path = SAVE_DIR / 'transfer_joint_seq.pt'
            torch.save({
                'state_3to2': state_3to2.state_dict(),
                'state_2to3': state_2to3.state_dict(),
                'action_2to3': action_2to3.state_dict(),
                'action_3to2': action_3to2.state_dict(),
            }, save_path)
            # Also save directional checkpoints compatible with existing eval scripts
            td_2to3 = SAVE_DIR / 'transfer_2to3_seq.pt'
            torch.save({
                'state_mapper': state_3to2.state_dict(),
                'action_mapper': action_2to3.state_dict(),
            }, td_2to3)
            td_3to2 = SAVE_DIR / 'transfer_3to2_seq.pt'
            torch.save({
                'state_mapper': state_2to3.state_dict(),
                'action_mapper': action_3to2.state_dict(),
            }, td_3to2)
            print('  Saved best joint mappers ->', save_path)
            print('  Also saved directional checkpoints ->', td_2to3, td_3to2)

    print('Training complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train_joint(args)
