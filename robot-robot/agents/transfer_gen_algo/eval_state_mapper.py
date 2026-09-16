import pickle
import numpy as np
import torch
import sys
from pathlib import Path

# Ensure workspace root is on sys.path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.transfer_gen_algo.mapper_models import (
    StateMapperMLP,
    fk_from_arm_state_torch,
    project_mapped_arm_state_to_reference,
    angle_slice_for_arm_dim,
    velocity_slice_for_arm_dim,
    eff_slice_for_arm_dim,
)

DATA_PATH = Path("./data/DIRECT_GEN_ALGO/trajectories_aligned_ss.pkl")
MAPPER_PATH = Path("./data/DIRECT_GEN_ALGO/transfer_2to3_seq.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def angular_rmse_rad(pred_norm, targ_norm):
    # pred_norm and targ_norm are normalized in [-1,1] representing radians/pi
    pa = pred_norm * np.pi
    ta = targ_norm * np.pi
    diff = np.arctan2(np.sin(pa - ta), np.cos(pa - ta))
    return np.sqrt(np.mean(diff ** 2))


with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

s3_sp = data['segments_3dof'].reshape(-1, 8)
s3_tt = data['segments_3dof_temporal'].reshape(-1, 8)
s2_sp = data['segments_2dof'].reshape(-1, 6)

X_state = np.concatenate([s3_sp, s3_tt], axis=1)  # (N*L, 16)
y_state = s2_sp  # (N*L, 6)

n_samples = len(X_state)
indices = np.random.RandomState(42).permutation(n_samples)
split = int(0.9 * n_samples)
val_idx = indices[split:]

# Load mapper
checkpoint = torch.load(MAPPER_PATH, map_location=DEVICE)
state_mapper = StateMapperMLP(16, 6).to(DEVICE)
state_mapper.load_state_dict(checkpoint['state_mapper'])
state_mapper.eval()

angle_sl = angle_slice_for_arm_dim(6)
vel_sl = velocity_slice_for_arm_dim(6)
eff_sl = eff_slice_for_arm_dim(6)

preds = []
targs = []
refs = []

with torch.no_grad():
    for i in val_idx:
        x = torch.tensor(X_state[i:i+1], dtype=torch.float32).to(DEVICE)
        pred = state_mapper(x).cpu().numpy()[0]
        preds.append(pred)
        targ = y_state[i]
        t3_ref = s3_sp[i]
        t3_ref = t3_ref.reshape(1, -1)
        refs.append(t3_ref[0])
        targs.append(targ)

preds = np.stack(preds, axis=0)
targs = np.stack(targs, axis=0)
refs = np.stack(refs, axis=0)

# Pre-projection metrics
angle_rmse_pre = angular_rmse_rad(preds[..., angle_sl], targs[..., angle_sl])
vel_rmse_pre = np.sqrt(np.mean((preds[..., vel_sl] - targs[..., vel_sl]) ** 2))
eff_rmse_pre = np.sqrt(np.mean((preds[..., eff_sl] - targs[..., eff_sl]) ** 2))

# FK-based effector RMSE (predicted FK vs true FK)
with torch.no_grad():
    pred_fk = fk_from_arm_state_torch(torch.tensor(preds, dtype=torch.float32)).cpu().numpy()
    targ_fk = fk_from_arm_state_torch(torch.tensor(targs, dtype=torch.float32)).cpu().numpy()
fk_rmse_pre = np.sqrt(np.mean((pred_fk - targ_fk) ** 2))

# Post-projection
preds_proj = []
for p, ref in zip(preds, refs):
    pp = project_mapped_arm_state_to_reference(p, reference_arm_state=ref)
    preds_proj.append(pp)
preds_proj = np.stack(preds_proj, axis=0)

angle_rmse_post = angular_rmse_rad(preds_proj[..., angle_sl], targs[..., angle_sl])
vel_rmse_post = np.sqrt(np.mean((preds_proj[..., vel_sl] - targs[..., vel_sl]) ** 2))
eff_rmse_post = np.sqrt(np.mean((preds_proj[..., eff_sl] - targs[..., eff_sl]) ** 2))

with torch.no_grad():
    pred_fk_post = fk_from_arm_state_torch(torch.tensor(preds_proj, dtype=torch.float32)).cpu().numpy()
fk_rmse_post = np.sqrt(np.mean((pred_fk_post - targ_fk) ** 2))

print("Validation samples:", len(val_idx))
print(f"Angle RMSE (rad) pre: {angle_rmse_pre:.4f}, post: {angle_rmse_post:.4f}")
print(f"Angle RMSE (deg) pre: {angle_rmse_pre * 180.0 / np.pi:.2f}, post: {angle_rmse_post * 180.0 / np.pi:.2f}")
print(f"Velocity RMSE pre: {vel_rmse_pre:.4f}, post: {vel_rmse_post:.4f}")
print(f"Effector field RMSE pre: {eff_rmse_pre:.4f}, post: {eff_rmse_post:.4f}")
print(f"FK RMSE pre: {fk_rmse_pre:.6f}, post: {fk_rmse_post:.6f}")

# angle absolute error percentiles
ang_errs = np.abs(np.arctan2(np.sin(preds[..., angle_sl] * np.pi - targs[..., angle_sl] * np.pi), np.cos(preds[..., angle_sl] * np.pi - targs[..., angle_sl] * np.pi)))
ang_err_deg = (ang_errs * 180.0 / np.pi).reshape(-1)
for p in [50, 75, 90, 99]:
    print(f"Angle abs error {p}th percentile: {np.percentile(ang_err_deg, p):.2f} deg")
