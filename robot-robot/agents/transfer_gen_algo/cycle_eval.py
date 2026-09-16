import pickle
import numpy as np
import torch
from pathlib import Path
from agents.transfer_gen_algo.mapper_models import StateMapperMLP, ActionMapperMLP

DATA_PATH = Path("./data/DIRECT_GEN_ALGO/trajectories_aligned_ss.pkl")
M2_PATH = Path("./data/DIRECT_GEN_ALGO/transfer_2to3_seq.pt")
M3_PATH = Path("./data/DIRECT_GEN_ALGO/transfer_3to2_seq.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

s2_sp = data['segments_2dof'].reshape(-1, 6)
if data.get('segments_2dof_temporal') is None:
    s2_tt = s2_sp.copy()
else:
    s2_tt = data['segments_2dof_temporal'].reshape(-1, 6)

s3_sp = data['segments_3dof'].reshape(-1, 8)
if data.get('segments_3dof_temporal') is None:
    s3_tt = s3_sp.copy()
else:
    s3_tt = data['segments_3dof_temporal'].reshape(-1, 8)

a2 = data['segments_actions_2dof'].reshape(-1, 2)
a3 = data['segments_actions_3dof'].reshape(-1, 3)

n = len(s2_sp)
indices = np.random.RandomState(42).permutation(n)
split = int(0.9 * n)
val_idx = indices[split:]

# load mappers
m2 = torch.load(M2_PATH, map_location=DEVICE)
m3 = torch.load(M3_PATH, map_location=DEVICE)

state_3to2 = StateMapperMLP(16, 6).to(DEVICE)
state_3to2.load_state_dict(m2['state_mapper'])
state_3to2.eval()

action_2to3 = ActionMapperMLP(16, 4, 3).to(DEVICE)
action_2to3.load_state_dict(m2['action_mapper'])
action_2to3.eval()

state_2to3 = StateMapperMLP(12, 8).to(DEVICE)
state_2to3.load_state_dict(m3['state_mapper'])
state_2to3.eval()

action_3to2 = ActionMapperMLP(12, 3, 2).to(DEVICE)
action_3to2.load_state_dict(m3['action_mapper'])
action_3to2.eval()

mse = lambda a,b: np.mean((a-b)**2)

state_2_cycle_errs = []
state_3_cycle_errs = []
action_2_cycle_errs = []
action_3_cycle_errs = []
forward_s3_errs = []
forward_s2_errs = []

with torch.no_grad():
    for i in val_idx:
        # prepare tensors
        s2_sp_i = torch.tensor(s2_sp[i:i+1], dtype=torch.float32).to(DEVICE)
        s2_tt_i = torch.tensor(s2_tt[i:i+1], dtype=torch.float32).to(DEVICE)
        s3_sp_i = torch.tensor(s3_sp[i:i+1], dtype=torch.float32).to(DEVICE)
        s3_tt_i = torch.tensor(s3_tt[i:i+1], dtype=torch.float32).to(DEVICE)
        a2_i = torch.tensor(a2[i:i+1], dtype=torch.float32).to(DEVICE)
        a3_i = torch.tensor(a3[i:i+1], dtype=torch.float32).to(DEVICE)

        # 2 -> 3 -> 2 (state)
        x2 = torch.cat([s2_sp_i, s2_tt_i], dim=1)
        pred_s3 = state_2to3(x2)
        x3_from_pred = torch.cat([pred_s3, s3_tt_i], dim=1)
        pred_s2_back = state_3to2(x3_from_pred)

        s2_cycle = mse(pred_s2_back.cpu().numpy(), s2_sp_i.cpu().numpy())
        state_2_cycle_errs.append(s2_cycle)

        # forward s3 error
        fwd_s3 = mse(pred_s3.cpu().numpy(), s3_sp_i.cpu().numpy())
        forward_s3_errs.append(fwd_s3)

        # 3 -> 2 -> 3 (state)
        x3 = torch.cat([s3_sp_i, s3_tt_i], dim=1)
        pred_s2 = state_3to2(x3)
        x2_from_pred = torch.cat([pred_s2, s2_tt_i], dim=1)
        pred_s3_back = state_2to3(x2_from_pred)

        s3_cycle = mse(pred_s3_back.cpu().numpy(), s3_sp_i.cpu().numpy())
        state_3_cycle_errs.append(s3_cycle)

        fwd_s2 = mse(pred_s2.cpu().numpy(), s2_sp_i.cpu().numpy())
        forward_s2_errs.append(fwd_s2)

        # action cycle 2 -> 3 -> 2
        s3_comb = torch.cat([s3_sp_i, s3_tt_i], dim=1)
        a2_pad = torch.cat([a2_i, torch.zeros((1,2), dtype=torch.float32).to(DEVICE)], dim=1)
        pred_a3 = action_2to3(s3_comb, a2_pad)
        pred_a2_back = action_3to2(torch.cat([s2_sp_i, s2_tt_i], dim=1), pred_a3)
        action_2_cycle_errs.append(mse(pred_a2_back.cpu().numpy(), a2_i.cpu().numpy()))

        # action cycle 3 -> 2 -> 3
        pred_a2 = action_3to2(torch.cat([s2_sp_i, s2_tt_i], dim=1), a3_i)
        pad_pred_a2 = torch.cat([pred_a2, torch.zeros((1,2), dtype=torch.float32).to(DEVICE)], dim=1)
        pred_a3_back = action_2to3(s3_comb, pad_pred_a2)
        action_3_cycle_errs.append(mse(pred_a3_back.cpu().numpy(), a3_i.cpu().numpy()))

# summarize
print('Validation samples:', len(val_idx))
print('State 2->3->2 cycle MSE (mean):', float(np.mean(state_2_cycle_errs)))
print('State 3->2->3 cycle MSE (mean):', float(np.mean(state_3_cycle_errs)))
print('Forward s3 MSE (2->3 prediction vs true):', float(np.mean(forward_s3_errs)))
print('Forward s2 MSE (3->2 prediction vs true):', float(np.mean(forward_s2_errs)))
print('Action 2->3->2 cycle MSE (mean):', float(np.mean(action_2_cycle_errs)))
print('Action 3->2->3 cycle MSE (mean):', float(np.mean(action_3_cycle_errs)))

print('\nPercentiles for state 2-cycle MSE:')
for p in [50,75,90,99]:
    print(f' {p}th: {np.percentile(state_2_cycle_errs, p):.6f}')

print('\nPercentiles for action 2-cycle MSE:')
for p in [50,75,90,99]:
    print(f' {p}th: {np.percentile(action_2_cycle_errs, p):.6f}')
