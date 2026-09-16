import pickle
from pathlib import Path
import numpy as np
import math
from agents.transfer_gen_algo.mapper_models import (
    project_mapped_arm_state_to_reference,
    _project_angles_to_ee,
    _fk_np,
)

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
DATA_ROOT = ROOT / "data"

p = DATA_ROOT / "DIRECT_GEN_ALGO" / "diagnostics_30.pkl"
with open(p, 'rb') as f:
    d = pickle.load(f)
fe = d['failed_examples'][0]
step = fe['steps'][0]
raw = np.array(step['state_raw'], dtype=np.float32)
target = np.array(step['target_m'], dtype=np.float32)
arm3 = np.array(step['arm_3'], dtype=np.float32)

proj = project_mapped_arm_state_to_reference(raw, reference_arm_state=arm3)
q_from_q0 = _project_angles_to_ee(raw[:2] * 1.0, target)
q_from_zero = _project_angles_to_ee(np.zeros(2), target)
q_from_clipped = _project_angles_to_ee(np.clip(raw[:2], -1.0, 1.0) * np.pi, target)

print('raw normalized (raw):', raw[:2].tolist())
print('raw_norm * pi (q0 rad):', (raw[:2] * np.pi).tolist())
print('target_m (m):', target.tolist())
print('proj normalized:', proj[:2].tolist())
print('proj rad (proj_norm*pi):', (proj[:2] * np.pi).tolist())
print('proj deg (proj_norm*180):', (proj[:2] * 180.0).tolist())
print('ik q_from_q0 rad:', q_from_q0.tolist())
print('ik q_from_q0 deg:', (q_from_q0 * 180.0 / np.pi).tolist())
print('ik q_from_zero deg:', (q_from_zero * 180.0 / np.pi).tolist())
print('ik q_from_clipped rad:', q_from_clipped.tolist())
print('ik q_from_clipped deg:', (q_from_clipped * 180.0 / np.pi).tolist())
print('proj deg - ik deg:', (proj[:2] * 180.0 - q_from_zero * 180.0 / np.pi).tolist())

# replicate function body
mapped = raw.reshape(1, -1)
desired_ee = arm3.reshape(1, -1)[..., 6:8] * 3.0
out = np.clip(mapped.copy(), -1.0, 1.0)
angle_sl = slice(0, 2)
velocity_sl = slice(2, 4)
eff_sl = slice(4, 6)
for i in range(out.shape[0]):
    q0 = out[i, angle_sl] * np.pi
    q = _project_angles_to_ee(q0, desired_ee[i])
    out[i, angle_sl] = q / np.pi
    out[i, velocity_sl] = np.clip(out[i, velocity_sl], -1.0, 1.0)
    out[i, eff_sl] = np.clip(_fk_np(q) / 3.0, -1.0, 1.0)
print('replicated out angles norm:', out[0, angle_sl].tolist())
print('replicated out eff norm:', out[0, eff_sl].tolist())
print('desired_ee from arm3 (m):', desired_ee[0].tolist())
print('step[target_m] (m):', target.tolist())

# show both analytic branches for reference
x, y = float(target[0]), float(target[1])
l1, l2 = 1.5, 1.5
r = math.hypot(x, y)
reach = l1 + l2 - 1e-6
if r > reach:
    x *= reach / r
    y *= reach / r
    r = reach
cos_t2 = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
cos_t2 = max(-1.0, min(1.0, cos_t2))
t2a = math.acos(cos_t2)
t2b = -t2a
k1a = l1 + l2 * cos_t2
k2a = l2 * math.sin(t2a)
t1a = math.atan2(y, x) - math.atan2(k2a, k1a)
k1b = l1 + l2 * math.cos(t2b)
k2b = l2 * math.sin(t2b)
t1b = math.atan2(y, x) - math.atan2(k2b, k1b)
print('analytic branch A deg:', (t1a * 180.0 / math.pi, t2a * 180.0 / math.pi))
print('analytic branch B deg:', (t1b * 180.0 / math.pi, t2b * 180.0 / math.pi))
