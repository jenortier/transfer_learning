import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data"
MODEL_ROOT = DATA_ROOT / "models"
sys.path.insert(0, str(ROOT))

import argparse
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.env_reaching_2dof import ReachingEnv_2dof
from agents.transfer_gen_algo.mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    project_mapped_arm_state_to_reference,
    fk_from_arm_state_torch,
    MAX_REACH,
)


def ik_2dof_for_target(x, y, l1=1.5, l2=1.5):
    r = np.hypot(x, y)
    reach = l1 + l2 - 1e-6
    if r > reach:
        x = x * (reach / r)
        y = y * (reach / r)
        r = reach
    cos_t2 = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)
    t2 = float(np.arccos(cos_t2))
    k1 = l1 + l2 * cos_t2
    k2 = l2 * np.sin(t2)
    t1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
    return np.array([t1, t2], dtype=np.float32)


def angular_diff_rad(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))


def run_diagnostics(episodes=100, max_steps=200, out_path=None, device='cpu'):
    # Paths used in test harness
    POLICY_2DOF = MODEL_ROOT / "ppo_reach_2dof_1" / "best_model.zip"
    VECNORM_2DOF = MODEL_ROOT / "ppo_reach_2dof_1" / "vec_normalize.pkl"
    MAPPER_PATH = DATA_ROOT / "DIRECT_GEN_ALGO" / "transfer_2to3_seq.pt"

    # Load policy and vecnorm
    policy_2dof = PPO.load(str(POLICY_2DOF), device=device)
    venv_2 = DummyVecEnv([lambda: Monitor(ReachingEnv_2dof(render_mode=None))])
    vec_norm_2dof = torch.load if False else None
    # Use stable-baselines load pattern for VecNormalize
    from stable_baselines3.common.vec_env import VecNormalize
    vec_norm_2dof = VecNormalize.load(str(VECNORM_2DOF), venv=DummyVecEnv([lambda: Monitor(ReachingEnv_2dof(render_mode=None))]))
    vec_norm_2dof.training = False
    vec_norm_2dof.norm_reward = False

    # Load mappers
    checkpoint = torch.load(str(MAPPER_PATH), map_location=device)
    state_mapper = StateMapperMLP(16, 6).to(device)
    state_mapper.load_state_dict(checkpoint['state_mapper'])
    state_mapper.eval()
    action_mapper = ActionMapperMLP(16, 4, 3).to(device)
    action_mapper.load_state_dict(checkpoint['action_mapper'])
    action_mapper.eval()

    env = DummyVecEnv([lambda: Monitor(ReachingEnv_3dof(render_mode=None))])

    results = {
        'episodes': [],
        'angle_errors_deg': [],
        'pred_eff_to_target_m': [],
        'true_eff_to_target_m': [],
    }

    failed_examples = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        ep_min_dist = float('inf')
        ep_step_logs = []
        last_info = {}

        while not done and steps < max_steps:
            arm_3 = obs[0][:8]
            task = obs[0][8:]
            # combined input
            arm_3_combined = np.concatenate([arm_3, arm_3], axis=0).reshape(1, -1)

            with torch.no_grad():
                x_t = torch.from_numpy(arm_3_combined).float().to(device)
                raw = state_mapper(x_t).cpu().numpy().squeeze()
                proj = project_mapped_arm_state_to_reference(raw, reference_arm_state=arm_3)

            full_obs_2 = np.concatenate([proj.reshape(1, -1), task.reshape(1, -1)], axis=1)
            try:
                norm_obs = vec_norm_2dof.normalize_obs(full_obs_2)
            except Exception:
                # fallback to observation itself
                norm_obs = full_obs_2

            act2, _ = policy_2dof.predict(norm_obs, deterministic=True)
            act2 = np.asarray(act2).reshape(1, -1)
            pad_len = 4 - act2.shape[1]
            if pad_len > 0:
                act2 = np.concatenate([act2, np.zeros((act2.shape[0], pad_len), dtype=act2.dtype)], axis=1)

            with torch.no_grad():
                act3 = action_mapper(torch.from_numpy(arm_3_combined).float().to(device), torch.from_numpy(act2).float().to(device))
                act3 = act3.cpu().numpy()[0]

            # compute distances
            target_m = np.array([obs[0][10], obs[0][11]]) * MAX_REACH
            pred_eff_norm = fk_from_arm_state_torch(torch.tensor(proj.reshape(1, -1), dtype=torch.float32)).cpu().numpy()[0]
            pred_eff_m = pred_eff_norm * MAX_REACH
            true_eff_norm = obs[0][6:8]
            true_eff_m = true_eff_norm * MAX_REACH
            pred_to_target = float(np.linalg.norm(pred_eff_m - target_m))
            true_to_target = float(np.linalg.norm(true_eff_m - target_m))

            # reference 2-DoF IK for target
            ik_sol = ik_2dof_for_target(float(target_m[0]), float(target_m[1]))

            # angle errors (predicted proj angles vs IK sol)
            pred_angles = proj[:2] * np.pi
            ang_diff = angular_diff_rad(pred_angles, ik_sol)
            ang_err_deg = np.abs(ang_diff) * 180.0 / np.pi

            step_log = {
                'step': steps,
                'arm_3': arm_3.copy(),
                'arm_3_combined': arm_3_combined.copy(),
                'state_raw': raw.copy(),
                'state_proj': proj.copy(),
                'full_obs_2': full_obs_2.copy(),
                'norm_obs': norm_obs.copy(),
                'act2': act2.copy(),
                'act3': act3.copy(),
                'target_m': target_m.copy(),
                'pred_eff_m': pred_eff_m.copy(),
                'true_eff_m': true_eff_m.copy(),
                'pred_to_target_m': pred_to_target,
                'true_to_target_m': true_to_target,
                'ik_sol': ik_sol.copy(),
                'angle_err_deg': ang_err_deg.tolist(),
            }
            ep_step_logs.append(step_log)

            obs, _, dones, infos = env.step([act3])
            steps += 1
            done = dones[0]
            last_info = infos[0]
            ep_min_dist = min(ep_min_dist, last_info.get('dist', ep_min_dist))

        success = bool(last_info.get('target_reached', False))
        results['episodes'].append({'episode': ep, 'success': success, 'steps': steps, 'min_dist': ep_min_dist, 'final_info': last_info})

        # aggregate per-episode stats
        preds = np.array([s['pred_to_target_m'] for s in ep_step_logs]) if ep_step_logs else np.array([np.nan])
        results['pred_eff_to_target_m'].append(float(np.nanmin(preds)))
        results['true_eff_to_target_m'].append(float(ep_min_dist))

        # aggregate angle errors
        ang_errs = np.array([s['angle_err_deg'] for s in ep_step_logs]).reshape(-1, 2) if ep_step_logs else np.zeros((0,2))
        if ang_errs.size:
            results['angle_errors_deg'].append({'median_deg': float(np.median(ang_errs)), '90th_deg': float(np.percentile(ang_errs, 90))})
        else:
            results['angle_errors_deg'].append({'median_deg': None, '90th_deg': None})

        if not success and len(failed_examples) < 5:
            failed_examples.append({'episode': ep, 'steps': ep_step_logs})

    env.close()

    # summary print
    successes = sum(1 for e in results['episodes'] if e['success'])
    print(f"Episodes: {episodes}  Successes: {successes}  Success rate: {successes/episodes*100:.2f}%")

    pred_mins = np.array(results['pred_eff_to_target_m'])
    true_mins = np.array(results['true_eff_to_target_m'])
    print(f"Predicted eff min dist (median): {np.nanmedian(pred_mins):.4f} m")
    print(f"True eff min dist (median): {np.nanmedian(true_mins):.4f} m")

    # angle error summary
    medians = [d['median_deg'] for d in results['angle_errors_deg'] if d['median_deg'] is not None]
    p90s = [d['90th_deg'] for d in results['angle_errors_deg'] if d['90th_deg'] is not None]
    if medians:
        print(f"Angle error median (deg) across episodes (median of medians): {np.median(medians):.2f}")
    if p90s:
        print(f"Angle error 90th perc across episodes (median): {np.median(p90s):.2f}")

    out = {'results': results, 'failed_examples': failed_examples}
    if out_path:
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)
        print(f"Saved diagnostics to {out_path}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--out', type=str, default=str(DATA_ROOT / "DIRECT_GEN_ALGO" / "diagnostics_transfer_2to3.pkl"))
    args = parser.parse_args()
    run_diagnostics(episodes=args.episodes, max_steps=args.max_steps, out_path=args.out, device='cpu')
