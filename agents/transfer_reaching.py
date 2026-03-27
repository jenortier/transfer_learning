"""
Transfer Learning — Reaching Task
===================================
Apply the 2-DoF reaching policy to the 3-DoF environment via learned mappers.

Full pipeline (per timestep):
  1. obs_3dof              ← raw observation from env_3dof
  2. obs_2dof_equiv        = state_mapper(obs_3dof)
  3. obs_2dof_norm         = vec_norm_2dof.normalize_obs(obs_2dof_equiv)
  4. action_2dof           = policy_2dof(obs_2dof_norm)   [deterministic]
  5. action_3dof           = action_mapper(obs_3dof, action_2dof)  [context-aware]
  6. obs_3dof, …           = env_3dof.step(action_3dof)

No new learning — only inference through the pre-trained components.
Output format matches test_2dof.py / test_3dof.py for easy comparison.
"""

import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.arm3dof_env import Arm3DoFEnv
from envs.arm2dof_env import Arm2DoFEnv

# Import mapper architectures (must match training definitions)
from agents.eq_mappings import StateMapperMLP, ActionMapperMLP


# ============================================================================
# TRANSFER POLICY WRAPPER
# ============================================================================

class TransferPolicy:
    """
    Wraps the 2-DoF policy + state mapper + action mapper for 3-DoF execution.

    Inference pipeline (see module docstring).
    """

    def __init__(
        self,
        policy_2dof_path: str,
        vecnorm_2dof_path: str,
        state_mapper_path: str,
        action_mapper_path: str,
        device: str = "cpu",
    ):
        self.device = device

        # ---- 2-DoF PPO policy ----
        self.policy_2dof = PPO.load(policy_2dof_path, device=device)

        # ---- 2-DoF VecNormalize (inference only) ----
        def make_dummy_2dof():
            return Monitor(Arm2DoFEnv(render_mode=None))
        venv = DummyVecEnv([make_dummy_2dof])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        # ---- State Mapper: obs_3dof (10) → obs_2dof (9) ----
        self.state_mapper = StateMapperMLP(input_dim=10, output_dim=9).to(device)
        self.state_mapper.load_state_dict(
            torch.load(state_mapper_path, map_location=device)
        )
        self.state_mapper.eval()

        # ---- Action Mapper: (obs_3dof [10], action_2dof [2]) → action_3dof [3] ----
        self.action_mapper = ActionMapperMLP(
            state_dim=10, action_2dof_dim=2, output_dim=3
        ).to(device)
        self.action_mapper.load_state_dict(
            torch.load(action_mapper_path, map_location=device)
        )
        self.action_mapper.eval()

        print("[Transfer] All components loaded successfully")
        print(f"  policy_2dof    : {policy_2dof_path}")
        print(f"  vecnorm_2dof   : {vecnorm_2dof_path}")
        print(f"  state_mapper   : {state_mapper_path}")
        print(f"  action_mapper  : {action_mapper_path}")

    @torch.no_grad()
    def predict(self, raw_obs_3dof: np.ndarray) -> np.ndarray:
        """
        Full transfer pipeline for one observation.

        Args:
            raw_obs_3dof: shape (10,) or (1, 10) — raw (unnormalized) 3-DoF obs

        Returns:
            action_3dof: shape (1, 3) — action to apply in env_3dof
        """
        if raw_obs_3dof.ndim == 1:
            raw_obs_3dof = raw_obs_3dof.reshape(1, -1)  # (1, 10)

        obs_t = torch.tensor(raw_obs_3dof, dtype=torch.float32, device=self.device)

        # Step 1 — map 3-DoF obs to 2-DoF obs equivalent (raw scale)
        obs_2dof_equiv = self.state_mapper(obs_t).cpu().numpy()   # (1, 9)

        # Step 2 — normalize with 2-DoF VecNormalize running stats
        obs_2dof_norm = self.vec_norm_2dof.normalize_obs(obs_2dof_equiv)  # (1, 9)

        # Step 3 — query 2-DoF policy
        action_2dof, _ = self.policy_2dof.predict(obs_2dof_norm, deterministic=True)
        # action_2dof: (1, 2)

        # Step 4 — context-aware action mapping → 3-DoF action
        a2_t = torch.tensor(action_2dof, dtype=torch.float32, device=self.device)  # (1, 2)
        action_3dof = self.action_mapper(obs_t, a2_t).cpu().numpy()  # (1, 3)

        return action_3dof  # (1, 3)


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    # ----- Configuration -----
    DEVICE       = "cpu"
    NUM_EPISODES = 1000   # same order of magnitude as test_2dof / test_3dof

    POLICY_2DOF_PATH  = "./models/ppo_reach_2dof/best_model.zip"
    VECNORM_2DOF_PATH = "./models/ppo_reach_2dof/vec_normalize.pkl"
    STATE_MAPPER_PATH  = "./data/transfer_learning/state_mapper.pt"
    ACTION_MAPPER_PATH = "./data/transfer_learning/action_mapper.pt"

    # ----- Sanity checks -----
    for p in [POLICY_2DOF_PATH, VECNORM_2DOF_PATH, STATE_MAPPER_PATH, ACTION_MAPPER_PATH]:
        if not Path(p).exists():
            print(f"❌  File not found: {p}")
            return

    print("\n" + "="*70)
    print("TRANSFER LEARNING TEST: 2-DoF Reaching → 3-DoF Environment")
    print("="*70)

    # ----- Load transfer policy -----
    print("\n[1] Loading transfer components...")
    policy = TransferPolicy(
        policy_2dof_path=POLICY_2DOF_PATH,
        vecnorm_2dof_path=VECNORM_2DOF_PATH,
        state_mapper_path=STATE_MAPPER_PATH,
        action_mapper_path=ACTION_MAPPER_PATH,
        device=DEVICE,
    )

    # ----- Create 3-DoF env (no VecNormalize — we get raw obs) -----
    print("\n[2] Creating 3-DoF environment (raw observations)...")
    def make_env():
        return Monitor(Arm3DoFEnv(render_mode=None))
    env = DummyVecEnv([make_env])

    # ----- Evaluation loop -----
    print(f"\n[3] Running {NUM_EPISODES} episodes...\n")

    successes = 0
    steps_on_success   = []
    final_dist_failure = []

    for ep in range(NUM_EPISODES):
        obs   = env.reset()           # obs shape: (1, 10), raw
        done  = False
        step  = 0
        info_last = {}

        while not done:
            action_3dof = policy.predict(obs[0])           # (1, 3)
            obs, _, dones, infos = env.step(action_3dof)
            step += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_on_success.append(step)
        else:
            final_dist_failure.append(info_last.get("dist", float("nan")))

        if (ep + 1) % 200 == 0:
            print(f"  Progress: {ep + 1}/{NUM_EPISODES} episodes  "
                  f"(running success rate: {100 * successes / (ep + 1):.1f}%)")

    # ----- Results (same format as test_2dof.py / test_3dof.py) -----
    rate = 100.0 * successes / NUM_EPISODES
    print("\n" + "="*45)
    print(f"  Épisodes testés       : {NUM_EPISODES}")
    print(f"  Réussites             : {successes}")
    print(f"  Taux de réussite      : {rate:.1f}%")
    print("-"*45)
    if steps_on_success:
        print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
        print(f"  Steps min / max       : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
    else:
        print("  Steps moyens (succès) : N/A  (0 succès)")
    if final_dist_failure:
        print(f"  Distance moy (échec)  : {np.mean(final_dist_failure):.4f} m")
        print(f"  Distance min / max    : "
              f"{np.min(final_dist_failure):.4f} / {np.max(final_dist_failure):.4f} m")
    print("="*45 + "\n")

    env.close()


if __name__ == "__main__":
    main()
