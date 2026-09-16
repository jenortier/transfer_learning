"""
Transfer UNN policy trained on 3-DoF to 2-DoF environment (velocity-based).
"""

import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from unn.bases_unn import CartesianStateEncoder
from unn.unn_policy import UNNPolicy


def main():
    device  = "cpu"
    encoder = CartesianStateEncoder(max_reach=3.0)

    print("Loading UNN policy trained on 3-DoF...")
    policy_3to2 = UNNPolicy.load(
        "./data/UNN/unn_pushball_3dof", name="final",
        encoder=encoder, device=device
    )

    # Override robot geometry / dynamics for the 2-DoF target arm
    tmp_env = PushBallEnv_2dof()
    policy_3to2.link_lengths = [float(tmp_env.l1), float(tmp_env.l2)]
    policy_3to2.omega_max    = float(tmp_env.omega_max)
    policy_3to2.dt           = float(tmp_env.dt)
    policy_3to2.n_joints     = 2
    policy_3to2.arm_obs_size = 6
    # v_max_ee stays the same (6.0 m/s) — morphology-agnostic latent space

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_2dof(render_mode=None, max_steps=150))])

    N_EPISODES = 20000
    successes  = 0
    steps_ok   = []
    dist_fail  = []

    print(f"\nRunning {N_EPISODES} transfer episodes (UNN 3→2 DoF)...\n")

    for ep in range(N_EPISODES):
        obs      = env.reset()[0]
        done     = False
        step     = 0
        info_last = {}
        while not done:
            action = policy_3to2.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action.reshape(1, -1))
            obs       = obs[0]
            step     += 1
            done      = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_ok.append(step)
        else:
            dist_fail.append(info_last.get("dist_ball_target", float("nan")))

        if (ep + 1) % 200 == 0:
            print(f"  {ep+1}/{N_EPISODES} – success rate: {100*successes/(ep+1):.1f}%")

    rate = 100 * successes / N_EPISODES
    print("=" * 70)
    print(f"  UNN Transfer 3→2 DoF — Success rate: {rate:.1f}%")
    if steps_ok:
        print(f"  Avg steps (success): {np.mean(steps_ok):.1f}")
    if dist_fail:
        print(f"  Avg dist (failure):  {np.mean(dist_fail):.4f} m")
    print("=" * 70)
    env.close()


if __name__ == "__main__":
    main()
