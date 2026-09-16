"""
Transfer UNN policy trained on 2-DoF to 3-DoF environment (velocity-based).
"""

import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from unn.bases_unn import CartesianStateEncoder
from unn.unn_policy import UNNPolicy


def main():
    device  = "cpu"
    encoder = CartesianStateEncoder(max_reach=3.0)

    # Load policy trained on 2-DoF
    policy_2to3 = UNNPolicy.load(
        "./data/UNN/unn_pushball_2dof", name="final",
        encoder=encoder, device=device
    )

    # Override robot geometry / dynamics for the 3-DoF target arm
    tmp_env = PushBallEnv_3dof()
    policy_2to3.link_lengths = [float(tmp_env.l1), float(tmp_env.l2), float(tmp_env.l3)]
    policy_2to3.omega_max    = float(tmp_env.omega_max)
    policy_2to3.dt           = float(tmp_env.dt)
    policy_2to3.n_joints     = 3
    policy_2to3.arm_obs_size = 8
    # v_max_ee stays the same (6.0 m/s) — the latent space is morphology-agnostic
    # If the 3-DoF arm has different max_reach you can update it here:
    # policy_2to3.v_max_ee = float(tmp_env.omega_max) * float(tmp_env.max_reach)

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None, max_steps=150))])

    N_EPISODES = 20000
    successes  = 0

    for ep in range(N_EPISODES):
        obs  = env.reset()[0]
        done = False
        while not done:
            action = policy_2to3.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action.reshape(1, -1))
            obs  = obs[0]
            done = dones[0]
        if infos[0].get("target_reached", False):
            successes += 1
        if (ep + 1) % 200 == 0:
            print(f"{ep+1}/{N_EPISODES} – success rate: {100*successes/(ep+1):.1f}%")

    print(f"\nFinal transfer (2→3 DoF) success rate: {100*successes/N_EPISODES:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
