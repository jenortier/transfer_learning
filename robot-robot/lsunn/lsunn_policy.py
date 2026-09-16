"""
LS-UNN Policy — PPO agent operating in the shared latent space.

Architecture (Beaussant et al., 2024), Eq. (9):

    Bi(x_r) --> z --> U(z, o_tau) --> z_d --> Bo(z_d) --> x_rd --> action

The key point is that the PPO agent (the UNN task module) acts *inside* the
latent space: its action IS the desired latent state z_d. Both its observation
[z, o_tau] and its action z_d therefore have the same dimension for every robot
sharing the latent space, which is exactly what makes the module pluggable
between a 2-DoF and a 3-DoF arm.

The previous version left the raw environment action space untouched, so PPO
was producing joint velocities directly (dimension n_joints) and Bo was never
called: the task module was robot-specific and could not be transferred.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from lsunn.bases_vae import BaseVAE


# ============================================================================
# Constants
# ============================================================================

# Bound of the UNN action space. The KL term regularises the latent space
# towards N(0, I), so |z| > 3 is essentially never visited during the bases
# training; letting PPO sample there would ask the decoders to extrapolate on
# a region where the two latent spaces are not aligned (cf. Appendix B of the
# paper: "it is crucial for a successful transfer that the UNN does not wander
# outside of the latent space region seen during bases training").
LATENT_ACTION_BOUND = 3.0

# The environments expect a NORMALISED velocity command in [-1, 1], and the arm
# observation stores the velocity as dtheta / OMEGA_MAX (see trajectories.py).
# Both use the same convention, so the decoded velocity is passed as is.
# Set this to OMEGA_MAX (2.0) only if your environment expects raw rad/s.
ACTION_VELOCITY_SCALE = 1.0


# ============================================================================
# Helpers
# ============================================================================

def decode_velocity(
    base_vae: BaseVAE,
    z_d: np.ndarray,
    n_joints: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Bo(z_d) -> x_rd -> joint velocity command.

    x_rd follows the arm_obs layout built in trajectories.py:
        [ theta_1..n / pi , dtheta_1..n / OMEGA_MAX , x_ee , y_ee ]
    so the velocity command is the slice [n : 2n]. The decoded joint positions
    are ignored: the robots are velocity-controlled and Bo is not surjective,
    so asking it for an exact reachable configuration is ill-posed
    (Appendix C of the paper).
    """
    z_d = np.asarray(z_d, dtype=np.float32)
    x_rd = base_vae.decode_np(z_d, device=device)
    if x_rd.ndim == 1:
        vel = x_rd[n_joints:2 * n_joints]
    else:
        vel = x_rd[:, n_joints:2 * n_joints]
    return np.clip(vel * ACTION_VELOCITY_SCALE, -1.0, 1.0).astype(np.float32)


def check_transfer_compatibility(
    ppo_policy: PPO,
    base_vae: BaseVAE,
    task_dim: int,
) -> None:
    """
    Verify that a trained task module can be plugged onto a given pair of bases.

    This is the sanity check that a 2 <-> 3 DoF transfer relies on: the UNN
    observation and action spaces must be robot-agnostic.
    """
    expected_obs = base_vae.latent_dim + task_dim
    got_obs = int(np.prod(ppo_policy.observation_space.shape))
    if got_obs != expected_obs:
        raise ValueError(
            f"UNN observation space mismatch: the task module expects {got_obs} "
            f"dims, these bases + task provide {expected_obs} "
            f"(latent_dim={base_vae.latent_dim} + task_dim={task_dim}). "
            "Both robots must share the same latent dimension and the same "
            "task observation."
        )

    expected_act = base_vae.latent_dim
    got_act = int(np.prod(ppo_policy.action_space.shape))
    if got_act != expected_act:
        raise ValueError(
            f"UNN action space mismatch: the task module outputs {got_act} dims, "
            f"the latent space has {expected_act}. A module whose action "
            "dimension equals the number of joints was trained in the 'direct' "
            "(non-transferable) setting and must be retrained with LatentEnv."
        )


# ============================================================================
# Latent environment (training side)
# ============================================================================

class LatentEnv(gym.Wrapper):
    """
    Turns a raw robot environment into the latent MDP the UNN is trained on:

        observation : [z, o_tau]   latent robot state + task observation
        action      : z_d          desired latent robot state

    Bi is applied on reset/step to build the observation, Bo is applied on the
    incoming action to recover the joint velocity command. The bases are frozen
    and used in inference mode only.
    """

    def __init__(
        self,
        env: gym.Env,
        base_vae: BaseVAE,
        device: str = "cpu",
        n_joints: Optional[int] = None,
        stochastic_encoding: bool = True,
        latent_bound: float = LATENT_ACTION_BOUND,
    ):
        super().__init__(env)

        self.base_vae = base_vae
        self.device = device
        self.base_vae.eval()
        for p in self.base_vae.parameters():
            p.requires_grad = False

        # Number of joints: read it from the raw action space rather than
        # guessing it from the state dimension (which breaks as soon as the
        # bases encode more than the kinematic arm_obs).
        self.raw_action_space = env.action_space
        if n_joints is None:
            n_joints = int(np.prod(self.raw_action_space.shape))
        self.n_joints = n_joints

        self.arm_dim = base_vae.state_dim
        self.latent_dim = base_vae.latent_dim
        self.task_dim = env.observation_space.shape[0] - self.arm_dim
        if self.task_dim < 0:
            raise ValueError(
                f"Environment observation ({env.observation_space.shape[0]}) is "
                f"smaller than the encoded arm observation ({self.arm_dim})."
            )

        self.stochastic_encoding = stochastic_encoding
        self.latent_bound = float(latent_bound)

        # --- Robot-agnostic spaces: identical for the 2-DoF and the 3-DoF arm.
        self.observation_space = spaces.Box(
            -10.0, 10.0, (self.latent_dim + self.task_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -self.latent_bound, self.latent_bound, (self.latent_dim,), dtype=np.float32
        )

        self.last_raw_obs: Optional[np.ndarray] = None

    # -- encoding --------------------------------------------------------
    def set_stochastic_encoding(self, stochastic: bool) -> None:
        """
        z ~ N(mu, sigma^2) during the UNN training (acts as a domain
        randomisation and improves zero-shot transfer), z = mu when fine-tuning
        on the target robot or at evaluation time (Section 5.4 / 7.3).
        """
        self.stochastic_encoding = bool(stochastic)

    @torch.no_grad()
    def _build_ppo_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        arm_obs = obs[:self.arm_dim]
        task_obs = obs[self.arm_dim:]
        z = self.base_vae.encode_np(
            arm_obs, device=self.device, stochastic=self.stochastic_encoding
        )
        return np.concatenate([np.asarray(z).ravel(), task_obs]).astype(np.float32)

    # -- gym API ---------------------------------------------------------
    @torch.no_grad()
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_raw_obs = np.asarray(obs, dtype=np.float32).copy()
        return self._build_ppo_obs(obs), info

    @torch.no_grad()
    def step(self, z_d):
        z_d = np.asarray(z_d, dtype=np.float32).reshape(-1)
        if z_d.shape[0] != self.latent_dim:
            raise ValueError(
                f"Expected a latent action of dim {self.latent_dim}, got {z_d.shape[0]}."
            )

        # Stay inside the region covered during the bases training.
        z_d = np.clip(z_d, -self.latent_bound, self.latent_bound)

        action = decode_velocity(self.base_vae, z_d, self.n_joints, self.device)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_raw_obs = np.asarray(obs, dtype=np.float32).copy()

        info["z_d"] = z_d
        info["joint_action"] = action

        return self._build_ppo_obs(obs), reward, terminated, truncated, info


# ============================================================================
# Full policy (inference / transfer side)
# ============================================================================

class UNNPolicy:
    """
    Complete LS-UNN policy, mirroring exactly what LatentEnv does at training
    time:
      1. split the raw observation into (x_r, o_tau)
      2. encode x_r -> z with Bi
      3. normalise [z, o_tau] with the VecNormalize statistics of the task module
      4. PPO -> desired latent state z_d
      5. decode z_d -> x_rd with Bo and extract the velocity command

    The VecNormalize statistics belong to the *task module*, not to the robot:
    they were fitted on [z, o_tau], which is robot-agnostic, so the source
    statistics are the ones to reuse after a transfer.
    """

    def __init__(
        self,
        base_vae: BaseVAE,
        ppo_policy: PPO,
        vec_normalize: Optional[VecNormalize] = None,
        device: str = "cpu",
        stochastic_encoding: bool = False,   # False = use mu (deterministic)
        n_joints: Optional[int] = None,
        task_dim: Optional[int] = None,
        latent_bound: float = LATENT_ACTION_BOUND,
    ):
        self.device = device
        self.base_vae = base_vae
        self.ppo_policy = ppo_policy
        self.vec_normalize = vec_normalize
        self.stochastic_encoding = stochastic_encoding
        self.latent_bound = float(latent_bound)

        self.base_vae.eval()
        self.ppo_policy.policy.eval()

        self.arm_dim = base_vae.state_dim
        self.latent_dim = base_vae.latent_dim

        if n_joints is None:
            raise ValueError(
                "n_joints must be given explicitly (2 or 3): it cannot be "
                "safely deduced from the encoded state dimension."
            )
        self.n_joints = n_joints

        # Deduce the task dimension from the policy if not provided, then check
        # that the module really is pluggable on these bases.
        if task_dim is None:
            task_dim = int(np.prod(ppo_policy.observation_space.shape)) - self.latent_dim
        self.task_dim = task_dim
        check_transfer_compatibility(ppo_policy, base_vae, self.task_dim)

    @torch.no_grad()
    def predict(
        self,
        raw_obs: np.ndarray,
        deterministic: bool = True,
        return_components: bool = False,
    ):
        raw_obs = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        arm_obs = raw_obs[:self.arm_dim]
        task_obs = raw_obs[self.arm_dim:]

        # 1. Bi
        z = np.asarray(
            self.base_vae.encode_np(
                arm_obs, device=self.device, stochastic=self.stochastic_encoding
            )
        ).ravel()

        # 2. UNN observation, normalised with the task module statistics
        obs = np.concatenate([z, task_obs]).astype(np.float32)
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)

        # 3. U
        z_d, _ = self.ppo_policy.predict(obs, deterministic=deterministic)
        z_d = np.clip(
            np.asarray(z_d, dtype=np.float32).ravel(),
            -self.latent_bound, self.latent_bound,
        )

        # 4. Bo -> velocity command
        action = decode_velocity(self.base_vae, z_d, self.n_joints, self.device)

        if return_components:
            x_rd = self.base_vae.decode_np(z_d, device=self.device)
            return action, z, z_d, x_rd
        return action

    # -- persistence -----------------------------------------------------
    def save(self, save_dir: str):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.ppo_policy.save(save_path / "policy")
        if self.vec_normalize is not None:
            with open(save_path / "vec_normalize.pkl", "wb") as f:
                pickle.dump(self.vec_normalize, f)
        print(f"  UNN Policy saved -> {save_path}")


def load_unn_policy(
    policy_dir: str,
    base_vae: BaseVAE,
    n_joints: int,
    task_dim: Optional[int] = None,
    device: str = "cpu",
    stochastic_encoding: bool = False,
) -> UNNPolicy:
    """
    Load a task module trained on one robot and plug it onto the bases of
    another one.

        base_3dof = BaseVAE(8, LATENT_DIM, HIDDEN_DIM); base_3dof.load_state_dict(...)
        policy = load_unn_policy("./data/LSUNN/lsunn_2dof", base_3dof, n_joints=3)

    The VecNormalize object is unpickled directly: normalize_obs() only needs
    obs_rms, so no vectorised environment has to be rebuilt for inference.
    """
    policy_dir = Path(policy_dir)
    ppo = PPO.load(policy_dir / "policy", device=device)

    vec = None
    vec_path = policy_dir / "vec_normalize.pkl"
    if vec_path.exists():
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)
        vec.training = False
        vec.norm_reward = False

    return UNNPolicy(
        base_vae=base_vae,
        ppo_policy=ppo,
        vec_normalize=vec,
        device=device,
        stochastic_encoding=stochastic_encoding,
        n_joints=n_joints,
        task_dim=task_dim,
    )
