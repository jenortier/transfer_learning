import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from gymnasium import spaces

from envs.arm_3dof import Arm3DoF
from direct_method.mapper_models import (
    StateMapperMLP, ActionMapperConditionedMLP,
    ARM_OBS_2DOF, ARM_OBS_3DOF, MAX_REACH,
)

# ── Link lengths (must match mapper_models FK equations) ──────────────────────
_L1_2 = _L2_2 = 1.5          # 2-DoF arm  (l1 = l2 = 1.5 m)
_L1_3 = _L2_3 = _L3_3 = 1.0  # 3-DoF arm  (l1 = l2 = l3 = 1.0 m)


# ── Geometry & similarity helpers ─────────────────────────────────────────────

def _joint_pos_3dof(arm_state: np.ndarray) -> np.ndarray:
    """Cartesian positions of the three joints for a normalised 3-DoF arm state.

    Returns array of shape (3, 2): [joint-1, joint-2, end-effector].
    Angles are stored normalised by π in arm_state[0:3].
    """
    t1 = float(arm_state[0]) * np.pi
    t2 = float(arm_state[1]) * np.pi
    t3 = float(arm_state[2]) * np.pi
    p1 = np.array([_L1_3 * np.cos(t1),
                   _L1_3 * np.sin(t1)], dtype=np.float32)
    p2 = p1 + np.array([_L2_3 * np.cos(t1 + t2),
                        _L2_3 * np.sin(t1 + t2)], dtype=np.float32)
    p3 = p2 + np.array([_L3_3 * np.cos(t1 + t2 + t3),
                        _L3_3 * np.sin(t1 + t2 + t3)], dtype=np.float32)
    return np.stack([p1, p2, p3])  # (3, 2) -> [joint1, joint2, end-effector]


def _geometric_similarity_3dof(sa: np.ndarray, sb: np.ndarray) -> float:
    """geometric_rec = exp(-d_shape).

    d_shape = mean Cartesian distance over ALL joint positions (joint-1,
    joint-2, end-effector), normalised by MAX_REACH. Captures global arm
    posture, independent of the number of joints.
    """
    ja = _joint_pos_3dof(sa)
    jb = _joint_pos_3dof(sb)
    d_shape = float(np.mean(np.linalg.norm(ja - jb, axis=1))) / MAX_REACH
    return float(np.exp(-d_shape))


def _effector_similarity_3dof(sa: np.ndarray, sb: np.ndarray) -> float:
    """effector_rec = exp(-d_ee).

    d_ee = end-effector-only Cartesian distance, normalised by MAX_REACH.
    Explicitly task-relevant: two similar postures can still reach very
    different points.
    """
    ja = _joint_pos_3dof(sa)
    jb = _joint_pos_3dof(sb)
    d_ee = float(np.linalg.norm(ja[-1] - jb[-1])) / MAX_REACH
    return float(np.exp(-d_ee))


def _velocity_similarity(va: np.ndarray, vb: np.ndarray) -> float:
    """Similarity between two normalised velocity/action vectors (in [-1, 1]).

    Normalised so that the maximum possible distance maps to exp(-1) ≈ 0.37.
    """
    n = max(len(va), 1)
    d = float(np.linalg.norm(va - vb)) / (2.0 * np.sqrt(n))
    return float(np.exp(-d))


# ── Mapper loading helpers ────────────────────────────────────────────────────

def _load_state_mapper(in_dim: int, out_dim: int,
                         path, device: str = 'cpu', hidden: int = 256):
    """Return a StateMapperMLP loaded from *path*, or None if unavailable."""
    if path is None or not os.path.exists(path):
        return None
    m = StateMapperMLP(in_dim, out_dim, hidden=hidden)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def _load_action_mapper(nq_src: int, nq_tgt: int,
                         path, device: str = 'cpu'):
    """Return an ActionMapperConditionedMLP loaded from *path*, or None if unavailable."""
    if path is None or not os.path.exists(path):
        return None
    m = ActionMapperConditionedMLP(nq_src, nq_tgt)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


# ── Environment ───────────────────────────────────────────────────────────────

class PushBallEnv_3dof(Arm3DoF):
    """
    Push-ball task for a 3-DoF arm.

    Extended from the original *_rec version with:
      • Joint-limit-free dynamics (unrestricted angle updates).
      • Mapper-based reconstruction reward that encourages the agent to produce
        states and actions surviving the 3 → 2 → 3 round-trip through
        pre-trained state/action mapper networks.

    Round-trip (forward → inverse):
        s3  ──SM_3→2──► s2  ──SM_2→3──► s3*
        a3  ──AM_3→2(s2)──► a2  ──AM_2→3(s3*)──► a3*

    Reconstruction reward — ONE global coefficient against the rest of the
    reward, and ONE ratio deciding the effector/geometric split inside the
    state-reconstruction term (state and action reconstruction are then
    averaged 50/50):

        state_sim    = eff_ratio * effector_rec + (1 - eff_ratio) * geometric_rec
        recon_term   = 0.5 * state_sim + 0.5 * velocity_rec
        recon_reward = w_recon * recon_term

      • geometric_rec ∈ (0, 1] — global arm-posture similarity (d_shape),  state recon.
      • effector_rec  ∈ (0, 1] — end-effector-only similarity (d_ee),      state recon.
      • velocity_rec  ∈ (0, 1] — joint-velocity round-trip similarity,    action recon.
      • eff_ratio ∈ [0, 1]     — share of effector_rec inside state_sim
                                 (1.0 → 100% effector / 0% geometric, 0.5 → 50/50,
                                 0.6 → 60% effector / 40% geometric, etc.).
      • w_recon                — overall scale of the reconstruction term
                                 relative to the other reward terms (near, ctrl,
                                 progress, alignment, success...).

    If any mapper is not loaded, recon_reward is 0.
    """

    def __init__(
        self,
        render_mode=None,
        max_steps=100,
        # ── mapper checkpoints (None → reconstruction reward disabled) ──
        state_mapper_3to2_path=None,
        state_mapper_2to3_path=None,
        action_mapper_3to2_path=None,
        action_mapper_2to3_path=None,
        # ── reconstruction-reward hyperparameters ────────────────────────
        w_recon=1.0,    # global scale of recon_reward vs. the rest of the reward
        eff_ratio=0.5,  # share of effector_rec inside state_sim (rest → geometric_rec)
    ):
        super().__init__(render_mode=render_mode)

        if not (0.0 <= eff_ratio <= 1.0):
            print(f"[PushBallEnv_3dof] WARNING: eff_ratio={eff_ratio} is "
                  f"outside [0, 1]; clipping.")
            eff_ratio = float(np.clip(eff_ratio, 0.0, 1.0))

        # Reconstruction-reward hyperparameters
        self.w_recon   = w_recon
        self.eff_ratio = eff_ratio

        # ── Load mapper networks ─────────────────────────────────────────────
        # State mappers:  s3 (8D) ↔ s2 (6D)
        # Action mapper 3→2:  AM(dtheta_3, theta_3, theta_2) → dtheta_2
        # Action mapper 2→3:  AM(dtheta_2, theta_2, theta_3) → dtheta_3
        device = 'cpu'
        self._sm_3to2 = _load_state_mapper(
            ARM_OBS_3DOF, ARM_OBS_2DOF, state_mapper_3to2_path, device, hidden=256)
        self._sm_2to3 = _load_state_mapper(
            ARM_OBS_2DOF, ARM_OBS_3DOF, state_mapper_2to3_path, device, hidden=256)
        self._am_3to2 = _load_action_mapper(
            3, 2, action_mapper_3to2_path, device)
        self._am_2to3 = _load_action_mapper(
            2, 3, action_mapper_2to3_path, device)

        self._mappers_ready = all(m is not None for m in [
            self._sm_3to2, self._sm_2to3,
            self._am_3to2, self._am_2to3,
        ])
        if not self._mappers_ready:
            missing = [name for name, m in zip(
                ['SM_3to2', 'SM_2to3', 'AM_3to2', 'AM_2to3'],
                [self._sm_3to2, self._sm_2to3,
                 self._am_3to2, self._am_2to3],
            ) if m is None]
            print(f"[PushBallEnv_3dof] Reconstruction reward DISABLED "
                  f"(missing mappers: {missing})")

        # ── Dynamics ─────────────────────────────────────────────────────────
        self.omega_max = 2.0
        self.dt        = 0.05
        self.max_steps = max_steps

        # Contact
        self.eff_radius        = 0.05
        self.ball_radius       = 0.10
        self.contact_threshold = self.eff_radius + self.ball_radius

        # Success criterion
        self.epsilon = 0.10

        # Task reward weights
        self.w_near        = 0.6
        self.w_ctrl        = 0.03
        self.bonus_success = 400.0

        # Observation space  (unchanged from original)
        arm_high  = np.ones(self.arm_obs_size, dtype=np.float32)
        task_high = np.ones(4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([-arm_high, -task_high]),
            high=np.concatenate([ arm_high,  task_high]),
            dtype=np.float32,
        )

        # Task state
        self.ball   = np.zeros(2)
        self.target = np.zeros(2)
        self.prev_dist_ball_target = 0.0
        self.prev_dist_eff_ball    = 0.0

    # ── Mapper round-trip ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_reconstruction_reward(self,
                                        arm_state: np.ndarray,
                                        action: np.ndarray):
        """
        Run the 3 → 2 → 3 round-trip and return the three decoupled
        similarity terms plus the final weighted reconstruction reward.

        Chain (all normalised):
          s3 ──SM_3→2──► s2 ──SM_2→3──► s3*
          a3 ──AM_3→2(s2)──► a2 ──AM_2→3(s3*)──► a3*

        Returns
        -------
        geometric_rec, effector_rec, velocity_rec, state_sim, recon_reward : float
            All 0.0 if mappers are not loaded.
        """
        if not self._mappers_ready:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        s3 = torch.tensor(arm_state, dtype=torch.float32).unsqueeze(0)
        a3 = torch.tensor(action,    dtype=torch.float32).unsqueeze(0)

        # State round-trip: 3 → 2 → 3
        s2       = self._sm_3to2(s3)
        s3_recon = self._sm_2to3(s2)

        # Action round-trip: 3 → 2 → 3  (conditioned action mappers)
        # AM_3to2(dtheta_3, theta_3, theta_2) -> dtheta_2
        # AM_2to3(dtheta_2, theta_2, theta_3) -> dtheta_3
        theta_3 = s3[:, :3]           # (1, 3)
        theta_2 = s2[:, :2]           # (1, 2)
        theta_3_recon = s3_recon[:, :3]  # (1, 3)

        a2       = self._am_3to2(a3, theta_3, theta_2)
        a3_recon = self._am_2to3(a2, theta_2, theta_3_recon)

        s3_recon_np = s3_recon.squeeze(0).cpu().numpy()
        a3_recon_np = a3_recon.squeeze(0).cpu().numpy()

        geometric_rec = _geometric_similarity_3dof(arm_state, s3_recon_np)
        effector_rec  = _effector_similarity_3dof(arm_state, s3_recon_np)
        velocity_rec  = _velocity_similarity(action, a3_recon_np)

        state_sim    = (self.eff_ratio * effector_rec
                         + (1.0 - self.eff_ratio) * geometric_rec)
        recon_term   = 0.5 * state_sim + 0.5 * velocity_rec
        recon_reward = self.w_recon * recon_term

        return geometric_rec, effector_rec, velocity_rec, state_sim, recon_reward

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        r_t = float(self.np_random.uniform(0.4 * self.max_reach, 0.75 * self.max_reach))
        a_t = float(self.np_random.uniform(-np.pi, np.pi))
        self.target = np.array([r_t * np.cos(a_t), r_t * np.sin(a_t)])

        for _ in range(1000):
            r_b = float(self.np_random.uniform(0.3 * self.max_reach, 0.75 * self.max_reach))
            a_b = float(self.np_random.uniform(-np.pi, np.pi))
            candidate = np.array([r_b * np.cos(a_b), r_b * np.sin(a_b)])
            if np.linalg.norm(candidate - self.target) >= 0.3:
                self.ball = candidate
                break
        else:
            self.ball = self.target + np.array([0.4, 0.0])

        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        self.prev_dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        self.prev_dist_eff_ball    = float(np.linalg.norm(eff - self.ball))

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Capture arm state BEFORE dynamics update (paired with the current action
        # for the mapper round-trip, consistent with the training distribution).
        arm_state_before = self._get_obs()[:self.arm_obs_size].copy()

        # ── Angle update  — NO joint-limit clamping ──────────────────────────
        new_theta1 = self.theta1 + action[0] * self.omega_max * self.dt
        new_theta2 = self.theta2 + action[1] * self.omega_max * self.dt
        new_theta3 = self.theta3 + action[2] * self.omega_max * self.dt

        self.dtheta1 = (new_theta1 - self.theta1) / self.dt
        self.dtheta2 = (new_theta2 - self.theta2) / self.dt
        self.dtheta3 = (new_theta3 - self.theta3) / self.dt
        self.theta1  = new_theta1
        self.theta2  = new_theta2
        self.theta3  = new_theta3

        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        # ── Ball contact ─────────────────────────────────────────────────────
        vec_eff_ball = self.ball - eff
        contact_dist = float(np.linalg.norm(vec_eff_ball))
        if contact_dist < self.contact_threshold and contact_dist > 1e-6:
            normal    = vec_eff_ball / contact_dist
            self.ball = self.ball + normal * (self.contact_threshold - contact_dist)

        # ── Distances ────────────────────────────────────────────────────────
        dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        dist_eff_ball    = float(np.linalg.norm(self.ball - eff))
        if dist_eff_ball < 0.25:
            dist_eff_ball = 0.1

        # ── Alignment (eff→ball aligned with ball→target) ────────────────────
        v_eb = self.ball - eff
        v_bt = self.target - self.ball
        if np.linalg.norm(v_eb) > 1e-3 and np.linalg.norm(v_bt) > 1e-3:
            alignment = float(np.dot(v_eb / np.linalg.norm(v_eb),
                                     v_bt / np.linalg.norm(v_bt)))
        else:
            alignment = 0.0

        progress_ball_target = self.prev_dist_ball_target - dist_ball_target

        # ── Base reward ──────────────────────────────────────────────────────
        reward  = -self.w_near * dist_eff_ball
        reward += -self.w_ctrl * np.sum(np.square(action))
        reward +=  30.0 * progress_ball_target
        reward +=   0.6 * alignment

        success  = dist_ball_target < self.epsilon
        ball_out = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05
        if success:
            reward += self.bonus_success
        if ball_out:
            reward -= 5.0

        # ── Reconstruction reward : w_recon * (0.5*state_sim + 0.5*velocity_rec)
        # where state_sim = eff_ratio*effector_rec + (1-eff_ratio)*geometric_rec.
        # action already contains normalised joint velocities (same as the
        # velocity slice of the arm state), so it is the correct input for the
        # action mapper.
        geometric_rec, effector_rec, velocity_rec, state_sim, recon_reward = \
            self._compute_reconstruction_reward(arm_state_before, action)
        reward += recon_reward

        self.prev_dist_ball_target = dist_ball_target
        self.prev_dist_eff_ball    = dist_eff_ball

        terminated = success or ball_out
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "target_reached":       success,
            "dist_ball_target":     dist_ball_target,
            "dist_eff_ball":        dist_eff_ball,
            "alignment":            alignment,
            "progress_ball_target": progress_ball_target,
            "geometric_rec":        geometric_rec,
            "effector_rec":         effector_rec,
            "velocity_rec":         velocity_rec,
            "state_sim":            state_sim,
            "recon_reward":         recon_reward,
            "theta1": self.theta1,
            "theta2": self.theta2,
            "theta3": self.theta3,
            "ball":   self.ball.copy(),
        }

    def _get_obs(self):
        """12-D observation: 8-D arm (inherited) + 4-D task."""
        arm_obs  = super()._get_obs()
        task_obs = np.array([
            np.clip(self.ball[0] / self.max_reach, -1., 1.),
            np.clip(self.ball[1] / self.max_reach, -1., 1.),
            self.target[0] / self.max_reach,
            self.target[1] / self.max_reach,
        ], dtype=np.float32)
        return np.concatenate([arm_obs, task_obs])

    def render(self):
        if self.render_mode != "human":
            return
        plt.clf()
        j1 = np.array([self.l1 * np.cos(self.theta1),
                        self.l1 * np.sin(self.theta1)])
        j2 = j1 + np.array([self.l2 * np.cos(self.theta1 + self.theta2),
                             self.l2 * np.sin(self.theta1 + self.theta2)])
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        plt.plot([0, j1[0]],   [0, j1[1]],   'r-', lw=4, label='Link 1')
        plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'g-', lw=4, label='Link 2')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'b-', lw=4, label='Link 3')
        plt.gca().add_patch(plt.Circle(eff, self.eff_radius * 2,
                                       color='red', alpha=0.6))
        ball_out   = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05
        ball_color = 'red' if ball_out else 'dodgerblue'
        plt.gca().add_patch(plt.Circle(self.ball, self.ball_radius * 1.5,
                                       color=ball_color, alpha=0.6))
        plt.plot(self.target[0], self.target[1], 'o', markersize=18, label='Target')
        plt.gca().add_patch(plt.Circle(self.target, self.epsilon,
                                       color='green', fill=False,
                                       linestyle='--', lw=1.5))
        vec = self.target - self.ball
        if np.linalg.norm(vec) > 1e-3:
            v = vec / np.linalg.norm(vec) * 0.3
            plt.arrow(self.ball[0], self.ball[1], v[0], v[1],
                      head_width=0.08, head_length=0.04,
                      fc='green', ec='green', alpha=0.4)
        dist_ball_target = np.linalg.norm(self.ball - self.target)
        plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect("equal")
        plt.title(f"Step {self.step_count} | d(ball,tgt)={dist_ball_target:.3f} m"
                  + (" | ⚠️ OUT" if ball_out else ""))
        plt.legend(loc="upper right", fontsize=8)
        plt.pause(0.001)
