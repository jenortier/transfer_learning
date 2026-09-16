import numpy as np
import torch
import torch.nn as nn

ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8
MAX_REACH = 3.0


class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionMapperMLP(nn.Module):
    def __init__(self, state_dim: int, act_in_dim: int, act_out_dim: int,
                 hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, act_out_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, act_in: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, act_in], dim=-1))


def dof_from_arm_dim(arm_dim: int) -> int:
    if arm_dim == ARM_OBS_2DOF:
        return 2
    if arm_dim == ARM_OBS_3DOF:
        return 3
    raise ValueError(f"Unsupported arm observation dim: {arm_dim}")


def eff_slice_for_arm_dim(arm_dim: int) -> slice:
    return slice(4, 6) if arm_dim == ARM_OBS_2DOF else slice(6, 8)


def angle_slice_for_arm_dim(arm_dim: int) -> slice:
    return slice(0, 2) if arm_dim == ARM_OBS_2DOF else slice(0, 3)


def velocity_slice_for_arm_dim(arm_dim: int) -> slice:
    return slice(2, 4) if arm_dim == ARM_OBS_2DOF else slice(3, 6)


def _wrap_pi_np(angles: np.ndarray) -> np.ndarray:
    return ((angles + np.pi) % (2.0 * np.pi)) - np.pi


def _fk_np(angles: np.ndarray) -> np.ndarray:
    """Forward kinematics in meters for 2DoF or 3DoF normalized arm angles."""
    dof = angles.shape[-1]
    if dof == 2:
        t1 = angles[..., 0]
        t2 = angles[..., 1]
        x = 1.5 * np.cos(t1) + 1.5 * np.cos(t1 + t2)
        y = 1.5 * np.sin(t1) + 1.5 * np.sin(t1 + t2)
    elif dof == 3:
        t1 = angles[..., 0]
        t2 = angles[..., 1]
        t3 = angles[..., 2]
        x = np.cos(t1) + np.cos(t1 + t2) + np.cos(t1 + t2 + t3)
        y = np.sin(t1) + np.sin(t1 + t2) + np.sin(t1 + t2 + t3)
    else:
        raise ValueError(f"Unsupported dof: {dof}")
    return np.stack([x, y], axis=-1).astype(np.float32)


def _jacobian_np(angles: np.ndarray) -> np.ndarray:
    """Planar end-effector Jacobian in meters/radian for one state."""
    if len(angles) == 2:
        t1, t2 = angles
        return np.array([
            [-1.5 * np.sin(t1) - 1.5 * np.sin(t1 + t2), -1.5 * np.sin(t1 + t2)],
            [ 1.5 * np.cos(t1) + 1.5 * np.cos(t1 + t2),  1.5 * np.cos(t1 + t2)],
        ], dtype=np.float32)

    t1, t2, t3 = angles
    a = t1
    b = t1 + t2
    c = t1 + t2 + t3
    return np.array([
        [-np.sin(a) - np.sin(b) - np.sin(c), -np.sin(b) - np.sin(c), -np.sin(c)],
        [ np.cos(a) + np.cos(b) + np.cos(c),  np.cos(b) + np.cos(c),  np.cos(c)],
    ], dtype=np.float32)


def _project_angles_to_ee(
    angles: np.ndarray,
    desired_ee: np.ndarray,
    damping: float = 0.08,
    max_iters: int = 12,
    max_delta: float = 0.35,
) -> np.ndarray:
    """Damped least-squares IK, initialized by the mapper output."""
    q = _wrap_pi_np(angles.astype(np.float32))
    target = desired_ee.astype(np.float32).copy()

    reach = MAX_REACH - 1e-4
    norm = float(np.linalg.norm(target))
    if norm > reach:
        target *= reach / norm

    # Fast analytic IK for 2-DoF planar arm (no iteration)
    if q.shape[-1] == 2:
        x, y = float(target[0]), float(target[1])
        l1 = 1.5
        l2 = 1.5
        r = float(np.hypot(x, y))
        # clamp reach
        reach = l1 + l2 - 1e-6
        if r > reach:
            x *= reach / r
            y *= reach / r
            r = reach

        # cosine law for theta2
        cos_t2 = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_t2 = float(np.clip(cos_t2, -1.0, 1.0))
        t2 = float(np.arccos(cos_t2))

        # compute theta1 using atan2 and law of cosines
        k1 = l1 + l2 * cos_t2
        k2 = l2 * np.sin(t2)
        t1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))

        q2 = _wrap_pi_np(np.array([t1, t2], dtype=np.float32))
        return q2.astype(np.float32)

    for _ in range(max_iters):
        err = target - _fk_np(q)
        if float(np.linalg.norm(err)) < 1e-4:
            break

        j = _jacobian_np(q)
        lhs = j @ j.T + (damping ** 2) * np.eye(2, dtype=np.float32)
        dq = j.T @ np.linalg.solve(lhs, err)

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_delta:
            dq *= max_delta / dq_norm

        q = _wrap_pi_np(q + dq)

    return q.astype(np.float32)


def arm_end_effector_np(arm_state: np.ndarray) -> np.ndarray:
    """Read the normalized arm end-effector fields and return meters."""
    arm_state = np.asarray(arm_state, dtype=np.float32)
    return arm_state[..., eff_slice_for_arm_dim(arm_state.shape[-1])] * MAX_REACH


def project_mapped_arm_state_to_reference(
    mapped_arm_state: np.ndarray,
    reference_arm_state: np.ndarray | None = None,
) -> np.ndarray:
    """
    Repair a mapped arm state so it is kinematically valid.

    The mapper remains task-agnostic: it only predicts a source-arm posture.
    The source end-effector is forced to match the current target-arm
    end-effector, so task observations such as target/ball positions stay
    meaningful for reaching, pushball, and future tasks.
    """
    mapped = np.asarray(mapped_arm_state, dtype=np.float32)
    single = mapped.ndim == 1
    if single:
        mapped = mapped.reshape(1, -1)

    if reference_arm_state is None:
        desired_ee = arm_end_effector_np(mapped)
    else:
        ref = np.asarray(reference_arm_state, dtype=np.float32)
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)
        desired_ee = arm_end_effector_np(ref)

    out = np.clip(mapped.copy(), -1.0, 1.0)
    arm_dim = out.shape[-1]
    angle_sl = angle_slice_for_arm_dim(arm_dim)
    velocity_sl = velocity_slice_for_arm_dim(arm_dim)
    eff_sl = eff_slice_for_arm_dim(arm_dim)

    for i in range(out.shape[0]):
        q0 = out[i, angle_sl] * np.pi
        q = _project_angles_to_ee(q0, desired_ee[i])
        out[i, angle_sl] = q / np.pi
        out[i, velocity_sl] = np.clip(out[i, velocity_sl], -1.0, 1.0)
        out[i, eff_sl] = np.clip(_fk_np(q) / MAX_REACH, -1.0, 1.0)

    return out[0] if single else out


def fk_from_arm_state_torch(arm_state: torch.Tensor) -> torch.Tensor:
    """Differentiable FK in normalized coordinates from normalized angles."""
    arm_dim = arm_state.shape[-1]
    if arm_dim == ARM_OBS_2DOF:
        t1 = arm_state[..., 0] * torch.pi
        t2 = arm_state[..., 1] * torch.pi
        x = 1.5 * torch.cos(t1) + 1.5 * torch.cos(t1 + t2)
        y = 1.5 * torch.sin(t1) + 1.5 * torch.sin(t1 + t2)
    elif arm_dim == ARM_OBS_3DOF:
        t1 = arm_state[..., 0] * torch.pi
        t2 = arm_state[..., 1] * torch.pi
        t3 = arm_state[..., 2] * torch.pi
        x = torch.cos(t1) + torch.cos(t1 + t2) + torch.cos(t1 + t2 + t3)
        y = torch.sin(t1) + torch.sin(t1 + t2) + torch.sin(t1 + t2 + t3)
    else:
        raise ValueError(f"Unsupported arm observation dim: {arm_dim}")

    return torch.stack([x, y], dim=-1) / MAX_REACH


def rebuild_arm_state_with_fk(arm_state: torch.Tensor) -> torch.Tensor:
    """Replace eff_x/eff_y with FK(theta), keeping angles and velocities."""
    eff_sl = eff_slice_for_arm_dim(arm_state.shape[-1])
    out = arm_state.clone()
    out[..., eff_sl] = fk_from_arm_state_torch(out)
    return torch.clamp(out, -1.0, 1.0)


def arm_effector_fields_torch(arm_state: torch.Tensor) -> torch.Tensor:
    return arm_state[..., eff_slice_for_arm_dim(arm_state.shape[-1])]
