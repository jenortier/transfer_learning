"""Cinématique planaire partagée pour les bras 2DoF et 3DoF.

Source unique de vérité pour :
  - les constantes géométriques et de normalisation (identiques aux envs) ;
  - la cinématique directe FK(theta) ;
  - les Jacobiens J(theta) tels que v_eff = J(theta) @ dtheta ;
  - la pseudo-inverse amortie (damped least squares) ;
  - les helpers de normalisation/dénormalisation des états.

Toutes les fonctions torch sont batchées (theta de forme (..., nq)) et
différentiables, donc utilisables directement dans les losses d'entraînement.
Les versions NumPy acceptent un vecteur simple (nq,) ou un batch (N, nq).

Auto-test :
    cd robot-robot && python3 -m direct_method.kinematics
"""

import numpy as np
import torch

# ── Constantes (doivent rester identiques à envs/arm_2dof.py et arm_3dof.py) ──
L_2DOF = (1.5, 1.5)
L_3DOF = (1.0, 1.0, 1.0)
OMEGA_MAX = 2.0    # rad/s
MAX_REACH = 3.0    # m
DT = 0.05          # s


# ══════════════════════════════════════════════════════════════════
# Cinématique générique (torch, batchée, différentiable)
# ══════════════════════════════════════════════════════════════════

def _reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Somme cumulée depuis la fin sur la dernière dimension."""
    return torch.flip(torch.cumsum(torch.flip(x, [-1]), dim=-1), [-1])


def fk(theta: torch.Tensor, lengths) -> torch.Tensor:
    """Position de l'effecteur. theta: (..., nq) -> (..., 2)."""
    cum = torch.cumsum(theta, dim=-1)
    lens = theta.new_tensor(lengths)
    x = (lens * torch.cos(cum)).sum(dim=-1)
    y = (lens * torch.sin(cum)).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


def jacobian(theta: torch.Tensor, lengths) -> torch.Tensor:
    """Jacobien planaire. theta: (..., nq) -> (..., 2, nq).

    J[0, j] = -sum_{k>=j} l_k sin(theta_1+..+theta_k)
    J[1, j] =  sum_{k>=j} l_k cos(theta_1+..+theta_k)
    """
    cum = torch.cumsum(theta, dim=-1)
    lens = theta.new_tensor(lengths)
    jx = -_reverse_cumsum(lens * torch.sin(cum))
    jy = _reverse_cumsum(lens * torch.cos(cum))
    return torch.stack([jx, jy], dim=-2)


def eff_vel(theta: torch.Tensor, dtheta: torch.Tensor, lengths) -> torch.Tensor:
    """Vitesse cartésienne de l'effecteur v = J(theta) @ dtheta. -> (..., 2)."""
    return (jacobian(theta, lengths) @ dtheta.unsqueeze(-1)).squeeze(-1)


def damped_pinv(J: torch.Tensor, lam: float = 0.05) -> torch.Tensor:
    """Pseudo-inverse amortie J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹. J: (..., 2, nq) -> (..., nq, 2).

    L'amortissement λ borne le gain près des singularités (bras tendu) au prix
    d'une légère erreur de suivi partout ailleurs.
    """
    Jt = J.transpose(-1, -2)
    eye = torch.eye(J.shape[-2], dtype=J.dtype, device=J.device)
    return Jt @ torch.linalg.inv(J @ Jt + (lam ** 2) * eye)


# ── Raccourcis par robot (torch) ──────────────────────────────────

def fk_2dof(theta): return fk(theta, L_2DOF)
def fk_3dof(theta): return fk(theta, L_3DOF)
def jacobian_2dof(theta): return jacobian(theta, L_2DOF)
def jacobian_3dof(theta): return jacobian(theta, L_3DOF)
def eff_vel_2dof(theta, dtheta): return eff_vel(theta, dtheta, L_2DOF)
def eff_vel_3dof(theta, dtheta): return eff_vel(theta, dtheta, L_3DOF)


# ══════════════════════════════════════════════════════════════════
# Versions NumPy (wrappers sur torch, pour scripts/éval/transfert)
# ══════════════════════════════════════════════════════════════════

def _np_wrap(fn, *arrays):
    tensors = [torch.as_tensor(np.asarray(a), dtype=torch.float64) for a in arrays]
    with torch.no_grad():
        out = fn(*tensors)
    return out.numpy()


def fk_2dof_np(theta): return _np_wrap(fk_2dof, theta)
def fk_3dof_np(theta): return _np_wrap(fk_3dof, theta)
def jacobian_2dof_np(theta): return _np_wrap(jacobian_2dof, theta)
def jacobian_3dof_np(theta): return _np_wrap(jacobian_3dof, theta)
def eff_vel_2dof_np(theta, dtheta): return _np_wrap(eff_vel_2dof, theta, dtheta)
def eff_vel_3dof_np(theta, dtheta): return _np_wrap(eff_vel_3dof, theta, dtheta)


def damped_pinv_np(J, lam: float = 0.05):
    return _np_wrap(lambda j: damped_pinv(j, lam), J)


# ══════════════════════════════════════════════════════════════════
# Normalisation (convention des envs et des fichiers de trajectoires)
# ══════════════════════════════════════════════════════════════════
# theta_norm = theta / pi ; dtheta_norm = dtheta / OMEGA_MAX ; eff_norm = eff / MAX_REACH

def denorm_theta(theta_norm): return theta_norm * np.pi
def norm_theta(theta): return theta / np.pi
def denorm_dtheta(dtheta_norm): return dtheta_norm * OMEGA_MAX
def norm_dtheta(dtheta): return dtheta / OMEGA_MAX
def denorm_eff(eff_norm): return eff_norm * MAX_REACH
def norm_eff(eff): return eff / MAX_REACH


def split_state(state, nq):
    """Découpe un bloc état [theta(nq), dtheta(nq), eff(2)] (normalisé ou non)."""
    return state[..., :nq], state[..., nq:2 * nq], state[..., 2 * nq:2 * nq + 2]


# ══════════════════════════════════════════════════════════════════
# Auto-test
# ══════════════════════════════════════════════════════════════════

def _self_test():
    from envs.arm_2dof import Arm2DoF
    from envs.arm_3dof import Arm3DoF

    rng = np.random.default_rng(0)
    env2, env3 = Arm2DoF(), Arm3DoF()
    max_fk_err = 0.0
    max_vel_err = 0.0

    for _ in range(200):
        t2 = rng.uniform(-np.pi, np.pi, 2)
        t3 = rng.uniform(-np.pi, np.pi, 3)
        d2 = rng.uniform(-OMEGA_MAX, OMEGA_MAX, 2)
        d3 = rng.uniform(-OMEGA_MAX, OMEGA_MAX, 3)

        # FK vs envs
        max_fk_err = max(max_fk_err,
                         np.abs(fk_2dof_np(t2) - env2.forward_kinematics(*t2)).max(),
                         np.abs(fk_3dof_np(t3) - env3.forward_kinematics(*t3)).max())

        # v_eff vs envs (get_end_effector_vel)
        env2.theta1, env2.theta2 = t2
        env2.dtheta1, env2.dtheta2 = d2
        env3.theta1, env3.theta2, env3.theta3 = t3
        env3.dtheta1, env3.dtheta2, env3.dtheta3 = d3
        max_vel_err = max(max_vel_err,
                          np.abs(eff_vel_2dof_np(t2, d2) - env2.get_end_effector_vel()).max(),
                          np.abs(eff_vel_3dof_np(t3, d3) - env3.get_end_effector_vel()).max())

    print(f"FK    max |err| vs envs : {max_fk_err:.2e}")
    print(f"v_eff max |err| vs envs : {max_vel_err:.2e}")
    assert max_fk_err < 1e-5 and max_vel_err < 1e-10

    # Jacobien vs différences finies (batché)
    theta = torch.tensor(rng.uniform(-np.pi, np.pi, (64, 3)), dtype=torch.float64)
    J = jacobian_3dof(theta)
    eps = 1e-6
    max_j_err = 0.0
    for j in range(3):
        dt = torch.zeros_like(theta)
        dt[:, j] = eps
        num = (fk_3dof(theta + dt) - fk_3dof(theta - dt)) / (2 * eps)
        max_j_err = max(max_j_err, (num - J[:, :, j]).abs().max().item())
    print(f"Jacobien 3DoF max |err| vs diff. finies : {max_j_err:.2e}")
    assert max_j_err < 1e-6

    # damped_pinv : reconstruction de v hors singularités
    theta = torch.tensor(rng.uniform(-np.pi, np.pi, (2000, 3)), dtype=torch.float64)
    v = torch.tensor(rng.uniform(-1.0, 1.0, (2000, 2)), dtype=torch.float64)
    J = jacobian_3dof(theta)
    dq = (damped_pinv(J, 0.05) @ v.unsqueeze(-1)).squeeze(-1)
    v_rec = (J @ dq.unsqueeze(-1)).squeeze(-1)
    # manipulabilité = sqrt(det(J Jᵀ)) : on exclut les configurations singulières
    manip = torch.sqrt(torch.det(J @ J.transpose(-1, -2)))
    ok = manip > 0.5
    err = (v_rec - v).norm(dim=-1)
    print(f"damped_pinv (manip>0.5, {int(ok.sum())} pts) : err médiane={err[ok].median():.2e}, max={err[ok].max():.2e}")
    # l'amortissement introduit une erreur en lam²/(sigma²+lam²) : ~5 % au pire ici
    assert err[ok].median() < 1e-2 and err[ok].max() < 1e-1

    print("Auto-test kinematics : OK")


if __name__ == "__main__":
    _self_test()
