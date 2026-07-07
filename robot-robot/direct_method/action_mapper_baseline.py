"""Baseline analytique pour le mapping d'actions entre bras 2DoF et 3DoF.

Principe : la seule grandeur commune aux deux robots est la vitesse cartésienne
de l'effecteur. On convertit donc l'action source en vitesse effecteur via le
Jacobien source, puis on la reconvertit en vitesses articulaires cibles via la
pseudo-inverse amortie du Jacobien cible :

    v          = J_src(theta_src) @ dtheta_src
    dtheta_tgt = J_tgt(theta_tgt)⁺ @ v  [+ (I − J⁺J) @ z  pour le 3DoF]

Aucun entraînement requis. Sert de référence de performance minimale pour tout
mapper appris, et de mapper de secours dans le script de transfert.

Auto-test :
    cd robot-robot && python3 -m direct_method.action_mapper_baseline
"""

import numpy as np
import torch

from direct_method.kinematics import (
    L_2DOF, L_3DOF, OMEGA_MAX,
    jacobian, eff_vel, damped_pinv,
)

_LENGTHS = {2: L_2DOF, 3: L_3DOF}


class JacobianActionMapper:
    """Mapping d'action analytique par Jacobien, même interface qu'un mapper appris.

    direction : "2to3" ou "3to2".
    lam       : amortissement de la pseudo-inverse (gère les singularités).
    null_space_gain : si > 0 (3DoF cible uniquement), ajoute un terme de
        null-space z = -gain * theta_tgt qui ramène doucement la posture vers
        theta = 0 sans modifier la vitesse de l'effecteur.

    Appel : __call__(dtheta_src, theta_src, theta_tgt) -> dtheta_tgt
    en unités réelles (rad, rad/s), formes (nq,) ou (N, nq).
    """

    def __init__(self, direction: str, lam: float = 0.05, null_space_gain: float = 0.0):
        if direction == "2to3":
            self.nq_src, self.nq_tgt = 2, 3
        elif direction == "3to2":
            self.nq_src, self.nq_tgt = 3, 2
        else:
            raise ValueError(f"direction inconnue: {direction!r} (attendu '2to3' ou '3to2')")
        self.lam = lam
        self.null_space_gain = null_space_gain

    def map_torch(self, dtheta_src: torch.Tensor, theta_src: torch.Tensor,
                  theta_tgt: torch.Tensor) -> torch.Tensor:
        """Version torch batchée (différentiable), unités réelles."""
        len_src = _LENGTHS[self.nq_src]
        len_tgt = _LENGTHS[self.nq_tgt]

        v = eff_vel(theta_src, dtheta_src, len_src)              # (..., 2)
        J_tgt = jacobian(theta_tgt, len_tgt)                     # (..., 2, nq_tgt)
        J_pinv = damped_pinv(J_tgt, self.lam)                    # (..., nq_tgt, 2)
        dtheta_tgt = (J_pinv @ v.unsqueeze(-1)).squeeze(-1)

        if self.null_space_gain > 0.0 and self.nq_tgt > 2:
            eye = torch.eye(self.nq_tgt, dtype=J_tgt.dtype, device=J_tgt.device)
            N = eye - J_pinv @ J_tgt                             # projecteur null-space
            z = -self.null_space_gain * theta_tgt
            dtheta_tgt = dtheta_tgt + (N @ z.unsqueeze(-1)).squeeze(-1)

        return dtheta_tgt

    def __call__(self, dtheta_src, theta_src, theta_tgt):
        """Version NumPy, unités réelles. Clippe la sortie à ±OMEGA_MAX."""
        single = np.asarray(dtheta_src).ndim == 1
        t = lambda a: torch.as_tensor(np.atleast_2d(np.asarray(a, dtype=np.float64)))
        with torch.no_grad():
            out = self.map_torch(t(dtheta_src), t(theta_src), t(theta_tgt)).numpy()
        out = np.clip(out, -OMEGA_MAX, OMEGA_MAX)
        return out[0] if single else out


def _self_test():
    """Sur des états aléatoires : la vitesse effecteur cible doit reproduire la source."""
    from direct_method.kinematics import eff_vel_2dof_np, eff_vel_3dof_np, jacobian_3dof_np

    rng = np.random.default_rng(1)
    n = 5000
    t2 = rng.uniform(-np.pi, np.pi, (n, 2))
    d2 = rng.uniform(-1.0, 1.0, (n, 2))
    t3 = rng.uniform(-np.pi, np.pi, (n, 3))
    d3 = rng.uniform(-1.0, 1.0, (n, 3))

    # 2 -> 3
    m23 = JacobianActionMapper("2to3", lam=0.05)
    d3_pred = m23(d2, t2, t3)
    v_src = eff_vel_2dof_np(t2, d2)
    v_tgt = eff_vel_3dof_np(t3, d3_pred)
    err23 = np.linalg.norm(v_tgt - v_src, axis=-1)

    # le 3DoF cible peut être singulier : on ventile par manipulabilité
    J3 = jacobian_3dof_np(t3)
    manip = np.sqrt(np.linalg.det(J3 @ np.transpose(J3, (0, 2, 1))))
    ok = manip > 0.5
    print(f"2->3 : |v_tgt - v_src| médiane={np.median(err23[ok]):.2e} "
          f"(hors singularités, {ok.sum()} pts) ; médiane globale={np.median(err23):.2e}")
    assert np.median(err23[ok]) < 2e-2

    # 3 -> 2 : le 2DoF ne peut pas toujours suivre (portée/direction), mais hors
    # singularités du 2DoF l'erreur doit rester faible
    m32 = JacobianActionMapper("3to2", lam=0.05)
    d2_pred = m32(d3, t3, t2)
    v_src = eff_vel_3dof_np(t3, d3)
    v_tgt = eff_vel_2dof_np(t2, d2_pred)
    err32 = np.linalg.norm(v_tgt - v_src, axis=-1)
    # manipulabilité du 2DoF : |sin(theta2)| * l1 * l2
    manip2 = np.abs(np.sin(t2[:, 1])) * L_2DOF[0] * L_2DOF[1]
    ok2 = manip2 > 0.5
    print(f"3->2 : |v_tgt - v_src| médiane={np.median(err32[ok2]):.2e} "
          f"(hors singularités, {ok2.sum()} pts) ; médiane globale={np.median(err32):.2e}")
    assert np.median(err32[ok2]) < 5e-2

    # null-space : ne doit pas changer la vitesse effecteur (test sur la version
    # non clippée map_torch ; le clipping à ±omega_max de __call__ peut, lui,
    # altérer la composante tâche si une articulation sature)
    m23_ns = JacobianActionMapper("2to3", lam=0.05, null_space_gain=0.5)
    tt3 = torch.as_tensor(t3)
    d3_plain = m23.map_torch(torch.as_tensor(d2), torch.as_tensor(t2), tt3).numpy()
    d3_ns = m23_ns.map_torch(torch.as_tensor(d2), torch.as_tensor(t2), tt3).numpy()
    dv = np.linalg.norm(eff_vel_3dof_np(t3, d3_ns) - eff_vel_3dof_np(t3, d3_plain), axis=-1)
    print(f"null-space : |Δv_eff| max={dv[ok].max():.2e} (doit être ~0 hors singularités)")
    assert dv[ok].max() < 5e-2

    print("Auto-test baseline Jacobien : OK")


if __name__ == "__main__":
    _self_test()
