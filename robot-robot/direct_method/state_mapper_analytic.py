"""State mapper analytique 3DoF -> 2DoF par cinématique inverse exacte.

Principe : la grandeur commune aux deux robots est la position/vitesse
cartésienne de l'effecteur. Plutôt que d'apprendre la correspondance d'états,
on la calcule :

    eff        = FK3(theta_3)                       [exacte, partagée]
    theta_2    = IK2(eff)                           [forme fermée, 2 branches]
    dtheta_2   = J2(theta_2)⁺ @ (J3(theta_3) @ dtheta_3)

L'IK 2DoF a deux solutions (coude haut / coude bas). Le mapper est *stateful* :
au premier appel d'un épisode il choisit la branche dont la courbure correspond
à celle du bras 3DoF (signe de theta2+theta3), puis il conserve la branche la
plus proche de la posture précédente pour garantir la continuité temporelle
(pas de saut de posture ni de vitesse dans les observations de la politique).

Erreur de position : 0 par construction (hors zone inatteignable, inexistante
ici car max_reach est identique). Erreur de vitesse : celle de la pseudo-inverse
amortie, bornée près des singularités du 2DoF (bras tendu r≈3, replié r≈0).

Interface : entrées/sorties NORMALISÉES comme les blocs bras des envs
    entrée  8D : [theta/pi (3), dtheta/omega_max (3), eff/max_reach (2)]
    sortie  6D : [theta/pi (2), dtheta/omega_max (2), eff/max_reach (2)]

Auto-test :
    cd robot-robot && python3 -m direct_method.state_mapper_analytic
"""

import numpy as np

from direct_method.kinematics import (
    L_2DOF, L_3DOF, OMEGA_MAX, MAX_REACH,
    fk_2dof_np, fk_3dof_np, jacobian_2dof_np, jacobian_3dof_np, damped_pinv_np,
)


def _wrap(a):
    """Ramène un angle dans [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def ik_2dof_branches(eff):
    """Les deux solutions IK du bras 2DoF pour une position d'effecteur (m).

    Retourne une liste de 1 ou 2 configurations [theta1, theta2].
    La cible est toujours atteignable ici (l1 = l2 -> portée [0, 3] = celle du
    3DoF) ; cos(theta2) est clippé pour absorber les erreurs d'arrondi au bord.
    """
    l1, l2 = L_2DOF
    x, y = float(eff[0]), float(eff[1])
    r2 = x * x + y * y

    c2 = np.clip((r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    t2 = float(np.arccos(c2))                       # branche coude "+" ; l'autre est -t2

    out = []
    for s in ((+1.0,) if t2 < 1e-9 else (+1.0, -1.0)):
        th2 = s * t2
        th1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(th2), l1 + l2 * np.cos(th2))
        out.append(np.array([_wrap(th1), th2]))
    return out


def ik_2r_branches(target, l1, l2):
    """Les deux solutions IK d'un bras 2R quelconque (longueurs l1, l2).

    Comme ik_2dof_branches mais paramétré par les longueurs — utilisé pour le
    poignet du bras 3DoF (l1 = l2 = 1). Retourne 1 ou 2 configs [theta1, theta2].
    """
    x, y = float(target[0]), float(target[1])
    r2 = x * x + y * y
    c2 = np.clip((r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    t2 = float(np.arccos(c2))

    out = []
    for s in ((+1.0,) if t2 < 1e-9 else (+1.0, -1.0)):
        th2 = s * t2
        th1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(th2), l1 + l2 * np.cos(th2))
        out.append(np.array([_wrap(th1), th2]))
    return out


class AnalyticStateMapper3to2:
    """Mapper d'état 3DoF -> 2DoF exact, même convention normalisée que les envs.

    lam : amortissement de la pseudo-inverse pour dtheta_2 (singularités 2DoF).
    Appeler reset() au début de chaque épisode (réinitialise le suivi de branche).
    """

    def __init__(self, lam: float = 0.05):
        self.lam = lam
        self._prev_theta2 = None

    def reset(self):
        self._prev_theta2 = None

    def _pick_branch(self, branches, theta_3):
        if len(branches) == 1:
            return branches[0]
        if self._prev_theta2 is None:
            # premier pas : courbure du coude 2DoF alignée sur celle du 3DoF
            bend = theta_3[1] + theta_3[2]
            sign = 1.0 if bend >= 0.0 else -1.0
            return branches[0] if branches[0][1] * sign >= 0.0 else branches[1]
        # continuité : branche la plus proche de la posture précédente
        dists = [np.abs(_wrap(b - self._prev_theta2)).sum() for b in branches]
        return branches[int(np.argmin(dists))]

    def __call__(self, arm_obs_3):
        """arm_obs_3 : bloc bras 3DoF normalisé (8,) -> bloc 2DoF normalisé (6,)."""
        arm_obs_3 = np.asarray(arm_obs_3, dtype=np.float64)
        theta_3 = arm_obs_3[:3] * np.pi
        dtheta_3 = arm_obs_3[3:6] * OMEGA_MAX

        eff = fk_3dof_np(theta_3)                   # exact (les colonnes obs sont clippées)
        theta_2 = self._pick_branch(ik_2dof_branches(eff), theta_3)
        self._prev_theta2 = theta_2

        v_eff = jacobian_3dof_np(theta_3) @ dtheta_3
        dtheta_2 = damped_pinv_np(jacobian_2dof_np(theta_2), self.lam) @ v_eff
        dtheta_2 = np.clip(dtheta_2, -OMEGA_MAX, OMEGA_MAX)

        return np.concatenate([
            theta_2 / np.pi,
            dtheta_2 / OMEGA_MAX,
            eff / MAX_REACH,
        ]).astype(np.float32)


class AnalyticStateMapper2to3:
    """State mapper analytique 2DoF -> 3DoF, exact en position, même convention.

    Le 3DoF est redondant (3 articulations pour une position 2D) : la position
    d'effecteur ne suffit pas à fixer la posture. La redondance (1 ddl) est
    résolue analytiquement en orientant le *dernier* lien radialement,
    phi = atan2(eff_y, eff_x). Le poignet est alors à un rayon |r - l3| toujours
    atteignable par les deux premiers liens (l1 = l2 = 1 -> portée [0, 2], et
    |r - 1| <= 2 pour r dans [0, 3]) : la position reconstruite est donc EXACTE
    partout, et le choix de posture est déterministe et continu dans le temps.

        eff       = FK2(theta_2)                        [exacte, partagée]
        phi       = atan2(eff_y, eff_x)                 [dernier lien radial]
        wrist     = eff - l3 * [cos phi, sin phi]
        theta1,2  = IK2R(wrist ; l1, l2)                [2 branches, continuité]
        theta3    = phi - theta1 - theta2
        dtheta_3  = J3(theta_3)⁺ @ (J2(theta_2) @ dtheta_2)   [min-norm amortie]

    lam : amortissement de la pseudo-inverse pour dtheta_3.
    Appeler reset() au début de chaque épisode (réinitialise le suivi de branche).

    Interface : entrées/sorties NORMALISÉES comme les blocs bras des envs
        entrée  6D : [theta/pi (2), dtheta/omega_max (2), eff/max_reach (2)]
        sortie  8D : [theta/pi (3), dtheta/omega_max (3), eff/max_reach (2)]
    """

    def __init__(self, lam: float = 0.05):
        self.lam = lam
        self._prev_theta3 = None

    def reset(self):
        self._prev_theta3 = None

    def _pick_branch(self, branches, bend_sign):
        if len(branches) == 1:
            return branches[0]
        if self._prev_theta3 is None:
            # premier pas : courbure du coude alignée sur celle du bras 2DoF source
            return branches[0] if branches[0][1] * bend_sign >= 0.0 else branches[1]
        # continuité : branche la plus proche de la posture (2 premiers axes) précédente
        prev = self._prev_theta3[:2]
        dists = [np.abs(_wrap(b - prev)).sum() for b in branches]
        return branches[int(np.argmin(dists))]

    def __call__(self, arm_obs_2):
        """arm_obs_2 : bloc bras 2DoF normalisé (6,) -> bloc 3DoF normalisé (8,)."""
        arm_obs_2 = np.asarray(arm_obs_2, dtype=np.float64)
        theta_2 = arm_obs_2[:2] * np.pi
        dtheta_2 = arm_obs_2[2:4] * OMEGA_MAX

        eff = fk_2dof_np(theta_2)                   # exact (les colonnes obs sont clippées)
        l1, l2, l3 = L_3DOF
        r = float(np.linalg.norm(eff))
        phi = float(np.arctan2(eff[1], eff[0])) if r > 1e-9 else 0.0
        wrist = eff - l3 * np.array([np.cos(phi), np.sin(phi)])

        bend_sign = 1.0 if theta_2[1] >= 0.0 else -1.0
        th12 = self._pick_branch(ik_2r_branches(wrist, l1, l2), bend_sign)
        theta_3 = np.array([th12[0], th12[1], _wrap(phi - th12[0] - th12[1])])
        self._prev_theta3 = theta_3

        v_eff = jacobian_2dof_np(theta_2) @ dtheta_2
        dtheta_3 = damped_pinv_np(jacobian_3dof_np(theta_3), self.lam) @ v_eff
        dtheta_3 = np.clip(dtheta_3, -OMEGA_MAX, OMEGA_MAX)

        return np.concatenate([
            theta_3 / np.pi,
            dtheta_3 / OMEGA_MAX,
            eff / MAX_REACH,
        ]).astype(np.float32)


def _self_test():
    from direct_method.kinematics import fk_2dof_np, eff_vel_2dof_np, eff_vel_3dof_np, DT

    rng = np.random.default_rng(0)
    mapper = AnalyticStateMapper3to2()

    max_fk_err = 0.0
    vel_errs, manips, jumps, radii = [], [], [], []

    for _ in range(50):                             # 50 trajectoires lisses de 100 pas
        theta_3 = rng.uniform(-np.pi, np.pi, 3)
        dtheta_3 = rng.uniform(-1.0, 1.0, 3)
        mapper.reset()
        prev_t2 = None
        for _ in range(100):
            dtheta_3 = np.clip(dtheta_3 + rng.normal(0, 0.15, 3), -OMEGA_MAX, OMEGA_MAX)
            theta_3 = _wrap(theta_3 + dtheta_3 * DT)

            obs8 = np.concatenate([theta_3 / np.pi, dtheta_3 / OMEGA_MAX,
                                   fk_3dof_np(theta_3) / MAX_REACH])
            out6 = mapper(obs8)
            theta_2 = out6[:2] * np.pi
            dtheta_2 = out6[2:4] * OMEGA_MAX

            # 1) position : exacte par construction
            max_fk_err = max(max_fk_err, float(np.linalg.norm(
                fk_2dof_np(theta_2) - fk_3dof_np(theta_3))))
            # 2) vitesse : bornée hors singularités du 2DoF
            vel_errs.append(float(np.linalg.norm(
                eff_vel_2dof_np(theta_2, dtheta_2) - eff_vel_3dof_np(theta_3, dtheta_3))))
            manips.append(abs(np.sin(theta_2[1])) * L_2DOF[0] * L_2DOF[1])
            # 3) continuité de branche
            if prev_t2 is not None:
                jumps.append(float(np.abs(_wrap(theta_2 - prev_t2)).max()))
                radii.append(float(np.linalg.norm(fk_3dof_np(theta_3))))
            prev_t2 = theta_2

    vel_errs, manips, jumps, radii = map(np.asarray, (vel_errs, manips, jumps, radii))
    ok = manips > 0.5
    far = radii > 0.3                  # près de r=0 le 2DoF doit pivoter vite : saut légitime
    print(f"position : max ||FK2(IK2) - FK3|| = {max_fk_err:.2e} m   (float32 en sortie)")
    print(f"vitesse  : ||v2 - v3|| médiane = {np.median(vel_errs):.2e} m/s "
          f"(hors singularités : {np.median(vel_errs[ok]):.2e}, {ok.sum()} pts)")
    print(f"continuité : saut max de theta_2 = {jumps[far].max():.3f} rad (r>0.3) ; "
          f"{jumps.max():.3f} rad (global, {int((~far).sum())} pts près du centre)")
    assert max_fk_err < 1e-5
    assert np.median(vel_errs[ok]) < 2e-2
    # un flip de branche vaudrait ~2|theta_2| (souvent > 2 rad) ; ~dtheta*dt + variation IK sinon
    assert jumps[far].max() < 0.8
    print("Auto-test state mapper analytique 3->2 : OK")

    # ── 2 -> 3 : position exacte par construction, vitesse reproduite ──
    mapper23 = AnalyticStateMapper2to3()
    max_fk_err23, vel_errs23 = 0.0, []
    for _ in range(50):
        theta_2 = rng.uniform(-np.pi, np.pi, 2)
        dtheta_2 = rng.uniform(-1.0, 1.0, 2)
        mapper23.reset()
        for _ in range(100):
            dtheta_2 = np.clip(dtheta_2 + rng.normal(0, 0.15, 2), -OMEGA_MAX, OMEGA_MAX)
            theta_2 = _wrap(theta_2 + dtheta_2 * DT)
            obs6 = np.concatenate([theta_2 / np.pi, dtheta_2 / OMEGA_MAX,
                                   fk_2dof_np(theta_2) / MAX_REACH])
            out8 = mapper23(obs6)
            theta_3 = out8[:3] * np.pi
            dtheta_3 = out8[3:6] * OMEGA_MAX
            max_fk_err23 = max(max_fk_err23, float(np.linalg.norm(
                fk_3dof_np(theta_3) - fk_2dof_np(theta_2))))
            vel_errs23.append(float(np.linalg.norm(
                eff_vel_3dof_np(theta_3, dtheta_3) - eff_vel_2dof_np(theta_2, dtheta_2))))
    vel_errs23 = np.asarray(vel_errs23)
    print(f"position : max ||FK3(map) - FK2|| = {max_fk_err23:.2e} m   (2->3, float32)")
    print(f"vitesse  : ||v3 - v2|| médiane = {np.median(vel_errs23):.2e} m/s (2->3)")
    assert max_fk_err23 < 1e-5
    assert np.median(vel_errs23) < 2e-2
    print("Auto-test state mapper analytique 2->3 : OK")


if __name__ == "__main__":
    _self_test()
