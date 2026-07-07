"""Visualise l'ACTION mapper : la vitesse d'effecteur cible reproduit-elle la source ?

Deux modes :
  - vue par état (--idx N, ou animation sans --idx) : bras source avec sa
    vitesse d'effecteur, bras cible (posture du dataset) avec la vitesse
    produite par l'action prédite + la vitesse source en fantôme. Le titre
    affiche l'erreur d'angle et le rapport de normes — exactement le
    protocole avec lequel le symptôme a été observé.
  - --summary : erreurs d'angle et de norme en fonction du rayon effecteur r
    sur le split de validation, en PNG. Permet de vérifier d'un coup d'œil
    que l'erreur d'angle près du centre et l'erreur de norme bras tendu ont
    disparu.

Usage :
    cd robot-robot && python3 -m direct_method.visualize_action_mapper --summary
    cd robot-robot && python3 -m direct_method.visualize_action_mapper \
        --mapper jacobian --direction 2to3 --idx 1200 --save
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from direct_method.dataset import TrajectoriesTrainingActionDataset, block_split
from direct_method.mapper_models import ActionMapperMLP, ActionMapperConditionedMLP
from direct_method.action_mapper_baseline import JacobianActionMapper
from direct_method.kinematics import (
    OMEGA_MAX, fk_2dof_np, fk_3dof_np, eff_vel_2dof_np, eff_vel_3dof_np)
from direct_method.visualize_velocity_mapper import (
    setup_axis, draw_arrow, joint_positions_2dof, joint_positions_3dof)

HIDDEN_DIM = 256
SEED = 0

FK_NP = {2: fk_2dof_np, 3: fk_3dof_np}
EFF_VEL_NP = {2: eff_vel_2dof_np, 3: eff_vel_3dof_np}


def load_mapper_fn(kind, direction, run_legacy, run_cond, lam):
    """Retourne fn(vel_src_norm, q_src_norm, q_tgt_norm) -> dtheta_tgt (rad/s)."""
    nq_src, nq_tgt = (2, 3) if direction == "2to3" else (3, 2)

    if kind == "jacobian":
        base = JacobianActionMapper(direction, lam=lam)
        return lambda vel, qs, qt: base(vel * OMEGA_MAX, qs * np.pi, qt * np.pi)

    if kind == "legacy":
        name = {"2to3": "action_mapper_r1_to_r2", "3to2": "action_mapper_r2_to_r1"}[direction]
        path = f"direct_method/runs/{run_legacy}/models/{name}.pt"
        m = ActionMapperMLP(nq_src, nq_tgt, HIDDEN_DIM)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def fn(vel, qs, qt):
            with torch.no_grad():
                return m(torch.from_numpy(vel)).numpy() * OMEGA_MAX
        return fn

    if kind == "conditioned":
        name = {"2to3": "action_mapper_2to3", "3to2": "action_mapper_3to2"}[direction]
        path = f"direct_method/runs/{run_cond}/models/{name}.pt"
        m = ActionMapperConditionedMLP(nq_src, nq_tgt, HIDDEN_DIM)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def fn(vel, qs, qt):
            with torch.no_grad():
                return m(torch.from_numpy(vel), torch.from_numpy(qs),
                         torch.from_numpy(qt)).numpy() * OMEGA_MAX
        return fn

    raise ValueError(kind)


def draw_arm(ax, nq, theta, title):
    if nq == 2:
        _, j1, eff = joint_positions_2dof(theta)
        pts = [(0, 0), tuple(j1), tuple(eff)]
        colors = ["#E05A3A", "#3A8FE0"]
    else:
        _, j1, j2, eff = joint_positions_3dof(theta)
        pts = [(0, 0), tuple(j1), tuple(j2), tuple(eff)]
        colors = ["#E05A3A", "#3A8FE0", "#9B5DE5"]
    setup_axis(ax, title)
    for (a, b), c in zip(zip(pts[:-1], pts[1:]), colors):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, linewidth=4.0)
    ax.plot(eff[0], eff[1], "o", color="#FFFFFF", markeredgecolor="#111111", markersize=9.0)
    return np.asarray(eff)


def angle_between_deg(v1, v2):
    n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    cos = np.clip((v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def main():
    p = argparse.ArgumentParser(description="Visualiser l'action mapper en espace effecteur.")
    p.add_argument("--mapper", choices=["jacobian", "legacy", "conditioned"], default="jacobian")
    p.add_argument("--direction", choices=["2to3", "3to2"], default="2to3")
    p.add_argument("--idx", type=int, default=None)
    p.add_argument("--summary", action="store_true",
                   help="PNG des erreurs angle/norme vs rayon (split de validation)")
    p.add_argument("--delay", type=float, default=0.5)
    p.add_argument("--save", action="store_true")
    p.add_argument("--run-action-legacy", default="run_02")
    p.add_argument("--run-action-cond", default="run_04_cond")
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--out", default="direct_method/runs/eval")
    args = p.parse_args()

    ds = TrajectoriesTrainingActionDataset(
        "direct_method/trajectories/robot_2dofs.txt",
        "direct_method/trajectories/robot_3dofs.txt", nq_r1=2, nq_r2=3)

    if args.direction == "2to3":
        nq_src, nq_tgt = 2, 3
        vel_src, q_src, q_tgt = ds.vel_r1, ds.q_r1, ds.q_r2
    else:
        nq_src, nq_tgt = 3, 2
        vel_src, q_src, q_tgt = ds.vel_r2, ds.q_r2, ds.q_r1

    fn = load_mapper_fn(args.mapper, args.direction,
                        args.run_action_legacy, args.run_action_cond, args.lam)

    theta_src = q_src * np.pi
    theta_tgt = q_tgt * np.pi
    v_src = EFF_VEL_NP[nq_src](theta_src, vel_src * OMEGA_MAX)

    # ── Mode résumé : erreurs vs rayon sur la validation ─────────────
    if args.summary:
        matplotlib.use("Agg")
        _, val, _ = block_split(ds, seed=SEED)
        idx = np.asarray(val.indices)
        dq_pred = fn(vel_src[idx], q_src[idx], q_tgt[idx])
        v_pred = EFF_VEL_NP[nq_tgt](theta_tgt[idx], dq_pred)
        vs = v_src[idx]

        r = np.linalg.norm(FK_NP[nq_src](theta_src[idx]), axis=-1)
        moving = np.linalg.norm(vs, axis=-1) > 0.05
        angle = angle_between_deg(v_pred[moving], vs[moving])
        norm_err = np.abs(np.linalg.norm(v_pred, axis=-1) - np.linalg.norm(vs, axis=-1))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(r[moving], angle, s=2, alpha=0.15, color="steelblue")
        axes[0].set_xlabel("rayon effecteur source r (m)")
        axes[0].set_ylabel("erreur d'angle v_pred/v_src (deg)")
        axes[1].scatter(r, norm_err, s=2, alpha=0.15, color="crimson")
        axes[1].set_xlabel("rayon effecteur source r (m)")
        axes[1].set_ylabel("| ‖v_pred‖ − ‖v_src‖ | (m/s)")
        # médianes par tranche de rayon
        bins = np.linspace(0, 3, 13)
        for ax, x, y in [(axes[0], r[moving], angle), (axes[1], r, norm_err)]:
            med = [np.median(y[(x >= a) & (x < b)]) if ((x >= a) & (x < b)).any() else np.nan
                   for a, b in zip(bins[:-1], bins[1:])]
            ax.plot(0.5 * (bins[:-1] + bins[1:]), med, "o-", color="#222222",
                    linewidth=1.5, label="médiane par tranche")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"Action mapper [{args.mapper}] {args.direction} — "
                     f"erreur d'angle médiane {np.median(angle):.2f}°, "
                     f"erreur de norme médiane {np.median(norm_err):.4f} m/s")
        plt.tight_layout()
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"action_mapper_{args.mapper}_{args.direction}_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Résumé -> {path}")
        return

    # ── Vue par état / animation ─────────────────────────────────────
    def render_frame(ax_src, ax_tgt, i):
        dq_pred = fn(vel_src[i:i + 1], q_src[i:i + 1], q_tgt[i:i + 1])[0]
        v_pred = EFF_VEL_NP[nq_tgt](theta_tgt[i], dq_pred)
        vs = v_src[i]
        ang = angle_between_deg(v_pred[None], vs[None])[0] \
            if np.linalg.norm(vs) > 1e-6 else 0.0

        eff_s = draw_arm(ax_src, nq_src, theta_src[i],
                         f"Source {nq_src}DoF — idx={i}\n|v_src|={np.linalg.norm(vs):.3f} m/s")
        draw_arrow(ax_src, eff_s, vs, "#FFD166", "v_src")
        ax_src.legend(loc="upper left", fontsize=7)

        eff_t = draw_arm(ax_tgt, nq_tgt, theta_tgt[i],
                         f"Cible {nq_tgt}DoF [{args.mapper}]\n"
                         f"angle(v_pred, v_src)={ang:.2f}°  "
                         f"|v_pred|={np.linalg.norm(v_pred):.3f} m/s")
        draw_arrow(ax_tgt, eff_t, vs, "#BBBBBB", "v_src (réf)")
        draw_arrow(ax_tgt, eff_t, v_pred, "#FFD166", "v_pred")
        ax_tgt.legend(loc="upper left", fontsize=7)

    if args.idx is not None:
        if args.save:
            matplotlib.use("Agg")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        render_frame(ax1, ax2, args.idx)
        fig.suptitle(f"Action mapper [{args.mapper}] {args.direction}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        if args.save:
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out,
                                f"action_mapper_{args.mapper}_{args.direction}_idx{args.idx}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Image -> {path}")
        else:
            plt.show()
        return

    matplotlib.use("QtAgg")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def update(i):
        ax1.cla()
        ax2.cla()
        render_frame(ax1, ax2, i)
        return ax1, ax2

    anim = FuncAnimation(fig, update, frames=len(vel_src), interval=args.delay * 1000.0)
    plt.show()


if __name__ == "__main__":
    main()
