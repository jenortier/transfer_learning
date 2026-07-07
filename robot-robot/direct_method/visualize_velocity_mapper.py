"""Visualise les vitesses d'effecteur reconstruites par le STATE mapper.

NB: malgré son nom historique, ce script visualise bien le state mapper
(état 2DoF -> état 3DoF prédit), pas l'action mapper autonome — pour ce
dernier, voir visualize_action_mapper.py.

Usage :
    cd robot-robot && python3 -m direct_method.visualize_velocity_mapper --idx 100
    (par défaut : animation sur toute la trajectoire)
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from direct_method.mapper_models import StateMapperMLP
from direct_method.kinematics import (
    L_2DOF, L_3DOF, OMEGA_MAX, MAX_REACH,
    fk_2dof_np, fk_3dof_np, eff_vel_2dof_np, eff_vel_3dof_np,
)

R1_NAME = "robot_2dofs"
STATE_DIM_R1 = 6
STATE_DIM_R2 = 8
HIDDEN_DIM = 256
PI = np.pi


def load_txt(path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def normalize_2dof(raw):
    """[θ, dθ, ...] bruts -> état 6D normalisé, effecteur recalculé par FK.

    Les colonnes 5-6 des fichiers sont en encodage fenêtre (eff+5)/10 : on
    recalcule l'effecteur depuis les angles, comme dans dataset.py.
    """
    state = np.zeros((len(raw), STATE_DIM_R1), dtype=np.float32)
    state[:, 0:2] = raw[:, 0:2] / PI
    state[:, 2:4] = raw[:, 2:4] / OMEGA_MAX
    state[:, 4:6] = fk_2dof_np(raw[:, 0:2]) / MAX_REACH
    return state


def denormalize_2dof(state):
    out = state.copy()
    out[0:2] *= PI
    out[2:4] *= OMEGA_MAX
    out[4:6] *= MAX_REACH
    return out


def denormalize_3dof(state):
    out = state.copy()
    out[0:3] *= PI
    out[3:6] *= OMEGA_MAX
    out[6:8] *= MAX_REACH
    return out


def joint_positions_2dof(theta):
    j1 = np.array([L_2DOF[0] * np.cos(theta[0]), L_2DOF[0] * np.sin(theta[0])])
    return np.array([0.0, 0.0]), j1, fk_2dof_np(theta[:2])


def joint_positions_3dof(theta):
    j1 = np.array([L_3DOF[0] * np.cos(theta[0]), L_3DOF[0] * np.sin(theta[0])])
    j2 = j1 + np.array([
        L_3DOF[1] * np.cos(theta[0] + theta[1]),
        L_3DOF[1] * np.sin(theta[0] + theta[1]),
    ])
    return np.array([0.0, 0.0]), j1, j2, fk_3dof_np(theta[:3])


def setup_axis(ax, title):
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(theta) * MAX_REACH, np.sin(theta) * MAX_REACH,
            color="#555555", linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)


def draw_arrow(ax, origin, velocity, color, label, coef=1.0):
    speed = float(np.linalg.norm(velocity))
    ax.quiver(origin[0], origin[1],
              coef * velocity[0], coef * velocity[1],
              angles="xy", scale_units="xy", scale=1.0,
              headwidth=4.0, headlength=5.0, color=color,
              label=f"{label} |v|={speed:.3f} m/s")


def draw_2dof(ax, state, title):
    theta = state[:2]
    velocity = eff_vel_2dof_np(theta, state[2:4])
    _, j1, eff = joint_positions_2dof(theta)

    setup_axis(ax, title)
    ax.plot([0.0, j1[0]], [0.0, j1[1]], color="#E05A3A", linewidth=4.0, label="Link 1")
    ax.plot([j1[0], eff[0]], [j1[1], eff[1]], color="#3A8FE0", linewidth=4.0, label="Link 2")
    ax.plot(eff[0], eff[1], "o", color="#FFFFFF", markeredgecolor="#111111",
            markersize=9.0, label="End-effector")
    draw_arrow(ax, eff, velocity, "#FFD166", "Velocity")
    ax.legend(loc="upper left", fontsize=7)


def draw_3dof(ax, state, title):
    theta = state[:3]
    velocity = eff_vel_3dof_np(theta, state[3:6])
    _, j1, j2, eff = joint_positions_3dof(theta)

    setup_axis(ax, title)
    ax.plot([0.0, j1[0]], [0.0, j1[1]], color="#E05A3A", linewidth=4.0, label="Link 1")
    ax.plot([j1[0], j2[0]], [j1[1], j2[1]], color="#3A8FE0", linewidth=4.0, label="Link 2")
    ax.plot([j2[0], eff[0]], [j2[1], eff[1]], color="#9B5DE5", linewidth=4.0, label="Link 3")
    ax.plot(eff[0], eff[1], "o", color="#FFFFFF", markeredgecolor="#111111",
            markersize=9.0, label="End-effector")
    draw_arrow(ax, eff, velocity, "#FFD166", "Velocity")
    ax.legend(loc="upper left", fontsize=7)


def state_title_2dof(state, idx):
    velocity = eff_vel_2dof_np(state[:2], state[2:4])
    return (
        f"2-DoF env — état source idx={idx}\n"
        f"θ1={np.degrees(state[0]):.1f}°  θ2={np.degrees(state[1]):.1f}°  "
        f"ω1={state[2]:.2f} rad/s  ω2={state[3]:.2f} rad/s\n"
        f"v_eff=({velocity[0]:.3f}, {velocity[1]:.3f}) m/s  |v|={np.linalg.norm(velocity):.3f} m/s"
    )


def state_title_3dof(state, idx):
    velocity = eff_vel_3dof_np(state[:3], state[3:6])
    return (
        f"3-DoF env — état reconstruit par mapper idx={idx}\n"
        f"θ1={np.degrees(state[0]):.1f}°  θ2={np.degrees(state[1]):.1f}°  θ3={np.degrees(state[2]):.1f}°  "
        f"ω1={state[3]:.2f} rad/s  ω2={state[4]:.2f} rad/s  ω3={state[5]:.2f} rad/s\n"
        f"v_eff=({velocity[0]:.3f}, {velocity[1]:.3f}) m/s  |v|={np.linalg.norm(velocity):.3f} m/s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualiser les vitesses reconstruites par le state mapper.")
    parser.add_argument("--idx", type=int, default=None,
                        help="Index de l'état à afficher. Si absent, animation complète.")
    parser.add_argument("--run-id", default="run_03_kin",
                        help="ID du run d'entraînement, ex: run_03_kin ou run_01.")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Délai entre deux images pour l'animation, en secondes.")
    parser.add_argument("--save", action="store_true", help="Sauvegarder l'image PNG générée.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device PyTorch.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    traj_path = base_dir / "trajectories" / f"{R1_NAME}.txt"
    model_path = base_dir / "runs" / args.run_id / "models" / "state_mapper_r1_to_r2.pt"
    plot_dir = base_dir / "runs" / args.run_id / "plots"

    if not traj_path.exists():
        raise FileNotFoundError(f"Fichier trajectoire introuvable: {traj_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle state mapper introuvable: {model_path}\n"
            "Lance d'abord: python3 -m direct_method.train_states_mapper")

    raw_2dof = load_txt(traj_path)
    if raw_2dof.shape[1] < STATE_DIM_R1:
        raise ValueError(f"Le fichier {traj_path} doit contenir au moins {STATE_DIM_R1} colonnes.")

    state_2dof_norm = normalize_2dof(raw_2dof)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    mapper = StateMapperMLP(STATE_DIM_R1, STATE_DIM_R2, HIDDEN_DIM).to(device)
    mapper.load_state_dict(torch.load(model_path, map_location=device))
    mapper.eval()

    with torch.no_grad():
        x = torch.from_numpy(state_2dof_norm).float().to(device)
        pred_3dof_norm = mapper(x).cpu().numpy()

    state_2dof = np.array([denormalize_2dof(s) for s in state_2dof_norm], dtype=np.float32)
    pred_3dof = np.array([denormalize_3dof(s) for s in pred_3dof_norm], dtype=np.float32)

    if args.idx is None:
        matplotlib.use("QtAgg")
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
        plt.tight_layout()

        def update(i):
            ax2.cla()
            ax3.cla()
            draw_2dof(ax2, state_2dof[i], state_title_2dof(state_2dof[i], i))
            draw_3dof(ax3, pred_3dof[i], state_title_3dof(pred_3dof[i], i))
            return ax2, ax3

        anim = FuncAnimation(fig, update, frames=len(raw_2dof), interval=args.delay * 1000.0)
        plt.show()
        return

    idx = args.idx
    if idx < 0 or idx >= len(raw_2dof):
        raise ValueError(f"Index hors limites: {idx}. Le fichier contient {len(raw_2dof)} états.")

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
    draw_2dof(ax2, state_2dof[idx], state_title_2dof(state_2dof[idx], idx))
    draw_3dof(ax3, pred_3dof[idx], state_title_3dof(pred_3dof[idx], idx))
    fig.suptitle("State mapper direct_method: vitesses 2-DoF → 3-DoF",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if args.save:
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / "state_mapper_velocity_reconstruction.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Image sauvegardée → {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
