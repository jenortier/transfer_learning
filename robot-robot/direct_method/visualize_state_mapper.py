import argparse
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from direct_method.mapper_models import StateMapperMLP


R1_NAME = "robot_2dofs"
R2_NAME = "robot_3dofs"
STATE_DIM_R1 = 6
STATE_DIM_R2 = 8
HIDDEN_DIM = 256

PI = np.pi
OMEGA_MAX = 2.0
MAX_REACH = 3.0

L1_2, L2_2 = 1.5, 1.5
L1_3, L2_3, L3_3 = 1.0, 1.0, 1.0


def load_txt(path):
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def normalize_2dof(raw):
    state = np.zeros((len(raw), STATE_DIM_R1), dtype=np.float32)
    state[:, 0:2] = raw[:, 0:2] / PI
    state[:, 2:4] = raw[:, 2:4] / OMEGA_MAX
    state[:, 4:6] = raw[:, 4:6] / MAX_REACH
    return state


def denormalize_2dof(state):
    return np.array([
        state[0] * PI,
        state[1] * PI,
        state[2] * OMEGA_MAX,
        state[3] * OMEGA_MAX,
        state[4] * MAX_REACH,
        state[5] * MAX_REACH,
    ], dtype=np.float32)


def denormalize_3dof(state):
    return np.array([
        state[0] * PI,
        state[1] * PI,
        state[2] * PI,
        state[3] * OMEGA_MAX,
        state[4] * OMEGA_MAX,
        state[5] * OMEGA_MAX,
        state[6] * MAX_REACH,
        state[7] * MAX_REACH,
    ], dtype=np.float32)


def fk_2dof(theta):
    t1, t2 = theta[:2]
    x = L1_2 * np.cos(t1) + L2_2 * np.cos(t1 + t2)
    y = L1_2 * np.sin(t1) + L2_2 * np.sin(t1 + t2)
    return np.array([x, y])


def fk_3dof(theta):
    t1, t2, t3 = theta[:3]
    x = L1_3 * np.cos(t1) + L2_3 * np.cos(t1 + t2) + L3_3 * np.cos(t1 + t2 + t3)
    y = L1_3 * np.sin(t1) + L2_3 * np.sin(t1 + t2) + L3_3 * np.sin(t1 + t2 + t3)
    return np.array([x, y])


def joint_positions_2dof(theta):
    j1 = np.array([L1_2 * np.cos(theta[0]), L1_2 * np.sin(theta[0])])
    eff = fk_2dof(theta)
    return np.array([0.0, 0.0]), j1, eff


def joint_positions_3dof(theta):
    j1 = np.array([L1_3 * np.cos(theta[0]), L1_3 * np.sin(theta[0])])
    j2 = j1 + np.array([
        L2_3 * np.cos(theta[0] + theta[1]),
        L2_3 * np.sin(theta[0] + theta[1]),
    ])
    eff = fk_3dof(theta)
    return np.array([0.0, 0.0]), j1, j2, eff


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


def draw_2dof(ax, theta, title):
    setup_axis(ax, title)
    _, j1, eff = joint_positions_2dof(theta)
    ax.plot([0.0, j1[0]], [0.0, j1[1]], color="#E05A3A", linewidth=4.0, label="Link 1")
    ax.plot([j1[0], eff[0]], [j1[1], eff[1]], color="#3A8FE0", linewidth=4.0, label="Link 2")
    ax.plot(eff[0], eff[1], "o", color="#FFFFFF", markeredgecolor="#111111",
            markersize=9.0, label="End-effector")
    ax.legend(loc="upper left", fontsize=8)


def draw_3dof(ax, theta, title):
    setup_axis(ax, title)
    _, j1, j2, eff = joint_positions_3dof(theta)
    ax.plot([0.0, j1[0]], [0.0, j1[1]], color="#E05A3A", linewidth=4.0, label="Link 1")
    ax.plot([j1[0], j2[0]], [j1[1], j2[1]], color="#3A8FE0", linewidth=4.0, label="Link 2")
    ax.plot([j2[0], eff[0]], [j2[1], eff[1]], color="#9B5DE5", linewidth=4.0, label="Link 3")
    ax.plot(eff[0], eff[1], "o", color="#FFFFFF", markeredgecolor="#111111",
            markersize=9.0, label="End-effector")
    ax.legend(loc="upper left", fontsize=8)


def state_title_2dof(theta, idx):
    return (
        f"2-DoF env — état source idx={idx}\n"
        f"θ1={np.degrees(theta[0]):.1f}°  θ2={np.degrees(theta[1]):.1f}°  "
        f"ω1={theta[2]:.2f} rad/s  ω2={theta[3]:.2f} rad/s"
    )


def state_title_3dof(theta, idx):
    return (
        f"3-DoF env — état reconstruit par mapper idx={idx}\n"
        f"θ1={np.degrees(theta[0]):.1f}°  θ2={np.degrees(theta[1]):.1f}°  θ3={np.degrees(theta[2]):.1f}°  "
        f"ω1={theta[3]:.2f} rad/s  ω2={theta[4]:.2f} rad/s  ω3={theta[5]:.2f} rad/s"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualiser l'état 2-DoF et sa reconstruction 3-DoF par le state mapper.")
    parser.add_argument("--idx", type=int, default=None, help="Index de l'état à afficher. Par défaut: milieu de la trajectoire.")
    parser.add_argument("--run-id", default="run_01", help="ID du run d'entraînement, ex: run_01.")
    parser.add_argument("--delay", type=float, default=0.5, help="Délai entre deux images pour l'animation, en secondes.")
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
            "Lance d'abord: python train_states_mapper.py"
        )

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
        idx = len(raw_2dof) // 2
    else:
        idx = args.idx
        if idx < 0 or idx >= len(raw_2dof):
            raise ValueError(f"Index hors limites: {idx}. Le fichier contient {len(raw_2dof)} états.")

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
    plt.tight_layout()

    def update(i):
        ax2.cla()
        ax3.cla()
        draw_2dof(ax2, state_2dof[i], state_title_2dof(state_2dof[i], i))
        draw_3dof(ax3, pred_3dof[i], state_title_3dof(pred_3dof[i], i))
        return ax2, ax3

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=len(raw_2dof), interval=args.delay * 1000.0)
    plt.show()

    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))
    draw_2dof(ax2, state_2dof[idx], state_title_2dof(state_2dof[idx], idx))
    draw_3dof(ax3, pred_3dof[idx], state_title_3dof(pred_3dof[idx], idx))
    fig.suptitle("State mapper direct_method: 2-DoF → 3-DoF", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if args.save:
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / "state_mapper_reconstruction.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Image sauvegardée → {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
