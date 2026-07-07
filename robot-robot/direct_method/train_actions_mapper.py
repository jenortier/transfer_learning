"""Entraînement des action mappers conditionnés par l'état (2DoF <-> 3DoF).

Reformulation par rapport à l'ancienne version (MSE articulaire pure sur
[dtheta_src] -> [dtheta_tgt]) :

  1. Le modèle reçoit les angles source ET cible en plus de l'action source :
     ce sont eux qui déterminent les Jacobiens, donc l'effet cartésien de
     l'action. Sans eux, le mapping est sous-déterminé par construction.

  2. La loss principale est CARTÉSIENNE : on pénalise l'écart entre la vitesse
     d'effecteur produite par l'action prédite sur le robot cible et celle du
     robot source :
         v_src  = J_src(theta_src) @ dtheta_src
         v_pred = J_tgt(theta_tgt) @ dtheta_tgt_pred
     La MSE articulaire vers les démonstrations n'est plus qu'un régularisateur
     faible (le dataset 3DoF n'est qu'UNE solution parmi l'infinité de postures
     redondantes qui donnent la même vitesse d'effecteur).

Usage :
    cd robot-robot && python3 -m direct_method.train_actions_mapper
"""

import os
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from direct_method.dataset import TrajectoriesTrainingActionDataset, block_split
from direct_method.mapper_models import ActionMapperConditionedMLP
from direct_method.kinematics import eff_vel_2dof, eff_vel_3dof, OMEGA_MAX

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
R1_NAME = "robot_2dofs"
R2_NAME = "robot_3dofs"
NQ_R1 = 2
NQ_R2 = 3
HIDDEN_DIM = 256

PATH_R1_TRAJ = f"direct_method/trajectories/{R1_NAME}.txt"
PATH_R2_TRAJ = f"direct_method/trajectories/{R2_NAME}.txt"

NUM_EPOCHS = int(os.environ.get("DM_EPOCHS", 150))   # DM_EPOCHS=5 pour un smoke test
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
SEED = 0
PATIENCE = 15          # early stopping sur la loss cartésienne de validation
RUN_ID = os.environ.get("DM_RUN_ID", "run_04_cond")

# Poids de la loss (la partie cartésienne domine volontairement)
W_V = 1.0        # Huber sur le vecteur vitesse effecteur
W_ANGLE = 0.5    # 1 - cos(v_pred, v_src)
W_SPEED = 0.25   # | ||v_pred|| - ||v_src|| |
W_JOINT = 0.1    # MSE articulaire vers les démonstrations (régularisateur)
W_REG = 1e-4     # ||dtheta_pred||^2 (préférence norme minimale)

SPEED_EPS = 0.05  # m/s : en dessous, la direction n'a pas de sens -> pas de loss d'angle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output dirs
RUN_DIR = os.path.join("direct_method/runs", RUN_ID)
MODEL_DIR = os.path.join(RUN_DIR, "models")
LOG_DIR = os.path.join(RUN_DIR, "logs")
PLOT_DIR = os.path.join(RUN_DIR, "plots")

EFF_VEL = {2: eff_vel_2dof, 3: eff_vel_3dof}


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
def compute_losses(model, vel_src, q_src, vel_tgt, q_tgt, nq_src, nq_tgt):
    """Retourne (loss_totale, composantes) pour un batch, une direction.

    Toutes les grandeurs cartésiennes sont calculées en unités réelles
    (rad, rad/s, m/s) via la cinématique différentiable.
    """
    pred_norm = model(vel_src, q_src, q_tgt)                     # dtheta_tgt / omega_max

    theta_src = q_src * torch.pi
    theta_tgt = q_tgt * torch.pi
    v_src = EFF_VEL[nq_src](theta_src, vel_src * OMEGA_MAX)
    v_pred = EFF_VEL[nq_tgt](theta_tgt, pred_norm * OMEGA_MAX)

    loss_v = F.huber_loss(v_pred, v_src)

    speed_src = v_src.norm(dim=-1)
    speed_pred = v_pred.norm(dim=-1)
    loss_speed = (speed_pred - speed_src).abs().mean()

    valid = speed_src > SPEED_EPS
    if valid.any():
        cos = F.cosine_similarity(v_pred[valid], v_src[valid], dim=-1, eps=1e-8)
        loss_angle = (1.0 - cos).mean()
    else:
        loss_angle = v_pred.new_zeros(())

    loss_joint = F.mse_loss(pred_norm, vel_tgt)
    loss_reg = pred_norm.pow(2).mean()

    cart = W_V * loss_v + W_ANGLE * loss_angle + W_SPEED * loss_speed
    total = cart + W_JOINT * loss_joint + W_REG * loss_reg

    comps = {
        "total": total.item(), "cart": cart.item(),
        "v": loss_v.item(), "angle": loss_angle.item(),
        "speed": loss_speed.item(), "joint": loss_joint.item(),
    }
    return total, cart, comps


def run_epoch(model, loader, direction, optimizer=None):
    """Une époque train (si optimizer) ou eval. Retourne les composantes moyennes."""
    training = optimizer is not None
    model.train(training)
    sums, n = {}, 0
    with torch.set_grad_enabled(training):
        for vel_r1, q_r1, vel_r2, q_r2 in loader:
            vel_r1, q_r1 = vel_r1.to(DEVICE), q_r1.to(DEVICE)
            vel_r2, q_r2 = vel_r2.to(DEVICE), q_r2.to(DEVICE)
            if direction == "2to3":
                args = (vel_r1, q_r1, vel_r2, q_r2, NQ_R1, NQ_R2)
            else:
                args = (vel_r2, q_r2, vel_r1, q_r1, NQ_R2, NQ_R1)
            total, _, comps = compute_losses(model, *args)
            if training:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
            for k, v in comps.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
    return {k: v / n for k, v in sums.items()}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for d in (MODEL_DIR, LOG_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)

    full_dataset = TrajectoriesTrainingActionDataset(
        PATH_R1_TRAJ, PATH_R2_TRAJ, nq_r1=NQ_R1, nq_r2=NQ_R2)
    data_train, data_val, data_test = block_split(full_dataset, seed=SEED)
    print(f"Dataset: {len(full_dataset)} lignes -> "
          f"train {len(data_train)} / val {len(data_val)} / test {len(data_test)} (split par blocs)")

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    directions = {
        "2to3": ActionMapperConditionedMLP(NQ_R1, NQ_R2, HIDDEN_DIM).to(DEVICE),
        "3to2": ActionMapperConditionedMLP(NQ_R2, NQ_R1, HIDDEN_DIM).to(DEVICE),
    }
    optims = {k: torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
              for k, m in directions.items()}
    scheds = {k: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=5)
              for k, o in optims.items()}
    best = {k: {"cart": float("inf"), "state": None, "epoch": -1} for k in directions}
    bad_epochs = {k: 0 for k in directions}

    LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
    comp_keys = ["total", "cart", "v", "angle", "speed", "joint"]
    fields = ["epoch"] + [f"{p}_{d}_{k}" for d in directions for p in ("train", "val")
                          for k in comp_keys]
    history = {d: {"train": [], "val": []} for d in directions}

    with open(LOG_PATH, "w", newline="") as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=fields)
        log_writer.writeheader()

        pbar = tqdm(range(NUM_EPOCHS), desc="Training conditioned action mappers")
        for epoch in pbar:
            row = {"epoch": epoch + 1}
            postfix = {}
            for d, model in directions.items():
                if bad_epochs[d] > PATIENCE:
                    continue  # direction déjà convergée (early stop)
                tr = run_epoch(model, train_loader, d, optims[d])
                va = run_epoch(model, val_loader, d)
                history[d]["train"].append(tr)
                history[d]["val"].append(va)
                scheds[d].step(va["cart"])

                if va["cart"] < best[d]["cart"]:
                    best[d] = {"cart": va["cart"], "epoch": epoch + 1,
                               "state": {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}}
                    bad_epochs[d] = 0
                else:
                    bad_epochs[d] += 1

                for k in comp_keys:
                    row[f"train_{d}_{k}"] = tr[k]
                    row[f"val_{d}_{k}"] = va[k]
                postfix[d] = f"val_cart={va['cart']:.5f}"

            log_writer.writerow(row)
            log_file.flush()
            pbar.set_postfix(postfix)

            if all(bad_epochs[d] > PATIENCE for d in directions):
                print(f"\nEarly stopping à l'époque {epoch + 1}.")
                break

    # ── Sauvegarde des meilleurs modèles + config ─────────────────
    name_map = {"2to3": "action_mapper_2to3", "3to2": "action_mapper_3to2"}
    for d, model in directions.items():
        state = best[d]["state"] or model.state_dict()
        path = os.path.join(MODEL_DIR, f"{name_map[d]}.pt")
        torch.save(state, path)
        print(f"[{d}] meilleur modèle (époque {best[d]['epoch']}, "
              f"val cart {best[d]['cart']:.6f}) -> {path}")

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump({
            "model": "ActionMapperConditionedMLP",
            "hidden": HIDDEN_DIM, "nq_r1": NQ_R1, "nq_r2": NQ_R2,
            "weights": {"w_v": W_V, "w_angle": W_ANGLE, "w_speed": W_SPEED,
                        "w_joint": W_JOINT, "w_reg": W_REG},
            "speed_eps": SPEED_EPS, "seed": SEED, "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE, "epochs_max": NUM_EPOCHS,
            "best": {d: {"epoch": best[d]["epoch"], "val_cart": best[d]["cart"]}
                     for d in directions},
        }, f, indent=2)

    # ── Courbes ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    for ax, d in zip(axes, directions):
        ep = range(1, len(history[d]["train"]) + 1)
        ax.plot(ep, [h["cart"] for h in history[d]["train"]], label="train (cartésien)",
                color="steelblue", linewidth=1.8)
        ax.plot(ep, [h["cart"] for h in history[d]["val"]], label="val (cartésien)",
                color="darkorange", linewidth=1.8, linestyle="--")
        ax.plot(ep, [h["joint"] for h in history[d]["val"]], label="val (MSE articulaire)",
                color="seagreen", linewidth=1.2, linestyle=":")
        ax.set_title(f"Action mapper {d}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "loss_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Courbes -> {plot_path}")
    print(f"\nDone. Sorties dans {RUN_DIR}")
