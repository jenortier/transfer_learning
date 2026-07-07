"""Entraînement des state mappers 2DoF <-> 3DoF avec losses cinématiques.

Améliorations par rapport à l'ancienne version (MSE pure sur le vecteur état) :

  1. Loss FK : la position d'effecteur RECALCULÉE depuis les angles prédits
     (FK(theta_pred)) doit coïncider avec la position d'effecteur source. La MSE
     seule autorise des angles légèrement faux dont l'effet cartésien est grand.

  2. Loss vitesse : J(theta_pred) @ dtheta_pred doit reproduire la vitesse
     cartésienne d'effecteur du robot source. C'est la grandeur que le transfert
     doit réellement conserver.

  3. Loss cycle : map_back(map(s)) ≈ s, les deux directions étant entraînées
     conjointement (un seul optimiseur sur les deux réseaux).

  4. Split par blocs contigus (pas de fuite train/val entre timesteps voisins),
     scheduler LR et early stopping.

Le format d'entrée/sortie des modèles reste l'état 6D/8D normalisé : compatible
avec les scripts de transfert et de visualisation existants.

Usage :
    cd robot-robot && python3 -m direct_method.train_states_mapper
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

from direct_method.dataset import TrajectoriesTrainingDataset, block_split
from direct_method.mapper_models import StateMapperMLP
from direct_method.kinematics import (
    fk_2dof, fk_3dof, eff_vel_2dof, eff_vel_3dof,
    OMEGA_MAX, MAX_REACH, split_state,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
R1_NAME = "robot_2dofs"
R2_NAME = "robot_3dofs"
NQ_R1 = 2
NQ_R2 = 3
STATE_DIM_R1 = 2 * NQ_R1 + 2   # 6  [θ1, θ2, dθ1, dθ2, x, y]
STATE_DIM_R2 = 2 * NQ_R2 + 2   # 8  [θ1, θ2, θ3, dθ1, dθ2, dθ3, x, y]
HIDDEN_DIM = 256

PATH_R1_TRAJ = f"direct_method/trajectories/{R1_NAME}.txt"
PATH_R2_TRAJ = f"direct_method/trajectories/{R2_NAME}.txt"

NUM_EPOCHS = int(os.environ.get("DM_EPOCHS", 150))   # DM_EPOCHS=5 pour un smoke test
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
SEED = 0
PATIENCE = 15
RUN_ID = os.environ.get("DM_RUN_ID", "run_03_kin")

# Poids des losses cinématiques (la MSE état garde un poids implicite de 1.0)
W_FK = 0.5      # ||FK(θ_pred) - eff_src||²        (mètres²)
W_VEL = 0.25    # ||J(θ_pred)@dθ_pred - v_src||²   ((m/s)²)
W_CYCLE = 0.1   # ||map_back(map(s)) - s||²        (espace normalisé)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUN_DIR = os.path.join("direct_method/runs", RUN_ID)
MODEL_DIR = os.path.join(RUN_DIR, "models")
LOG_DIR = os.path.join(RUN_DIR, "logs")
PLOT_DIR = os.path.join(RUN_DIR, "plots")

FK = {2: fk_2dof, 3: fk_3dof}
EFF_VEL = {2: eff_vel_2dof, 3: eff_vel_3dof}


def source_cartesian(state_norm, nq):
    """Position (m) et vitesse (m/s) d'effecteur d'un état normalisé (vérité terrain)."""
    q, dq, eff = split_state(state_norm, nq)
    v = EFF_VEL[nq](q * torch.pi, dq * OMEGA_MAX)
    return eff * MAX_REACH, v


def kinematic_losses(state_pred_norm, nq_pred, eff_ref, v_ref):
    """Losses FK et vitesse de l'état prédit contre les références cartésiennes source."""
    q, dq, _ = split_state(state_pred_norm, nq_pred)
    theta = q * torch.pi
    loss_fk = F.mse_loss(FK[nq_pred](theta), eff_ref)
    loss_vel = F.mse_loss(EFF_VEL[nq_pred](theta, dq * OMEGA_MAX), v_ref)
    return loss_fk, loss_vel


def compute_losses(model_12, model_21, s1, s2):
    """Loss conjointe des deux directions sur un batch. Retourne (total, composantes)."""
    pred2 = model_12(s1)
    pred1 = model_21(s2)

    loss_mse = F.mse_loss(pred2, s2) + F.mse_loss(pred1, s1)

    eff1, v1 = source_cartesian(s1, NQ_R1)
    eff2, v2 = source_cartesian(s2, NQ_R2)
    fk_12, vel_12 = kinematic_losses(pred2, NQ_R2, eff1, v1)
    fk_21, vel_21 = kinematic_losses(pred1, NQ_R1, eff2, v2)
    loss_fk = fk_12 + fk_21
    loss_vel = vel_12 + vel_21

    loss_cycle = F.mse_loss(model_21(pred2), s1) + F.mse_loss(model_12(pred1), s2)

    total = loss_mse + W_FK * loss_fk + W_VEL * loss_vel + W_CYCLE * loss_cycle
    comps = {"total": total.item(), "mse": loss_mse.item(), "fk": loss_fk.item(),
             "vel": loss_vel.item(), "cycle": loss_cycle.item()}
    return total, comps


def run_epoch(model_12, model_21, loader, optimizer=None):
    training = optimizer is not None
    model_12.train(training)
    model_21.train(training)
    sums, n = {}, 0
    with torch.set_grad_enabled(training):
        for s1, s2 in loader:
            s1, s2 = s1.to(DEVICE), s2.to(DEVICE)
            total, comps = compute_losses(model_12, model_21, s1, s2)
            if training:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
            for k, v in comps.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
    return {k: v / n for k, v in sums.items()}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    for d in (MODEL_DIR, LOG_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)

    full_dataset = TrajectoriesTrainingDataset(
        PATH_R1_TRAJ, PATH_R2_TRAJ, nq_r1=NQ_R1, nq_r2=NQ_R2)
    data_train, data_val, data_test = block_split(full_dataset, seed=SEED)
    print(f"Dataset: {len(full_dataset)} lignes -> "
          f"train {len(data_train)} / val {len(data_val)} / test {len(data_test)} (split par blocs)")

    train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model_r1_to_r2 = StateMapperMLP(STATE_DIM_R1, STATE_DIM_R2, HIDDEN_DIM).to(DEVICE)
    model_r2_to_r1 = StateMapperMLP(STATE_DIM_R2, STATE_DIM_R1, HIDDEN_DIM).to(DEVICE)

    # Un seul optimiseur : la loss cycle couple les deux réseaux.
    optimizer = torch.optim.Adam(
        list(model_r1_to_r2.parameters()) + list(model_r2_to_r1.parameters()),
        lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best = {"total": float("inf"), "epoch": -1, "state_12": None, "state_21": None}
    bad_epochs = 0

    comp_keys = ["total", "mse", "fk", "vel", "cycle"]
    fields = ["epoch"] + [f"{p}_{k}" for p in ("train", "val") for k in comp_keys]
    history = {"train": [], "val": []}

    LOG_PATH = os.path.join(LOG_DIR, "training_log.csv")
    with open(LOG_PATH, "w", newline="") as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=fields)
        log_writer.writeheader()

        pbar = tqdm(range(NUM_EPOCHS), desc="Training state mappers (kinematic)")
        for epoch in pbar:
            tr = run_epoch(model_r1_to_r2, model_r2_to_r1, train_loader, optimizer)
            va = run_epoch(model_r1_to_r2, model_r2_to_r1, val_loader)
            history["train"].append(tr)
            history["val"].append(va)
            scheduler.step(va["total"])

            if va["total"] < best["total"]:
                best = {"total": va["total"], "epoch": epoch + 1,
                        "state_12": {k: v.detach().cpu().clone()
                                     for k, v in model_r1_to_r2.state_dict().items()},
                        "state_21": {k: v.detach().cpu().clone()
                                     for k, v in model_r2_to_r1.state_dict().items()}}
                bad_epochs = 0
            else:
                bad_epochs += 1

            row = {"epoch": epoch + 1}
            row.update({f"train_{k}": tr[k] for k in comp_keys})
            row.update({f"val_{k}": va[k] for k in comp_keys})
            log_writer.writerow(row)
            log_file.flush()
            pbar.set_postfix({"val": f"{va['total']:.5f}", "fk": f"{va['fk']:.5f}",
                              "vel": f"{va['vel']:.5f}"})

            if bad_epochs > PATIENCE:
                print(f"\nEarly stopping à l'époque {epoch + 1}.")
                break

    # ── Sauvegarde (mêmes noms de fichiers qu'avant : compat transfert/visu) ──
    for name, state in [("state_mapper_r1_to_r2", best["state_12"]),
                        ("state_mapper_r2_to_r1", best["state_21"])]:
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save(state, path)
        print(f"Meilleur modèle (époque {best['epoch']}, val {best['total']:.6f}) -> {path}")

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump({
            "model": "StateMapperMLP", "hidden": HIDDEN_DIM,
            "state_dim_r1": STATE_DIM_R1, "state_dim_r2": STATE_DIM_R2,
            "weights": {"w_fk": W_FK, "w_vel": W_VEL, "w_cycle": W_CYCLE},
            "seed": SEED, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "epochs_max": NUM_EPOCHS,
            "best": {"epoch": best["epoch"], "val_total": best["total"]},
        }, f, indent=2)

    # ── Courbes ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ep = range(1, len(history["train"]) + 1)
    ax.plot(ep, [h["total"] for h in history["train"]], label="train total", color="steelblue")
    ax.plot(ep, [h["total"] for h in history["val"]], label="val total",
            color="darkorange", linestyle="--")
    for key, color in [("fk", "seagreen"), ("vel", "crimson"), ("cycle", "purple")]:
        ax.plot(ep, [h[key] for h in history["val"]], label=f"val {key}",
                linewidth=1.0, linestyle=":", color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("State mappers avec losses cinématiques")
    ax.legend()
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "loss_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Courbes -> {plot_path}")
    print(f"\nDone. Sorties dans {RUN_DIR}")
