"""Évaluation offline des mappers d'état et d'action, en espace effecteur.

Compare, sur le split de validation (mêmes blocs/seed que les trainers) :
  - action mapper `legacy`      : ActionMapperMLP non conditionné (run_02) ;
  - action mapper `jacobian`    : baseline analytique pseudo-inverse amortie ;
  - action mapper `conditioned` : ActionMapperConditionedMLP (run_04_cond) ;
  - state mapper(s)             : run_01 (legacy) et/ou run_03_kin.

Métriques (unités réelles) :
  - MSE articulaire normalisée (vs la solution du dataset — indicative seulement,
    la redondance 3DoF admet d'autres solutions valides) ;
  - ‖v_pred − v_src‖ : erreur vectorielle de vitesse effecteur (m/s) ;
  - | ‖v_pred‖ − ‖v_src‖ | : erreur de norme (m/s) ;
  - angle(v_pred, v_src) en degrés (si ‖v_src‖ > 0.05 m/s) ;
  - state mapper : ‖FK(θ_pred) − eff_src‖ en mètres.

Ventilation par zone du rayon effecteur source r = ‖eff_src‖ :
  centre (r < 1.0, bras plié) / milieu / bord (r > 2.5, bras presque tendu).
C'est exactement la ventilation du symptôme observé.

Inclut aussi un diagnostic d'alignement des fichiers de trajectoires
(‖eff_2dof − eff_3dof‖ et ‖v_2 − v_3‖ ligne à ligne).

Usage :
    cd robot-robot && python3 -m direct_method.eval_mappers
    (options : --skip-missing est implicite, les mappers absents sont ignorés)
"""

import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from direct_method.dataset import (
    TrajectoriesTrainingDataset, TrajectoriesTrainingActionDataset, block_split)
from direct_method.mapper_models import (
    StateMapperMLP, ActionMapperMLP, ActionMapperConditionedMLP)
from direct_method.action_mapper_baseline import JacobianActionMapper
from direct_method.kinematics import (
    fk_2dof_np, fk_3dof_np, eff_vel_2dof_np, eff_vel_3dof_np,
    OMEGA_MAX, MAX_REACH)

NQ_R1, NQ_R2 = 2, 3
HIDDEN = 256
SEED = 0
SPEED_EPS = 0.05

FK_NP = {2: fk_2dof_np, 3: fk_3dof_np}
EFF_VEL_NP = {2: eff_vel_2dof_np, 3: eff_vel_3dof_np}

ZONES = [("centre  (r<1.0)", 0.0, 1.0),
         ("milieu  (1.0-2.5)", 1.0, 2.5),
         ("bord    (r>2.5)", 2.5, np.inf)]


def zone_masks(r):
    return [(name, (r >= lo) & (r < hi)) for name, lo, hi in ZONES]


def stats(x, mask=None):
    x = x[mask] if mask is not None else x
    if len(x) == 0:
        return dict(mean=np.nan, median=np.nan, p90=np.nan, n=0)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), n=int(len(x)))


def action_metrics(dq_pred, theta_src, dq_src_real, theta_tgt, dq_tgt_data_real,
                   nq_src, nq_tgt):
    """Toutes les métriques action, en unités réelles."""
    v_src = EFF_VEL_NP[nq_src](theta_src, dq_src_real)
    v_pred = EFF_VEL_NP[nq_tgt](theta_tgt, dq_pred)

    vel_err = np.linalg.norm(v_pred - v_src, axis=-1)
    speed_src = np.linalg.norm(v_src, axis=-1)
    speed_pred = np.linalg.norm(v_pred, axis=-1)
    speed_err = np.abs(speed_pred - speed_src)

    moving = speed_src > SPEED_EPS
    cos = np.full(len(v_src), np.nan)
    denom = speed_src[moving] * np.maximum(speed_pred[moving], 1e-9)
    cos[moving] = np.clip((v_pred[moving] * v_src[moving]).sum(-1) / denom, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))

    joint_mse = np.mean((dq_pred - dq_tgt_data_real) ** 2 / OMEGA_MAX ** 2, axis=-1)
    return vel_err, speed_err, angle_deg, joint_mse, moving


def print_zone_table(title, r, metrics):
    """metrics: dict name -> (values, mask_extra|None)."""
    print(f"\n  {title}")
    header = f"    {'zone':<20}" + "".join(f"{k:>22}" for k in metrics)
    print(header)
    for zname, zmask in zone_masks(r) + [("TOUTES ZONES", np.ones_like(r, bool))]:
        cells = []
        for k, (vals, extra) in metrics.items():
            m = zmask if extra is None else (zmask & extra)
            s = stats(vals, m)
            cells.append(f"{s['median']:>9.4f}/{s['p90']:<8.4f}" if s["n"] else f"{'—':>18}")
        print(f"    {zname:<20}" + "".join(f"{c:>22}" for c in cells))
    print("    (format: médiane/p90)")


# ══════════════════════════════════════════════════════════════════
# Évaluation des action mappers
# ══════════════════════════════════════════════════════════════════

def eval_action_mappers(args, csv_rows):
    ds = TrajectoriesTrainingActionDataset(
        args.traj_r1, args.traj_r2, nq_r1=NQ_R1, nq_r2=NQ_R2)
    _, val, _ = block_split(ds, seed=SEED)
    idx = np.asarray(val.indices)
    print(f"\n{'=' * 78}\nACTION MAPPERS — {len(idx)} échantillons de validation\n{'=' * 78}")

    data = {
        "2to3": dict(nq_src=2, nq_tgt=3,
                     vel_src=ds.vel_r1[idx], q_src=ds.q_r1[idx],
                     vel_tgt=ds.vel_r2[idx], q_tgt=ds.q_r2[idx]),
        "3to2": dict(nq_src=3, nq_tgt=2,
                     vel_src=ds.vel_r2[idx], q_src=ds.q_r2[idx],
                     vel_tgt=ds.vel_r1[idx], q_tgt=ds.q_r1[idx]),
    }

    # Mappers disponibles -> fonction (vel_src_n, q_src_n, q_tgt_n) -> dq_pred réel
    def make_legacy(direction):
        name = {"2to3": "action_mapper_r1_to_r2", "3to2": "action_mapper_r2_to_r1"}[direction]
        path = os.path.join("direct_method/runs", args.run_action_legacy, "models", f"{name}.pt")
        if not os.path.exists(path):
            return None
        dims = (2, 3) if direction == "2to3" else (3, 2)
        m = ActionMapperMLP(*dims, HIDDEN)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def f(vel_src, q_src, q_tgt):
            with torch.no_grad():
                out = m(torch.from_numpy(vel_src)).numpy()
            return out * OMEGA_MAX
        return f

    def make_conditioned(direction):
        name = {"2to3": "action_mapper_2to3", "3to2": "action_mapper_3to2"}[direction]
        path = os.path.join("direct_method/runs", args.run_action_cond, "models", f"{name}.pt")
        if not os.path.exists(path):
            return None
        dims = (2, 3) if direction == "2to3" else (3, 2)
        m = ActionMapperConditionedMLP(*dims, HIDDEN)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def f(vel_src, q_src, q_tgt):
            with torch.no_grad():
                out = m(torch.from_numpy(vel_src), torch.from_numpy(q_src),
                        torch.from_numpy(q_tgt)).numpy()
            return out * OMEGA_MAX
        return f

    def make_jacobian(direction):
        base = JacobianActionMapper(direction, lam=args.lam)

        def f(vel_src, q_src, q_tgt):
            return base(vel_src * OMEGA_MAX, q_src * np.pi, q_tgt * np.pi)
        return f

    for direction, d in data.items():
        theta_src = d["q_src"] * np.pi
        theta_tgt = d["q_tgt"] * np.pi
        dq_src = d["vel_src"] * OMEGA_MAX
        dq_tgt_data = d["vel_tgt"] * OMEGA_MAX
        r = np.linalg.norm(FK_NP[d["nq_src"]](theta_src), axis=-1)

        print(f"\n─── Direction {direction} "
              f"(source {d['nq_src']}DoF -> cible {d['nq_tgt']}DoF) ───")

        mappers = {"legacy": make_legacy(direction),
                   "jacobian": make_jacobian(direction),
                   "conditioned": make_conditioned(direction)}
        for mname, fn in mappers.items():
            if fn is None:
                print(f"\n  [{mname}] modèle absent — ignoré")
                continue
            dq_pred = fn(d["vel_src"], d["q_src"], d["q_tgt"])
            vel_err, speed_err, angle, joint_mse, moving = action_metrics(
                dq_pred, theta_src, dq_src, theta_tgt, dq_tgt_data,
                d["nq_src"], d["nq_tgt"])

            print_zone_table(
                f"[{mname}]",
                r,
                {"‖v_pred−v_src‖ (m/s)": (vel_err, None),
                 "err norme (m/s)": (speed_err, None),
                 "angle (deg)": (angle, moving),
                 "MSE artic.": (joint_mse, None)})

            for zname, zmask in zone_masks(r) + [("all", np.ones_like(r, bool))]:
                csv_rows.append({
                    "kind": "action", "mapper": mname, "direction": direction,
                    "zone": zname.split()[0],
                    "vel_err_median": stats(vel_err, zmask)["median"],
                    "vel_err_p90": stats(vel_err, zmask)["p90"],
                    "speed_err_median": stats(speed_err, zmask)["median"],
                    "angle_deg_median": stats(angle, zmask & moving)["median"],
                    "joint_mse_median": stats(joint_mse, zmask)["median"],
                    "n": int(zmask.sum()),
                })


# ══════════════════════════════════════════════════════════════════
# Évaluation des state mappers
# ══════════════════════════════════════════════════════════════════

def eval_state_mappers(args, csv_rows):
    ds = TrajectoriesTrainingDataset(
        args.traj_r1, args.traj_r2, nq_r1=NQ_R1, nq_r2=NQ_R2)
    _, val, _ = block_split(ds, seed=SEED)
    idx = np.asarray(val.indices)
    s1 = ds.trajectories_r1[idx]
    s2 = ds.trajectories_r2[idx]
    print(f"\n{'=' * 78}\nSTATE MAPPERS — {len(idx)} échantillons de validation\n{'=' * 78}")

    for run_id in args.run_states:
        mdir = os.path.join("direct_method/runs", run_id, "models")
        p12 = os.path.join(mdir, "state_mapper_r1_to_r2.pt")
        p21 = os.path.join(mdir, "state_mapper_r2_to_r1.pt")
        if not (os.path.exists(p12) and os.path.exists(p21)):
            print(f"\n[{run_id}] modèles absents — ignoré")
            continue

        m12 = StateMapperMLP(6, 8, HIDDEN)
        m12.load_state_dict(torch.load(p12, map_location="cpu"))
        m21 = StateMapperMLP(8, 6, HIDDEN)
        m21.load_state_dict(torch.load(p21, map_location="cpu"))
        m12.eval()
        m21.eval()

        with torch.no_grad():
            pred2 = m12(torch.from_numpy(s1)).numpy()
            pred1 = m21(torch.from_numpy(s2)).numpy()

        for direction, (src, pred, nq_src, nq_tgt, tgt) in {
                "2to3": (s1, pred2, 2, 3, s2),
                "3to2": (s2, pred1, 3, 2, s1)}.items():
            theta_src = src[:, :nq_src] * np.pi
            dq_src = src[:, nq_src:2 * nq_src] * OMEGA_MAX
            eff_src = src[:, 2 * nq_src:] * MAX_REACH
            v_src = EFF_VEL_NP[nq_src](theta_src, dq_src)
            r = np.linalg.norm(eff_src, axis=-1)

            theta_pred = pred[:, :nq_tgt] * np.pi
            dq_pred = pred[:, nq_tgt:2 * nq_tgt] * OMEGA_MAX
            fk_pred = FK_NP[nq_tgt](theta_pred)
            v_pred = EFF_VEL_NP[nq_tgt](theta_pred, dq_pred)

            pos_err = np.linalg.norm(fk_pred - eff_src, axis=-1)
            vel_err = np.linalg.norm(v_pred - v_src, axis=-1)
            state_mse = np.mean((pred - tgt) ** 2, axis=-1)

            print(f"\n─── [{run_id}] state mapper {direction} ───")
            print_zone_table(
                f"[{run_id} / {direction}]", r,
                {"‖FK(θ̂)−eff_src‖ (m)": (pos_err, None),
                 "‖v_pred−v_src‖ (m/s)": (vel_err, None),
                 "MSE état (norm.)": (state_mse, None)})

            for zname, zmask in zone_masks(r) + [("all", np.ones_like(r, bool))]:
                csv_rows.append({
                    "kind": "state", "mapper": run_id, "direction": direction,
                    "zone": zname.split()[0],
                    "pos_err_median": stats(pos_err, zmask)["median"],
                    "pos_err_p90": stats(pos_err, zmask)["p90"],
                    "vel_err_median": stats(vel_err, zmask)["median"],
                    "vel_err_p90": stats(vel_err, zmask)["p90"],
                    "state_mse_median": stats(state_mse, zmask)["median"],
                    "n": int(zmask.sum()),
                })


# ══════════════════════════════════════════════════════════════════
# Diagnostic d'alignement des données
# ══════════════════════════════════════════════════════════════════

def eval_alignment(args, out_dir):
    raw1 = np.loadtxt(args.traj_r1, dtype=np.float64)[:, :6]
    raw2 = np.loadtxt(args.traj_r2, dtype=np.float64)[:, :8]
    n = min(len(raw1), len(raw2))
    raw1, raw2 = raw1[:n], raw2[:n]

    # Positions/vitesses effecteur recalculées par FK/Jacobien depuis les
    # angles (les colonnes eff des fichiers sont en encodage fenêtre
    # (eff+5)/10, pas en mètres — voir dataset.py).
    eff1 = fk_2dof_np(raw1[:, :2])
    eff2 = fk_3dof_np(raw2[:, :3])
    v1 = eff_vel_2dof_np(raw1[:, :2], raw1[:, 2:4])
    v2 = eff_vel_3dof_np(raw2[:, :3], raw2[:, 3:6])

    pos_gap = np.linalg.norm(eff1 - eff2, axis=-1)
    vel_gap = np.linalg.norm(v1 - v2, axis=-1)

    print(f"\n{'=' * 78}\nALIGNEMENT DES TRAJECTOIRES ({n} lignes appariées)\n{'=' * 78}")
    for name, g in [("‖FK2(θ) − FK3(θ)‖ (m)", pos_gap),
                    ("‖v_2dof − v_3dof‖ (m/s)", vel_gap)]:
        print(f"  {name:<28} médiane={np.median(g):.4f}  p90={np.percentile(g, 90):.4f}  "
              f"max={g.max():.4f}")
    print("  -> Si ces écarts sont grands, les cibles articulaires du dataset sont un")
    print("     signal bruité : garder W_JOINT faible et privilégier la loss cartésienne.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, g, title in [(axes[0], pos_gap, "‖eff_2dof − eff_3dof‖ (m)"),
                         (axes[1], vel_gap, "‖v_2 − v_3‖ (m/s)")]:
        ax.hist(g, bins=80, color="steelblue", alpha=0.85)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "alignment_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Histogrammes -> {path}")


# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--traj-r1", default="direct_method/trajectories/robot_2dofs.txt")
    p.add_argument("--traj-r2", default="direct_method/trajectories/robot_3dofs.txt")
    p.add_argument("--run-action-legacy", default="run_02")
    p.add_argument("--run-action-cond", default="run_04_cond")
    p.add_argument("--run-states", nargs="*", default=["run_01", "run_03_kin"],
                   help="runs de state mappers à évaluer")
    p.add_argument("--lam", type=float, default=0.05, help="amortissement du baseline")
    p.add_argument("--out", default="direct_method/runs/eval")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_rows = []

    eval_alignment(args, args.out)
    eval_action_mappers(args, csv_rows)
    eval_state_mappers(args, csv_rows)

    if csv_rows:
        keys = sorted({k for row in csv_rows for k in row})
        csv_path = os.path.join(args.out, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nMétriques CSV -> {csv_path}")


if __name__ == "__main__":
    main()
