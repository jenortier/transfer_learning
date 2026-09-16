"""
replay_transfer_reach_2to3dof.py
Rejoue les épisodes sauvegardés par runs_transfer_reach_2to3dof.py.
Politique 2-DOF transférée, exécutée dans l'environnement 3-DOF.
Avec subplot du reward cumulé.
"""

import argparse
import glob
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Replay transfer reaching 2→3 DOF")
group = parser.add_mutually_exclusive_group()
group.add_argument("--success", action="store_true", default=False)
group.add_argument("--fail",    action="store_true", default=False)
parser.add_argument("--ep",    type=int,   default=None)
parser.add_argument("--delay", type=float, default=0.03)
args = parser.parse_args()

mode     = "fail" if args.fail else "success"
data_dir = f"runs/transfer_reach_2to3dof/{mode}"

# ── Chargement ────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
if not files:
    raise FileNotFoundError(
        f"Aucun fichier dans '{data_dir}'. Lancez runs_transfer_reach_2to3dof.py d'abord."
    )
if args.ep is not None:
    target_name = f"ep_{args.ep:03d}.npz"
    files = [f for f in files if os.path.basename(f) == target_name]
    if not files:
        raise FileNotFoundError(f"Épisode {args.ep:03d} introuvable.")

print(f"Replay [{mode.upper()}] — {len(files)} épisode(s) — transfer reaching 2→3 DOF\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fk3(t1, t2, t3, l1, l2, l3):
    x = l1*np.cos(t1) + l2*np.cos(t1+t2) + l3*np.cos(t1+t2+t3)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2) + l3*np.sin(t1+t2+t3)
    return np.array([x, y])

# ── Replay avec subplot ──────────────────────────────────────────────────────
fig, (ax_robot, ax_reward) = plt.subplots(1, 2, figsize=(10, 6))
plt.ion()

for file_idx, fpath in enumerate(files):
    if not plt.fignum_exists(fig.number):
        print("Fermeture fenêtre")
        break

    ep_name = os.path.basename(fpath)
    d = np.load(fpath, allow_pickle=True)

    target  = d["target"]
    thetas  = d["thetas"]    # (T, 3)
    dists   = d["dists"]
    rewards = d["rewards"]
    l1      = float(d["l1"])
    l2      = float(d["l2"])
    l3      = float(d["l3"])
    epsilon = float(d["epsilon"])
    total_r = float(d["total_reward"])
    success = bool(d["success"])

    label = "✓ Succès" if success else "✗ Échec"
    print(f"[{file_idx+1}/{len(files)}]  {ep_name}  |  {label}  |  "
          f"steps={len(thetas)}  reward={total_r:+.1f}  dist_finale={dists[-1]:.4f} m")

    cum_rewards = np.cumsum(rewards)

    for step_i, (t1, t2, t3) in enumerate(thetas):
        if not plt.fignum_exists(fig.number):
            print("Fermeture fenêtre")
            break

        ax_robot.cla()
        j1  = np.array([l1*np.cos(t1), l1*np.sin(t1)])
        j2  = j1 + np.array([l2*np.cos(t1+t2), l2*np.sin(t1+t2)])
        eff = fk3(t1, t2, t3, l1, l2, l3)
        dist = dists[step_i]

        ax_robot.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4)
        ax_robot.plot([j1[0], j2[0]], [j1[1], j2[1]], 'b-', lw=4)
        ax_robot.plot([j2[0], eff[0]], [j2[1], eff[1]], 'm-', lw=4)

        eff_color = "lime" if dist < epsilon else "red"
        ax_robot.plot(eff[0], eff[1], "o", color=eff_color, markersize=10, label="End-effector")
        ax_robot.plot(target[0], target[1], "go", markersize=12, label="Target")
        ax_robot.add_patch(plt.Circle(target, epsilon,
                                      color="green", fill=False, linestyle="--", lw=1.5))

        past_effs = np.array([fk3(thetas[i,0], thetas[i,1], thetas[i,2], l1, l2, l3)
                               for i in range(step_i + 1)])
        ax_robot.plot(past_effs[:, 0], past_effs[:, 1], "b-", lw=1, alpha=0.3)

        ax_robot.set_xlim(-3.5, 3.5)
        ax_robot.set_ylim(-3.5, 3.5)
        ax_robot.set_aspect("equal")
        ax_robot.set_title(
            f"[Transfer 2→3 DOF]  {ep_name}  [{mode}]  step {step_i+1}/{len(thetas)}\n"
            f"dist={dist:.3f} m"
        )
        ax_robot.legend(loc="upper left", fontsize=8)

        ax_reward.cla()
        ax_reward.plot(np.arange(1, step_i+2), cum_rewards[:step_i+1], 'b-', lw=2)
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Cumulative reward")
        ax_reward.set_title(f"Reward cumulé (final = {total_r:+.1f})")
        ax_reward.grid(True, alpha=0.3)
        ax_reward.set_xlim(0, len(thetas))
        if len(cum_rewards[:step_i+1]) > 0:
            ymin = min(0, np.min(cum_rewards[:step_i+1]) - 1)
            ymax = max(0, np.max(cum_rewards[:step_i+1]) + 1)
            ax_reward.set_ylim(ymin, ymax)

        plt.pause(args.delay)

    if not plt.fignum_exists(fig.number):
        break
    time.sleep(1.0)

plt.ioff()
plt.show()
print("\nReplay terminé.")
