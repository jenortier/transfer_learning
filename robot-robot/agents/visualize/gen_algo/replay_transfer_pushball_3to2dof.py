"""
replay_transfer_pushball_3to2dof.py
Rejoue les épisodes sauvegardés par runs_transfer_pushball_3to2dof.py.
Politique pushball 3-DOF transférée, exécutée dans l'environnement 2-DOF.
Avec subplot du reward cumulé.
"""

import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Replay transfer pushball 3→2 DOF")
group = parser.add_mutually_exclusive_group()
group.add_argument("--success", action="store_true", default=False)
group.add_argument("--fail",    action="store_true", default=False)
parser.add_argument("--ep",    type=int,   default=None)
parser.add_argument("--delay", type=float, default=0.05)
args = parser.parse_args()

mode     = "fail" if args.fail else "success"
data_dir = f"runs/transfer_pushball_3to2dof/{mode}"

# ── Chargement ────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(data_dir, "ep_*.npz")))
if not files:
    raise FileNotFoundError(
        f"Aucun fichier dans '{data_dir}'. Lancez runs_transfer_pushball_3to2dof.py d'abord."
    )
if args.ep is not None:
    target_name = f"ep_{args.ep:03d}.npz"
    files = [f for f in files if os.path.basename(f) == target_name]
    if not files:
        raise FileNotFoundError(f"Épisode {args.ep:03d} introuvable.")

print(f"Replay [{mode.upper()}] — {len(files)} épisode(s) — transfer pushball 3→2 DOF\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fk2(t1, t2, l1, l2):
    x = l1*np.cos(t1) + l2*np.cos(t1+t2)
    y = l1*np.sin(t1) + l2*np.sin(t1+t2)
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

    target         = d["target"]
    thetas         = d["thetas"]          # (T, 2)
    balls          = d["balls"]
    rewards        = d["rewards"]
    dists_ball_tgt = d["dists_ball_tgt"]
    l1             = float(d["l1"])
    l2             = float(d["l2"])
    epsilon        = float(d["epsilon"])
    eff_radius     = float(d["eff_radius"])
    ball_radius    = float(d["ball_radius"])
    max_reach      = float(d["max_reach"])
    total_r        = float(d["total_reward"])
    success        = bool(d["success"])

    label = "✓ Succès" if success else "✗ Échec"
    print(f"[{file_idx+1}/{len(files)}]  {ep_name}  |  {label}  |  "
          f"steps={len(thetas)}  reward={total_r:+.1f}  "
          f"dist_finale={dists_ball_tgt[-1]:.4f} m")

    cum_rewards = np.cumsum(rewards)

    for step_i, (t1, t2) in enumerate(thetas):
        if not plt.fignum_exists(fig.number):
            print("Fermeture fenêtre")
            break

        ax_robot.cla()
        j1  = np.array([l1*np.cos(t1), l1*np.sin(t1)])
        eff = fk2(t1, t2, l1, l2)
        ball          = balls[step_i]
        dist_ball_tgt = dists_ball_tgt[step_i]
        ball_out      = float(np.linalg.norm(ball)) > max_reach * 1.05

        ax_robot.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4, label='Link 1')
        ax_robot.plot([j1[0], eff[0]], [j1[1], eff[1]], 'b-', lw=4, label='Link 2')
        ax_robot.add_patch(plt.Circle(eff, eff_radius * 2, color='red', alpha=0.6))

        ball_color = 'red' if ball_out else 'dodgerblue'
        ax_robot.add_patch(plt.Circle(ball, ball_radius * 1.5, color=ball_color, alpha=0.6))

        ax_robot.plot(target[0], target[1], 'o', markersize=18, label='Target', color='orange')
        ax_robot.add_patch(plt.Circle(target, epsilon,
                                      color='green', fill=False, linestyle='--', lw=1.5))

        vec = target - ball
        if np.linalg.norm(vec) > 1e-3:
            v = vec / np.linalg.norm(vec) * 0.3
            ax_robot.arrow(ball[0], ball[1], v[0], v[1],
                           head_width=0.08, head_length=0.04,
                           fc='green', ec='green', alpha=0.4)

        ax_robot.plot(balls[:step_i+1, 0], balls[:step_i+1, 1],
                      '--', color='dodgerblue', lw=1, alpha=0.4, label='Traj. balle')

        ax_robot.set_xlim(-3.5, 3.5)
        ax_robot.set_ylim(-3.5, 3.5)
        ax_robot.set_aspect("equal")
        ax_robot.set_title(
            f"[Transfer 3→2 DOF]  {ep_name}  [{mode}]  step {step_i+1}/{len(thetas)}\n"
            f"d(ball,tgt)={dist_ball_tgt:.3f} m"
            + ("  ⚠️ OUT" if ball_out else "")
        )
        ax_robot.legend(loc="upper right", fontsize=8)

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
