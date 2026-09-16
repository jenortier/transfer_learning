"""Visualisation du transfert push-ball : PPO 3DoF exécutée sur le bras 2DoF.

Affiche côte à côte, dans le même repère :
  - le bras 2DoF réel (celui qui agit dans l'environnement) ;
  - le bras 3DoF *virtuel* reconstruit par le state mapper (ce que "voit" la
    politique PPO) — en pointillés ;
  - la balle, la cible (avec son rayon de succès epsilon) ;
  - la vitesse d'effecteur commandée par la politique (via J3 sur le bras
    virtuel) et celle réellement exécutée par le 2DoF (via J2) : si le
    mapping est bon, les deux flèches se superposent.

Même chaîne et mêmes options que transfer_pushball_3to2dof.py.

Usage :
    # fenêtre interactive
    cd robot-robot && python3 -m direct_method.visualize_transfer_3to2dof --episodes 3

    # GIF par épisode (sans écran) -> direct_method/runs/eval/transfer_gifs/
    MPLBACKEND=Agg python3 -m direct_method.visualize_transfer_3to2dof --episodes 3 --save
"""

import argparse
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_pushball_2dof_rec import PushBallEnv_2dof
from envs.env_pushball_3dof_rec import PushBallEnv_3dof
from direct_method.transfer_pushball_3to2dof import (
    load_state_mapper, load_action_mapper, R1_STATE_DIM)
from direct_method.kinematics import (
    L_2DOF, L_3DOF, OMEGA_MAX, MAX_REACH,
    jacobian_2dof_np, jacobian_3dof_np)

ARROW_SCALE = 0.5   # s : longueur de flèche = vitesse (m/s) * échelle


def joints_2dof(theta):
    j1 = np.array([L_2DOF[0] * np.cos(theta[0]), L_2DOF[0] * np.sin(theta[0])])
    eff = j1 + np.array([L_2DOF[1] * np.cos(theta[0] + theta[1]),
                         L_2DOF[1] * np.sin(theta[0] + theta[1])])
    return np.array([[0.0, 0.0], j1, eff])


def joints_3dof(theta):
    pts = [np.array([0.0, 0.0])]
    cum = 0.0
    for li, ti in zip(L_3DOF, theta):
        cum += ti
        pts.append(pts[-1] + np.array([li * np.cos(cum), li * np.sin(cum)]))
    return np.array(pts)


def draw_frame(ax, env, theta_3, v_cmd, v_real, title):
    import matplotlib.pyplot as plt

    ax.cla()
    # espace de travail
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(a) * MAX_REACH, np.sin(a) * MAX_REACH,
            color="#777777", linestyle=":", linewidth=1.0, alpha=0.6,
            label="Espace de travail (r = 3 m)")

    # bras 2DoF réel
    p2 = joints_2dof([env.theta1, env.theta2])
    ax.plot(p2[:, 0], p2[:, 1], "-o", color="#1f77b4", linewidth=4.0,
            markersize=5, label="Bras 2DoF réel")

    # bras 3DoF virtuel (état vu par la politique PPO)
    p3 = joints_3dof(theta_3)
    ax.plot(p3[:, 0], p3[:, 1], "--o", color="#9B5DE5", linewidth=2.5,
            markersize=5, alpha=0.85, label="Bras 3DoF virtuel (state mapper)")

    eff = p2[-1]
    ax.add_patch(plt.Circle(eff, env.eff_radius, color="#1f77b4", alpha=0.9,
                            label="Effecteur"))
    ax.add_patch(plt.Circle(env.ball, env.ball_radius, color="#E05A3A",
                            alpha=0.85, label="Balle"))
    ax.add_patch(plt.Circle(env.target, env.epsilon, color="green", fill=False,
                            linestyle="--", linewidth=1.8,
                            label=f"Cible (succès < {env.epsilon:.2f} m)"))
    ax.plot(*env.target, "+", color="green", markersize=10, markeredgewidth=2)

    # vitesses effecteur : commandée (politique via bras virtuel) vs exécutée
    if np.linalg.norm(v_cmd) > 1e-3:
        ax.annotate("", xy=eff + v_cmd * ARROW_SCALE, xytext=eff,
                    arrowprops=dict(arrowstyle="-|>", color="#9B5DE5", lw=2.2))
    if np.linalg.norm(v_real) > 1e-3:
        ax.annotate("", xy=eff + v_real * ARROW_SCALE, xytext=eff,
                    arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2.2,
                                    alpha=0.9))
    # entrées de légende des flèches (proxys)
    ax.plot([], [], color="#9B5DE5", lw=2.2,
            label=f"v_eff commandée (PPO, ‖v‖={np.linalg.norm(v_cmd):.2f} m/s)")
    ax.plot([], [], color="#1f77b4", lw=2.2,
            label=f"v_eff exécutée (2DoF, ‖v‖={np.linalg.norm(v_real):.2f} m/s)")

    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-3.4, 3.4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)


def main():
    p = argparse.ArgumentParser(
        description="Visualiser le transfert push-ball 3DoF -> 2DoF (bras réel + bras virtuel).")
    p.add_argument("--mapper", choices=["legacy", "jacobian", "conditioned"],
                   default="jacobian")
    p.add_argument("--run-state", default="analytic",
                   help="run d'un state mapper appris (run_01, run_03_kin) ou "
                        "'analytic' (IK redondante, hors distribution -> dégradé)")
    p.add_argument("--no-eff-passthrough", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.03,
                   help="pause entre frames en mode interactif (s)")
    p.add_argument("--save", action="store_true",
                   help="écrire un GIF par épisode dans runs/eval/transfer_gifs/ "
                        "(utiliser MPLBACKEND=Agg sans écran)")
    p.add_argument("--fps", type=int, default=20, help="images/s des GIF")
    p.add_argument("--model-path", default="data/models/ppo_pushball_3dof_rec_1/ppo_pushball_final.zip")
    p.add_argument("--vecnorm-path", default="data/models/ppo_pushball_3dof_rec_1/vec_normalize.pkl")
    p.add_argument("--run-action-legacy", default="run_02")
    p.add_argument("--run-action-cond", default="run_04_cond")
    p.add_argument("--lam", type=float, default=0.05)
    args = p.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    interactive = matplotlib.get_backend().lower() != "agg" and not args.save

    # ── Politique + VecNormalize (mêmes chargements que le transfert) ──
    model = PPO.load(args.model_path, device="cpu", custom_objects={
        "learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2})
    vn3 = VecNormalize.load(args.vecnorm_path,
                            DummyVecEnv([lambda: PushBallEnv_3dof(None)]))
    vn3.training = False

    state_fn, state_reset = load_state_mapper(
        args.run_state, args.lam, eff_passthrough=not args.no_eff_passthrough)
    action_fn = load_action_mapper(args.mapper, args.run_action_legacy,
                                   args.run_action_cond, args.lam)

    env = PushBallEnv_2dof(render_mode=None, max_steps=args.max_steps)
    gif_dir = Path(__file__).resolve().parent / "runs" / "eval" / "transfer_gifs"

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    if interactive:
        plt.ion()
        plt.show()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state_reset()
        frames = []
        done, step, info = False, 0, {}

        while not done:
            arm_obs_2 = obs[:R1_STATE_DIM].astype(np.float32)
            task_obs = obs[R1_STATE_DIM:]

            state_r2 = state_fn(arm_obs_2)
            obs12 = np.concatenate([state_r2, task_obs]).astype(np.float32)
            obs12_n = vn3.normalize_obs(obs12[None, :])[0]
            action_3, _ = model.predict(obs12_n, deterministic=True)
            action_2 = np.clip(action_fn(action_3, state_r2, arm_obs_2), -1.0, 1.0)

            # vitesses effecteur pour l'affichage
            theta_3 = state_r2[:3] * np.pi
            theta_2 = arm_obs_2[:2] * np.pi
            v_cmd = jacobian_3dof_np(theta_3) @ (action_3 * OMEGA_MAX)
            v_real = jacobian_2dof_np(theta_2) @ (action_2 * OMEGA_MAX)

            obs, reward, terminated, truncated, info = env.step(action_2)
            done = terminated or truncated
            step += 1

            title = (f"Transfert push-ball 2DoF ← PPO 3DoF | action: {args.mapper} | "
                     f"état: {args.run_state}\n"
                     f"épisode {ep + 1}/{args.episodes} — pas {step} | "
                     f"d(balle, cible) = {info['dist_ball_target']:.3f} m")
            if done:
                title += ("  ->  SUCCÈS" if info.get("target_reached")
                          else "  ->  ÉCHEC (balle sortie)"
                          if float(np.linalg.norm(env.ball)) > env.max_reach * 1.05
                          else "  ->  ÉCHEC (temps écoulé)")

            if not plt.fignum_exists(fig.number):
                break
            
            draw_frame(ax, env, theta_3, v_cmd, v_real, title)

            if interactive:
                plt.pause(args.delay if not done else 1.5)
            if args.save:
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())

        result = "succes" if info.get("target_reached") else "echec"
        
        if not plt.fignum_exists(fig.number):
            print("\nFermeture fenêtre")
            break
        
        print(f"épisode {ep + 1}: {result} en {step} pas "
              f"(d finale = {info['dist_ball_target']:.3f} m)")

        if args.save and frames:
            from PIL import Image
            gif_dir.mkdir(parents=True, exist_ok=True)
            path = gif_dir / (f"transfer_3to2_{args.mapper}_{args.run_state}"
                              f"_ep{ep + 1:02d}_{result}.gif")
            imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE)
                    for f in frames]
            # dernière image tenue 1,5 s pour lire le verdict
            imgs[0].save(path, save_all=True, append_images=imgs[1:],
                         duration=[1000 // args.fps] * (len(imgs) - 1) + [1500],
                         loop=0)
            print(f"  GIF → {path}")

    if interactive:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
