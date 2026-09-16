"""Évaluation quantitative du transfert push-ball : PPO 3DoF exécutée sur le bras 2DoF.

Chaîne strictement identique à visualize_transfer_3to2dof.py (même env brut,
même seeding par épisode `seed + ep`, mêmes mappers), mais sans matplotlib :
on mesure le taux de réussite sur N épisodes au lieu d'afficher les bras.

En bonus, on quantifie ce que la visualisation montrait avec les deux flèches :
l'écart entre la vitesse d'effecteur commandée par la politique (J3 sur le bras
3DoF virtuel) et celle réellement exécutée par le 2DoF (J2). Si le mapping
d'action est bon, l'erreur relative tend vers 0 et le cosinus vers 1.

Usage :
    cd robot-robot && python3 -m direct_method.eval_transfer_3to2dof \
        --mapper conditioned --run-state run_03_kin --episodes 200 --seed 0
"""

import argparse
import os

import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from direct_method.transfer_pushball_3to2dof import (
    load_state_mapper, load_action_mapper, R1_STATE_DIM)
from direct_method.kinematics import (
    OMEGA_MAX, jacobian_2dof_np, jacobian_3dof_np)


def main():
    p = argparse.ArgumentParser(
        description="Taux de réussite du transfert push-ball 3DoF -> 2DoF.")
    p.add_argument("--mapper", choices=["legacy", "jacobian", "conditioned"],
                   default="jacobian")
    p.add_argument("--run-state", default="analytic",
                   help="run d'un state mapper appris (run_01, run_03_kin) ou "
                        "'analytic' (IK redondante, hors distribution -> dégradé)")
    p.add_argument("--no-eff-passthrough", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-path", default="data/models/ppo_pushball_3dof_1/ppo_pushball_final.zip")
    p.add_argument("--vecnorm-path", default="data/models/ppo_pushball_3dof_1/vec_normalize.pkl")
    p.add_argument("--run-action-legacy", default="run_02")
    p.add_argument("--run-action-cond", default="run_04_cond")
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--progress", type=int, default=25,
                   help="afficher un point d'étape tous les N épisodes (0 = jamais)")
    args = p.parse_args()

    # ── Politique 3DoF + stats VecNormalize associées ─────────────────
    model = PPO.load(args.model_path, device="cpu", custom_objects={
        "learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2})

    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        vn3 = VecNormalize.load(args.vecnorm_path,
                                DummyVecEnv([lambda: PushBallEnv_3dof(None)]))
        vn3.training = False
        normalize_obs = vn3.normalize_obs
    else:
        print(f"ATTENTION: vec_normalize.pkl introuvable ({args.vecnorm_path}).\n"
              "Obs transmises brutes : résultats dégradés.")
        normalize_obs = lambda x: x

    state_fn, state_reset = load_state_mapper(
        args.run_state, args.lam, eff_passthrough=not args.no_eff_passthrough)
    action_fn = load_action_mapper(args.mapper, args.run_action_legacy,
                                   args.run_action_cond, args.lam)

    env = PushBallEnv_2dof(render_mode=None, max_steps=args.max_steps)

    successes = 0
    steps_on_success = []
    final_dist_on_failure = []
    n_ball_out = 0
    n_timeout = 0
    v_err_rel = []      # ‖v_real - v_cmd‖ / ‖v_cmd‖
    v_cos = []          # cos(v_real, v_cmd)

    print(f"Transfert push-ball 2DoF <- PPO 3DoF | action={args.mapper} | "
          f"état={args.run_state} | {args.episodes} épisodes | "
          f"seeds={args.seed}..{args.seed + args.episodes - 1}\n")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state_reset()
        done, step, info = False, 0, {}

        while not done:
            arm_obs_2 = obs[:R1_STATE_DIM].astype(np.float32)
            task_obs = obs[R1_STATE_DIM:]

            # 1) état 2DoF -> état 3DoF virtuel
            state_r2 = state_fn(arm_obs_2)

            # 2) obs 3DoF complète + normalisation VecNormalize
            obs12 = np.concatenate([state_r2, task_obs]).astype(np.float32)
            obs12_n = normalize_obs(obs12[None, :])[0]

            # 3) action 3DoF
            action_3, _ = model.predict(obs12_n, deterministic=True)

            # 4) action 3DoF -> action 2DoF
            action_2 = np.clip(action_fn(action_3, state_r2, arm_obs_2), -1.0, 1.0)

            # suivi de la vitesse d'effecteur (les deux flèches de la visu)
            theta_3 = state_r2[:3] * np.pi
            theta_2 = arm_obs_2[:2] * np.pi
            v_cmd = jacobian_3dof_np(theta_3) @ (action_3 * OMEGA_MAX)
            v_real = jacobian_2dof_np(theta_2) @ (action_2 * OMEGA_MAX)
            n_cmd = float(np.linalg.norm(v_cmd))
            n_real = float(np.linalg.norm(v_real))
            if n_cmd > 1e-3:
                v_err_rel.append(float(np.linalg.norm(v_real - v_cmd)) / n_cmd)
                if n_real > 1e-3:
                    v_cos.append(float(v_cmd @ v_real) / (n_cmd * n_real))

            obs, reward, terminated, truncated, info = env.step(action_2)
            done = terminated or truncated
            step += 1

        if info.get("target_reached", False):
            successes += 1
            steps_on_success.append(step)
        else:
            final_dist_on_failure.append(info.get("dist_ball_target", float("nan")))
            if float(np.linalg.norm(env.ball)) > env.max_reach * 1.05:
                n_ball_out += 1
            else:
                n_timeout += 1

        if args.progress and (ep + 1) % args.progress == 0:
            print(f"  {ep + 1:4d}/{args.episodes} épisodes — "
                  f"réussite courante {100 * successes / (ep + 1):5.1f}%")

    # ── Résultats ─────────────────────────────────────────────────────
    n = args.episodes
    rate = 100 * successes / n
    # intervalle de Wald à 95 % (indicatif, n = 200)
    ci = 196 * np.sqrt((rate / 100) * (1 - rate / 100) / n)

    print()
    print("=" * 52)
    print(f"  Sens                  : 2DoF réel <- politique 3DoF")
    print(f"  State mapper          : {args.run_state}")
    print(f"  Action mapper         : {args.mapper}")
    print(f"  Épisodes testés       : {n}")
    print(f"  Réussites             : {successes}")
    print(f"  Taux de réussite      : {rate:.1f}%  (±{ci:.1f} pts, IC95)")
    print("-" * 52)
    if steps_on_success:
        print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
        print(f"  Steps min / max       : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
    if final_dist_on_failure:
        print(f"  Échecs balle sortie   : {n_ball_out}")
        print(f"  Échecs temps écoulé   : {n_timeout}")
        print(f"  Distance moy (échec)  : {np.nanmean(final_dist_on_failure):.4f} m")
        print(f"  Distance min / max    : {np.nanmin(final_dist_on_failure):.4f} / "
              f"{np.nanmax(final_dist_on_failure):.4f} m")
    if v_err_rel:
        print("-" * 52)
        print(f"  Erreur v_eff relative : moy {np.mean(v_err_rel):.3f} | "
              f"médiane {np.median(v_err_rel):.3f} | p90 {np.percentile(v_err_rel, 90):.3f}")
        print(f"  cos(v_cmd, v_real)    : moy {np.mean(v_cos):.3f} | "
              f"p10 {np.percentile(v_cos, 10):.3f}")
    print("=" * 52)

    env.close()


if __name__ == "__main__":
    main()
