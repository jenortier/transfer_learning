"""Transfert push-ball : politique PPO 2DoF exécutée sur le bras 3DoF.

Flux par pas de temps :
    obs_3dof (12D brute) = [bloc bras 8D | balle 2D | cible 2D]
      -> state_mapper_r2_to_r1(bloc bras 8D) -> bloc bras 2DoF 6D
      -> obs_2dof (10D) = [bloc 6D | balle | cible]
      -> normalisation VecNormalize (stats du run PPO 2DoF)   <- indispensable :
         la politique a été entraînée sur des obs normalisées
      -> PPO 2DoF -> action_2 (= dtheta_2 / omega_max)
      -> action mapper (--mapper) -> action_3 -> env 3DoF

Mappers d'action disponibles :
    legacy      : ActionMapperMLP non conditionné (run_02) — référence historique
    jacobian    : baseline analytique J3⁺ @ (J2 @ dθ2) — aucun entraînement requis
    conditioned : ActionMapperConditionedMLP (run_04_cond) — conditionné par θ2 et θ3

Corrections par rapport à l'ancienne version :
    - dimensions d'état 6/8 (l'ancien script utilisait 4/6 : incompatible avec
      les state mappers entraînés) ;
    - normalisation VecNormalize des obs mappées avant model.predict (l'ancien
      script donnait des obs brutes à une politique entraînée en obs normalisées) ;
    - suppression du hack `clamp(-2, 2) * 0.5` sur l'action ;
    - le VecNormalize 2DoF n'est plus chargé sur l'env 3DoF (shapes 10 vs 12).

Usage :
    cd robot-robot && python3 -m direct_method.transfer_pushball_3to2dof \
        --mapper jacobian --episodes 100 --seed 0
"""

import argparse

import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_3dof import PushBallEnv_3dof
from envs.env_pushball_2dof import PushBallEnv_2dof
from direct_method.mapper_models import (
    StateMapperMLP, ActionMapperMLP, ActionMapperConditionedMLP)
from direct_method.action_mapper_baseline import JacobianActionMapper
from direct_method.kinematics import OMEGA_MAX

# Dimensions des blocs bras (voir envs/arm_2dof.py et arm_3dof.py)
R1_STATE_DIM = 6    # [θ1, θ2, dθ1, dθ2, x, y] normalisé
R2_STATE_DIM = 8    # [θ1, θ2, θ3, dθ1, dθ2, dθ3, x, y] normalisé
HIDDEN_DIM = 256


def load_state_mapper(run_id):
    path = f"direct_method/runs/{run_id}/models/state_mapper_r2_to_r1.pt"
    m = StateMapperMLP(R2_STATE_DIM, R1_STATE_DIM, HIDDEN_DIM)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def load_action_mapper(kind, run_legacy, run_cond, lam):
    """Retourne fn(action_2, state_r1_6d, arm_obs_3dof_8d) -> action_3 (normalisée)."""
    if kind == "legacy":
        path = f"direct_method/runs/{run_legacy}/models/action_mapper_r1_to_r2.pt"
        m = ActionMapperMLP(2, 3, HIDDEN_DIM)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def fn(action_2, state_r1, arm_obs_3):
            with torch.no_grad():
                a = torch.from_numpy(action_2.astype(np.float32)).unsqueeze(0)
                return m(a).squeeze(0).numpy()
        return fn

    if kind == "conditioned":
        path = f"direct_method/runs/{run_cond}/models/action_mapper_2to3.pt"
        m = ActionMapperConditionedMLP(2, 3, HIDDEN_DIM)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def fn(action_2, state_r1, arm_obs_3):
            with torch.no_grad():
                a = torch.from_numpy(action_2.astype(np.float32)).unsqueeze(0)
                q2 = torch.from_numpy(state_r1[:2].astype(np.float32)).unsqueeze(0)
                q3 = torch.from_numpy(arm_obs_3[:3].astype(np.float32)).unsqueeze(0)
                return m(a, q2, q3).squeeze(0).numpy()
        return fn

    if kind == "jacobian":
        base = JacobianActionMapper("2to3", lam=lam)

        def fn(action_2, state_r1, arm_obs_3):
            dtheta_2 = action_2 * OMEGA_MAX
            theta_2 = state_r1[:2] * np.pi     # posture 2DoF mappée
            theta_3 = arm_obs_3[:3] * np.pi    # posture 3DoF réelle
            dtheta_3 = base(dtheta_2, theta_2, theta_3)
            return dtheta_3 / OMEGA_MAX
        return fn

    raise ValueError(f"mapper inconnu: {kind}")


def main():
    p = argparse.ArgumentParser(description="Transfert push-ball 2DoF -> 3DoF")
    p.add_argument("--mapper", choices=["legacy", "jacobian", "conditioned"],
                   default="jacobian")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model-path", default="data/models/ppo_pushball_2dof_1/best_model.zip")
    p.add_argument("--vecnorm-path", default="data/models/ppo_pushball_2dof_1/vec_normalize.pkl")
    p.add_argument("--run-state", default="run_03_kin",
                   help="run du state mapper (ex: run_01 pour l'ancien)")
    p.add_argument("--run-action-legacy", default="run_02")
    p.add_argument("--run-action-cond", default="run_04_cond")
    p.add_argument("--lam", type=float, default=0.05)
    args = p.parse_args()

    # ── Politique 2DoF + stats VecNormalize associées ────────────────
    model = PPO.load(args.model_path, custom_objects={
        "learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2})

    # VecNormalize chargé sur un env 2DoF factice : sert uniquement à
    # normaliser les obs mappées avec les stats vues par la politique.
    import os
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        vn2 = VecNormalize.load(args.vecnorm_path,
                                DummyVecEnv([lambda: PushBallEnv_2dof(None)]))
        vn2.training = False
        normalize_obs = vn2.normalize_obs
    else:
        print(f"ATTENTION: vec_normalize.pkl introuvable ({args.vecnorm_path}).\n"
              "La politique a probablement été entraînée avec VecNormalize : sans les\n"
              "stats, les résultats seront dégradés. Obs transmises brutes.")
        normalize_obs = lambda x: x

    # ── Mappers ──────────────────────────────────────────────────────
    state_mapper = load_state_mapper(args.run_state)
    action_fn = load_action_mapper(args.mapper, args.run_action_legacy,
                                   args.run_action_cond, args.lam)

    # ── Env 3DoF, obs brutes (pas de VecNormalize : shapes incompatibles
    #    et les mappers sont entraînés sur les obs physiquement normalisées) ──
    render_mode = "human" if args.render else None
    env = DummyVecEnv([lambda: Monitor(
        PushBallEnv_3dof(render_mode=render_mode, max_steps=args.max_steps))])
    env.seed(args.seed)

    successes = 0
    steps_on_success = []
    final_dist_on_failure = []

    print(f"Transfert push-ball 3DoF <- PPO 2DoF | mapper={args.mapper} | "
          f"state mapper={args.run_state} | {args.episodes} épisodes | seed={args.seed}\n")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        step_count = 0
        info_last = {}

        while not done:
            obs12 = obs[0]
            arm_obs_3 = obs12[:R2_STATE_DIM].astype(np.float32)
            task_obs = obs12[R2_STATE_DIM:]

            # 1) état 3DoF -> état 2DoF
            with torch.no_grad():
                state_r1 = state_mapper(
                    torch.from_numpy(arm_obs_3).unsqueeze(0)).squeeze(0).numpy()

            # 2) obs 2DoF complète + normalisation VecNormalize
            obs10 = np.concatenate([state_r1, task_obs]).astype(np.float32)
            obs10_n = normalize_obs(obs10[None, :])[0]

            # 3) action 2DoF
            action_2, _ = model.predict(obs10_n, deterministic=True)

            # 4) action 2DoF -> action 3DoF
            action_3 = np.clip(action_fn(action_2, state_r1, arm_obs_3), -1.0, 1.0)

            obs, reward, dones, infos = env.step(action_3.reshape(1, -1))
            if args.render:
                env.envs[0].render()
            step_count += 1
            done = dones[0]
            info_last = infos[0]

        if info_last.get("target_reached", False):
            successes += 1
            steps_on_success.append(step_count)
        else:
            final_dist_on_failure.append(info_last.get("dist_ball_target", float("nan")))

    # ── Résultats ─────────────────────────────────────────────────────
    success_rate = 100 * successes / args.episodes
    print("=" * 45)
    print(f"  Mapper d'action       : {args.mapper}")
    print(f"  Épisodes testés       : {args.episodes}")
    print(f"  Réussites             : {successes}")
    print(f"  Taux de réussite      : {success_rate:.1f}%")
    print("-" * 45)
    if steps_on_success:
        print(f"  Steps moyens (succès) : {np.mean(steps_on_success):.1f}")
        print(f"  Steps min / max       : {np.min(steps_on_success)} / {np.max(steps_on_success)}")
    if final_dist_on_failure:
        print(f"  Distance moy (échec)  : {np.nanmean(final_dist_on_failure):.4f} m")
        print(f"  Distance min / max    : {np.nanmin(final_dist_on_failure):.4f} / "
              f"{np.nanmax(final_dist_on_failure):.4f} m")
    print("=" * 45)

    env.close()


if __name__ == "__main__":
    main()
