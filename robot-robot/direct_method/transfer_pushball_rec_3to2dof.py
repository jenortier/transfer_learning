"""Transfert push-ball : politique PPO 3DoF exécutée sur le bras 2DoF.

Flux par pas de temps :
    obs_2dof (10D brute) = [bloc bras 6D | balle 2D | cible 2D]
      -> state mapper 2->3 (bloc bras 6D) -> bloc bras 3DoF 8D
      -> obs_3dof (12D) = [bloc 8D | balle | cible]
      -> normalisation VecNormalize (stats du run PPO 3DoF)   <- indispensable :
         la politique a été entraînée sur des obs normalisées
      -> PPO 3DoF -> action_3 (= dtheta_3 / omega_max)
      -> action mapper (--mapper) -> action_2 -> env 2DoF

State mappers disponibles (--run-state) :
    <run_id>          : StateMapperMLP appris (ex: run_03_kin (défaut), run_01).
                        Les composantes effecteur de l'état mappé sont alors
                        remplacées par l'effecteur 2DoF réel (identique pour
                        les deux robots) sauf si --no-eff-passthrough.
    analytic          : IK redondante 2->3 résolue analytiquement (dernier lien
                        radial) + vitesses par Jacobiens, aucun entraînement
                        (voir state_mapper_analytic.py). ATTENTION : contrairement
                        au sens 3->2 (IK 2DoF unique et exacte), le 3DoF est
                        redondant ; la posture reconstruite est exacte en position
                        mais arbitraire, donc HORS DISTRIBUTION pour la politique
                        3DoF -> transfert dégradé. Préférer un state mapper appris
                        ici (run_03_kin : ~80 % contre ~0 % en analytic sur 10 ép.).

Mappers d'action disponibles (--mapper) :
    legacy      : ActionMapperMLP non conditionné (run_02) — référence historique
    jacobian    : baseline analytique J2⁺ @ (J3 @ dθ3) — aucun entraînement requis
    conditioned : ActionMapperConditionedMLP (run_04_cond) — conditionné par θ2 et θ3

Corrections par rapport à l'ancienne version :
    - dimensions d'état 6/8 (l'ancien script utilisait 4/6 : incompatible avec
      les state mappers entraînés) ;
    - normalisation VecNormalize des obs mappées avant model.predict (l'ancien
      script donnait des obs brutes à une politique entraînée en obs normalisées) ;
    - suppression du hack `clamp(-2, 2) * 0.5` sur l'action ;
    - le VecNormalize 3DoF n'est plus chargé sur l'env 2DoF (shapes 12 vs 10).

Usage :
    cd robot-robot && python3 -m direct_method.transfer_pushball_3to2dof \
        --mapper conditioned --episodes 100 --seed 0
"""

import argparse

import numpy as np
import torch
torch.set_num_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof_rec import PushBallEnv_2dof
from envs.env_pushball_3dof_rec import PushBallEnv_3dof
from direct_method.mapper_models import (
    StateMapperMLP, ActionMapperMLP, ActionMapperConditionedMLP)
from direct_method.action_mapper_baseline import JacobianActionMapper
from direct_method.state_mapper_analytic import AnalyticStateMapper2to3
from direct_method.kinematics import OMEGA_MAX

# Dimensions des blocs bras (voir envs/arm_2dof.py et arm_3dof.py)
R1_STATE_DIM = 6    # [θ1, θ2, dθ1, dθ2, x, y] normalisé          (bras 2DoF réel)
R2_STATE_DIM = 8    # [θ1, θ2, θ3, dθ1, dθ2, dθ3, x, y] normalisé  (bras 3DoF virtuel)
HIDDEN_DIM = 256


def load_state_mapper(run_state, lam, eff_passthrough):
    """Retourne (fn(arm_obs_2_6d) -> state_r2_8d, reset_fn)."""
    if run_state == "analytic":
        mapper = AnalyticStateMapper2to3(lam=lam)
        return mapper, mapper.reset

    path = f"direct_method/runs/{run_state}/models/state_mapper_r1_to_r2.pt"
    m = StateMapperMLP(R1_STATE_DIM, R2_STATE_DIM, HIDDEN_DIM)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()

    def fn(arm_obs_2):
        with torch.no_grad():
            state_r2 = m(torch.from_numpy(arm_obs_2).unsqueeze(0)).squeeze(0).numpy()
        if eff_passthrough:
            # l'effecteur est identique pour les deux robots (même convention
            # eff/max_reach) : autant transmettre la valeur exacte plutôt que
            # la reconstruction du réseau.
            state_r2[6:8] = arm_obs_2[4:6]
        return state_r2

    return fn, lambda: None


def load_action_mapper(kind, run_legacy, run_cond, lam):
    """Retourne fn(action_3, state_r2_8d, arm_obs_2dof_6d) -> action_2 (normalisée)."""
    if kind == "legacy":
        path = f"direct_method/runs/{run_legacy}/models/action_mapper_r2_to_r1.pt"
        m = ActionMapperMLP(3, 2, HIDDEN_DIM)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def fn(action_3, state_r2, arm_obs_2):
            with torch.no_grad():
                a = torch.from_numpy(action_3.astype(np.float32)).unsqueeze(0)
                return m(a).squeeze(0).numpy()
        return fn

    if kind == "conditioned":
        path = f"direct_method/runs/{run_cond}/models/action_mapper_3to2.pt"
        m = ActionMapperConditionedMLP(3, 2, HIDDEN_DIM)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()

        def fn(action_3, state_r2, arm_obs_2):
            with torch.no_grad():
                a = torch.from_numpy(action_3.astype(np.float32)).unsqueeze(0)
                q3 = torch.from_numpy(state_r2[:3].astype(np.float32)).unsqueeze(0)
                q2 = torch.from_numpy(arm_obs_2[:2].astype(np.float32)).unsqueeze(0)
                return m(a, q3, q2).squeeze(0).numpy()
        return fn

    if kind == "jacobian":
        base = JacobianActionMapper("3to2", lam=lam)

        def fn(action_3, state_r2, arm_obs_2):
            dtheta_3 = action_3 * OMEGA_MAX
            theta_3 = state_r2[:3] * np.pi     # posture 3DoF mappée (virtuelle)
            theta_2 = arm_obs_2[:2] * np.pi    # posture 2DoF réelle
            dtheta_2 = base(dtheta_3, theta_3, theta_2)
            return dtheta_2 / OMEGA_MAX
        return fn

    raise ValueError(f"mapper inconnu: {kind}")


def main():
    p = argparse.ArgumentParser(description="Transfert push-ball 3DoF -> 2DoF")
    p.add_argument("--mapper", choices=["legacy", "jacobian", "conditioned"],
                   default="jacobian")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model-path", default="data/models/ppo_pushball_3dof_1/ppo_pushball_final.zip")
    p.add_argument("--vecnorm-path", default="data/models/ppo_pushball_3dof_1/vec_normalize.pkl")
    p.add_argument("--run-state", default="analytic",
                   help="run d'un state mapper appris (run_01, run_03_kin) ou "
                        "'analytic' (IK redondante 2->3, hors distribution pour la "
                        "politique 3DoF -> transfert dégradé, cf. docstring)")
    p.add_argument("--no-eff-passthrough", action="store_true",
                   help="ne pas remplacer l'effecteur mappé par l'effecteur réel "
                        "(state mappers appris uniquement)")
    p.add_argument("--run-action-legacy", default="run_02")
    p.add_argument("--run-action-cond", default="run_04_cond")
    p.add_argument("--lam", type=float, default=0.05)
    args = p.parse_args()

    # ── Politique 3DoF + stats VecNormalize associées ────────────────
    model = PPO.load(args.model_path, custom_objects={
        "learning_rate": 3e-4, "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2})

    # VecNormalize chargé sur un env 3DoF factice : sert uniquement à
    # normaliser les obs mappées avec les stats vues par la politique.
    import os
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        run_state = args.run_state
        run_action_cond = args.run_action_cond
        state_mapper_3to2_path = f"direct_method/runs/{run_state}/models/state_mapper_r2_to_r1.pt"
        state_mapper_2to3_path = f"direct_method/runs/{run_state}/models/state_mapper_r1_to_r2.pt"
        action_mapper_3to2_path = f"direct_method/runs/{run_action_cond}/models/action_mapper_3to2.pt"
        action_mapper_2to3_path = f"direct_method/runs/{run_action_cond}/models/action_mapper_2to3.pt"
        vn3 = VecNormalize.load(args.vecnorm_path,
                                DummyVecEnv([lambda: PushBallEnv_3dof(
                                    state_mapper_3to2_path=state_mapper_3to2_path,
                                    state_mapper_2to3_path=state_mapper_2to3_path,
                                    action_mapper_3to2_path=action_mapper_3to2_path,
                                    action_mapper_2to3_path=action_mapper_2to3_path,
                                )]))
        vn3.training = False
        normalize_obs = vn3.normalize_obs
    else:
        print(f"ATTENTION: vec_normalize.pkl introuvable ({args.vecnorm_path}).\n"
              "La politique a probablement été entraînée avec VecNormalize : sans les\n"
              "stats, les résultats seront dégradés. Obs transmises brutes.")
        normalize_obs = lambda x: x

    # ── Mappers ──────────────────────────────────────────────────────
    state_fn, state_reset = load_state_mapper(
        args.run_state, args.lam, eff_passthrough=not args.no_eff_passthrough)
    action_fn = load_action_mapper(args.mapper, args.run_action_legacy,
                                   args.run_action_cond, args.lam)

    # ── Env 2DoF, obs brutes (pas de VecNormalize : shapes incompatibles
    #    et les mappers sont entraînés sur les obs physiquement normalisées) ──
    # Passer les chemins des mappers pour activer la reconstruction reward
    run_state = args.run_state
    run_action_cond = args.run_action_cond
    state_mapper_2to3_path = f"direct_method/runs/{run_state}/models/state_mapper_r1_to_r2.pt"
    state_mapper_3to2_path = f"direct_method/runs/{run_state}/models/state_mapper_r2_to_r1.pt"
    action_mapper_2to3_path = f"direct_method/runs/{run_action_cond}/models/action_mapper_2to3.pt"
    action_mapper_3to2_path = f"direct_method/runs/{run_action_cond}/models/action_mapper_3to2.pt"

    render_mode = "human" if args.render else None
    env = DummyVecEnv([lambda: Monitor(
        PushBallEnv_2dof(
            render_mode=render_mode,
            max_steps=args.max_steps,
            state_mapper_2to3_path=state_mapper_2to3_path,
            state_mapper_3to2_path=state_mapper_3to2_path,
            action_mapper_2to3_path=action_mapper_2to3_path,
            action_mapper_3to2_path=action_mapper_3to2_path,
        ))])
    env.seed(args.seed)

    successes = 0
    steps_on_success = []
    final_dist_on_failure = []

    print(f"Transfert push-ball 2DoF <- PPO 3DoF | mapper={args.mapper} | "
          f"state mapper={args.run_state} | {args.episodes} épisodes | seed={args.seed}\n")

    for ep in range(args.episodes):
        obs = env.reset()
        state_reset()
        done = False
        step_count = 0
        info_last = {}

        while not done:
            obs10 = obs[0]
            arm_obs_2 = obs10[:R1_STATE_DIM].astype(np.float32)
            task_obs = obs10[R1_STATE_DIM:]

            # 1) état 2DoF -> état 3DoF
            state_r2 = state_fn(arm_obs_2)

            # 2) obs 3DoF complète + normalisation VecNormalize
            obs12 = np.concatenate([state_r2, task_obs]).astype(np.float32)
            obs12_n = normalize_obs(obs12[None, :])[0]

            # 3) action 3DoF
            action_3, _ = model.predict(obs12_n, deterministic=True)

            # 4) action 3DoF -> action 2DoF
            action_2 = np.clip(action_fn(action_3, state_r2, arm_obs_2), -1.0, 1.0)

            obs, reward, dones, infos = env.step(action_2.reshape(1, -1))
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
