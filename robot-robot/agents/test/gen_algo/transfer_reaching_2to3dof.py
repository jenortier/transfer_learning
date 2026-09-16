import argparse
from collections import deque
import numpy as np
import os
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
DATA_ROOT = ROOT / "data"
MODEL_ROOT = DATA_ROOT / "models"

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.env_reaching_2dof import ReachingEnv_2dof
from agents.transfer_gen_algo.mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    project_mapped_arm_state_to_reference,
)


class ReachingTransfer2to3:
    def __init__(self, policy_2dof_path: str, vecnorm_2dof_path: str,
                 mapper_path: str, device="cpu"):
        self.device = device

        self.policy_2dof = PPO.load(policy_2dof_path, device=device)

        # VecNormalize 2DoF
        venv = DummyVecEnv([lambda: Monitor(ReachingEnv_2dof(render_mode=None))])
        self.vec_norm_2dof = VecNormalize.load(vecnorm_2dof_path, venv=venv)
        self.vec_norm_2dof.training = False
        self.vec_norm_2dof.norm_reward = False
        venv.close()

        # Chargement des mappers depuis le fichier unique
        checkpoint = torch.load(mapper_path, map_location=device)
        # Le checkpoint attendu : entrée 16D (8 spatial + 8 temporal)
        self.state_mapper = StateMapperMLP(16, 6).to(device)
        self.state_mapper.load_state_dict(checkpoint["state_mapper"])
        self.state_mapper.eval()

        # action_mapper attend state_dim + act_in_dim = 20 (ici 16 + 4)
        self.action_mapper = ActionMapperMLP(16, 4, 3).to(device)
        self.action_mapper.load_state_dict(checkpoint["action_mapper"])
        self.action_mapper.eval()
        self.history = deque(maxlen=1)

        print("Reaching Transfer 2->3 loaded successfully")

    def _build_history_features(self, arm_state: np.ndarray) -> np.ndarray:
        """
        Extrait les features temporelles (velocities) à partir de l'état courant
        et les combine avec les features spatiales pour former l'entrée du mapper.
        Le mapper attend 16 dims (8 spatial + 8 temporal).
        """
        if len(self.history) == 0:
            self.history.append(arm_state.copy())

        self.history.append(arm_state.copy())

        hist = np.concatenate(list(self.history), axis=1)
        return np.concatenate([arm_state, hist], axis=1)

    @torch.no_grad()
    def predict(self, obs_3dof: np.ndarray) -> np.ndarray:
        if obs_3dof.ndim == 1:
            obs_3dof = obs_3dof.reshape(1, -1)

        arm_3 = obs_3dof[:, :8]      # 8D arm state
        task = obs_3dof[:, 8:]       # 5D task for reaching

        # Construire l'entrée combinée spatiale + temporelle pour le state mapper
        # Le mapper attend 16 dims (8 spatial + 8 temporal)
        arm_3_combined = self._build_history_features(arm_3)

        arm_2 = self.state_mapper(torch.from_numpy(arm_3_combined).float().to(self.device))
        arm_2 = project_mapped_arm_state_to_reference(
            arm_2.cpu().numpy(),
            reference_arm_state=arm_3,
        )

        full_obs_2 = np.concatenate([arm_2, task], axis=1)
        norm_obs = self.vec_norm_2dof.normalize_obs(full_obs_2)

        act_2, _ = self.policy_2dof.predict(norm_obs, deterministic=True)

        # Pad action from 2D policy to expected input size (4) if necessary
        act_2 = np.asarray(act_2)
        if act_2.ndim == 1:
            act_2 = act_2.reshape(1, -1)
        pad_len = 4 - act_2.shape[1]
        if pad_len > 0:
            act_2 = np.concatenate([act_2, np.zeros((act_2.shape[0], pad_len), dtype=act_2.dtype)], axis=1)

        # Pour l'action mapper, on utilise également l'entrée combinée
        act_3 = self.action_mapper(
            torch.from_numpy(arm_3_combined).float().to(self.device),
            torch.from_numpy(act_2).float().to(self.device)
        )
        return act_3.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate direct reaching transfer 2DoF -> 3DoF.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode.")
    args = parser.parse_args()

    POLICY_2DOF   = str(MODEL_ROOT / "ppo_reach_2dof_1" / "best_model.zip")
    VECNORM_2DOF  = str(MODEL_ROOT / "ppo_reach_2dof_1" / "vec_normalize.pkl")
    MAPPER_PATH   = str(DATA_ROOT / "DIRECT_GEN_ALGO" / "transfer_2to3_seq.pt")

    transfer = ReachingTransfer2to3(POLICY_2DOF, VECNORM_2DOF, MAPPER_PATH)

    env = DummyVecEnv([lambda: Monitor(ReachingEnv_3dof(render_mode=None))])
    n_episodes = args.episodes
    max_steps = args.max_steps

    successes = 0
    steps_success = []

    for ep in tqdm(range(n_episodes), desc="Reaching 2->3 Transfer Test"):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = transfer.predict(obs[0])
            obs, _, dones, infos = env.step([action])
            steps += 1
            done = dones[0]
            info = infos[0]

        if info.get("target_reached", False):
            successes += 1
            steps_success.append(steps)

    rate = successes / n_episodes * 100
    print(f"\n=== REACHING TRANSFER 2->3 RESULTS ===")
    print(f"Success Rate : {rate:.2f}%  ({successes}/{n_episodes})")
    if steps_success:
        print(f"Avg steps (success) : {np.mean(steps_success):.1f}")
    print("="*50)

    env.close()


if __name__ == "__main__":
    main()
