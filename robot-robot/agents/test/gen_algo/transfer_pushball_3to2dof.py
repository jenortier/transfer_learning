from collections import deque
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "data" / "models").exists())
DATA_ROOT = ROOT / "data"
MODEL_ROOT = DATA_ROOT / "models"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from envs.env_pushball_2dof import PushBallEnv_2dof
from envs.env_pushball_3dof import PushBallEnv_3dof
from agents.transfer_gen_algo.mapper_models import (
    StateMapperMLP,
    ActionMapperMLP,
    project_mapped_arm_state_to_reference,
)


class PushBallTransfer3to2:
    def __init__(self, policy_3dof_path: str, vecnorm_3dof_path: str,
                 mapper_path: str, device="cpu"):
        self.device = device

        self.policy_3dof = PPO.load(policy_3dof_path, device=device)

        # VecNormalize 3DoF
        venv = DummyVecEnv([lambda: Monitor(PushBallEnv_3dof(render_mode=None))])
        self.vec_norm_3dof = VecNormalize.load(vecnorm_3dof_path, venv=venv)
        self.vec_norm_3dof.training = False
        self.vec_norm_3dof.norm_reward = False
        venv.close()

        # Chargement des mappers depuis le fichier unique
        checkpoint = torch.load(mapper_path, map_location=device)
        state_input_dim = checkpoint["state_mapper"]["net.0.weight"].shape[1]
        action_input_dim = checkpoint["action_mapper"]["net.0.weight"].shape[1]
        if state_input_dim < 6 or (state_input_dim - 6) % 6 != 0:
            raise ValueError(f"Unsupported 2-DoF mapper input dimension: {state_input_dim}")
        if action_input_dim != state_input_dim + 3:
            raise ValueError(
                f"Mapper dimensions disagree: state={state_input_dim}, action={action_input_dim}"
            )

        self.state_input_dim = state_input_dim
        self.history_len = (state_input_dim - 6) // 6
        # Older checkpoints use 12D (current state + one history state), while
        # newer sequence checkpoints use 30D (current state + four history states).
        self.state_mapper = StateMapperMLP(state_input_dim, 8).to(device)
        self.state_mapper.load_state_dict(checkpoint["state_mapper"])
        self.state_mapper.eval()

        self.action_mapper = ActionMapperMLP(state_input_dim, 3, 2).to(device)
        self.action_mapper.load_state_dict(checkpoint["action_mapper"])
        self.action_mapper.eval()

        self.history = deque(maxlen=self.history_len)
        print("PushBall Transfer 3->2 loaded successfully")

    def _build_history_features(self, arm_state: np.ndarray) -> np.ndarray:
        """
        Construit l'entrée du mapper selon la largeur du checkpoint chargé.
        """
        if len(self.history) == 0:
            for _ in range(self.history_len):
                self.history.append(arm_state.copy())

        self.history.append(arm_state.copy())

        hist = np.concatenate(list(self.history), axis=1)
        return np.concatenate([arm_state, hist], axis=1)

    @torch.no_grad()
    def predict(self, obs_2dof: np.ndarray) -> np.ndarray:
        if obs_2dof.ndim == 1:
            obs_2dof = obs_2dof.reshape(1, -1)

        arm_2 = obs_2dof[:, :6]      # 6D arm state
        task = obs_2dof[:, 6:]       # 4D task for pushball

        arm_2_combined = self._build_history_features(arm_2)

        arm_3 = self.state_mapper(torch.from_numpy(arm_2_combined).float().to(self.device))
        arm_3 = project_mapped_arm_state_to_reference(
            arm_3.cpu().numpy(),
            reference_arm_state=arm_2,
        )

        full_obs_3 = np.concatenate([arm_3, task], axis=1)
        norm_obs = self.vec_norm_3dof.normalize_obs(full_obs_3)

        act_3, _ = self.policy_3dof.predict(norm_obs, deterministic=True)

        act_2 = self.action_mapper(
            torch.from_numpy(arm_2_combined).float().to(self.device),
            torch.from_numpy(act_3).float().to(self.device)
        )
        return act_2.cpu().numpy()[0]


def main():
    POLICY_3DOF   = str(MODEL_ROOT / "ppo_pushball_3dof_1" / "best_model.zip")
    VECNORM_3DOF  = str(MODEL_ROOT / "ppo_pushball_3dof_1" / "vec_normalize.pkl")
    MAPPER_PATH   = str(DATA_ROOT / "DIRECT_GEN_ALGO" / "transfer_3to2_seq.pt")

    transfer = PushBallTransfer3to2(POLICY_3DOF, VECNORM_3DOF, MAPPER_PATH)

    env = DummyVecEnv([lambda: Monitor(PushBallEnv_2dof(render_mode=None))])
    n_episodes = 500
    max_steps = 400

    successes = 0
    steps_success = []

    for ep in tqdm(range(n_episodes), desc="PushBall 3->2 Transfer Test"):
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
    print(f"\n=== PUSHBALL TRANSFER 3->2 RESULTS ===")
    print(f"Success Rate : {rate:.2f}%  ({successes}/{n_episodes})")
    if steps_success:
        print(f"Avg steps (success) : {np.mean(steps_success):.1f}")
    print("="*50)

    env.close()


if __name__ == "__main__":
    main()
