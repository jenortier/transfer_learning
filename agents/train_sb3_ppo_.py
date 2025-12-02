import os
from torch import nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from envs.arm2d_env import Arm2DEnv
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter


# --- TensorBoard Writer ---
writer = SummaryWriter(log_dir="logs/rl_experiment") 


# --- Callback custom pour logguer reward ---
class TBCallback(BaseCallback):
	def __init__(self, verbose=0):
		super().__init__(verbose)

	def _on_step(self):
		# reward moyen fourni par SB3
		if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
			rew = self.model.logger.name_to_value["rollout/ep_rew_mean"]
			writer.add_scalar("Reward/episode", rew, self.num_timesteps)

		return True


# --- Environnements ---
train_env = Arm2DEnv(render_mode=None)
eval_env = Monitor(Arm2DEnv(render_mode=None))


# --- Dossier sauvegarde ---
log_dir = "models/"
os.makedirs(log_dir, exist_ok=True)


# --- Callback évaluation ---
eval_callback = EvalCallback(
	eval_env,
	best_model_save_path=log_dir,
	log_path=log_dir,
	eval_freq=2000,
	deterministic=True,
	render=False
)

# --- Callback fusionné (Eval + TensorBoard custom) ---
callbacks = [eval_callback, TBCallback()]


# --- Agent PPO ---
policy_kwargs = {
	"activation_fn": nn.Tanh,
	"net_arch": [128, 128, 64],
}

model = PPO(
	"MlpPolicy",
	train_env,
	verbose=1,
	tensorboard_log=log_dir,
	policy_kwargs=policy_kwargs
)

# --- Entraînement ---
model.learn(
	total_timesteps=100000,
	callback=callbacks
)

# --- Fermeture propre du writer ---
writer.close()

# --- Sauvegarde modèle ---
model.save(os.path.join(log_dir, "ppo_arm2d"))
print(log_dir)



