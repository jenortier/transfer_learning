import os
from torch import nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from envs.arm2d_env_v3 import Arm2DEnv
from torch.utils.tensorboard import SummaryWriter


env = Arm2DEnv
#env = gym.make("CartPole-v1", render_mode="rgb_array")

# --- TensorBoard Writer ---
writer = SummaryWriter(log_dir="logs/rl_experiment") 

# --- Linear schedule pour ent_coef (SB3) ---
def linear_schedule(start, end):
	def f(progress_remaining: float):
		return end + (start - end) * progress_remaining
	return f

# --- Callback custom pour logguer reward ---
class TBCallback(BaseCallback):
	def __init__(self, verbose=0):
		super().__init__(verbose)

	def _on_step(self):
		if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
			rew = self.model.logger.name_to_value["rollout/ep_rew_mean"]
			writer.add_scalar("Reward/episode", rew, self.num_timesteps)
		return True

# --- Callback scheduler ent_coef ---
class EntCoefCallback(BaseCallback):
	def __init__(self, schedule, verbose=0):
		super().__init__(verbose)
		self.schedule = schedule

	def _on_step(self) -> bool:
		progress = 1 - self.num_timesteps / self.model._total_timesteps
		self.model.ent_coef = self.schedule(progress)
		return True


# --- Environnements ---
train_env = env(render_mode=None)
eval_env = Monitor(env(render_mode=None))


# --- Dossier sauvegarde ---
log_dir = "models/"
os.makedirs(log_dir, exist_ok=True)


# --- Callback évaluation ---
eval_callback = EvalCallback(
	eval_env,
	best_model_save_path=log_dir,
	log_path=log_dir,
	eval_freq=2000,
	n_eval_episodes=10,
	deterministic=True,
	render=False
)

callbacks = [
	eval_callback,
	TBCallback(),
	EntCoefCallback(linear_schedule(0.001, 0.00001))
]


# --- Agent PPO ---
policy_kwargs = {
	"activation_fn": nn.Tanh,
	"net_arch": [256, 256, 256, 256],
}

model = PPO(
	"MlpPolicy",
	train_env,
	n_steps=2048,
	batch_size=512,
	n_epochs=10,
	learning_rate=1e-4,
	clip_range=0.2,
	ent_coef=0.001,
	vf_coef=0.7,
	clip_range_vf=0.2,
	target_kl=0.02,
	verbose=1,
	tensorboard_log=log_dir,
	policy_kwargs=policy_kwargs
)


# --- Entraînement ---
model.learn(
	total_timesteps=1_250_000,
	callback=callbacks
)

writer.close()

model.save(os.path.join(log_dir, "ppo_arm2d"))
print(log_dir)

