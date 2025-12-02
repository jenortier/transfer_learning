import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Arm2DEnv(gym.Env):

#environnement Gymnasium 2D : bras deux articulations
#objectif : atteindre la cible avec leffecteur


	metadata = {"render_modes": ["human"], "render_fps": 30}

	def __init__(self, render_mode=None):
		super().__init__()

# def bras 
		self.l1 = 1.0
		self.l2 = 1.0
		self.max_reach = self.l1 + self.l2

#limites angles
		self.theta_min = -np.pi
		self.theta_max = np.pi

# action : delta angles
		self.action_space = spaces.Box(
			low=np.array([-1, -1], dtype=np.float32),
			high=np.array([1, 1], dtype=np.float32)
		)

# Observation = angles + erreur relative (dx, dy)
		self.observation_space = spaces.Box(
			low=np.array([self.theta_min, self.theta_min, -2.0, -2.0], dtype=np.float32),
			high=np.array([self.theta_max, self.theta_max,  2.0,  2.0], dtype=np.float32)
		)

		self.theta1 = 0.0
		self.theta2 = 0.0
		self.target = np.zeros(2, dtype=np.float32)

		self.step_count = 0
		self.max_steps = 500 ###############################

		self.render_mode = render_mode


#-----------------------------------------------
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)
		self.step_count = 0

# angles de depart
		self.theta1 = self.np_random.uniform(-np.pi, np.pi)
		self.theta2 = self.np_random.uniform(-np.pi, np.pi)

# cible
#		radius = self.np_random.uniform(0.1, self.max_reach)
#		angle = self.np_random.uniform(-np.pi, np.pi)
#		self.target = np.abs(np.array([radius*np.cos(angle), radius*np.sin(angle)], dtype=np.float32))

#		self.target = [1, 0.5]
		

		self.target = self.np_random.uniform(0.3, 1.4, size=2)

		
		return self._get_obs(), {}


# --------------------------------------------------------
	def step(self, action):
		self.step_count += 1
		action[:]*=0.1

# Mise à jour des angles
		self.theta1 = np.clip(self.theta1 + action[0], self.theta_min, self.theta_max)
		self.theta2 = np.clip(self.theta2 + action[1], self.theta_min, self.theta_max)

		eff = self.forward_kinematics(self.theta1, self.theta2)

# --- Distance à la cible ---
		dist = np.linalg.norm(eff - self.target)

# --- Récompense principale : plus proche = mieux ---
		reward = -dist/4

# --- Bonus de réussite ---
		if dist < 0.05:
			reward += 1.0

# --- Mémorise la distance pour le prochain step ---
		self.prev_dist = dist


# Succès
		done = False
		if dist < 0.05:
			done = True 
		truncated = self.step_count >= self.max_steps

		return self._get_obs(), reward, done, truncated, {"effector_pos": eff}


#-----------------------------------------------------------
	def _get_obs(self):
		eff = self.forward_kinematics(self.theta1, self.theta2)
		dx, dy = self.target - eff

		return np.array([self.theta1/3.14, self.theta2/3.14, self.target[0],self.target[1]], dtype=np.float32)


# ----------------------------------------------------------------------
	def forward_kinematics(self, t1, t2):
		x = self.l1*np.cos(t1) + self.l2*np.cos(t1+t2)
		y = self.l1*np.sin(t1) + self.l2*np.sin(t1+t2)

		return np.array([x, y], dtype=np.float32)


# -----------------------------------------------------------------------------
	def render(self):
		if self.render_mode != "human":
			return

		import matplotlib.pyplot as plt

		plt.clf()
		j1 = np.array([0, 0])
		j2 = np.array([self.l1*np.cos(self.theta1), self.l1*np.sin(self.theta1)])
		eff = self.forward_kinematics(self.theta1, self.theta2)

		plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'r-', lw=4)
		plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'b-', lw=4)

		# Cible
		plt.plot(self.target[0], self.target[1], 'go', markersize=12)

		plt.xlim(-2.2, 2.2)
		plt.ylim(-2.2, 2.2)
		plt.gca().set_aspect("equal")
		plt.pause(0.001)

