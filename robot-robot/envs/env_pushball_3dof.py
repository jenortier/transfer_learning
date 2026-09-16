import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from envs.arm_3dof import Arm3DoF

class PushBallEnv_3dof(Arm3DoF):
    def __init__(self, render_mode=None, max_steps = 100):
        super().__init__(render_mode=render_mode)

        # --- Hyperparamètres identiques à 2DoF ---
        self.omega_max = 2.0       # vitesse max
        self.dt        = 0.05      # pas de simulation
        self.max_steps = max_steps

        # --- Contact ---
        self.eff_radius        = 0.05
        self.ball_radius       = 0.10
        self.contact_threshold = self.eff_radius + self.ball_radius

        # --- Succès ---
        self.epsilon = 0.10

        # --- Poids des récompenses identiques à 2DoF ---
        self.w_near        = 0.6      # proximité bras-balle
        self.w_ctrl        = 0.03     # contrôle action
        self.bonus_success = 400.0    # bonus succès

        # --- Observation ---
        # 8D bras (hérité de 3DoF) + 4D tâche = 12D
        arm_high  = np.ones(self.arm_obs_size, dtype=np.float32)
        task_high = np.ones(4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([-arm_high, -task_high]),
            high=np.concatenate([ arm_high,  task_high]),
            dtype=np.float32,
        )

        # --- État ---
        self.ball    = np.zeros(2)
        self.target  = np.zeros(2)
        self.prev_dist_ball_target = 0.0
        self.prev_dist_eff_ball    = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # --- Cible aléatoire identique à 2DoF ---
        r_t = float(self.np_random.uniform(0.4 * self.max_reach, 0.75 * self.max_reach))
        a_t = float(self.np_random.uniform(-np.pi, np.pi))
        self.target = np.array([r_t * np.cos(a_t), r_t * np.sin(a_t)])

        # --- Balle à distance minimale de 0.3m ---
        for _ in range(1000):
            r_b = float(self.np_random.uniform(0.3 * self.max_reach, 0.75 * self.max_reach))
            a_b = float(self.np_random.uniform(-np.pi, np.pi))
            candidate = np.array([r_b * np.cos(a_b), r_b * np.sin(a_b)])
            if np.linalg.norm(candidate - self.target) >= 0.3:
                self.ball = candidate
                break
        else:
            self.ball = self.target + np.array([0.4, 0.0])

        # --- Distance initiale ---
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        self.prev_dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        self.prev_dist_eff_ball    = float(np.linalg.norm(eff - self.ball))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action        = np.clip(action, -1.0, 1.0)
        action_scaled = action * 0.1

        # --- Mise à jour angles identique à 2DoF mais pour 3 articulations ---
        new_theta1 = self.theta1 + action[0] * self.omega_max * self.dt
        new_theta2 = self.theta2 + action[1] * self.omega_max * self.dt
        new_theta3 = self.theta3 + action[2] * self.omega_max * self.dt

        self.dtheta1 = (new_theta1 - self.theta1) / self.dt
        self.dtheta2 = (new_theta2 - self.theta2) / self.dt
        self.dtheta3 = (new_theta3 - self.theta3) / self.dt

        self.theta1 = new_theta1
        self.theta2 = new_theta2
        self.theta3 = new_theta3

        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        # --- Contact identique à 2DoF ---
        vec_eff_ball = self.ball - eff
        contact_dist = float(np.linalg.norm(vec_eff_ball))
        if contact_dist < self.contact_threshold and contact_dist > 1e-6:
            normal = vec_eff_ball / contact_dist
            self.ball += normal * (self.contact_threshold - contact_dist)

        # --- Distances ---
        dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        dist_eff_ball    = float(np.linalg.norm(self.ball - eff))
        
        ##if dist_eff_ball < 0.25:
        ##    dist_eff_ball = 0.1

        # --- Alignement ---
        v_eb = self.ball - eff
        v_bt = self.target - self.ball
        alignment = float(np.dot(v_eb / np.linalg.norm(v_eb), v_bt / np.linalg.norm(v_bt))) \
            if np.linalg.norm(v_eb) > 1e-3 and np.linalg.norm(v_bt) > 1e-3 else 0.0

        # --- Récompense (identique à 2DoF) ---
        reward_near = -self.w_near * dist_eff_ball
        reward_ctrl = -self.w_ctrl * np.sum(np.square(action))
        reward = reward_near + reward_ctrl
        progress_ball_target = self.prev_dist_ball_target - dist_ball_target
        reward += 30.0 * progress_ball_target
        reward += 0.6 * alignment

        success  = dist_ball_target < self.epsilon
        ball_out = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05
        if success:
            reward += self.bonus_success
        if ball_out:
            reward -= 5.0

        self.prev_dist_ball_target = dist_ball_target
        self.prev_dist_eff_ball    = dist_eff_ball

        terminated = success or ball_out
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "target_reached":       success,
            "dist_ball_target":     dist_ball_target,
            "dist_eff_ball":        dist_eff_ball,
            "alignment":            alignment,
            "progress_ball_target": progress_ball_target,
            "theta1": self.theta1,
            "theta2": self.theta2,
            "theta3": self.theta3,
            "ball":   self.ball.copy(),
        }

    # ------------------------------------------------------------------
    def _get_obs(self):
        arm_obs = super()._get_obs()
        task_obs = np.array([
            np.clip(self.ball[0] / self.max_reach, -1., 1.),
            np.clip(self.ball[1] / self.max_reach, -1., 1.),
            self.target[0] / self.max_reach,
            self.target[1] / self.max_reach,
        ], dtype=np.float32)
        return np.concatenate([arm_obs, task_obs])
    def render(self):
        if self.render_mode != "human":
            return

        plt.clf()

        # Positions des articulations
        j1 = np.array([self.l1 * np.cos(self.theta1),
                       self.l1 * np.sin(self.theta1)])
        j2 = j1 + np.array([self.l2 * np.cos(self.theta1 + self.theta2),
                            self.l2 * np.sin(self.theta1 + self.theta2)])
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)

        # Tracé des segments du bras
        plt.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4, label='Link 1')
        plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'g-', lw=4, label='Link 2')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'b-', lw=4, label='Link 3')

        # Effecteur
        plt.gca().add_patch(plt.Circle(eff, self.eff_radius * 2, color='red', alpha=0.6))

        # Balle
        ball_out   = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05
        ball_color = 'red' if ball_out else 'dodgerblue'
        plt.gca().add_patch(plt.Circle(self.ball, self.ball_radius * 1.5, color=ball_color, alpha=0.6))

        # Cible
        plt.plot(self.target[0], self.target[1], 'o', markersize=18, label='Target')
        plt.gca().add_patch(plt.Circle(self.target, self.epsilon, color='green', fill=False, linestyle='--', lw=1.5))

        # Flèche balle → cible
        vec = self.target - self.ball
        if np.linalg.norm(vec) > 1e-3:
            v = vec / np.linalg.norm(vec) * 0.3
            plt.arrow(self.ball[0], self.ball[1], v[0], v[1],
                      head_width=0.08, head_length=0.04, fc='green', ec='green', alpha=0.4)

        dist_ball_target = np.linalg.norm(self.ball - self.target)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect("equal")
        plt.title(f"Step {self.step_count} | d(ball,tgt)={dist_ball_target:.3f} m"
                  + (" | ⚠️ OUT" if ball_out else ""))
        plt.legend(loc="upper right", fontsize=8)
        plt.pause(0.001)
