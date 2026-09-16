import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from envs.arm_2dof import Arm2DoF


class PushBallEnv_2dof(Arm2DoF):
    def __init__(self, render_mode=None, max_steps = 100):
        super().__init__(render_mode=render_mode)

        # Dynamique spécifique (commande en vitesse)
        self.omega_max = 2.0
        self.dt        = 0.05
        self.max_steps = max_steps

        # Physique du contact
        self.eff_radius        = 0.05
        self.ball_radius       = 0.10
        self.contact_threshold = self.eff_radius + self.ball_radius

        # Succès
        self.epsilon = 0.10

        # Poids des récompenses
        self.w_near        = 0.6
        self.w_ctrl        = 0.03
        self.bonus_success = 400.0 	## 30.0->100.0

        # Observation 10D = 6D bras (hérité) + 4D tâche
        # Indices tâche (arm_obs_size + i) :
        #   0  ball_x / max_reach    [-1, 1]  (clippé)
        #   1  ball_y / max_reach    [-1, 1]  (clippé)
        #   2  tgt_x  / max_reach    [-1, 1]
        #   3  tgt_y  / max_reach    [-1, 1]
        arm_high  = np.ones(self.arm_obs_size, dtype=np.float32)
        task_high = np.ones(4, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([-arm_high, -task_high]),
            high=np.concatenate([ arm_high,  task_high]),
            dtype=np.float32,
        )

        # État spécifique à la tâche
        self.ball   = np.zeros(2)
        self.target = np.zeros(2)
        self.prev_dist_ball_target = 0.0
        self.prev_dist_eff_ball    = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)   # angles aléatoires, step_count=0

        # Cible
        r_t = float(self.np_random.uniform(0.4 * self.max_reach, 0.75 * self.max_reach))
        a_t = float(self.np_random.uniform(-np.pi, np.pi))
        self.target = np.array([r_t * np.cos(a_t), r_t * np.sin(a_t)])

        # Balle à au moins 0.3 m de la cible
        for _ in range(1000):
            r_b = float(self.np_random.uniform(0.3 * self.max_reach, 0.75 * self.max_reach))
            a_b = float(self.np_random.uniform(-np.pi, np.pi))
            candidate = np.array([r_b * np.cos(a_b), r_b * np.sin(a_b)])
            if np.linalg.norm(candidate - self.target) >= 0.3:
                self.ball = candidate
                break
        else:
            self.ball = self.target + np.array([0.4, 0.0])

        eff = self.forward_kinematics(self.theta1, self.theta2)
        self.prev_dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        self.prev_dist_eff_ball    = float(np.linalg.norm(eff - self.ball))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action        = np.clip(action, -1.0, 1.0)
        action_scaled = action * 0.1

        # --- Cinématique (commande en vitesse) ---
        new_theta1 = self.theta1 + float(action[0]) * self.omega_max * self.dt
        new_theta2 = self.theta2 + float(action[1]) * self.omega_max * self.dt

        self.dtheta1 = (new_theta1 - self.theta1) / self.dt
        self.dtheta2 = (new_theta2 - self.theta2) / self.dt

        self.theta1 = new_theta1
        self.theta2 = new_theta2

        eff = self.forward_kinematics(self.theta1, self.theta2)

        # --- Contact effecteur → balle ---
        vec_eff_ball = self.ball - eff
        contact_dist = float(np.linalg.norm(vec_eff_ball))
        if contact_dist < self.contact_threshold and contact_dist > 1e-6:
            normal    = vec_eff_ball / contact_dist
            self.ball = self.ball + normal * (self.contact_threshold - contact_dist)

        # --- Distances ---
        dist_ball_target = float(np.linalg.norm(self.ball - self.target))
        dist_eff_ball    = float(np.linalg.norm(self.ball - eff))

        ##if dist_eff_ball < 0.25:
        ##    dist_eff_ball = 0.1

        # --- Alignement ---
        v_eb = self.ball - eff
        v_bt = self.target - self.ball
        if np.linalg.norm(v_eb) > 1e-3 and np.linalg.norm(v_bt) > 1e-3:
            alignment = float(np.dot(v_eb / np.linalg.norm(v_eb),
                                     v_bt / np.linalg.norm(v_bt)))
        else:
            alignment = 0.0

        # --- Récompense ---
        reward_near = -self.w_near * dist_eff_ball
        reward_ctrl = -self.w_ctrl * float(np.sum(np.square(action)))
        reward = reward_near + reward_ctrl

        progress_ball_target = self.prev_dist_ball_target - dist_ball_target
##            if progress_ball_target > 0:
        ##penalty_neg_progress = 2.5 if progress_ball_target < 0.0 else 1.0
        reward += 30.0 *  progress_ball_target 	## 50.0->30.0

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
            "ball":   self.ball.copy(),
        }

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Observation 10D.

        [0:6]  Observation bras normalisée (héritée de Arm2DoF)
               → voir Arm2DoF._get_obs() pour le détail

        [6:10] Observation tâche push-ball :
        Index  Grandeur         Normalisation    Plage
        ───────────────────────────────────────────────
          6    ball_x           / max_reach      [-1,   1  ]
          7    ball_y           / max_reach      [-1,   1  ]
          8    tgt_x            / max_reach      [-1,   1  ]
          9    tgt_y            / max_reach      [-1,   1  ]
        """
        arm_obs  = super()._get_obs()
        task_obs = np.array([
            np.clip(self.ball[0]   / self.max_reach, -1., 1.),
            np.clip(self.ball[1]   / self.max_reach, -1., 1.),
            self.target[0] / self.max_reach,
            self.target[1] / self.max_reach,
        ], dtype=np.float32)
        return np.concatenate([arm_obs, task_obs])

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        plt.clf()
        j2  = np.array([self.l1 * np.cos(self.theta1),
                        self.l1 * np.sin(self.theta1)])
        eff = self.forward_kinematics(self.theta1, self.theta2)

        plt.plot([0, j2[0]], [0, j2[1]], 'r-', lw=4, label='Link 1')
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'b-', lw=4, label='Link 2')
        plt.gca().add_patch(plt.Circle(eff, self.eff_radius * 2,
                                       color='red', alpha=0.6))

        ball_out   = float(np.linalg.norm(self.ball)) > self.max_reach * 1.05
        ball_color = 'red' if ball_out else 'dodgerblue'
        plt.gca().add_patch(plt.Circle(self.ball, self.ball_radius * 1.5,
                                       color=ball_color, alpha=0.6))

        plt.plot(self.target[0], self.target[1], 'o', markersize=18, label='Target')
        plt.gca().add_patch(plt.Circle(self.target, self.epsilon,
                                       color='green', fill=False, linestyle='--', lw=1.5))

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
