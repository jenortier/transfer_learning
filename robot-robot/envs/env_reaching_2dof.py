import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from envs.arm_2dof import Arm2DoF


class ReachingEnv_2dof(Arm2DoF):
    ALPHA       = 10.0
    LAMBDA_CTRL = 0.05
    R_SUCCESS   = 5.0
    EPSILON     = 0.05      # 5cm
    DELTA_MAX   = 0.1

    def __init__(self, render_mode=None,
                 alpha=ALPHA, lambda_ctrl=LAMBDA_CTRL,
                 r_success=R_SUCCESS, epsilon=EPSILON,
                 delta_max=DELTA_MAX, max_steps=200):
        super().__init__(render_mode=render_mode)

        self.alpha       = alpha
        self.lambda_ctrl = lambda_ctrl
        self.r_success   = r_success
        self.epsilon     = epsilon
        self.delta_max   = delta_max
        self.max_steps   = max_steps

        # Observation 11D = 6D bras (hérité) + 5D tâche
        # Indices tâche (arm_obs_size + i) :
        #   0  dx/r        erreur cartésienne x     [-2, 2]
        #   1  dy/r        erreur cartésienne y     [-2, 2]
        #   2  tgt_x/r     cible x                  [-1, 1]
        #   3  tgt_y/r     cible y                  [-1, 1]
        #   4  dist/r      distance absolue          [0, 2]
        arm_high  = np.ones(self.arm_obs_size, dtype=np.float32)
        task_low  = np.array([-1., -1., -1., -1.,  0.], dtype=np.float32)
        task_high = np.array([ 1.,  1.,  1.,  1.,  1.], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([-arm_high, task_low]),
            high=np.concatenate([ arm_high, task_high]),
            dtype=np.float32,
        )

        self.target    = np.zeros(2, dtype=np.float32)
        self.prev_dist = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)   # réinitialise le bras
        radius = self.np_random.uniform(0.15, self.max_reach)
        angle  = self.np_random.uniform(-np.pi, np.pi)
        self.target = np.array([radius * np.cos(angle),
                                radius * np.sin(angle)], dtype=np.float32)
        eff = self.forward_kinematics(self.theta1, self.theta2)
        self.prev_dist = float(np.linalg.norm(eff - self.target))
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        delta = np.clip(action, -1.0, 1.0) * self.delta_max
        new_t1 = self.theta1 + delta[0]
        new_t2 = self.theta2 + delta[1]

        # dtheta stocké comme vitesse angulaire (rad/s), identique à PushBall,
        # pour que le bloc arm_obs soit cohérent entre les deux tâches.
        # Plage : delta/dt = 0.1/0.05 = 2 rad/s → normalisé par omega_max → [-1, 1]
        self.dtheta1 = (new_t1 - self.theta1) / self.dt
        self.dtheta2 = (new_t2 - self.theta2) / self.dt
        self.theta1  = new_t1
        self.theta2  = new_t2

        eff  = self.forward_kinematics(self.theta1, self.theta2)
        dist = float(np.linalg.norm(eff - self.target))

        progress = self.prev_dist - dist
        reward  = self.alpha * progress / self.max_reach		# ALPHA       = 10.0
        reward -= self.lambda_ctrl * float(np.dot(action, action))	# LAMBDA_CTRL = 0.05

        success = dist < self.epsilon					# EPSILON     = 0.05  -> 5cm
        if success:
            reward += self.r_success					# R_SUCCESS   = 5.0

        self.prev_dist = dist
        terminated = success
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "target_reached": success,
            "dist": dist,
            "theta1": self.theta1,
            "theta2": self.theta2,
        }

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Observation 11D.

        [0:6]  Observation bras normalisée (héritée de Arm2DoF)
               → voir Arm2DoF._get_obs() pour le détail

        [6:11] Observation tâche reaching :
        Index  Grandeur         Normalisation      Plage
        ─────────────────────────────────────────────────
          6    dx (tgt-eff x)   / (2·max_reach)   [-1, 1]
          7    dy (tgt-eff y)   / (2·max_reach)   [-1, 1]
          8    tgt_x            / max_reach        [-1, 1]
          9    tgt_y            / max_reach        [-1, 1]
         10    dist eff-tgt     / (2·max_reach)   [ 0, 1]
        """
        arm_obs = super()._get_obs()
        eff     = self.forward_kinematics(self.theta1, self.theta2)
        dx, dy  = self.target - eff
        dist    = float(np.linalg.norm(eff - self.target))
        norm2   = 2.0 * self.max_reach
        task_obs = np.array([
            np.clip(dx / norm2, -1., 1.),
            np.clip(dy / norm2, -1., 1.),
            self.target[0] / self.max_reach,
            self.target[1] / self.max_reach,
            dist / norm2,
        ], dtype=np.float32)
        return np.concatenate([arm_obs, task_obs])

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        plt.clf()

        j1  = np.array([self.l1 * np.cos(self.theta1),
                        self.l1 * np.sin(self.theta1)])
        eff = self.forward_kinematics(self.theta1, self.theta2)
        dist = np.linalg.norm(eff - self.target)

        plt.plot([0, j1[0]], [0, j1[1]], 'r-', lw=4)
        plt.plot([j1[0], eff[0]], [j1[1], eff[1]], 'b-', lw=4)

        color = 'lime' if dist < self.epsilon else 'red'
        plt.plot(eff[0], eff[1], 'o', color=color, markersize=10, label="End-effector")
        plt.plot(self.target[0], self.target[1], 'go', markersize=12, label="Target")

        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.gca().set_aspect("equal")
        plt.title(
            f"Step {self.step_count}  |  d={dist:.3f}m  |  "
            f"θ1={np.degrees(self.theta1):.0f}°  θ2={np.degrees(self.theta2):.0f}°"
        )
        plt.legend(loc="upper left")
        plt.pause(0.001)
