import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Arm3DoFEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    ALPHA        = 10.0
    LAMBDA_CTRL  = 0.05
    R_SUCCESS    = 5.0    # aligné avec arm2dof
    EPSILON      = 0.05
    DELTA_MAX    = 0.1

    def __init__(self, render_mode=None,
                 alpha=ALPHA,
                 lambda_ctrl=LAMBDA_CTRL,
                 r_success=R_SUCCESS,
                 epsilon=EPSILON,
                 delta_max=DELTA_MAX):
        super().__init__()

        self.l1 = 1.0
        self.l2 = 1.0
        self.l3 = 1.0
        self.max_reach = self.l1 + self.l2 + self.l3   # 3.0 m

        self.theta1_min = -np.pi
        self.theta1_max =  np.pi
        self.theta2_min = -3.0
        self.theta2_max =  3.0
        self.theta3_min = -3.0
        self.theta3_max =  3.0

        self.alpha       = alpha
        self.lambda_ctrl = lambda_ctrl
        self.r_success   = r_success
        self.epsilon     = epsilon
        self.delta_max   = delta_max

        self.max_steps   = 200
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # [q1/pi, q2/pi, q3/pi, dx/max, dy/max, eff_x/max, eff_y/max,
        #  tgt_x/max, tgt_y/max, dist/max]  → 10 obs
        obs_low  = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0], dtype=np.float32)
        obs_high = np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.theta1     = 0.0
        self.theta2     = 0.0
        self.theta3     = 0.0
        self.target     = np.zeros(2, dtype=np.float32)
        self.prev_dist  = 0.0
        self.step_count = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.theta1 = self.np_random.uniform(-np.pi, np.pi)
        self.theta2 = self.np_random.uniform(-3.0, 3.0)
        self.theta3 = self.np_random.uniform(-3.0, 3.0)

        radius = self.np_random.uniform(0.1, self.max_reach)
        angle  = self.np_random.uniform(-np.pi, np.pi)
        self.target = np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
        )

        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        self.prev_dist = float(np.linalg.norm(eff - self.target))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        delta = np.clip(action, -1.0, 1.0) * self.delta_max
        new_t1 = np.clip(self.theta1 + delta[0], self.theta1_min, self.theta1_max)
        new_t2 = np.clip(self.theta2 + delta[1], self.theta2_min, self.theta2_max)
        new_t3 = np.clip(self.theta3 + delta[2], self.theta3_min, self.theta3_max)

        # Pénalise les actions bloquées aux limites
        at_limit = (
            float(abs(new_t1 - self.theta1) < 1e-6 and abs(delta[0]) > 0.01) +
            float(abs(new_t2 - self.theta2) < 1e-6 and abs(delta[1]) > 0.01) +
            float(abs(new_t3 - self.theta3) < 1e-6 and abs(delta[2]) > 0.01)
        )

        self.theta1 = new_t1
        self.theta2 = new_t2
        self.theta3 = new_t3

        eff  = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        dist = float(np.linalg.norm(eff - self.target))

        # Reward normalisée (identique à arm2dof)
        progress = self.prev_dist - dist
        reward   = self.alpha * progress / self.max_reach
        reward  -= self.lambda_ctrl * float(np.dot(action, action))
        reward  -= 0.02 * at_limit
        reward  -= 0.1 * np.tanh(dist) 

        success = dist < self.epsilon
        if success:
            reward += self.r_success

        self.prev_dist = dist
        terminated = success
        truncated  = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "target_reached": success,
            "dist": dist,
        }

    # ------------------------------------------------------------------
    def _get_obs(self):
        eff    = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        dx, dy = self.target - eff
        dist   = float(np.linalg.norm(eff - self.target))
        return np.array([
            self.theta1 / np.pi,
            self.theta2 / np.pi,
            self.theta3 / np.pi,
            dx / self.max_reach,
            dy / self.max_reach,
            eff[0] / self.max_reach,
            eff[1] / self.max_reach,
            self.target[0] / self.max_reach,
            self.target[1] / self.max_reach,
            dist / self.max_reach,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def forward_kinematics(self, t1, t2, t3):
        x = (  self.l1 * np.cos(t1)
             + self.l2 * np.cos(t1 + t2)
             + self.l3 * np.cos(t1 + t2 + t3))
        y = (  self.l1 * np.sin(t1)
             + self.l2 * np.sin(t1 + t2)
             + self.l3 * np.sin(t1 + t2 + t3))
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        import matplotlib.pyplot as plt
        plt.clf()

        j1  = np.array([self.l1 * np.cos(self.theta1),
                         self.l1 * np.sin(self.theta1)])
        j2  = j1 + np.array([self.l2 * np.cos(self.theta1 + self.theta2),
                               self.l2 * np.sin(self.theta1 + self.theta2)])
        eff = self.forward_kinematics(self.theta1, self.theta2, self.theta3)
        dist = np.linalg.norm(eff - self.target)

        plt.plot([0,     j1[0]], [0,     j1[1]], 'r-', lw=4)
        plt.plot([j1[0], j2[0]], [j1[1], j2[1]], 'b-', lw=4)
        plt.plot([j2[0], eff[0]], [j2[1], eff[1]], 'm-', lw=4)

        color = 'lime' if dist < self.epsilon else 'red'
        plt.plot(eff[0],         eff[1],         'o', color=color, markersize=10, label="End-effector")
        plt.plot(self.target[0], self.target[1], 'go', markersize=12,             label="Target")

        plt.xlim(-3.2, 3.2)
        plt.ylim(-3.2, 3.2)
        plt.gca().set_aspect("equal")
        plt.title(
            f"Step {self.step_count}  |  d={dist:.3f}m  |  "
            f"θ1={np.degrees(self.theta1):.0f}°  "
            f"θ2={np.degrees(self.theta2):.0f}°  "
            f"θ3={np.degrees(self.theta3):.0f}°"
        )
        plt.legend(loc="upper left")
        plt.pause(0.001)
