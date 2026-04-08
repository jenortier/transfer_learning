import numpy as np
from envs.arm2dof_env import Arm2DoFEnv


class Arm2DoFPersistentEnv(Arm2DoFEnv):
    """
    Environnement 2DoF dérivé qui ne réinitialise PAS la position du bras
    et de la cible lorsque celle-ci est atteinte. L'effecteur reste sur la cible.
    """

    def __init__(self, render_mode=None, **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self._target_reached = False

    def reset(self, seed=None, options=None):
        """Reset initial : initialise aléatoirement comme la classe parente"""
        obs, info = super().reset(seed=seed, options=options)
        self._target_reached = False
        return obs, info

    def step(self, action):
        """
        Surcharge de step() : si la cible est déjà atteinte, on reste en place
        sans réinitialiser aléatoirement.
        """
        # Si la cible a déjà été atteinte, on maintient la position
        if self._target_reached:
            obs = self._get_obs()
            # Petite récompense pour maintenir la position
            reward = 0.1
            self._target_reached = False
            return obs, reward, False, False, {
                "target_reached": True,
                "dist": float(np.linalg.norm(
                    self.forward_kinematics(self.theta1, self.theta2) - self.target
                )),
            }

        # Sinon, comportement normal
        obs, reward, terminated, truncated, info = super().step(action)

        # Si on vient d'atteindre la cible, on marque le flag
        if info.get("target_reached", False):
            self._target_reached = True
            # On ne termine pas l'épisode, on continue
            terminated = False

        return obs, reward, terminated, truncated, info
