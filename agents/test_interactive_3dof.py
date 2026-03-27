import time
import numpy as np
import matplotlib.pyplot as plt
from envs.arm3dof_env import Arm3DoFEnv
from stable_baselines3 import PPO

# Charger le modèle entraîné
model_path = "models/ppo_reach_3dof/best_model.zip" 
model = PPO.load(model_path)

# Créer l'environnement
env = Arm3DoFEnv(render_mode="human")

# Fonction pour définir la cible au clic
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        env.target = np.array([event.xdata, event.ydata], dtype=np.float32)
        print(f"Cible définie en ({event.xdata:.2f}, {event.ydata:.2f})")


obs, _ = env.reset()
env.render()

# Connecter callback sur la figure
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', onclick)

# Boucle principale interactive
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    env.render()
    time.sleep(0.02)

    if done:
        # Sauvegarder la cible cliquée avant le reset
        clicked_target = env.target.copy()
        obs, _ = env.reset()
        # Restaurer la cible définie par le clic
        env.target = clicked_target
