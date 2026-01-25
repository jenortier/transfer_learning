import numpy as np
import matplotlib.pyplot as plt
from envs.arm2d_env import Arm2DEnv
from stable_baselines3 import PPO

			# Charger le modèle
model_path = "models/ppo_arm2d.zip"
model = PPO.load(model_path)

			# Créer l'environnement
env = Arm2DEnv(render_mode="human")

			# Configurer Matplotlib pour rendu interactif
plt.ion()
fig, ax = plt.subplots()
target = env.target.copy()  # cible initiale

			# Fonction pour dessiner le bras et la cible
def render_arm():
    ax.clear()
    # positions articulations
    joint1 = np.array([env.l1*np.cos(env.theta1), env.l1*np.sin(env.theta1)])
    joint2 = env.forward_kinematics(env.theta1, env.theta2)
    # segments du bras
    ax.plot([0, joint1[0]], [0, joint1[1]], 'r-', lw=4)
    ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'b-', lw=4)
    # cible
    ax.plot(target[0], target[1], 'go', markersize=10)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.draw()
    plt.pause(0.001)

			# Fonction pour définir la cible au clic
def onclick(event):
    global target
    if event.xdata is not None and event.ydata is not None:
        target = np.array([event.xdata, event.ydata])
        env.target = target.copy()
        print(f"Cible définie en {target}")

fig.canvas.mpl_connect('button_press_event', onclick)

			# Boucle principale interactive 
obs, _ = env.reset()
while True:
    # Calculer l'action avec le modèle
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    render_arm()

    if done:
        # Reset l'environnement pour continuer à viser la nouvelle cible
        obs, _ = env.reset()
        env.target = target.copy()

