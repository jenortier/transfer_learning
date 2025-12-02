import numpy as np
import matplotlib.pyplot as plt
from envs.arm2d_env import Arm2DEnv
from stable_baselines3 import PPO

# --- Charger le modèle entraîné ---
model_path = "models/ppo_arm2d.zip"  # chemin vers le modèle sauvegardé
model = PPO.load(model_path)

# --- Créer un environnement pour le test ---
env = Arm2DEnv(render_mode="human")

# --- Configurer Matplotlib pour rendu en temps réel ---
plt.ion()
fig, ax = plt.subplots()

def render_arm(env, ax):
    """
    Fonction pour dessiner le bras et la cible.
    """
    ax.clear()
    # positions articulations
    joint1 = np.array([env.l1*np.cos(env.theta1), env.l1*np.sin(env.theta1)])
    joint2 = env.forward_kinematics(env.theta1, env.theta2)
    # segments du bras
    ax.plot([0, joint1[0]], [0, joint1[1]], 'r-', lw=4)
    ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'b-', lw=4)
    # cible
    ax.plot(env.target[0], env.target[1], 'go', markersize=10)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.draw()
    plt.pause(0.001)

# --- Tester plusieurs épisodes ---
num_tests = 5
max_steps = 150

for test_ep in range(num_tests):
    # Réinitialiser l'environnement avec positions aléatoires
    obs, _ = env.reset()
    total_reward = 0
    
    # Afficher l'épisode
    print(f"Test épisode {test_ep+1} : cible = {env.target}")

    for step in range(max_steps):
        # Choisir l'action selon le modèle entraîné
        action, _ = model.predict(obs, deterministic=True)
        #print(obs)
        #rint(action)
        #action[:]=0
        obs, reward, done, truncated, info = env.step(action)
        #print(reward)
        total_reward += reward

        # Affichage en temps réel
        render_arm(env, ax)

    if done:
        print(f"  - Cible atteinte ! Récompense = {total_reward:.2f}")
            
    else:
        print(f"  - Échec après {max_steps} steps. Récompense = {total_reward:.2f}")

plt.ioff()
plt.show()

