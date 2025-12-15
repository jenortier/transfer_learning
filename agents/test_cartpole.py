import numpy as np
import matplotlib.pyplot as plt
from envs.cartpole_env import CartPoleEnv
from stable_baselines3 import PPO

# --- Charger le modèle entraîné ---
model_path = "models/ppo_cartpole.zip"
model = PPO.load(model_path)

# --- Créer un environnement avec rendu image ---
env = CartPoleEnv(render_mode="rgb_array")

# --- Matplotlib : affichage interactif ---
plt.ion()
fig, ax = plt.subplots()
img_disp = ax.imshow(np.zeros((400, 600, 3), dtype=np.uint8))
ax.axis("off")


def render_cartpole(env, img_disp):
    """Affiche la frame retournée par render() dans matplotlib."""
    frame = env.render()  # retourne un array RGB
    img_disp.set_data(frame)
    plt.draw()
    plt.pause(0.0001)


# --- Paramètres test ---
num_tests = 5
max_steps = 100

for test_ep in range(num_tests):
    obs, _ = env.reset()
    total_reward = 0

    print(f"\n=== Test épisode {test_ep+1} ===")

    for step in range(max_steps):

        # Action choisie par le modèle
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Affichage visuel
        render_cartpole(env, img_disp)

        if done or truncated:
            break

    print(f"  → Terminé en {step+1} steps, Reward = {total_reward:.2f}")

plt.ioff()
plt.show()
env.close()

