# record_pushball_episodes.py
import os
import numpy as np
import pickle
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.env_pushball_2dof import PushBallEnv_2dof


def record_episodes(
    model_path,
    vec_normalize_path,
    num_episodes=1000,
    max_steps_per_episode=200,
    output_dir="./data/demonstrations/pushball_2dof",
    render=False,
):
    """
    Enregistre des épisodes d'un agent PPO pré-entraîné.
    
    Args:
        model_path: Chemin vers le modèle PPO
        vec_normalize_path: Chemin vers le fichier VecNormalize
        num_episodes: Nombre d'épisodes à enregistrer
        max_steps_per_episode: Nombre max de pas par épisode
        output_dir: Dossier de sortie
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le modèle
    model = PPO.load(model_path)
    
    # Créer l'environnement
    env = PushBallEnv_2dof(None, max_steps=max_steps_per_episode)
    
    # Encapsuler avec VecNormalize
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False
    env.norm_reward = False  # Désactiver la normalisation des récompenses
    
    episodes_data = []
    
    print(f"Enregistrement de {num_episodes} épisodes...")

    episode_idx = 0
    
    while episode_idx < num_episodes:
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Stocker les données de l'épisode
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_infos = []
        
        while not done and steps < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=False)
            obs_next, reward, done, info = env.step(action)
            
            episode_obs.append(obs.flatten())
            episode_actions.append(action.flatten())
            episode_rewards.append(reward)
            episode_dones.append(done)
            episode_infos.append(info[0] if isinstance(info, list) else info)
            
            episode_reward += reward
            obs = obs_next
            steps += 1
            
            '''
            # Affichage de progression
            if episode_idx % 50 == 0 and steps % 10 == 0:
                print(f"Episode {episode_idx+1}/{num_episodes}, Step {steps}")#, Reward: {episode_reward:.2f}")
            '''
        
        # Stocker l'épisode
        episode = {
            "obs": np.array(episode_obs, dtype=np.float32),
            "actions": np.array(episode_actions, dtype=np.float32),
            "rewards": np.array(episode_rewards, dtype=np.float32),
            "dones": np.array(episode_dones, dtype=bool),
            "infos": episode_infos,
            "total_reward": episode_reward,
            "length": steps,
        }
        success = info[0].get("target_reached", False)
        if success:      # enregistre uniquement les success
            episodes_data.append(episode)
            episode_idx += 1  # Incrémenter l'index de l'épisode uniquement si succès
        
        # Sauvegarder périodiquement
        if (episode_idx + 1) % 100 == 0:
            print(f"Progress: {episode_idx+1}/{num_episodes} épisodes enregistrés")
    
    # Sauvegarder toutes les données
    output_file = os.path.join(output_dir, "demonstrations.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(episodes_data, f)
    
    # Statistiques
    total_rewards = [ep["total_reward"] for ep in episodes_data]
    print(f"\n--- Enregistrement terminé ---")
    print(f"Épisodes enregistrés: {len(episodes_data)}")
    print(f"Récompense moyenne: {np.mean(total_rewards):.2f}")
    print(f"Récompense min: {np.min(total_rewards):.2f}")
    print(f"Récompense max: {np.max(total_rewards):.2f}")
    print(f"Longueur moyenne des épisodes: {np.mean([ep['length'] for ep in episodes_data]):.2f}")
    print(f"Fichier sauvegardé: {output_file}")
    
    env.close()
    return episodes_data


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "./database/models/ppo_pushball_2dof_1/ppo_pushball_final.zip"
    VEC_NORMALIZE_PATH = "./database/models/ppo_pushball_2dof_1/vec_normalize.pkl"
    NUM_EPISODES = 2000
    OUTPUT_DIR = "./database/pushball_2dof"
    
    record_episodes(
        model_path=MODEL_PATH,
        vec_normalize_path=VEC_NORMALIZE_PATH,
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR,
        render=False
    )