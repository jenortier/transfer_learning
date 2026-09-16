# generate_aligned_trajectories.py
import os
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_reaching_2dof import ReachingEnv_2dof
from envs.env_reaching_3dof import ReachingEnv_3dof
from envs.reaching_alignment_2to3 import ReachingAlignment2to3Env


# ==========================================================
# CPU SETUP
# ==========================================================
torch.set_num_threads(16)


# ==========================================================
# PARAMETERS
# ==========================================================
N_EPISODES = 10000
MAX_STEPS = 50


# ==========================================================
# LOAD MODELS
# ==========================================================
def load_expert_2dof():
    """Charge le modèle expert 2DoF"""
    model_path = "./data/models/ppo_reach_2dof_1/ppo_reach_final.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expert 2DoF model not found at {model_path}")
    return PPO.load(model_path)


def load_student_3dof():
    """Charge le modèle student 3DoF (après alignment 2→3)"""
    model_path = "./data/models/align_2to3_1/align_2to3_final.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Student 3DoF model not found at {model_path}")
    return PPO.load(model_path)


# ==========================================================
# COLLECT TRAJECTORIES
# ==========================================================
def collect_aligned_trajectories(n_episodes=10000, max_steps=50):
    """Collecte les paires (s2, s3) à partir de l'environnement d'alignement"""
    
    print("Loading models...")
    expert_2dof = load_expert_2dof()
    student_3dof = load_student_3dof()
    
    print("Creating environments...")
    env2 = ReachingEnv_2dof()
    env3 = ReachingEnv_3dof()
    
    env = ReachingAlignment2to3Env(
        expert_env=env2,
        expert_policy=expert_2dof,
        lambda_pos=0.5,
        lambda_vel=0.1,
        deterministic_expert=True,
        render_mode=None
    )
    
    print(f"Collecting {n_episodes} episodes...")
    
    s2_list = []
    s3_list = []
    
    for episode in tqdm(range(n_episodes), desc="Collecting trajectories"):
        obs = env.reset()[0]
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action, _ = student_3dof.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extraire UNIQUEMENT les états du bras (sans la cible, etc.)
            # Pour 2DoF: arm_obs_size = 6 (positions et vitesses des articulations + position EE)
            # Pour 3DoF: arm_obs_size = 8 (positions et vitesses des articulations + position EE)
            
            # L'observation de l'expert 2DoF contient plus que l'état du bras
            # On extrait seulement les 6 premières dimensions pour 2DoF
            expert_full_obs = env.expert_env._get_obs()
            student_full_obs = env._get_obs()
            
            # Extraire seulement les états des bras
            # Pour 2DoF: les 6 premières dimensions sont [θ1/π, θ2/π, dθ1/ω_max, dθ2/ω_max, x/r, y/r]
            s2_arm = expert_full_obs[:6]  # 6 dims pour 2DoF
            
            # Pour 3DoF: les 8 premières dimensions sont [θ1/π, θ2/π, θ3/π, dθ1/ω_max, dθ2/ω_max, dθ3/ω_max, x/r, y/r]
            s3_arm = student_full_obs[:8]  # 8 dims pour 3DoF
            
            s2_list.append(s2_arm)
            s3_list.append(s3_arm)
            
            obs = next_obs
            steps += 1
    
    s2_array = np.array(s2_list, dtype=np.float32)
    s3_array = np.array(s3_list, dtype=np.float32)
    
    print(f"Collected {len(s2_array)} samples")
    print(f"s2 shape: {s2_array.shape}")  # Devrait être (N, 6)
    print(f"s3 shape: {s3_array.shape}")  # Devrait être (N, 8)
    
    return s2_array, s3_array


# ==========================================================
# SAVE DATA
# ==========================================================
def save_aligned_data(s2, s3, output_path="./data/aligned_traj_2to3.npy"):
    """Sauvegarde les données alignées dans un fichier unique"""
    
    # Concaténer s2 et s3: [s2 (6 dims) | s3 (8 dims)] -> total 14 dims
    aligned_data = np.concatenate([s2, s3], axis=1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, aligned_data)
    print(f"Data saved to {output_path}")
    print(f"Data shape: {aligned_data.shape}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    
    if not os.path.exists("./data/models/ppo_reach_2dof_1/ppo_reach_final.zip"):
        print("ERROR: Expert 2DoF model not found!")
        exit(1)
    
    if not os.path.exists("./data/models/align_2to3_1/align_2to3_final.zip"):
        print("ERROR: Student 3DoF alignment model not found!")
        print("Please run train_reaching_align2-3.py first to train the alignment.")
        exit(1)
    
    try:
        s2, s3 = collect_aligned_trajectories(N_EPISODES, MAX_STEPS)
        save_aligned_data(s2, s3)
        print("\n✅ Data generation complete!")
        print(f"Now you can run: python3 -m agents.mappers.train_mapper")
    except Exception as e:
        print(f"Error during data collection: {e}")
        import traceback
        traceback.print_exc()
