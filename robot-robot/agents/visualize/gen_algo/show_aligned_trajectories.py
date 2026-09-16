"""
Visualisation des trajectoires alignées 2-DoF et 3-DoF à partir des fichiers générés
par l'algorithme génétique d'Antonio.

Lecture des fichiers TXT dans data/DIRECT_GEN_ALGO/ et affichage des trajectoires
des effecteurs des deux robots en parallèle.

Usage:
    python -m lsunn.visualization.show_aligned_trajectories --file 2dof_file.txt --file3 3dof_file.txt
    python -m lsunn.visualization.show_aligned_trajectories --idx 0
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation

# Dimensions
MAX_REACH = 3.0
OMEGA_MAX = 2.0
DT = 0.05


def normalize_angle(angle_rad):
    """Normalize angle to [-π, π] then divide by π → [-1, 1]"""
    wrapped = ((angle_rad + np.pi) % (2 * np.pi)) - np.pi
    return wrapped / np.pi


def denormalize_angle(angle_norm):
    """Convert normalized angle back to radians."""
    return angle_norm * np.pi


def normalize_velocity(vel_rad_s):
    """Normalize velocity by ω_max → [-1, 1]"""
    return np.clip(vel_rad_s / OMEGA_MAX, -1.0, 1.0)


def denormalize_velocity(vel_norm):
    """Convert normalized velocity back to rad/s."""
    return vel_norm * OMEGA_MAX


def normalize_position(pos_m):
    """Normalize position by max_reach → [-1, 1]"""
    return np.clip(pos_m / MAX_REACH, -1.0, 1.0)


def denormalize_position(pos_norm):
    """Convert normalized position back to meters."""
    return pos_norm * MAX_REACH


def load_2dof_txt(file_path):
    """Load 2-DoF trajectory from TXT file."""
    data = np.loadtxt(file_path, delimiter='\t')
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Extract columns: [θ1, θ2, dθ1, dθ2, eff_x, eff_y]
    theta1_rad = data[:, 0]
    theta2_rad = data[:, 1]
    dtheta1_rad_s = data[:, 2]
    dtheta2_rad_s = data[:, 3]
    eff_x_m = data[:, 4]
    eff_y_m = data[:, 5]
    
    # Normalize
    theta1_norm = normalize_angle(theta1_rad)
    theta2_norm = normalize_angle(theta2_rad)
    dtheta1_norm = normalize_velocity(dtheta1_rad_s)
    dtheta2_norm = normalize_velocity(dtheta2_rad_s)
    eff_x_norm = normalize_position(eff_x_m)
    eff_y_norm = normalize_position(eff_y_m)
    
    # Stack
    arm_obs = np.stack([
        theta1_norm, theta2_norm,
        dtheta1_norm, dtheta2_norm,
        eff_x_norm, eff_y_norm
    ], axis=-1)
    
    return {
        'theta1_rad': theta1_rad,
        'theta2_rad': theta2_rad,
        'dtheta1_rad_s': dtheta1_rad_s,
        'dtheta2_rad_s': dtheta2_rad_s,
        'eff_m': np.stack([eff_x_m, eff_y_m], axis=-1),
        'arm_obs_norm': arm_obs,
        'n_steps': len(theta1_rad)
    }


def load_3dof_txt(file_path):
    """Load 3-DoF trajectory from TXT file."""
    data = np.loadtxt(file_path, delimiter='\t')
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Extract columns: [θ1, θ2, θ3, dθ1, dθ2, dθ3, eff_x, eff_y]
    theta1_rad = data[:, 0]
    theta2_rad = data[:, 1]
    theta3_rad = data[:, 2]
    dtheta1_rad_s = data[:, 3]
    dtheta2_rad_s = data[:, 4]
    dtheta3_rad_s = data[:, 5]
    eff_x_m = data[:, 6]
    eff_y_m = data[:, 7]
    
    # Normalize
    theta1_norm = normalize_angle(theta1_rad)
    theta2_norm = normalize_angle(theta2_rad)
    theta3_norm = normalize_angle(theta3_rad)
    dtheta1_norm = normalize_velocity(dtheta1_rad_s)
    dtheta2_norm = normalize_velocity(dtheta2_rad_s)
    dtheta3_norm = normalize_velocity(dtheta3_rad_s)
    eff_x_norm = normalize_position(eff_x_m)
    eff_y_norm = normalize_position(eff_y_m)
    
    # Stack
    arm_obs = np.stack([
        theta1_norm, theta2_norm, theta3_norm,
        dtheta1_norm, dtheta2_norm, dtheta3_norm,
        eff_x_norm, eff_y_norm
    ], axis=-1)
    
    return {
        'theta1_rad': theta1_rad,
        'theta2_rad': theta2_rad,
        'theta3_rad': theta3_rad,
        'dtheta1_rad_s': dtheta1_rad_s,
        'dtheta2_rad_s': dtheta2_rad_s,
        'dtheta3_rad_s': dtheta3_rad_s,
        'eff_m': np.stack([eff_x_m, eff_y_m], axis=-1),
        'arm_obs_norm': arm_obs,
        'n_steps': len(theta1_rad)
    }


def forward_kinematics_2dof(theta1_rad, theta2_rad, l1=1.5, l2=1.5):
    """Compute end-effector position for 2-DoF arm."""
    x = l1 * np.cos(theta1_rad) + l2 * np.cos(theta1_rad + theta2_rad)
    y = l1 * np.sin(theta1_rad) + l2 * np.sin(theta1_rad + theta2_rad)
    return np.array([x, y])


def forward_kinematics_3dof(theta1_rad, theta2_rad, theta3_rad, l1=1.0, l2=1.0, l3=1.0):
    """Compute end-effector position for 3-DoF arm."""
    x = (l1 * np.cos(theta1_rad) + l2 * np.cos(theta1_rad + theta2_rad) +
         l3 * np.cos(theta1_rad + theta2_rad + theta3_rad))
    y = (l1 * np.sin(theta1_rad) + l2 * np.sin(theta1_rad + theta2_rad) +
         l3 * np.sin(theta1_rad + theta2_rad + theta3_rad))
    return np.array([x, y])


def get_joint_positions_2dof(theta1_rad, theta2_rad, l1=1.5, l2=1.5):
    """Get joint positions for 2-DoF arm."""
    j1 = np.array([l1 * np.cos(theta1_rad), l1 * np.sin(theta1_rad)])
    eff = forward_kinematics_2dof(theta1_rad, theta2_rad, l1, l2)
    return np.zeros(2), j1, eff


def get_joint_positions_3dof(theta1_rad, theta2_rad, theta3_rad, l1=1.0, l2=1.0, l3=1.0):
    """Get joint positions for 3-DoF arm."""
    j1 = np.array([l1 * np.cos(theta1_rad), l1 * np.sin(theta1_rad)])
    j2 = j1 + np.array([l2 * np.cos(theta1_rad + theta2_rad), l2 * np.sin(theta1_rad + theta2_rad)])
    eff = forward_kinematics_3dof(theta1_rad, theta2_rad, theta3_rad, l1, l2, l3)
    return np.zeros(2), j1, j2, eff


class TrajectoryViewer:
    COL_LINK1 = "#E05A3A"
    COL_LINK2 = "#3A8FE0"
    COL_LINK3 = "#9B5DE5"
    COL_EFF = "#FFFFFF"
    COL_TRAIL = "#888888"
    
    def __init__(self, traj_2dof, traj_3dof, title="Trajectoires alignées 2-DoF ↔ 3-DoF"):
        self.traj_2dof = traj_2dof
        self.traj_3dof = traj_3dof
        self.n_steps = min(traj_2dof['n_steps'], traj_3dof['n_steps'])
        self.current_step = 0
        
        # Create figure
        self.fig, (self.ax2, self.ax3) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Initialize trails
        self.trail_2dof = []
        self.trail_3dof = []
        self.max_trail_length = 200
        
        self.setup_axes()
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    def setup_axes(self):
        """Configure both axes."""
        for ax, title in [(self.ax2, "2-DoF Robot"), (self.ax3, "3-DoF Robot")]:
            ax.set_xlim(-3.3, 3.3)
            ax.set_ylim(-3.3, 3.3)
            ax.set_aspect("equal")
            ax.set_facecolor("#1a1a2e")
            ax.set_title(title, color="white", fontsize=12)
            
            # Add reachable workspace circle
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta)*MAX_REACH, np.sin(theta)*MAX_REACH,
                    color="#444", lw=1, ls="--", alpha=0.5)
            
            ax.tick_params(colors="#888")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
    
    def draw_2dof(self, step):
        """Draw 2-DoF arm at given step."""
        self.ax2.cla()
        self.setup_axes()
        self.ax2.set_title(f"2-DoF Robot | step {step+1}/{self.n_steps}", color="white", fontsize=12)
        
        theta1 = self.traj_2dof['theta1_rad'][step]
        theta2 = self.traj_2dof['theta2_rad'][step]
        eff = self.traj_2dof['eff_m'][step]
        
        o, j1, eff_pos = get_joint_positions_2dof(theta1, theta2)
        
        # Draw links
        self.ax2.plot([o[0], j1[0]], [o[1], j1[1]], '-', color=self.COL_LINK1, lw=6, solid_capstyle='round')
        self.ax2.plot([j1[0], eff_pos[0]], [j1[1], eff_pos[1]], '-', color=self.COL_LINK2, lw=6, solid_capstyle='round')
        
        # Draw joints
        self.ax2.plot(j1[0], j1[1], 'o', color="#FFD700", markersize=8, zorder=6)
        
        # Draw end-effector
        self.ax2.plot(eff_pos[0], eff_pos[1], 'o', color=self.COL_EFF, markersize=12,
                      markeredgecolor="#222", markeredgewidth=1.5, zorder=7)
        
        # Draw trail
        trail_start = max(0, step - self.max_trail_length)
        trail_eff = self.traj_2dof['eff_m'][trail_start:step+1]
        if len(trail_eff) > 1:
            self.ax2.plot(trail_eff[:, 0], trail_eff[:, 1], '-', color=self.COL_TRAIL, lw=1.5, alpha=0.6)
        
        # Info text
        info = (f"θ1={np.degrees(theta1):+.0f}°  θ2={np.degrees(theta2):+.0f}°\n"
                f"v1={self.traj_2dof['dtheta1_rad_s'][step]:+.2f} rad/s  v2={self.traj_2dof['dtheta2_rad_s'][step]:+.2f} rad/s")
        self.ax2.text(0.02, 0.98, info, transform=self.ax2.transAxes,
                      fontsize=9, verticalalignment='top', color="#AAA",
                      bbox=dict(boxstyle='round', facecolor='#222', alpha=0.7))
    
    def draw_3dof(self, step):
        """Draw 3-DoF arm at given step."""
        self.ax3.cla()
        self.setup_axes()
        self.ax3.set_title(f"3-DoF Robot | step {step+1}/{self.n_steps}", color="white", fontsize=12)
        
        theta1 = self.traj_3dof['theta1_rad'][step]
        theta2 = self.traj_3dof['theta2_rad'][step]
        theta3 = self.traj_3dof['theta3_rad'][step]
        
        o, j1, j2, eff_pos = get_joint_positions_3dof(theta1, theta2, theta3)
        
        # Draw links
        self.ax3.plot([o[0], j1[0]], [o[1], j1[1]], '-', color=self.COL_LINK1, lw=6, solid_capstyle='round')
        self.ax3.plot([j1[0], j2[0]], [j1[1], j2[1]], '-', color=self.COL_LINK2, lw=6, solid_capstyle='round')
        self.ax3.plot([j2[0], eff_pos[0]], [j2[1], eff_pos[1]], '-', color=self.COL_LINK3, lw=6, solid_capstyle='round')
        
        # Draw joints
        self.ax3.plot(j1[0], j1[1], 'o', color="#FFD700", markersize=8, zorder=6)
        self.ax3.plot(j2[0], j2[1], 'o', color="#FFD700", markersize=7, zorder=6)
        
        # Draw end-effector
        self.ax3.plot(eff_pos[0], eff_pos[1], 'o', color=self.COL_EFF, markersize=12,
                      markeredgecolor="#222", markeredgewidth=1.5, zorder=7)
        
        # Draw trail
        trail_start = max(0, step - self.max_trail_length)
        trail_eff = self.traj_3dof['eff_m'][trail_start:step+1]
        if len(trail_eff) > 1:
            self.ax3.plot(trail_eff[:, 0], trail_eff[:, 1], '-', color=self.COL_TRAIL, lw=1.5, alpha=0.6)
        
        # Info text
        info = (f"θ1={np.degrees(theta1):+.0f}°  θ2={np.degrees(theta2):+.0f}°  θ3={np.degrees(theta3):+.0f}°\n"
                f"v1={self.traj_3dof['dtheta1_rad_s'][step]:+.2f}  v2={self.traj_3dof['dtheta2_rad_s'][step]:+.2f}  v3={self.traj_3dof['dtheta3_rad_s'][step]:+.2f} rad/s")
        self.ax3.text(0.02, 0.98, info, transform=self.ax3.transAxes,
                      fontsize=9, verticalalignment='top', color="#AAA",
                      bbox=dict(boxstyle='round', facecolor='#222', alpha=0.7))
    
    def update(self, step):
        """Update display for given step."""
        self.draw_2dof(step)
        self.draw_3dof(step)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def animate(self, delay=0.05):
        """Run animation."""
        plt.ion()
        plt.show()
        
        for step in range(self.n_steps):
            if not plt.fignum_exists(self.fig.number):
                break
            self.update(step)
            plt.pause(delay)
        
        plt.ioff()
        plt.show()


def find_matching_files(data_dir, idx=None, file_2dof=None, file_3dof=None):
    """Find matching trajectory files."""
    data_dir = Path(data_dir)
    
    if file_2dof and file_3dof:
        return Path(file_2dof), Path(file_3dof)
    
    # Find all 2-DoF and 3-DoF files
    files_2dof = sorted(data_dir.glob("*2dof*.txt"))
    files_3dof = sorted(data_dir.glob("*3dof*.txt"))
    
    if not files_2dof or not files_3dof:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    
    if idx is not None:
        if idx < len(files_2dof) and idx < len(files_3dof):
            return files_2dof[idx], files_3dof[idx]
        else:
            raise IndexError(f"Index {idx} out of range (max {min(len(files_2dof), len(files_3dof))-1})")
    
    # Default: use first pair
    return files_2dof[0], files_3dof[0]


def main():
    parser = argparse.ArgumentParser(description="Visualiser les trajectoires alignées 2-DoF et 3-DoF")
    parser.add_argument("--data_dir", type=str, default="./data/DIRECT_GEN_ALGO",
                        help="Répertoire contenant les fichiers TXT")
    parser.add_argument("--file_2dof", type=str, default=None,
                        help="Chemin vers le fichier 2-DoF (optionnel)")
    parser.add_argument("--file_3dof", type=str, default=None,
                        help="Chemin vers le fichier 3-DoF (optionnel)")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index de la paire à visualiser (si plusieurs fichiers)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Délai entre les frames (secondes)")
    
    args = parser.parse_args()
    
    # Find matching files
    file_2dof, file_3dof = find_matching_files(args.data_dir, args.idx, args.file_2dof, args.file_3dof)
    
    print(f"\n📂 Chargement des trajectoires:")
    print(f"   2-DoF: {file_2dof}")
    print(f"   3-DoF: {file_3dof}")
    
    # Load trajectories
    traj_2dof = load_2dof_txt(file_2dof)
    traj_3dof = load_3dof_txt(file_3dof)
    
    print(f"\n📊 Statistiques:")
    print(f"   2-DoF: {traj_2dof['n_steps']} steps")
    print(f"   3-DoF: {traj_3dof['n_steps']} steps")
    print(f"   Affichage: {min(traj_2dof['n_steps'], traj_3dof['n_steps'])} steps synchronisés")
    
    # Create and run viewer
    title = f"Trajectoires alignées | {file_2dof.name} ↔ {file_3dof.name}"
    viewer = TrajectoryViewer(traj_2dof, traj_3dof, title)
    
    print(f"\n▶️ Animation en cours... (delay={args.delay}s)")
    print("   Fermer la fenêtre pour quitter")
    
    viewer.animate(delay=args.delay)
    
    print("\n✅ Visualisation terminée")


if __name__ == "__main__":
    main()
