import os
import os
import matplotlib.pyplot as plt
from direct_method.dataset import TrajectoriesTrainingDataset


R1_NAME = "robot_2dofs"
R2_NAME = "robot_3dofs"

PATH_R1_TRAj = "direct_method/trajectories/{}.txt".format(R1_NAME)
PATH_R2_TRAj = "direct_method/trajectories/{}.txt".format(R2_NAME)

robots = {
    "Dof2": {
        "path": PATH_R1_TRAj,
        "state_dim": 4,
        "dof": 2
    },
    "Dof3": {
        "path": PATH_R2_TRAj,
        "state_dim": 6,
        "dof": 3
    },
}



for robot_name, cfg in robots.items():

    print(f"Plotando histogramas para {robot_name}")

    # carregar dados (ajuste conforme sua classe Dataset)
    dataset = TrajectoriesTrainingDataset(
        cfg["path"],
        cfg["path"],  # se for single-robot, ou adapte
        cfg["state_dim"],
        cfg["state_dim"]
    )

    data = dataset.trajectories_r1  # ou r2, conforme seu design
    dof = cfg["dof"]

    dir_path = f"direct_method/Data_Robots/{robot_name}/100k"
    os.makedirs(dir_path, exist_ok=True)

    for i in range(cfg["state_dim"]):
        plt.figure(figsize=(6, 4))
        plt.hist(data[:, i], bins=100, edgecolor="black")
        plt.grid(True)

        if i < dof:
            joint_id = i + 1
            plt.title(f"{robot_name} - Joint Position j{joint_id}")
            plt.xlabel("pos (rad)")
            filename = f"joint_pos_{joint_id}.png"
        else:
            joint_id = i - dof + 1
            plt.title(f"{robot_name} - Joint Velocity j{joint_id}")
            plt.xlabel("vel (rad/s)")
            filename = f"joint_vel_{joint_id}.png"

        plt.tight_layout()
        plt.savefig(os.path.join(dir_path, filename))
        plt.close()