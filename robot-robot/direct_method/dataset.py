import numpy as np
from torch.utils.data import Dataset, Subset

from direct_method.kinematics import fk_2dof_np, fk_3dof_np

_FK_NP = {2: fk_2dof_np, 3: fk_3dof_np}


def block_split(dataset, fracs=(0.85, 0.10, 0.05), block=200, seed=0):
    """Split train/val/test par BLOCS CONTIGUS de lignes plutôt que par lignes.

    Les fichiers de trajectoires sont des timesteps consécutifs : deux lignes
    voisines sont quasi identiques. Un random_split par lignes met donc des
    quasi-doublons du train dans la validation (fuite), ce qui rend la loss de
    validation artificiellement optimiste. Ici on découpe en blocs de `block`
    lignes consécutives et on répartit les blocs entiers entre les splits.

    Retourne des torch.utils.data.Subset (même usage que random_split).
    """
    n = len(dataset)
    n_blocks = (n + block - 1) // block
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)

    n_train = int(round(fracs[0] * n_blocks))
    n_val = int(round(fracs[1] * n_blocks))
    groups = (order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:])

    subsets = []
    for g in groups:
        idx = np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in g]) \
            if len(g) else np.array([], dtype=np.int64)
        subsets.append(Subset(dataset, idx.tolist()))
    return subsets


class TrajectoriesTrainingDataset(Dataset):
    """
    Each trajectory row is assumed to be laid out as:
        [theta_1 ... theta_nq | dtheta_1 ... dtheta_nq | eff_x  eff_y]
    i.e. nq joint positions, nq joint velocities, then 2 end-effector
    Cartesian coordinates  ->  total columns = 2*nq + 2.

    NOTE: nq_r1 / nq_r2 (not dim_R1/dim_R2) now drive every slice, so this
    works correctly whether the trailing eff_x/eff_y columns are included
    in the mapper's I/O dimension or not.
    """
    def __init__(self, trajectories_r1, trajectories_r2,
                 nq_r1, nq_r2,
                 omega_max=2.0, max_reach=3.0):
        self.nq_r1 = nq_r1
        self.nq_r2 = nq_r2

        dim_r1 = 2 * nq_r1 + 2
        dim_r2 = 2 * nq_r2 + 2

        self.trajectories_r1 = np.loadtxt(trajectories_r1, dtype=np.float32)[:, :dim_r1]
        self.trajectories_r2 = np.loadtxt(trajectories_r2, dtype=np.float32)[:, :dim_r2]

        # ── Validity mask (raw angular velocities must lie within +/- omega_max) ──
        vel_r1 = self.trajectories_r1[:, nq_r1:2 * nq_r1]
        vel_r2 = self.trajectories_r2[:, nq_r2:2 * nq_r2]

        valid_mask_r1 = np.all((vel_r1 >= -omega_max) & (vel_r1 <= omega_max), axis=1)
        valid_mask_r2 = np.all((vel_r2 >= -omega_max) & (vel_r2 <= omega_max), axis=1)
        valid_mask = valid_mask_r1 & valid_mask_r2

        self.trajectories_r1 = self.trajectories_r1[valid_mask]
        self.trajectories_r2 = self.trajectories_r2[valid_mask]

        # ── Colonnes effecteur : RECALCULÉES par cinématique directe ──────
        # Dans les fichiers de trajectoires, les 2 dernières colonnes ne sont
        # PAS l'effecteur en mètres : elles sont encodées en coordonnées
        # fenêtre (eff + 5) / 10 (vérifié numériquement, erreur < 1e-5).
        # L'ancienne normalisation "/ max_reach" produisait donc des features
        # dans [0.07, 0.27] alors qu'au transfert les envs fournissent
        # eff / max_reach dans [-1, 1] : gros décalage train/test.
        # On recalcule eff = FK(theta) (exact) puis on normalise / max_reach,
        # ce qui reproduit exactement le format d'observation des envs.
        eff_r1 = _FK_NP[nq_r1](self.trajectories_r1[:, :nq_r1]).astype(np.float32)
        eff_r2 = _FK_NP[nq_r2](self.trajectories_r2[:, :nq_r2]).astype(np.float32)
        self.trajectories_r1[:, 2 * nq_r1:2 * nq_r1 + 2] = eff_r1 / max_reach
        self.trajectories_r2[:, 2 * nq_r2:2 * nq_r2 + 2] = eff_r2 / max_reach

        # ── Normalisation ──────────────────────────────────────────────────
        # positions   -> / pi
        self.trajectories_r1[:, :nq_r1] /= np.pi
        self.trajectories_r2[:, :nq_r2] /= np.pi
        # velocities  -> / omega_max
        self.trajectories_r1[:, nq_r1:2 * nq_r1] /= omega_max
        self.trajectories_r2[:, nq_r2:2 * nq_r2] /= omega_max

    def __len__(self):
        return len(self.trajectories_r1)

    def __getitem__(self, idx):
        return self.trajectories_r1[idx], self.trajectories_r2[idx]


class TrajectoriesTrainingActionDataset(Dataset):
    """
    Extracts the VELOCITY block [dtheta_1 ... dtheta_nq] AND the joint
    angles [theta_1 ... theta_nq] from each trajectory file.

    The velocities are what the policy's `action` actually looks like at
    runtime in the env (a normalised commanded velocity in [-1, 1]). The
    angles are required to condition the action mapper: the effect of a
    joint velocity on the end-effector depends on the Jacobian J(theta),
    so a mapper that never sees theta cannot preserve the end-effector
    velocity.

    Returns per item: (vel_r1, q_r1, vel_r2, q_r2), all normalised
    (vel / omega_max, q / pi).
    """
    def __init__(self, trajectories_r1, trajectories_r2,
                 nq_r1, nq_r2, omega_max=2.0):
        dim_r1 = 2 * nq_r1 + 2
        dim_r2 = 2 * nq_r2 + 2

        raw_r1 = np.loadtxt(trajectories_r1, dtype=np.float32)[:, :dim_r1]
        raw_r2 = np.loadtxt(trajectories_r2, dtype=np.float32)[:, :dim_r2]

        q_r1 = raw_r1[:, :nq_r1]
        q_r2 = raw_r2[:, :nq_r2]
        vel_r1 = raw_r1[:, nq_r1:2 * nq_r1]
        vel_r2 = raw_r2[:, nq_r2:2 * nq_r2]

        # Same validity criterion as the state dataset, applied here too so
        # the action mapper isn't trained on out-of-range velocity outliers.
        valid_mask_r1 = np.all((vel_r1 >= -omega_max) & (vel_r1 <= omega_max), axis=1)
        valid_mask_r2 = np.all((vel_r2 >= -omega_max) & (vel_r2 <= omega_max), axis=1)
        valid_mask = valid_mask_r1 & valid_mask_r2

        self.q_r1 = q_r1[valid_mask] / np.pi
        self.q_r2 = q_r2[valid_mask] / np.pi
        self.vel_r1 = vel_r1[valid_mask] / omega_max
        self.vel_r2 = vel_r2[valid_mask] / omega_max

    def __len__(self):
        return len(self.vel_r1)

    def __getitem__(self, idx):
        return (self.vel_r1[idx], self.q_r1[idx],
                self.vel_r2[idx], self.q_r2[idx])