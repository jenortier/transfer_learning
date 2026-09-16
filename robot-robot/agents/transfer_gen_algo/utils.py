import numpy as np


def _make_strictly_increasing(cum_dist: np.ndarray) -> np.ndarray:
    """
    Garantit que cum_dist est strictement croissant en ajoutant un epsilon
    cumulatif aux doublons.  Nécessaire pour np.interp et interp1d.
    """
    out = cum_dist.copy()
    eps = 1e-8
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + eps
    return out


def spatial_sampling(trajectory: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """
    Ré-échantillonne une trajectoire (T, D) en num_samples points espacés
    régulièrement en distance curviligne (arc-length).

    Fixes :
      - trajectoire dégénérée (total_dist ≈ 0) → sous-échantillonnage uniforme
      - cum_dist avec doublons (bras immobile) → rendu strictement croissant
        avant interpolation, sinon np.interp produit des NaN silencieux
    """
    diffs    = np.diff(trajectory, axis=0)
    dists    = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = cum_dist[-1]

    if total_dist < 1e-9:
        idx = np.round(np.linspace(0, len(trajectory) - 1, num_samples)).astype(int)
        return trajectory[idx].astype(np.float32)

    cum_dist_strict = _make_strictly_increasing(cum_dist)
    target_dists    = np.linspace(0.0, cum_dist_strict[-1], num_samples)

    new_traj = np.zeros((num_samples, trajectory.shape[1]), dtype=np.float32)
    for d in range(trajectory.shape[1]):
        new_traj[:, d] = np.interp(target_dists, cum_dist_strict, trajectory[:, d])
    return new_traj


def spatial_sampling_aligned(states: np.ndarray, actions: np.ndarray,
                              num_samples: int = 50,
                              use_effector: bool = True):
    """
    Ré-échantillonne CONJOINTEMENT states (T, Ds) et actions (T, Da) en
    num_samples points, en utilisant la paramétrisation arc-length calculée
    sur les STATES uniquement.

    Cela garantit que states_ss[i] correspond bien à actions_ss[i], préservant
    l'alignement temporel (état_t, action_t) indispensable pour l'entraînement
    du mapper.

    Précondition : len(states) == len(actions)

    Fix critique : cum_dist rendu strictement croissant avant np.interp
    pour éviter les NaN silencieux quand le bras est immobile.
    """
    assert len(states) == len(actions), (
        f"states ({len(states)}) et actions ({len(actions)}) doivent avoir "
        "le même nombre de lignes."
    )

    # If requested, compute arc-length using only end-effector coords
    if use_effector:
        D = states.shape[1]
        if D == 6:
            eff_sl = slice(4, 6)
        elif D == 8:
            eff_sl = slice(6, 8)
        else:
            eff_sl = slice(D - 2, D)
        diffs = np.diff(states[:, eff_sl], axis=0)
    else:
        diffs = np.diff(states, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = cum_dist[-1]

    # Cas dégénéré : trajectoire plate
    if total_dist < 1e-9:
        idx = np.round(np.linspace(0, len(states) - 1, num_samples)).astype(int)
        return states[idx].astype(np.float32), actions[idx].astype(np.float32)

    # Rendre cum_dist strictement croissant (doublons = bras immobile)
    cum_dist_strict = _make_strictly_increasing(cum_dist)
    target_dists    = np.linspace(0.0, cum_dist_strict[-1], num_samples)

    states_out  = np.zeros((num_samples, states.shape[1]),  dtype=np.float32)
    actions_out = np.zeros((num_samples, actions.shape[1]), dtype=np.float32)

    for d in range(states.shape[1]):
        states_out[:, d] = np.interp(target_dists, cum_dist_strict, states[:, d])

    for d in range(actions.shape[1]):
        actions_out[:, d] = np.interp(target_dists, cum_dist_strict, actions[:, d])

    return states_out, actions_out


def temporal_sampling_aligned(states: np.ndarray, actions: np.ndarray,
                              num_samples: int = 50):
    """
    Time-based resampling: produce `num_samples` points uniformly in time
    (interpolated) while preserving alignment between `states` and `actions`.
    """
    assert len(states) == len(actions), (
        f"states ({len(states)}) and actions ({len(actions)}) must have the same length"
    )

    T = len(states)
    if T == 0:
        return (
            np.zeros((num_samples, states.shape[1]), dtype=np.float32),
            np.zeros((num_samples, actions.shape[1]), dtype=np.float32),
        )

    if T == 1:
        idx = np.zeros(num_samples, dtype=int)
        return states[idx].astype(np.float32), actions[idx].astype(np.float32)

    orig_idx = np.arange(T)
    target_idx = np.linspace(0, T - 1, num_samples)

    states_out = np.zeros((num_samples, states.shape[1]), dtype=np.float32)
    actions_out = np.zeros((num_samples, actions.shape[1]), dtype=np.float32)

    for d in range(states.shape[1]):
        states_out[:, d] = np.interp(target_idx, orig_idx, states[:, d])

    for d in range(actions.shape[1]):
        actions_out[:, d] = np.interp(target_idx, orig_idx, actions[:, d])

    return states_out, actions_out
