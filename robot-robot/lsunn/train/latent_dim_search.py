"""
Phase 1bis: selection of the latent dimension (Fig. A.10 of Beaussant et al.).

Trains one pair of bases per candidate dimension using EXACTLY the Phase 2
setup (BasesVAETrainer), so that the dimension selected here is the one the
bases will actually be trained with. Three quantities are measured on a
held-out validation set, with deterministic encoding (z = mu):

  - reconstruction error      Bo_r(Bi_r(x_r))    vs x_r
  - cross-reconstruction err. Bo_r2(Bi_r1(x_r1)) vs x_r2
  - latent gap                || mu_2dof - mu_3dof ||

The reconstruction error is the criterion of the paper, but the last two are
the ones that condition the 2 <-> 3 DoF transfer: a latent space can be a good
auto-encoder for each arm separately and still be badly aligned.

Usage:
    python -m lsunn.train.latent_dim_search
    python -m lsunn.train.latent_dim_search --dims 2 3 4 5 6 --seeds 3
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lsunn.bases_vae import BasesVAETrainer, DEFAULT_HIDDEN_DIM

EVAL_CHUNK = 8192


# ============================================================================
# Data
# ============================================================================

def load_trajectories(traj_path: str) -> dict:
    with open(traj_path, "rb") as f:
        return pickle.load(f)


def split_pairs(data: dict, val_ratio: float, max_samples: int, seed: int):
    """
    Split the paired dataset. The pairing must be preserved, so the SAME
    permutation is applied to both arms.
    """
    s2 = np.asarray(data["arm_states_2dof"], dtype=np.float32)
    s3 = np.asarray(data["arm_states_3dof"], dtype=np.float32)
    assert len(s2) == len(s3), "paired dataset expected"

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(s2))
    if max_samples and max_samples < len(perm):
        perm = perm[:max_samples]

    n_val = int(val_ratio * len(perm))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train = {
        "arm_states_2dof": s2[train_idx],
        "arm_states_3dof": s3[train_idx],
    }
    return train, s2[val_idx], s3[val_idx]


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def _encode_mu(vae, x: torch.Tensor) -> torch.Tensor:
    """Deterministic encoding: the mean of the posterior, no sampling."""
    _, mu, _ = vae.encoder(x)
    return mu


@torch.no_grad()
def evaluate_bases(base_2dof, base_3dof, s2_val, s3_val, device: str) -> dict:
    base_2dof.eval()
    base_3dof.eval()

    n = len(s2_val)
    acc = {"recon_2": 0.0, "recon_3": 0.0, "cross_2": 0.0, "cross_3": 0.0, "gap": 0.0}

    for start in range(0, n, EVAL_CHUNK):
        b2 = torch.as_tensor(s2_val[start:start + EVAL_CHUNK], device=device)
        b3 = torch.as_tensor(s3_val[start:start + EVAL_CHUNK], device=device)
        w = len(b2) / n

        mu2 = _encode_mu(base_2dof, b2)
        mu3 = _encode_mu(base_3dof, b3)

        acc["recon_2"] += w * F.mse_loss(base_2dof.decoder(mu2), b2).item()
        acc["recon_3"] += w * F.mse_loss(base_3dof.decoder(mu3), b3).item()
        # Cross: the latent produced by one arm is decoded in the other's space.
        acc["cross_2"] += w * F.mse_loss(base_2dof.decoder(mu3), b2).item()
        acc["cross_3"] += w * F.mse_loss(base_3dof.decoder(mu2), b3).item()
        acc["gap"] += w * torch.norm(mu2 - mu3, dim=-1).mean().item()

    return {
        "recon": 0.5 * (acc["recon_2"] + acc["recon_3"]),
        "cross": 0.5 * (acc["cross_2"] + acc["cross_3"]),
        "gap": acc["gap"],
    }


# ============================================================================
# Selection
# ============================================================================

def select_latent_dim(dims, errors, tol: float = 0.05):
    """
    Return (selected_dim, best_dim).

    best_dim minimises the validation error; selected_dim is the smallest
    dimension whose error stays within `tol` of that minimum, i.e. the smallest
    bottleneck that loses nothing meaningful. Taking the argmin of the finite
    difference, as in the first version, returns the steepest drop rather than
    the elbow, and ignores the over-fitting regime where the curve goes back up.
    """
    errors = np.asarray(errors, dtype=float)
    best = int(np.argmin(errors))
    threshold = errors[best] * (1.0 + tol)
    selected = next(i for i, e in enumerate(errors) if e <= threshold)
    return dims[selected], dims[best]


# ============================================================================
# Main
# ============================================================================

def run_search(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data = load_trajectories(args.traj)
    results = {d: {"recon": [], "cross": [], "gap": []} for d in args.dims}

    for seed in range(args.seeds):
        train, s2_val, s3_val = split_pairs(
            data, args.val_ratio, args.max_samples, seed=args.seed + seed
        )
        print(
            f"\n=== seed {seed} | train {len(train['arm_states_2dof']):,} "
            f"| val {len(s2_val):,} ==="
        )

        for dim in args.dims:
            torch.manual_seed(args.seed + seed)
            print(f"\n  latent_dim = {dim}")

            # Exactly the Phase 2 configuration (weight decay, cosine schedule,
            # alpha/beta/gamma/lambda), so the search is not biased by a
            # different optimisation setup.
            trainer = BasesVAETrainer(
                latent_dim=dim, hidden_dim=args.hidden_dim, device=device
            )
            trainer.train(
                train, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
            )

            m = evaluate_bases(
                trainer.base_2dof, trainer.base_3dof, s2_val, s3_val, device
            )
            for k, v in m.items():
                results[dim][k].append(v)
            print(
                f"    recon MSE {m['recon']:.6f} | cross MSE {m['cross']:.6f} "
                f"| latent gap {m['gap']:.4f}"
            )

    return results


def summarise(results, dims):
    mean = {k: np.array([np.mean(results[d][k]) for d in dims]) for k in ("recon", "cross", "gap")}
    std = {k: np.array([np.std(results[d][k]) for d in dims]) for k in ("recon", "cross", "gap")}
    return mean, std


def plot_results(dims, mean, std, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].errorbar(dims, mean["recon"], yerr=std["recon"], fmt="o-",
                     capsize=3, label="reconstruction")
    axes[0].errorbar(dims, mean["cross"], yerr=std["cross"], fmt="s--",
                     capsize=3, label="cross-reconstruction")
    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("MSE (validation)")
    axes[0].set_title("Reconstruction error vs latent dimension")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].errorbar(dims, mean["gap"], yerr=std["gap"], fmt="o-", capsize=3,
                     color="tab:red")
    axes[1].set_xlabel("Latent dimension")
    axes[1].set_ylabel(r"$\|\mu_{2DoF} - \mu_{3DoF}\|$")
    axes[1].set_title("Latent alignment (lower is better)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\n  Figure saved -> {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", default="./data/LSUNN/trajectories.pkl")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=200_000,
                        help="subsample the dataset to keep the search tractable "
                             "(0 = use everything)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of repetitions, averaged as in Fig. A.10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--out", default="latent_dim_search")
    args = parser.parse_args()

    results = run_search(args)
    mean, std = summarise(results, args.dims)
    plot_results(args.dims, mean, std, f"{args.out}.png")

    sel_recon, best_recon = select_latent_dim(args.dims, mean["recon"], args.tol)
    sel_cross, best_cross = select_latent_dim(args.dims, mean["cross"], args.tol)

    print("\n=== Recommendation ===")
    for i, d in enumerate(args.dims):
        print(f"  dim {d:>2} | recon {mean['recon'][i]:.6f} "
              f"| cross {mean['cross'][i]:.6f} | gap {mean['gap'][i]:.4f}")
    print(f"\n  reconstruction  : best {best_recon}, smallest within "
          f"{args.tol:.0%} -> {sel_recon}")
    print(f"  cross-reconstr. : best {best_cross}, smallest within "
          f"{args.tol:.0%} -> {sel_cross}")
    print(f"\n  -> set DEFAULT_LATENT_DIM = {max(sel_recon, sel_cross)} in bases_vae.py")

    with open(f"{args.out}.json", "w") as f:
        json.dump(
            {
                "dims": args.dims,
                "recon_mean": mean["recon"].tolist(),
                "cross_mean": mean["cross"].tolist(),
                "gap_mean": mean["gap"].tolist(),
                "recon_std": std["recon"].tolist(),
                "cross_std": std["cross"].tolist(),
                "gap_std": std["gap"].tolist(),
                "selected": {"recon": sel_recon, "cross": sel_cross},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
