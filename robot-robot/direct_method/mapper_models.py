import numpy as np
import torch
import torch.nn as nn

ARM_OBS_2DOF = 6
ARM_OBS_3DOF = 8
MAX_REACH = 3.0


class StateMapperMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionMapperMLP(nn.Module):
    """Ancien mapper d'action non conditionné (conservé pour recharger les
    runs existants, ex: run_02). Déficient par construction : il ne voit pas
    les angles, donc pas les Jacobiens — préférer ActionMapperConditionedMLP."""
    def __init__(self, act_in_dim: int, act_out_dim: int,
                 hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_in_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden,    hidden), nn.LeakyReLU(),
            nn.Linear(hidden, act_out_dim),
        )
    def forward(self, act_in: torch.Tensor) -> torch.Tensor:
        return self.net(act_in)


class ActionMapperConditionedMLP(nn.Module):
    """Mapper d'action conditionné par les configurations source ET cible.

    Entrées (toutes normalisées, conventions des fichiers de trajectoires) :
        dtheta_src : (B, nq_src)  vitesses articulaires source / omega_max
        theta_src  : (B, nq_src)  angles source / pi
        theta_tgt  : (B, nq_tgt)  angles cible  / pi
    Sortie :
        dtheta_tgt : (B, nq_tgt)  vitesses articulaires cible / omega_max

    Les angles sont featurisés en sin/cos DANS forward : représentation continue
    à la discontinuité -pi/pi, et le réseau reçoit exactement l'information qui
    détermine les Jacobiens J_src(theta_src) et J_tgt(theta_tgt).
    """

    def __init__(self, nq_src: int, nq_tgt: int, hidden: int = 256):
        super().__init__()
        self.nq_src = nq_src
        self.nq_tgt = nq_tgt
        in_dim = nq_src + 2 * nq_src + 2 * nq_tgt   # dtheta + sin/cos src + sin/cos tgt
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LeakyReLU(),
            nn.Linear(hidden, nq_tgt),
        )

    def forward(self, dtheta_src: torch.Tensor, theta_src: torch.Tensor,
                theta_tgt: torch.Tensor) -> torch.Tensor:
        ang_src = theta_src * torch.pi
        ang_tgt = theta_tgt * torch.pi
        x = torch.cat([
            dtheta_src,
            torch.sin(ang_src), torch.cos(ang_src),
            torch.sin(ang_tgt), torch.cos(ang_tgt),
        ], dim=-1)
        return self.net(x)

