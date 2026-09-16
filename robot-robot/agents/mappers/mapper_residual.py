# agents/mappers/mapper_residual.py
import torch
import torch.nn as nn


class MLPResidual(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ResidualMapper(nn.Module):
    def __init__(self, input_dim=6, output_dim=8):
        super().__init__()
        self.delta_net = MLPResidual(input_dim=input_dim, output_dim=output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, s2):
        """
        s2: (batch, 6) - état 2DoF
             [θ1/π, θ2/π, dθ1/ω_max, dθ2/ω_max, x/r, y/r]
        
        return: s3_hat = s2_expanded + delta
             [θ1/π, θ2/π, θ3/π, dθ1/ω_max, dθ2/ω_max, dθ3/ω_max, x/r, y/r]
        """
        batch_size = s2.shape[0]
        device = s2.device
        
        # Étendre s2 (6 dims) à 8 dims pour correspondre à s3
        s2_expanded = torch.zeros(batch_size, self.output_dim, device=device)
        s2_expanded[:, 0] = s2[:, 0]  # θ1
        s2_expanded[:, 1] = s2[:, 1]  # θ2
        # θ3 reste à 0
        s2_expanded[:, 3] = s2[:, 2]  # dθ1
        s2_expanded[:, 4] = s2[:, 3]  # dθ2
        # dθ3 reste à 0
        s2_expanded[:, 6] = s2[:, 4]  # x
        s2_expanded[:, 7] = s2[:, 5]  # y
        
        # Prédire le delta
        delta = self.delta_net(s2)
        
        # Ajouter le delta à s2 expansé
        s3_hat = s2_expanded + delta
        
        return s3_hat
