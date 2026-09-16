# agents/mappers/mapper_direct.py
import torch
import torch.nn as nn


class MLPMapper(nn.Module):
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


class DirectMapper(nn.Module):
    def __init__(self, input_dim=6, output_dim=8):
        super().__init__()
        self.mlp = MLPMapper(input_dim=input_dim, output_dim=output_dim)

    def forward(self, s2):
        """
        s2: (batch, 6) - état 2DoF
             [θ1/π, θ2/π, dθ1/ω_max, dθ2/ω_max, x/r, y/r]
        return: s3_hat (batch, 8) - état 3DoF prédit
             [θ1/π, θ2/π, θ3/π, dθ1/ω_max, dθ2/ω_max, dθ3/ω_max, x/r, y/r]
        """
        return self.mlp(s2)
