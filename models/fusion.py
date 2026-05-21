import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, embeddings):
        """
        embeddings: list of B × D
        """
        z = torch.cat(embeddings, dim=-1)
        z = self.net(z)
        return z