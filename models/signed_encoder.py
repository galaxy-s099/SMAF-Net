import torch
import torch.nn as nn
import torch.nn.functional as F


class SignedGraphEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, embedding_dim=128, dropout=0.5):
        super().__init__()

        self.num_nodes = num_nodes

        self.pos_linear = nn.Linear(num_nodes, hidden_dim)
        self.neg_linear = nn.Linear(num_nodes, hidden_dim)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, fc):
        """
        fc: B × N × N
        """
        A_pos = torch.clamp(fc, min=0.0)
        A_neg = torch.clamp(-fc, min=0.0)

        h_pos = F.relu(self.pos_linear(A_pos))
        h_neg = F.relu(self.neg_linear(A_neg))

        h = torch.cat([h_pos, h_neg], dim=-1)
        h = self.proj(h)

        # graph-level pooling
        graph_emb = h.mean(dim=1)

        return graph_emb