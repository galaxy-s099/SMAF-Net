import torch
import torch.nn as nn


def fc_to_signed_edge_vector(fc):
    """
    fc: B × N × N
    return: B × (2 * N * (N-1) / 2)
    """
    B, N, _ = fc.shape

    idx = torch.triu_indices(N, N, offset=1, device=fc.device)

    fc_pos = torch.clamp(fc, min=0.0)
    fc_neg = torch.clamp(-fc, min=0.0)

    pos_vec = fc_pos[:, idx[0], idx[1]]
    neg_vec = fc_neg[:, idx[0], idx[1]]

    return torch.cat([pos_vec, neg_vec], dim=-1)


class EdgeBranchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, fc):
        edge_vec = fc_to_signed_edge_vector(fc)
        z = self.encoder(edge_vec)
        return z