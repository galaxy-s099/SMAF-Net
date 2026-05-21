import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h):
        """
        h: B × N × D
        """
        score = self.attn(h)              # B × N × 1
        weight = torch.softmax(score, dim=1)
        graph_emb = torch.sum(weight * h, dim=1)
        return graph_emb, weight


class SignedGraphEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, embedding_dim=128, dropout=0.5):
        super().__init__()

        self.num_nodes = num_nodes

        self.pos_linear = nn.Linear(num_nodes, hidden_dim)
        self.neg_linear = nn.Linear(num_nodes, hidden_dim)

        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attn_pool = AttentionPooling(
            input_dim=embedding_dim,
            hidden_dim=max(embedding_dim // 2, 32)
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
        h = self.node_proj(h)

        graph_emb, attn_weight = self.attn_pool(h)

        return graph_emb