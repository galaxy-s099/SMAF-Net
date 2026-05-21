import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A, eps=1e-6):
    """
    A: B × N × N
    D^{-1/2} A D^{-1/2}
    """
    degree = A.sum(dim=-1)
    degree_inv_sqrt = torch.pow(degree + eps, -0.5)
    D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
    return torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)


class SignedGCNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5):
        super().__init__()

        self.pos_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neg_linear = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H, A_pos, A_neg):
        """
        H:     B × N × D
        A_pos: B × N × N
        A_neg: B × N × N
        """
        H_res = H

        H_pos = torch.bmm(A_pos, H)
        H_neg = torch.bmm(A_neg, H)

        H_pos = self.pos_linear(H_pos)
        H_neg = self.neg_linear(H_neg)

        H_out = torch.cat([H_pos, H_neg], dim=-1)
        H_out = self.out_proj(H_out)

        # residual connection，防止过平滑和塌缩
        H_out = self.norm(H_out + H_res)

        return H_out


class SignedGraphEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, embedding_dim=128, dropout=0.5):
        super().__init__()

        self.num_nodes = num_nodes

        # 每个节点初始特征是该节点的 FC row，所以输入维度是 num_nodes
        self.input_proj = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        self.layer1 = SignedGCNLayer(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.layer2 = SignedGCNLayer(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, fc):
        """
        fc: B × N × N
        """
        B, N, _ = fc.shape

        A_pos = torch.clamp(fc, min=0.0)
        A_neg = torch.clamp(-fc, min=0.0)

        # 正边加自环；负边不建议加自环
        I = torch.eye(N, device=fc.device).unsqueeze(0).expand(B, -1, -1)
        A_pos = A_pos + I

        A_pos = normalize_adj(A_pos)
        A_neg = normalize_adj(A_neg)

        # 关键改动：使用每个受试者自己的 FC row 作为节点特征
        H = self.input_proj(fc)

        H = self.layer1(H, A_pos, A_neg)
        H = self.layer2(H, A_pos, A_neg)

        graph_emb = H.mean(dim=1)
        graph_emb = self.graph_proj(graph_emb)

        return graph_emb