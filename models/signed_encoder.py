import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A, eps=1e-6):
    """
    A: B × N × N
    对邻接矩阵做 D^{-1/2} A D^{-1/2} 归一化
    """
    degree = A.sum(dim=-1)
    degree_inv_sqrt = torch.pow(degree + eps, -0.5)
    D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
    return torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)


class SignedGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()

        self.pos_linear = nn.Linear(in_dim, out_dim)
        self.neg_linear = nn.Linear(in_dim, out_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, H, A_pos, A_neg):
        """
        H:     B × N × D
        A_pos: B × N × N
        A_neg: B × N × N
        """
        H_pos = torch.bmm(A_pos, H)
        H_neg = torch.bmm(A_neg, H)

        H_pos = self.pos_linear(H_pos)
        H_neg = self.neg_linear(H_neg)

        H_out = torch.cat([H_pos, H_neg], dim=-1)
        H_out = self.out_proj(H_out)

        return H_out


class SignedGraphEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, embedding_dim=128, dropout=0.5):
        super().__init__()

        self.num_nodes = num_nodes

        # 每个 ROI 用可学习节点嵌入作为初始特征
        self.node_embedding = nn.Parameter(
            torch.randn(num_nodes, hidden_dim) * 0.02
        )

        self.layer1 = SignedGCNLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout
        )

        self.layer2 = SignedGCNLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
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

        # 加自环
        I = torch.eye(N, device=fc.device).unsqueeze(0).expand(B, -1, -1)
        A_pos = A_pos + I
        A_neg = A_neg + I

        # 图归一化
        A_pos = normalize_adj(A_pos)
        A_neg = normalize_adj(A_neg)

        # 初始节点特征：B × N × hidden_dim
        H = self.node_embedding.unsqueeze(0).expand(B, -1, -1)

        H = self.layer1(H, A_pos, A_neg)
        H = self.layer2(H, A_pos, A_neg)

        # mean pooling 作为图级表示
        graph_emb = H.mean(dim=1)
        graph_emb = self.graph_proj(graph_emb)

        return graph_emb