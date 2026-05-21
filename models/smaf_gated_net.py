import torch
import torch.nn as nn

from models.edge_encoder import EdgeBranchEncoder


class AtlasGatedFusion(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128, dropout=0.5):
        super().__init__()

        # 输入为三个 atlas embedding 拼接后的向量：B × 3D
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z_aal, z_cc200, z_ho):
        """
        z_*: B × D
        """
        z_concat = torch.cat([z_aal, z_cc200, z_ho], dim=-1)

        # B × 3
        gate_logits = self.gate_mlp(z_concat)
        gate_weight = torch.softmax(gate_logits, dim=-1)

        w_aal = gate_weight[:, 0].unsqueeze(-1)
        w_cc200 = gate_weight[:, 1].unsqueeze(-1)
        w_ho = gate_weight[:, 2].unsqueeze(-1)

        # B × D
        z_gate = w_aal * z_aal + w_cc200 * z_cc200 + w_ho * z_ho

        return z_gate, gate_weight


class SMAFGatedNet(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5
    ):
        super().__init__()

        # signed edge vector dimensions:
        # AAL:   116 * 115 / 2 * 2 = 13340
        # CC200: 200 * 199 / 2 * 2 = 39800
        # HO:    111 * 110 / 2 * 2 = 12210

        self.encoder_aal = EdgeBranchEncoder(
            input_dim=13340,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.encoder_cc200 = EdgeBranchEncoder(
            input_dim=39800,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.encoder_ho = EdgeBranchEncoder(
            input_dim=12210,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.gated_fusion = AtlasGatedFusion(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # z_raw = 3D, z_gate = D, total = 4D
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 2)
        )

    def forward(self, batch):
        z_aal = self.encoder_aal(batch["aal"])
        z_cc200 = self.encoder_cc200(batch["cc200"])
        z_ho = self.encoder_ho(batch["ho"])

        z_gate, gate_weight = self.gated_fusion(
            z_aal,
            z_cc200,
            z_ho
        )

        z_raw = torch.cat([z_aal, z_cc200, z_ho], dim=-1)

        z_final = torch.cat([z_raw, z_gate], dim=-1)

        logits = self.classifier(z_final)

        return logits