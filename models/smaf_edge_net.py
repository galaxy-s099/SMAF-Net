import torch
import torch.nn as nn

from models.edge_encoder import EdgeBranchEncoder
from models.cross_atlas_attention import CrossAtlasAttention


class SMAFEdgeNet(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5,
        num_heads=4
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

        self.cross_atlas_attention = CrossAtlasAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # v4.1:
        # raw branch embedding: 3 * embedding_dim
        # attention-enhanced embedding: 3 * embedding_dim
        # final fusion dim = 6 * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, batch):
        z_aal = self.encoder_aal(batch["aal"])
        z_cc200 = self.encoder_cc200(batch["cc200"])
        z_ho = self.encoder_ho(batch["ho"])

        # B × 3 × D
        z = torch.stack([z_aal, z_cc200, z_ho], dim=1)

        z_enhanced, attn_weight = self.cross_atlas_attention(z)

        # 原始三图谱表示
        z_raw = z.reshape(z.size(0), -1)

        # attention 增强后的三图谱表示
        z_attn = z_enhanced.reshape(z_enhanced.size(0), -1)

        # residual-style fusion
        z_final = torch.cat([z_raw, z_attn], dim=-1)

        logits = self.classifier(z_final)

        return logits