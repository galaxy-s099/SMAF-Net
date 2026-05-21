import torch
import torch.nn as nn


class CrossAtlasAttention(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.5):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, atlas_embeddings):
        """
        atlas_embeddings: B × M × D
        M = number of atlases, here 3
        """
        attn_out, attn_weight = self.attn(
            atlas_embeddings,
            atlas_embeddings,
            atlas_embeddings
        )

        z = self.norm1(atlas_embeddings + attn_out)

        ffn_out = self.ffn(z)
        z = self.norm2(z + ffn_out)

        return z, attn_weight