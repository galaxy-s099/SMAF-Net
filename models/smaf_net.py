import torch.nn as nn

from models.signed_encoder import SignedGraphEncoder
from models.fusion import ConcatFusion


class SMAFNetV1(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=128, dropout=0.5):
        super().__init__()

        self.encoder_aal = SignedGraphEncoder(
            num_nodes=116,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.encoder_cc200 = SignedGraphEncoder(
            num_nodes=200,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.encoder_ho = SignedGraphEncoder(
            num_nodes=111,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        self.fusion = ConcatFusion(
            input_dim=embedding_dim * 3,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, batch):
        z_aal = self.encoder_aal(batch["aal"])
        z_cc200 = self.encoder_cc200(batch["cc200"])
        z_ho = self.encoder_ho(batch["ho"])

        z = self.fusion([z_aal, z_cc200, z_ho])
        logits = self.classifier(z)

        return logits