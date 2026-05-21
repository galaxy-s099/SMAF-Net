import torch
import torch.nn as nn

from models.edge_encoder import EdgeBranchEncoder


class SMAFUncertaintyNet(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        embedding_dim=128,
        dropout=0.5,
        temperature=1.0
    ):
        super().__init__()

        self.temperature = temperature

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

        # 每个 atlas 分支单独分类
        self.classifier_aal = nn.Linear(embedding_dim, 2)
        self.classifier_cc200 = nn.Linear(embedding_dim, 2)
        self.classifier_ho = nn.Linear(embedding_dim, 2)

    def compute_energy(self, logits):
        """
        logits: B × C

        Energy score:
        E(x) = -T * logsumexp(logits / T)

        energy 越低，通常代表模型越自信。
        """
        T = self.temperature
        energy = -T * torch.logsumexp(logits / T, dim=1)
        return energy

    def forward(self, batch):
        z_aal = self.encoder_aal(batch["aal"])
        z_cc200 = self.encoder_cc200(batch["cc200"])
        z_ho = self.encoder_ho(batch["ho"])

        logits_aal = self.classifier_aal(z_aal)
        logits_cc200 = self.classifier_cc200(z_cc200)
        logits_ho = self.classifier_ho(z_ho)

        # B × 3 × 2
        logits_stack = torch.stack(
            [logits_aal, logits_cc200, logits_ho],
            dim=1
        )

        # B × 3
        energy_aal = self.compute_energy(logits_aal)
        energy_cc200 = self.compute_energy(logits_cc200)
        energy_ho = self.compute_energy(logits_ho)

        energy_stack = torch.stack(
            [energy_aal, energy_cc200, energy_ho],
            dim=1
        )

        # energy 越低越自信，因此用 -energy 做 softmax
        atlas_weight = torch.softmax(-energy_stack, dim=1)

        # B × 3 × 1
        weight = atlas_weight.unsqueeze(-1)

        # B × 2
        final_logits = torch.sum(weight * logits_stack, dim=1)

        return final_logits