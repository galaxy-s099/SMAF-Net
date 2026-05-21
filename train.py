import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from data.abide_dataset import ABIDEMultiAtlasDataset, load_labels
from models.smaf_net import SMAFNetV1
from utils.seed import set_seed
from utils.metrics import compute_metrics, summarize_results


def train_one_fold(
    data_root,
    train_idx,
    test_idx,
    seed,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=128,
    embedding_dim=128,
    dropout=0.5,
    device="cuda"
):
    set_seed(seed)

    train_dataset = ABIDEMultiAtlasDataset(data_root, train_idx)
    test_dataset = ABIDEMultiAtlasDataset(data_root, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = SMAFNetV1(
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            labels = batch["label"].to(device)

            for key in ["aal", "cc200", "ho"]:
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"].to(device)

            for key in ["aal", "cc200", "ho"]:
                batch[key] = batch[key].to(device)

            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds)
    )

    return metrics


def run_repeated_cv(config):
    data_root = config["data"]["data_root"]
    y = load_labels(data_root)

    seeds = config["train"]["seeds"]
    n_splits = config["train"]["n_splits"]

    all_results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            metrics = train_one_fold(
                data_root=data_root,
                train_idx=train_idx,
                test_idx=test_idx,
                seed=seed * 100 + fold,
                epochs=config["train"]["epochs"],
                batch_size=config["train"]["batch_size"],
                lr=config["train"]["lr"],
                weight_decay=config["train"]["weight_decay"],
                hidden_dim=config["model"]["hidden_dim"],
                embedding_dim=config["model"]["embedding_dim"],
                dropout=config["model"]["dropout"],
                device=device
            )

            all_results.append(metrics)

            print(
                f"Seed {seed} | Fold {fold}: "
                f"ACC={metrics['ACC']:.4f}, "
                f"AUC={metrics['AUC']:.4f}, "
                f"SEN={metrics['SEN']:.4f}, "
                f"SPE={metrics['SPE']:.4f}, "
                f"F1={metrics['F1']:.4f}"
            )

    summary = summarize_results(all_results)

    print("\n========== Final Result ==========")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    return all_results, summary