import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit
from copy import deepcopy

from data.abide_dataset import ABIDEMultiAtlasDataset, load_labels
from models.smaf_net import SMAFNetV1
from models.smaf_edge_net import SMAFEdgeNet
from models.smaf_uncertainty_net import SMAFUncertaintyNet
from models.smaf_gated_net import SMAFGatedNet
from utils.seed import set_seed
from utils.metrics import compute_metrics, summarize_results

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)

            for key in ["aal", "cc200", "ho"]:
                batch[key] = batch[key].to(device)

            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]

            preds = (probs >= threshold).long()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds)
    )

    return metrics, np.array(all_probs), np.array(all_labels)

def search_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_acc = -1.0

    for threshold in np.arange(0.30, 0.71, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        acc = (y_pred == y_true).mean()

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return float(best_threshold), float(best_acc)

def train_one_fold(
    data_root,
    train_idx,
    test_idx,
    seed,
    config,
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

    y_all = load_labels(data_root)

    # 从 train_idx 里再划分 train_sub / val_sub
    use_best_val = config["train"].get("use_best_val", False)
    val_ratio = config["train"].get("val_ratio", 0.15)

    if use_best_val:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed
        )

        train_local_idx, val_local_idx = next(
            splitter.split(np.zeros(len(train_idx)), y_all[train_idx])
        )

        train_sub_idx = train_idx[train_local_idx]
        val_sub_idx = train_idx[val_local_idx]
    else:
        train_sub_idx = train_idx
        val_sub_idx = test_idx

    train_dataset = ABIDEMultiAtlasDataset(data_root, train_sub_idx)
    val_dataset = ABIDEMultiAtlasDataset(data_root, val_sub_idx)
    test_dataset = ABIDEMultiAtlasDataset(data_root, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model_name = config["model"].get("model_name", "smaf_v1")

    if model_name == "smaf_edge_v4":
        model = SMAFEdgeNet(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_heads=config["model"].get("num_heads", 4)
        ).to(device)

    elif model_name == "smaf_uncertainty_v4_2":
        model = SMAFUncertaintyNet(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            temperature=config["model"].get("temperature", 1.0)
        ).to(device)

    elif model_name == "smaf_gated_v5":
        model = SMAFGatedNet(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout
        ).to(device)

    else:
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

    best_val_acc = -1.0
    best_state = None
    best_threshold = 0.5
    best_epoch = -1

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

        if use_best_val:
            val_metrics, val_probs, val_labels = evaluate_model(
                model,
                val_loader,
                device,
                threshold=0.5
            )

            threshold, threshold_acc = search_best_threshold(
                val_labels,
                val_probs
            )

            # 以 validation ACC 为选择依据
            if threshold_acc > best_val_acc:
                best_val_acc = threshold_acc
                best_threshold = threshold
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch

    if use_best_val and best_state is not None:
        model.load_state_dict(best_state)

        test_metrics, _, _ = evaluate_model(
            model,
            test_loader,
            device,
            threshold=best_threshold
        )

        test_metrics["Best_Epoch"] = best_epoch
        test_metrics["Best_Threshold"] = best_threshold
        test_metrics["Val_ACC"] = best_val_acc

        return test_metrics

    else:
        test_metrics, _, _ = evaluate_model(
            model,
            test_loader,
            device,
            threshold=0.5
        )

        return test_metrics


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
                config=config,
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