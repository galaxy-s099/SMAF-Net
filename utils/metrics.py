import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score


def compute_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    sen = recall_score(y_true, y_pred, pos_label=1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp + 1e-8)

    f1 = f1_score(y_true, y_pred)

    return {
        "ACC": acc,
        "AUC": auc,
        "SEN": sen,
        "SPE": spe,
        "F1": f1,
    }


def summarize_results(results):
    metric_keys = ["ACC", "AUC", "SEN", "SPE", "F1"]
    summary = {}

    for key in metric_keys:
        values = np.array([r[key] for r in results])
        summary[key + "_mean"] = values.mean()
        summary[key + "_std"] = values.std()

    return summary