from pathlib import Path

import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Class Weights
# =========================================================

def compute_pos_weight_from_dataset(dataset, target_classes):
    """
    Compute pos_weight for BCEWithLogitsLoss.

    Formula:
        pos_weight = negative_count / positive_count

    A higher value means positive examples for that class are rarer.
    """

    positive_counts = dataset.get_positive_counts()
    total_samples = len(dataset)

    pos_weights = []

    for class_name in target_classes:
        positive_count = int(positive_counts.get(class_name, 0))
        negative_count = total_samples - positive_count

        if positive_count <= 0:
            weight = 1.0
        else:
            weight = negative_count / positive_count

        pos_weights.append(float(weight))

    return torch.tensor(pos_weights, dtype=torch.float32)


# =========================================================
# Training / Validation
# =========================================================

def train_multilabel_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch=None,
):
    """
    Train multi-label model for one epoch.
    """

    model.train()

    running_loss = 0.0
    total_samples = 0

    iterator = dataloader

    if tqdm is not None:
        desc = "Training" if epoch is None else f"Training epoch {epoch}"
        iterator = tqdm(dataloader, desc=desc, leave=False)

    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    average_loss = running_loss / max(total_samples, 1)

    return average_loss


@torch.no_grad()
def validate_multilabel_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    target_classes=None,
    epoch=None,
):
    """
    Validate multi-label model for one epoch.
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_probabilities = []

    iterator = dataloader

    if tqdm is not None:
        desc = "Validation" if epoch is None else f"Validation epoch {epoch}"
        iterator = tqdm(dataloader, desc=desc, leave=False)

    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        probabilities = torch.sigmoid(outputs)
        predictions = sigmoid_outputs_to_predictions(outputs, threshold=threshold)

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(targets.detach().cpu())
        all_predictions.append(predictions.detach().cpu())
        all_probabilities.append(probabilities.detach().cpu())

    average_loss = running_loss / max(total_samples, 1)

    targets_tensor = torch.cat(all_targets, dim=0)
    predictions_tensor = torch.cat(all_predictions, dim=0)
    probabilities_tensor = torch.cat(all_probabilities, dim=0)

    metrics = compute_multilabel_metrics_from_predictions(
        y_true=targets_tensor,
        y_pred=predictions_tensor,
        target_classes=target_classes,
    )

    metrics["validation_loss"] = float(average_loss)

    return average_loss, metrics, targets_tensor, predictions_tensor, probabilities_tensor


def sigmoid_outputs_to_predictions(outputs, threshold=0.5):
    """
    Convert raw logits to binary predictions using sigmoid + threshold.
    """

    probabilities = torch.sigmoid(outputs)
    predictions = (probabilities >= threshold).float()

    return predictions


# =========================================================
# Metrics
# =========================================================

def compute_per_class_multilabel_precision_recall_f1(y_true, y_pred, target_classes):
    """
    Compute per-class precision, recall, F1, and support.

    y_true and y_pred shape:
        [num_samples, num_classes]
    """

    y_true = y_true.float()
    y_pred = y_pred.float()

    per_class_metrics = {}

    for class_index, class_name in enumerate(target_classes):
        true_col = y_true[:, class_index]
        pred_col = y_pred[:, class_index]

        true_positive = ((true_col == 1) & (pred_col == 1)).sum().item()
        false_positive = ((true_col == 0) & (pred_col == 1)).sum().item()
        false_negative = ((true_col == 1) & (pred_col == 0)).sum().item()
        true_negative = ((true_col == 0) & (pred_col == 0)).sum().item()

        support = int((true_col == 1).sum().item())

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_negative": int(true_negative),
        }

    return per_class_metrics


def compute_macro_micro_weighted_f1(per_class_metrics):
    """
    Compute macro, micro, and weighted precision/recall/F1.
    """

    if not per_class_metrics:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }

    precisions = []
    recalls = []
    f1_values = []
    supports = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for metrics in per_class_metrics.values():
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_values.append(metrics["f1"])
        supports.append(metrics["support"])

        total_tp += metrics["true_positive"]
        total_fp += metrics["false_positive"]
        total_fn += metrics["false_negative"]

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1_values) / len(f1_values)

    total_support = sum(supports)

    if total_support > 0:
        weighted_precision = sum(
            p * s for p, s in zip(precisions, supports)
        ) / total_support

        weighted_recall = sum(
            r * s for r, s in zip(recalls, supports)
        ) / total_support

        weighted_f1 = sum(
            f * s for f, s in zip(f1_values, supports)
        ) / total_support

    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }


def compute_multilabel_metrics_from_predictions(y_true, y_pred, target_classes):
    """
    Compute full multi-label validation metrics.

    Includes:
    - subset accuracy / exact-match accuracy
    - hamming accuracy
    - per-class metrics
    - micro/macro/weighted metrics
    """

    y_true = y_true.float()
    y_pred = y_pred.float()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )

    num_samples = y_true.shape[0]
    num_classes = y_true.shape[1]

    exact_matches = (y_true == y_pred).all(dim=1).sum().item()
    subset_accuracy = exact_matches / max(num_samples, 1)

    correct_labels = (y_true == y_pred).sum().item()
    total_labels = num_samples * num_classes
    hamming_accuracy = correct_labels / max(total_labels, 1)

    per_class_metrics = compute_per_class_multilabel_precision_recall_f1(
        y_true=y_true,
        y_pred=y_pred,
        target_classes=target_classes,
    )

    aggregate_metrics = compute_macro_micro_weighted_f1(per_class_metrics)

    metrics = {
        "subset_accuracy": float(subset_accuracy),
        "hamming_accuracy": float(hamming_accuracy),
        "per_class_metrics": per_class_metrics,
    }

    metrics.update(aggregate_metrics)

    return metrics


# =========================================================
# Checkpoint Utilities
# =========================================================

def save_checkpoint(path, checkpoint):
    """
    Save PyTorch checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    """
    Load PyTorch checkpoint.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================================================
# Plotting
# =========================================================

def plot_multilabel_training_curves(history, output_path, title):
    """
    Plot multi-label training curves.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        raise ValueError("Cannot plot empty history.")

    epochs = [row["epoch"] for row in history]

    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["validation_loss"] for row in history]
    val_macro_f1 = [row["validation_macro_f1"] for row in history]
    val_micro_f1 = [row["validation_micro_f1"] for row in history]
    val_subset_accuracy = [row["validation_subset_accuracy"] for row in history]
    val_hamming_accuracy = [row["validation_hamming_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, val_macro_f1, marker="o", label="Validation Macro-F1")
    axes[1].plot(epochs, val_micro_f1, marker="o", label="Validation Micro-F1")
    axes[1].set_title("F1 Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, val_subset_accuracy, marker="o", label="Subset Accuracy")
    axes[2].plot(epochs, val_hamming_accuracy, marker="o", label="Hamming Accuracy")
    axes[2].set_title("Accuracy Curves")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_history_csv(history, output_path):
    """
    Save training history CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path, index=False)
