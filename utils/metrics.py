"""
Evaluation metrics for ProtoEvoNet.

Provides accuracy, per-class metrics, AUROC for novelty detection, and
episode-level accuracy used during Phase 2 meta-learning evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Classification accuracy
# ---------------------------------------------------------------------------

def accuracy(
    logits_or_dists: torch.Tensor,
    labels: torch.Tensor,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Compute top-k accuracy from logits or negative-distance scores.

    Parameters
    ----------
    logits_or_dists:
        Shape ``(N, C)`` — higher scores are treated as more confident.
    labels:
        Shape ``(N,)`` — integer class indices.
    top_k:
        Which k values to compute accuracy for.

    Returns
    -------
    dict
        E.g. ``{"top1": 0.84, "top5": 0.97}``.
    """
    max_k = max(top_k)
    batch_size = labels.size(0)

    _, pred = logits_or_dists.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (max_k, N)
    correct = pred.eq(labels.view(1, -1).expand_as(pred))  # (max_k, N)

    results: Dict[str, float] = {}
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[f"top{k}"] = float(correct_k.mul_(100.0 / batch_size).item())
    return results


def per_class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """
    Return per-class accuracy as a dict mapping class_id → accuracy (%).

    Parameters
    ----------
    predictions:
        Shape ``(N,)`` — predicted class indices.
    labels:
        Shape ``(N,)`` — ground-truth class indices.
    num_classes:
        Total number of classes.
    """
    correct = predictions.eq(labels)
    per_class: Dict[int, float] = {}
    for c in range(num_classes):
        mask = labels.eq(c)
        if mask.sum().item() == 0:
            per_class[c] = float("nan")
        else:
            per_class[c] = float(correct[mask].float().mean().item() * 100.0)
    return per_class


# ---------------------------------------------------------------------------
# Episode accuracy (few-shot)
# ---------------------------------------------------------------------------

def episode_accuracy(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    """
    Compute accuracy for one few-shot episode using nearest-prototype
    (Euclidean) assignment.

    Parameters
    ----------
    query_embeddings:
        Shape ``(Q, D)`` — L2-normalised query embeddings.
    prototypes:
        Shape ``(N, D)`` — L2-normalised class prototypes.
    query_labels:
        Shape ``(Q,)`` — ground-truth class indices in ``[0, N)``.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    # Euclidean distance in L2-normalised space equals sqrt(2(1 - cos_sim))
    dists = torch.cdist(query_embeddings, prototypes)  # (Q, N)
    preds = dists.argmin(dim=1)                         # (Q,)
    acc = preds.eq(query_labels).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# Novelty detection — AUROC
# ---------------------------------------------------------------------------

def compute_auroc(
    scores: torch.Tensor,
    is_novel: torch.Tensor,
) -> float:
    """
    Compute Area Under the ROC Curve for novelty detection.

    Parameters
    ----------
    scores:
        Shape ``(N,)`` — higher score = more novel.
    is_novel:
        Shape ``(N,)`` — Boolean / 0-1 ground-truth novelty labels.

    Returns
    -------
    float
        AUROC in [0, 1].
    """
    scores = scores.float().cpu()
    labels = is_novel.float().cpu()

    # Sort by descending novelty score
    sorted_idx = scores.argsort(descending=True)
    labels_sorted = labels[sorted_idx]

    n_pos = labels.sum().item()
    n_neg = (1 - labels).sum().item()

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = labels_sorted.cumsum(0)
    fp = (1 - labels_sorted).cumsum(0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # Prepend (0, 0)
    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])

    # Trapezoidal integration
    auroc = torch.trapezoid(tpr, fpr).item()
    return float(auroc)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Build a ``(num_classes, num_classes)`` confusion matrix.

    Parameters
    ----------
    predictions:
        Shape ``(N,)`` — predicted class indices.
    labels:
        Shape ``(N,)`` — ground-truth class indices.
    num_classes:
        Total number of classes.

    Returns
    -------
    torch.Tensor
        Integer confusion matrix of shape ``(num_classes, num_classes)``.
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(labels.view(-1), predictions.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


# ---------------------------------------------------------------------------
# Novelty detection — FPR at fixed TPR (OOD detection standard)
# ---------------------------------------------------------------------------

def fpr_at_tpr(
    scores: torch.Tensor,
    is_novel: torch.Tensor,
    tpr_target: float = 0.95,
) -> float:
    """
    False-positive rate at a fixed true-positive rate level.

    This is the standard metric reported in out-of-distribution detection
    papers (e.g. FPR@95TPR).  A *lower* value is better.

    Parameters
    ----------
    scores:
        Shape ``(N,)`` — higher score = more novel.
    is_novel:
        Shape ``(N,)`` — Boolean / 0-1 ground-truth novelty labels.
    tpr_target:
        TPR level at which to read off the FPR (default 0.95).

    Returns
    -------
    float
        FPR in [0, 1].  Returns ``nan`` if either class is absent.
    """
    scores = scores.float().cpu()
    labels = is_novel.float().cpu()

    sorted_idx = scores.argsort(descending=True)
    labels_sorted = labels[sorted_idx]

    n_pos = labels.sum().item()
    n_neg = (1 - labels).sum().item()

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = labels_sorted.cumsum(0)
    fp = (1 - labels_sorted).cumsum(0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    mask = tpr >= tpr_target
    if not mask.any():
        return 1.0

    return float(fpr[mask][0].item())


# ---------------------------------------------------------------------------
# Per-class F1, precision, recall
# ---------------------------------------------------------------------------

def per_class_prf(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """
    Per-class precision, recall and F1.

    Parameters
    ----------
    predictions:
        Shape ``(N,)`` — predicted class indices.
    labels:
        Shape ``(N,)`` — ground-truth class indices.
    num_classes:
        Total number of classes.

    Returns
    -------
    dict
        ``{class_id: {"precision": …, "recall": …, "f1": …}}``.
    """
    results: Dict[int, Dict[str, float]] = {}
    for c in range(num_classes):
        tp = ((predictions == c) & (labels == c)).sum().item()
        fp = ((predictions == c) & (labels != c)).sum().item()
        fn = ((predictions != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        results[c] = {"precision": precision, "recall": recall, "f1": f1}
    return results


# ---------------------------------------------------------------------------
# Utility: format metric dict for logging
# ---------------------------------------------------------------------------

def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Return a compact human-readable string of metric values."""
    parts = [f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)
