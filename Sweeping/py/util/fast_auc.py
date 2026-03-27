from __future__ import annotations
import numpy as np
from typing import Optional, Sequence


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """
    Compute average ranks for ties (1-based ranks), vectorized.
    Returns an array of ranks aligned to `scores` order.
    """
    # Stable sort to keep tie groups deterministic
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    # Find tie groups and their counts
    uniq, counts = np.unique(sorted_scores, return_counts=True)
    cum_counts = np.cumsum(counts)
    starts = cum_counts - counts + 1  # 1-based
    ends = cum_counts                 # inclusive
    avg_ranks_per_group = (starts + ends) / 2.0

    # Map each element to its tie-group average rank
    group_ids = np.repeat(np.arange(uniq.size), counts)
    ranks_sorted = avg_ranks_per_group[group_ids]

    # Scatter back to original positions
    ranks = np.empty_like(ranks_sorted, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Fast binary ROC AUC via Mann–Whitney U with average-tie ranks.
    - y_true: array-like with 2 unique labels.
              If labels are not {0,1}, the "positive" class is taken
              as the lexicographically larger label (consistent with sklearn
              when pos_label=None and y_score is for the larger label).
    - y_score: array-like of scores/probabilities for the positive class.

    Raises:
        ValueError if y_true does not contain exactly 2 classes or if one class is missing.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    classes, y_idx = np.unique(y_true, return_inverse=True)
    if classes.size != 2:
        raise ValueError("binary_auc requires exactly 2 classes in y_true")
    n_pos = np.sum(y_idx == 1)
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("binary_auc requires at least one positive and one negative sample")

    ranks = _average_ranks(y_score)
    r_pos = ranks[y_idx == 1].sum()
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # numeric guard
    if auc < 0.0:
        auc = 0.0
    elif auc > 1.0:
        auc = 1.0
    return float(auc)


def multiclass_ovr_auc(
    y_true: np.ndarray,
    y_score_2d: np.ndarray,
    *,
    average: str = "weighted",
    classes: Optional[Sequence] = None,
) -> float:
    """
    Multiclass ROC AUC using One-vs-Rest and either 'weighted' or 'macro' averaging.

    Args:
        y_true: shape (n,)
        y_score_2d: shape (n, C) score/proba matrix, columns correspond to labels in `classes`
                    (or to sorted unique(y_true) if `classes` is None).
        average: 'weighted' | 'macro'
        classes: sequence of class labels for columns in y_score_2d (len == C).
                 If None, assumed to match np.unique(y_true).

    Raises:
        ValueError for mismatched shapes, missing classes in y_true, or invalid averaging.
    """
    y_true = np.asarray(y_true)
    S = np.asarray(y_score_2d)

    if S.ndim != 2:
        raise ValueError("multiclass_ovr_auc requires a 2D score matrix (n_samples, n_classes)")

    if classes is None:
        classes = np.unique(y_true)
    classes = np.asarray(classes)
    C = classes.size
    if S.shape[1] != C:
        raise ValueError(
            f"Score matrix columns ({S.shape[1]}) do not match number of classes ({C})"
        )

    # Map y_true labels to indices [0..C-1] according to `classes`
    # Build a dict for mapping
    class_to_index = {c: i for i, c in enumerate(classes)}
    try:
        y_idx = np.array([class_to_index[c] for c in y_true], dtype=int)
    except KeyError as e:
        raise ValueError(f"Label {e} in y_true is not present in provided `classes`.") from None

    counts = np.bincount(y_idx, minlength=C)
    if (counts == 0).any():
        raise ValueError("Each class must be present at least once in y_true for OvR AUC.")

    aucs = np.empty(C, dtype=float)
    for k in range(C):
        # positive: class k vs rest
        aucs[k] = binary_auc(y_idx == k, S[:, k])

    if average == "weighted":
        weights = counts / counts.sum()
        auc = float(np.dot(aucs, weights))
    elif average == "macro":
        auc = float(aucs.mean())
    else:
        raise ValueError("average must be 'weighted' or 'macro'")

    return auc


def normalize_auc_to_importance(auc: float) -> float:
    """
    Convert AUC to importance scale in [0,1]: max(0, 2*AUC - 1).
    """
    return float(max(0.0, 2.0 * auc - 1.0))