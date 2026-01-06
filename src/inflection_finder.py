# inflection_finder.py
import math
import numpy as np


# ---------- helpers for minima / rises ----------

def _local_minima_idxs(y: np.ndarray):
    """0-based indices of simple local minima."""
    out = []
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] <= y[i + 1]:
            out.append(i)
    return out


def _suffix_record_low_minima(min_idxs, y):
    """Keep minima that are record lows when scanning right->left. Return in ascending order."""
    kept, best = [], math.inf
    for idx in reversed(min_idxs):
        if y[idx] < best:
            kept.append(idx)
            best = y[idx]
    kept.reverse()
    return kept


def _compute_rises_slopes(kept_idxs, y):
    """
    Between consecutive kept minima (ascending indices):
      from_idx -> to_idx  (1-based reporting)
      rise  = y[to] - y[from]
      slope = rise / (to - from)
    """
    froms, tos, rises, slopes = [], [], [], []
    for a, b in zip(kept_idxs[:-1], kept_idxs[1:]):
        froms.append(a + 1)
        tos.append(b + 1)
        rise = float(y[b] - y[a])
        slope = rise / float(b - a)
        rises.append(rise)
        slopes.append(slope)
    return froms, tos, rises, slopes


def _pick_estimate_from_rises(froms, rises):
    """
    Scan rises left->right; on first decrease, return start of previous segment.
    If no decrease, return start of segment with largest rise.
    If empty, return None.
    """
    if not rises:
        return None
    for j in range(1, len(rises)):
        if rises[j] < rises[j - 1]:
            return froms[j - 1]
    return froms[int(np.argmax(rises))]


# ---------- main API ----------

def find_inflection(errs, g: int = 4):
    """
    Compute θ̂ using filtered minima and rises.

    Args:
        errs: 2D array-like (runs × windows). Medians across runs taken on axis=0.
        g: plot limit parameter (not used in this version, kept for compatibility).

    Returns:
        tuple: (theta_hat, medians, all_min_idxs, kept_min_idxs, froms, tos, rises, slopes)
               where theta_hat is (int, 1-based) or None if not enough minima to estimate.
    """
    errs = np.asarray(errs)
    medians = np.median(errs, axis=0)

    # minima & filtered minima
    all_min_idxs = _local_minima_idxs(medians)
    kept_min_idxs = _suffix_record_low_minima(all_min_idxs, medians)

    # rises/slopes & estimate
    froms, tos, rises, slopes = _compute_rises_slopes(kept_min_idxs, medians)
    theta_hat = _pick_estimate_from_rises(froms, rises)

    return theta_hat, medians, all_min_idxs, kept_min_idxs, froms, tos, rises, slopes

