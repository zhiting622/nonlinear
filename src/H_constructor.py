

import numpy as np
from math import floor

# Random seed is set in main_func_test.py for reproducibility

def _stochastic_round(value):
    base = floor(value)
    frac = value - base
    return base + (np.random.rand() < frac)

def _quantize_overlap_to_1_over_g(x, g):
    """
    Quantize x in [0,1] to {0, 1/g, 2/g, ..., 1} with unbiased stochastic rounding.
    """
    xg   = x * g
    base = floor(xg)
    frac = xg - base
    k    = base + (np.random.rand() < frac)  # choose base or base+1
    if k > g:  # only happens if x == 1 and numerical noise; clamp
        k = g
    return k / g

def construct_H(y, theta_hat, s, g=3):
    """
    Build H with cumulative left-hand zeros.
    Edge overlaps (alpha, beta) are quantized to multiples of 1/g
    before normalization; interior full blocks remain 1.

    Parameters
    ----------
    y          : 1-D array-like, seconds per beat (RR)
    theta_hat  : float, window length θ [s]
    s          : float, sampling interval between successive readings [s]
    g          : int, beats per coarse block (default 3)

    Returns
    -------
    H: list[list[float]]  (rows = len(y), tall-and-thin)
    """
    y = np.asarray(y, dtype=float)
    rows = []
    cum_blocks = 0.0  # blocks completed up to current row

    for beat_period in y:
        # 1) window length in "g-beat blocks"
        blocks_in_window = theta_hat / (g * beat_period)

        j_first = _stochastic_round(cum_blocks)
        j_last  = _stochastic_round(cum_blocks + blocks_in_window)

        # 2) assemble the row
        row = [0.0] * j_first
        raw_overlaps = []
        # compute raw overlaps for each touched unit block [j, j+1]
        win_start = cum_blocks
        win_end   = cum_blocks + blocks_in_window
        for j in range(j_first, j_last + 1):
            blk_start = j
            blk_end   = j + 1
            overlap = max(0.0, min(win_end, blk_end) - max(win_start, blk_start))
            if overlap > 0.0:
                raw_overlaps.append(overlap)

        if raw_overlaps:
            # Quantize every overlap to multiples of 1/g (edges become {k/g}, interior 1 stays 1)
            q_overlaps = [_quantize_overlap_to_1_over_g(w, g) for w in raw_overlaps]
            denom_q = sum(q_overlaps)

            # Safety: ensure at least one nonzero weight survives quantization
            if denom_q == 0.0:
                # promote the largest raw overlap to 1/g
                idx = int(np.argmax(raw_overlaps))
                q_overlaps[idx] = 1.0 / g
                denom_q = 1.0 / g

            # Normalize (sum to 1), then scale by 1/g (block-sum parameterization)
            row.extend([(wq / denom_q) / g for wq in q_overlaps])

        rows.append(row)

        # 3) advance cumulative block counter
        beats_since_last = s / beat_period
        cum_blocks += beats_since_last / g

    # 4) right-pad rows to equal length
    max_len = max(len(r) for r in rows)
    H = [r + [0.0] * (max_len - len(r)) for r in rows]
    return H
