"""
Piece utilities for data segmentation
"""

import numpy as np


def get_piece_starting_indices(y, piece_len: int, overlap_ratio: float):
    """
    Generate array of starting indices for each piece of y.
    
    Args:
        y: input array
        piece_len: length of each piece
        overlap_ratio: ratio of overlap between pieces
        
    Returns:
        List of starting indices for each piece
    """
    if piece_len <= 0:
        raise ValueError("piece_len must be a positive integer")
    
    n = len(y)
    step = piece_len * (1 - overlap_ratio)
    
    # Only windows that fully fit within y
    starting_indices = []
    for i in range(0, n - piece_len + 1, int(step)):
        starting_indices.append(i)
    
    return starting_indices



def get_num_pieces(y, piece_len: int, overlap_ratio: float = 0.0):
    """
    Get the total number of pieces that can be extracted from y.
    
    Args:
        y: input array
        piece_len: length of each piece
        overlap_ratio: ratio of overlap between pieces
        
    Returns:
        Number of pieces
    """
    starting_indices = get_piece_starting_indices(y, piece_len, overlap_ratio)
    return len(starting_indices)


def extract_and_resample_window(
    x_full,          # np.ndarray, shape (N,)
    t_full,          # np.ndarray, shape (N,), timestamps in seconds
    center_time,     # float, time of y_i
    theta_seconds,   # float, window length in seconds
    L                # int, fixed resampled length
) -> np.ndarray:
    """
    Extract a time-based window from x_full and resample it to fixed length L.
    
    Args:
        x_full: IBI sequence, shape (N,)
        t_full: timestamps in seconds, shape (N,)
        center_time: center time of the window (time of y_i)
        theta_seconds: window length in seconds
        L: fixed resampled length
    
    Returns:
        x_window_L: np.ndarray of shape (L,)
    
    Raises:
        ValueError: if window is empty
    """
    # Select window based on time: [center_time - theta_seconds, center_time]
    window_start = center_time - theta_seconds
    window_end = center_time
    
    # Find indices where t_full is in the window
    mask = (t_full >= window_start) & (t_full <= window_end)
    x_window = x_full[mask]
    
    # Check if window is empty
    if len(x_window) == 0:
        raise ValueError(f"Window is empty: center_time={center_time}, "
                        f"theta_seconds={theta_seconds}, "
                        f"window_range=[{window_start}, {window_end}]")
    
    # Get corresponding time points for the selected window
    t_window = t_full[mask]
    
    # Normalize time axis to [0, theta_seconds] for interpolation
    # Original sample points' time axis
    t_window_normalized = t_window - window_start  # Shift to start at 0
    
    # Target time axis: evenly spaced points in [0, theta_seconds]
    t_target = np.linspace(0, theta_seconds, L)
    
    # Resample using linear interpolation
    x_window_L = np.interp(t_target, t_window_normalized, x_window)
    
    return x_window_L
