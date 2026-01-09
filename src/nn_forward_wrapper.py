"""
NN forward wrapper for Phase C: Inverse optimization with frozen NN surrogate.

This module provides differentiable forward functions that slide a window-level NN
across time-based windows to generate full-sequence predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from src.piece_utils import extract_and_resample_window
from src.nn_forward import WindowNN


def torch_linear_resample_1d(x_varlen: torch.Tensor, L: int) -> torch.Tensor:
    """
    Differentiable 1D linear resampling from variable-length sequence to fixed length L.
    
    Args:
        x_varlen: torch.Tensor of shape (n,) where n can vary (variable length)
        L: target fixed length
    
    Returns:
        x_L: torch.Tensor of shape (L,)
    
    Raises:
        ValueError: if input is empty
    
    Implementation notes:
        - Uses piecewise linear interpolation to maintain gradients w.r.t. x_varlen
        - For n=1, repeats the single value L times
        - For n>1, performs linear interpolation between adjacent points
        - Vectorized implementation for efficiency
    """
    n = len(x_varlen)
    
    # Handle empty case
    if n == 0:
        raise ValueError(f"Cannot resample empty sequence to length {L}")
    
    # Handle single point case: repeat the value
    if n == 1:
        return x_varlen.repeat(L)
    
    # For n > 1: perform linear interpolation
    # Original positions: [0, 1, 2, ..., n-1] normalized to [0, 1]
    # Target positions: evenly spaced in [0, 1]
    u = torch.linspace(0.0, 1.0, n, device=x_varlen.device, dtype=x_varlen.dtype)
    v = torch.linspace(0.0, 1.0, L, device=x_varlen.device, dtype=x_varlen.dtype)
    
    # Vectorized implementation: find indices for each v_j
    # Expand dimensions for broadcasting: v (L,), u (n,)
    v_expanded = v.unsqueeze(1)  # (L, 1)
    u_expanded = u.unsqueeze(0)  # (1, n)
    
    # Find which interval each v_j falls into
    # For each v_j, find the largest i such that u[i] <= v_j
    # Create a mask: u[i] <= v_j
    mask = u_expanded <= v_expanded  # (L, n)
    
    # Get left indices (largest i where u[i] <= v_j)
    # Sum over axis 1 gives the count of u[i] <= v_j, minus 1 gives the index
    indices_left = mask.long().sum(dim=1) - 1  # (L,)
    
    # Handle boundary cases: v_j < u[0] or v_j >= u[n-1]
    boundary_left = (v < u[0])
    boundary_right = (v >= u[n-1])
    
    # Clamp indices to valid range [0, n-2] (for accessing pairs)
    # But we'll override with boundary handling
    indices_left = torch.clamp(indices_left, 0, n - 2)
    
    # Get left and right values for interpolation
    indices_right = indices_left + 1
    
    x_left = x_varlen[indices_left]  # (L,)
    x_right = x_varlen[indices_right]  # (L,)
    u_left = u[indices_left]  # (L,)
    u_right = u[indices_right]  # (L,)
    
    # Compute interpolation weights
    u_diff = u_right - u_left
    # Avoid division by zero (shouldn't happen for properly spaced u, but safety first)
    u_diff = torch.clamp(u_diff, min=1e-10)
    alpha = (v - u_left) / u_diff  # (L,)
    
    # Linear interpolation
    x_L = x_left + alpha * (x_right - x_left)
    
    # Apply boundary conditions (override interpolation at boundaries)
    x_L = torch.where(boundary_left, x_varlen[0], x_L)
    x_L = torch.where(boundary_right, x_varlen[n-1], x_L)
    
    return x_L


def load_frozen_window_nn(
    model_path: str, 
    L: int, 
    hidden_dim: int = 16, 
    device: str = "cpu"
) -> torch.nn.Module:
    """
    Load WindowNN from .pt file, set to eval mode, freeze parameters.
    
    Args:
        model_path: path to saved model state dict
        L: input dimension (fixed window length)
        hidden_dim: hidden dimension (must match training)
        device: device to load model on
    
    Returns:
        model: frozen WindowNN model on specified device
    """
    model = WindowNN(input_dim=L, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    return model


@torch.no_grad()
def nn_forward_full_sequence_no_grad(
    x_full_np: np.ndarray,    # (N,)
    t_full_np: np.ndarray,    # (N,)
    t_y_np: np.ndarray,       # (M,)
    theta_seconds: float,
    model: torch.nn.Module,
    L: int,
    time_offset: float = 0.0,
) -> np.ndarray:
    """
    Fast forward for evaluation only (no gradient). Returns y_hat (M-1,).
    
    Args:
        x_full_np: IBI sequence, shape (N,)
        t_full_np: timestamps in seconds, shape (N,)
        t_y_np: target time points for y, shape (M,) - uses indices 1 to M-1 (skips index 0)
        theta_seconds: window length in seconds
        model: frozen WindowNN model
        L: fixed window length for resampling
        time_offset: optional time offset (default 0.0, reserved for future use)
    
    Returns:
        y_hat: predicted y values, shape (M-1,) - skips index 0 (boundary effect, same as Phase B)
    """
    M = len(t_y_np)
    y_hat = []
    device = next(model.parameters()).device
    
    # Skip index 0 (boundary effect, same as Phase B)
    valid_indices = range(1, M)
    
    for i in valid_indices:
        center_time = t_y_np[i] + time_offset
        
        # Use numpy-based extraction (fast, no gradient needed)
        try:
            x_window_L_np = extract_and_resample_window(
                x_full_np, t_full_np, center_time, theta_seconds, L
            )
            
            # Convert to tensor and predict
            x_window_L_t = torch.FloatTensor(x_window_L_np).unsqueeze(0).to(device)
            y_pred = model(x_window_L_t).squeeze().cpu().item()
            y_hat.append(y_pred)
        except ValueError:
            # Skip if window is empty (should not happen for i > 0, but handle gracefully)
            raise ValueError(f"Empty window at index {i}, center_time={center_time:.2f}")
    
    return np.array(y_hat)


def nn_forward_full_sequence_with_grad(
    x_full_t: torch.Tensor,   # (N,), requires_grad=True
    t_full_np: np.ndarray,    # (N,), numpy (treated as constants)
    t_y_np: np.ndarray,       # (M,), numpy
    theta_seconds: float,
    model: torch.nn.Module,
    L: int,
    device: str = "cpu",
    time_offset: float = 0.0,
) -> torch.Tensor:
    """
    Differentiable forward used inside SGD. Returns y_hat_t: torch.Tensor shape (M-1,).
    
    Args:
        x_full_t: IBI sequence as torch tensor, shape (N,), requires_grad=True
        t_full_np: timestamps in seconds, shape (N,), numpy (constants)
        t_y_np: target time points for y, shape (M,), numpy - uses indices 1 to M-1 (skips index 0)
        theta_seconds: window length in seconds
        model: frozen WindowNN model (must be on same device as x_full_t)
        L: fixed window length for resampling
        device: device for computation
        time_offset: optional time offset (default 0.0, reserved for future use)
    
    Returns:
        y_hat_t: predicted y values, shape (M-1,), torch.Tensor with gradients - skips index 0 (boundary effect, same as Phase B)
    
    Notes:
        - Model parameters must be frozen (no gradients through model)
        - Window extraction uses numpy indices computed from timestamps
        - Resampling step uses torch-based linear interpolation (differentiable)
        - All operations maintain gradient flow w.r.t. x_full_t
    """
    M = len(t_y_np)
    y_hat_list = []
    
    # Ensure x_full_t is on correct device
    x_full_t = x_full_t.to(device)
    
    # Skip index 0 (boundary effect, same as Phase B)
    valid_indices = range(1, M)
    
    for i in valid_indices:
        center_time = t_y_np[i] + time_offset
        
        # Find window indices using numpy (constant, no gradient)
        window_start = center_time - theta_seconds
        window_end = center_time
        mask = (t_full_np >= window_start) & (t_full_np <= window_end)
        
        # Extract window from x_full_t (this maintains gradients)
        x_window_varlen = x_full_t[mask]
        
        # Check if window is empty
        if len(x_window_varlen) == 0:
            # Should not happen for i > 0 (since we skip index 0), but handle gracefully
            raise ValueError(
                f"Empty window at index {i}, center_time={center_time:.2f}, "
                f"window_range=[{window_start:.2f}, {window_end:.2f}]"
            )
        
        # Resample to fixed length L using differentiable torch interpolation
        x_window_L = torch_linear_resample_1d(x_window_varlen, L)
        
        # Add batch dimension and predict
        x_window_L_batch = x_window_L.unsqueeze(0)  # (1, L)
        y_pred = model(x_window_L_batch).squeeze()  # (1,) -> scalar
        
        y_hat_list.append(y_pred)
    
    # Stack all predictions into (M-1,) tensor (skipped index 0)
    if len(y_hat_list) == 0:
        raise ValueError("No valid predictions generated (all windows were empty)")
    
    y_hat_t = torch.stack(y_hat_list)
    return y_hat_t

