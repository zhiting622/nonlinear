"""
Evaluate NN forward surrogate on S2 data
Load trained model and evaluate on S2 data, plotting y_true vs y_pred curves.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import load_x
from src.y_generator import generate_y
from src.nn_forward import WindowNN
from src.piece_utils import extract_and_resample_window


def eval_nn_forward(
    subject_name="S2",
    model_path=None,
    theta_seconds=10.0,
    L=32,
    hidden_dim=16,
    s=2
):
    """
    Evaluate NN forward surrogate on test data.
    
    Args:
        subject_name: subject identifier (e.g., "S2")
        model_path: path to saved model. If None, uses default path.
        theta_seconds: window length in seconds (must match training)
        L: fixed input length (must match training)
        hidden_dim: hidden dimension of NN (must match training)
        s: sampling interval for generating y
    """
    print("=" * 60)
    print("Evaluating NN Forward Surrogate")
    print("=" * 60)
    print(f"Subject: {subject_name}")
    print(f"Theta (seconds): {theta_seconds}")
    print(f"Fixed window length L: {L}")
    print()
    
    # Step 1: Load IBI sequence
    print("Step 1: Loading IBI sequence...")
    x_full = load_x(subject_name)
    print(f"  Loaded x_full with length: {len(x_full)}")
    
    # Generate time stamps from cumulative sum of IBI
    t_full = np.cumsum(x_full)
    print(f"  Generated t_full (timestamps) with length: {len(t_full)}")
    print(f"  Time range: [{t_full[0]:.2f}, {t_full[-1]:.2f}] seconds")
    print()
    
    # Step 2: Generate HR ground truth
    print("Step 2: Generating HR ground truth...")
    y_full = generate_y(x_full, theta_seconds, s=s)
    y_full = np.array(y_full)
    
    # Generate time points for y: each y_i corresponds to time i * s
    t_y = np.array([i * s for i in range(len(y_full))])
    
    print(f"  Generated y_full with length: {len(y_full)}")
    print(f"  Generated t_y with length: {len(t_y)}")
    print(f"  y range: [{np.min(y_full):.4f}, {np.max(y_full):.4f}]")
    print()
    
    # Step 3: Load trained model
    print("Step 3: Loading trained model...")
    if model_path is None:
        results_dir = os.path.join(project_root, "results", "models")
        model_path = os.path.join(results_dir, f"window_nn_theta_{theta_seconds}.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = WindowNN(input_dim=L, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"  Model loaded from: {model_path}")
    print()
    
    # Step 4: Generate predictions
    print("Step 4: Generating predictions...")
    y_pred = []
    y_true = []
    valid_indices = []
    
    with torch.no_grad():
        for i in range(len(y_full)):
            center_time = t_y[i]
            target_y = y_full[i]
            
            try:
                # Extract and resample window
                x_window_L = extract_and_resample_window(
                    x_full, t_full, center_time, theta_seconds, L
                )
                
                # Predict
                x_tensor = torch.FloatTensor(x_window_L).unsqueeze(0)  # Add batch dimension
                y_pred_val = model(x_tensor).item()
                
                y_pred.append(y_pred_val)
                y_true.append(target_y)
                valid_indices.append(i)
                
            except ValueError as e:
                # Skip if window is empty
                print(f"  Warning: Skipping sample {i} (center_time={center_time:.2f}): {e}")
                continue
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    print(f"  Generated {len(y_pred)} predictions")
    print(f"  y_pred range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    print()
    
    # Step 5: Calculate metrics
    print("Step 5: Calculating metrics...")
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.4f}")
    print()
    
    # Step 6: Plot curves
    print("Step 6: Plotting curves...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both curves
    ax.plot(y_true, label='y_true', linewidth=1.5, alpha=0.8, color='blue')
    ax.plot(y_pred, label='y_pred', linewidth=1.5, alpha=0.8, color='red')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Value (seconds)', fontsize=12)
    ax.set_title(f'NN Prediction vs Ground Truth on {subject_name}\n(theta={theta_seconds}s, {len(y_pred)} samples)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Display metrics on plot
    metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.4f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    results_dir = os.path.join(project_root, "results", "models")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"eval_{subject_name}_y_true_vs_pred_theta_{theta_seconds}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved to: {plot_path}")
    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


def main():
    """Main function."""
    # Parameters (must match training parameters)
    subject_name = "S2"
    theta_seconds = 10.0  # window length in seconds
    L = 32                # fixed input length
    hidden_dim = 16       # hidden dimension
    s = 2                 # sampling interval for generating y
    
    eval_nn_forward(
        subject_name=subject_name,
        model_path=None,  # Will use default path
        theta_seconds=theta_seconds,
        L=L,
        hidden_dim=hidden_dim,
        s=s
    )


if __name__ == "__main__":
    main()

