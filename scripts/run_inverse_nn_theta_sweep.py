"""
Phase C: Theta sweep for inverse optimization.

Run inverse optimization for multiple candidate theta values and generate error curve.
"""

import sys
import os
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Removed multiprocessing - using serial execution for stability

# Optimize PyTorch for CPU performance
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(os.cpu_count() or 4)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import load_x
from src.y_generator import generate_y
from src.nn_forward_wrapper import (
    load_frozen_window_nn,
    nn_forward_full_sequence_with_grad
)
from src.SGD_solver import solve_x_sgd




def run_theta_sweep(
    subject="S1",
    theta_gt=10.0,
    theta_candidates=[4, 6, 8, 10, 12, 14, 16, 18, 20],
    L=16,
    hidden_dim=16,
    s=2,
    num_iters=1000,
    lr=1e-2,
    lam=0.01,
    bounds=(0.3, 2.0),
    device="cpu",
    verbose=False,
):
    """
    Run theta sweep for inverse optimization.
    
    Args:
        subject: subject identifier (e.g., "S1")
        theta_gt: ground truth theta used to generate teacher y_obs
        theta_candidates: list of candidate theta values to test
        L: fixed window length
        hidden_dim: hidden dimension of NN
        s: sampling interval for generating y
        num_iters: number of SGD iterations
        lr: learning rate
        lam: TV regularization parameter
        bounds: (lower, upper) bounds for x
        device: device for computation
        verbose: whether to print per-iteration progress (False for sweep)
    
    Returns:
        dict with results per theta
    """
    print("=" * 60)
    print(f"Theta Sweep: Inverse Optimization (Subject {subject})")
    print("=" * 60)
    print(f"Theta GT: {theta_gt}")
    print(f"Theta candidates: {theta_candidates}")
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    x_full_np = load_x(subject)
    t_full_np = np.cumsum(x_full_np)
    print(f"  Loaded x_full_np, shape: {x_full_np.shape}")
    print()
    
    # Step 2: Generate teacher y_obs using theta_gt (same for all theta candidates)
    print("Step 2: Generating teacher y_obs...")
    y_obs_np = generate_y(x_full_np, theta_gt, s=s)
    y_obs_np = np.array(y_obs_np)
    t_y_np = np.array([i * s for i in range(len(y_obs_np))])
    
    # Skip index 0 for y_obs (same as forward function, boundary effect)
    y_obs_np_valid = y_obs_np[1:]
    t_y_np_valid = t_y_np[1:]
    
    print(f"  Generated y_obs_np_valid, shape: {y_obs_np_valid.shape}")
    print(f"  Number of y points to process per forward: {len(y_obs_np_valid)}")
    print(f"  Total forward calls per theta: {num_iters}")
    print(f"  Estimated operations per theta: {num_iters * len(y_obs_np_valid)} window extractions")
    print()
    
    # Convert to torch tensor
    y_obs_t = torch.FloatTensor(y_obs_np_valid).to(device)
    
    # Convert numpy arrays to ensure they're pickle-able for multiprocessing
    x_full_np = np.array(x_full_np)
    t_full_np = np.array(t_full_np)
    y_obs_np_valid = np.array(y_obs_np_valid)
    t_y_np = np.array(t_y_np)
    
    # Step 3: Sweep over theta candidates (serial execution for stability)
    print("Step 3: Running theta sweep...")
    print(f"  Processing {len(theta_candidates)} theta candidates sequentially")
    print(f"  Each candidate will run {num_iters} SGD iterations")
    print(f"  Estimated time: ~{len(theta_candidates) * num_iters / 10:.0f} seconds (very rough estimate)")
    print()
    
    # Initialize x and y_obs tensors
    x_init = torch.FloatTensor(x_full_np.copy()).to(device)
    noise_std = 0.05 * x_init.std()
    x_init = x_init + noise_std * torch.randn_like(x_init)
    y_obs_t = torch.FloatTensor(y_obs_np_valid).to(device)
    
    # Load the trained model (use theta_gt=10.0 model for all candidates)
    results_dir = os.path.join(project_root, "results", "models")
    model_path = os.path.join(results_dir, f"window_nn_theta_{theta_gt}.pt")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Cannot proceed without the trained model.")
        return {}
    
    # Load the model once (same model for all theta candidates)
    print(f"Loading model: {model_path}")
    model = load_frozen_window_nn(model_path, L=L, hidden_dim=hidden_dim, device=device)
    print(f"  Using this model for all theta candidates")
    print()
    
    results = {}
    
    for idx, theta_candidate in enumerate(theta_candidates):
        print(f"Processing theta_candidate={theta_candidate} ({idx+1}/{len(theta_candidates)})...", end=" ", flush=True)
        
        try:
            # Use the same model (trained with theta=10), but forward uses theta_candidate
            # This tests how each candidate performs with the theta=10 trained model
            def forward_fn(x_t):
                return nn_forward_full_sequence_with_grad(
                    x_t, t_full_np, t_y_np, theta_candidate, model, L, device=device
                )
            
            # Run SGD optimization with minimal progress output
            import time
            start_time = time.time()
            
            result = solve_x_sgd(
                x_init=x_init.clone(),
                forward_fn=forward_fn,
                y_obs_t=y_obs_t,
                bounds=bounds,
                lam=lam,
                num_iters=num_iters,
                lr=lr,
                tol=1e-6,
                verbose=False,  # No per-iteration output
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract results
            y_hat = result['y_hat']
            final_loss = result['final_loss']
            initial_loss = result.get('initial_loss', final_loss)
            delta_x_final = result.get('delta_x_final', 0.0)
            
            # Compute additional metrics
            mse = np.mean((y_obs_np_valid - y_hat) ** 2)
            mae = np.mean(np.abs(y_obs_np_valid - y_hat))
            
            # Store results
            theta_key = str(theta_candidate)
            results[theta_key] = {
                "final_loss": float(final_loss),
                "initial_loss": float(initial_loss),
                "mse": float(mse),
                "mae": float(mae),
                "delta_x": float(delta_x_final),
            }
            
            print(f"DONE (final_loss={final_loss:.6f}, time={elapsed_time:.1f}s)", flush=True)
            
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 60)
    print("Theta sweep complete.")
    print("=" * 60)
    
    return results


def save_results(subject, theta_candidates, results):
    """Save results to JSON file."""
    output_dir = os.path.join(project_root, "results", "theta_sweep")
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, f"theta_sweep_results_S{subject}.json")
    
    output_data = {
        "theta_candidates": theta_candidates,
        "results": results
    }
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return json_path


def plot_error_curve(subject, theta_candidates, results):
    """Plot error curve (final_loss vs theta)."""
    # Extract theta values and corresponding final_loss values
    thetas = []
    losses = []
    
    for theta_str in [str(t) for t in theta_candidates]:
        if theta_str in results:
            thetas.append(float(theta_str))
            losses.append(results[theta_str]["final_loss"])
    
    if len(thetas) == 0:
        print("Warning: No valid results to plot")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thetas, losses, marker='o', linewidth=2, markersize=8)
    
    ax.set_xlabel("Theta (seconds)", fontsize=12)
    ax.set_ylabel("Final inverse loss", fontsize=12)
    ax.set_title(f"Theta Sweep Error Curve (Subject S{subject})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(project_root, "results", "theta_sweep")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"error_curve_final_loss_S{subject}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    """Main function."""
    # Parameters
    subject = "S2"
    theta_gt = 10.0
    theta_candidates = [4, 6, 8, 10, 12, 14, 16, 18]
    L = 16
    hidden_dim = 16
    s = 2
    num_iters = 1000
    lr = 1e-2
    lam = 0.01
    bounds = (0.3, 2.0)
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"Using CPU with {torch.get_num_threads()} threads")
    print()
    
    # Run theta sweep (serial execution for stability)
    results = run_theta_sweep(
        subject=subject,
        theta_gt=theta_gt,
        theta_candidates=theta_candidates,
        L=L,
        hidden_dim=hidden_dim,
        s=s,
        num_iters=num_iters,
        lr=lr,
        lam=lam,
        bounds=bounds,
        device=device,
        verbose=True,
    )
    
    # Save results
    json_path = save_results(subject, theta_candidates, results)
    print(f"Saved results to {json_path}")
    
    # Plot error curve
    plot_path = plot_error_curve(subject, theta_candidates, results)
    if plot_path:
        print(f"Saved plot to {plot_path}")
    
    print()
    print("Theta sweep complete.")
    print(f"Saved results to results/theta_sweep/")
    print()
    
    return results


if __name__ == "__main__":
    main()

