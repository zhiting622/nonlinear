"""
Phase C: Inverse optimization with frozen NN forward surrogate (minimal demo).

This script demonstrates end-to-end inverse optimization:
1. Load subject data (x_full, t_full)
2. Generate teacher y_obs using existing y_generator (theta_gt)
3. Load frozen NN model for candidate theta
4. Run SGD to optimize x_star
5. Plot results and save metrics
"""

import sys
import os
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optimize PyTorch for CPU performance
# Use all available CPU cores for matrix operations
torch.set_num_threads(os.cpu_count() or 4)  # Use all CPU cores
torch.set_num_interop_threads(os.cpu_count() or 4)  # For inter-op parallelism

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import load_x
from src.y_generator import generate_y
from src.nn_forward_wrapper import (
    load_frozen_window_nn,
    nn_forward_full_sequence_with_grad,
    nn_forward_full_sequence_no_grad
)
from src.SGD_solver import solve_x_sgd


def run_inverse_nn_single_theta(
    subject="S1",
    theta_gt=10.0,
    candidate_thetas=[10.0],  # Start with single theta for validation
    L=16,
    hidden_dim=16,
    s=2,
    num_iters=1000,
    lr=0.001,
    lam=0.01,  # TV regularization
    bounds=(0.9, 6.0),
    device="cpu",
    verbose=True,
):
    """
    Run inverse optimization for a single theta candidate.
    
    Args:
        subject: subject identifier (e.g., "S1")
        theta_gt: ground truth theta used to generate teacher y_obs
        candidate_thetas: list of candidate theta values to test
        L: fixed window length
        hidden_dim: hidden dimension of NN
        s: sampling interval for generating y
        num_iters: number of SGD iterations
        lr: learning rate
        lam: TV regularization parameter
        bounds: (lower, upper) bounds for x
        device: device for computation
        verbose: whether to print progress
    """
    print("=" * 60)
    print("Phase C: Inverse Optimization with Frozen NN Surrogate")
    print("=" * 60)
    print(f"Subject: {subject}")
    print(f"Theta GT: {theta_gt}")
    print(f"Candidate thetas: {candidate_thetas}")
    print(f"L: {L}, hidden_dim: {hidden_dim}")
    print()
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    x_full_np = load_x(subject)
    print(f"  Loaded x_full_np, shape: {x_full_np.shape}")
    
    # Generate timestamps
    t_full_np = np.cumsum(x_full_np)
    print(f"  Generated t_full_np, shape: {t_full_np.shape}")
    print(f"  Time range: [{t_full_np[0]:.2f}, {t_full_np[-1]:.2f}] seconds")
    print()
    
    # Step 2: Generate teacher y_obs using theta_gt
    print("Step 2: Generating teacher y_obs...")
    y_obs_np = generate_y(x_full_np, theta_gt, s=s)
    y_obs_np = np.array(y_obs_np)
    t_y_np = np.array([i * s for i in range(len(y_obs_np))])
    
    print(f"  Generated y_obs_np, shape: {y_obs_np.shape}")
    print(f"  Generated t_y_np, shape: {t_y_np.shape}")
    print(f"  y_obs range: [{np.min(y_obs_np):.4f}, {np.max(y_obs_np):.4f}]")
    print()
    
    # Skip index 0 for y_obs (same as forward function, boundary effect)
    # This ensures alignment between y_obs and y_hat
    y_obs_np_valid = y_obs_np[1:]  # Skip index 0
    t_y_np_valid = t_y_np[1:]  # Also skip for t_y (for reference)
    
    print(f"  After skipping index 0:")
    print(f"  y_obs_np_valid shape: {y_obs_np_valid.shape}")
    print(f"  t_y_np_valid shape: {t_y_np_valid.shape}")
    print()
    
    # Convert to torch tensors
    y_obs_t = torch.FloatTensor(y_obs_np_valid).to(device)
    
    # Results dictionary
    results = {}
    
    # Step 3: For each candidate theta, run inverse optimization
    for theta_candidate in candidate_thetas:
        print("=" * 60)
        print(f"Processing theta_candidate: {theta_candidate}")
        print("=" * 60)
        
        # Step 3.1: Load frozen NN model
        print("Step 3.1: Loading frozen NN model...")
        results_dir = os.path.join(project_root, "results", "models")
        model_path = os.path.join(results_dir, f"window_nn_theta_{theta_candidate}.pt")
        
        if not os.path.exists(model_path):
            print(f"  Warning: Model not found at {model_path}")
            print(f"  Skipping theta_candidate={theta_candidate}")
            continue
        
        model = load_frozen_window_nn(model_path, L=L, hidden_dim=hidden_dim, device=device)
        print(f"  Model loaded from: {model_path}")
        print(f"  Model device: {device}")
        print()
        
        # Step 3.2: Initialize x
        print("Step 3.2: Initializing x...")
        # Use original x_full as initialization, add controlled Gaussian noise to break trivial fixed point
        x_init = torch.FloatTensor(x_full_np.copy()).to(device)
        
        # Add controlled Gaussian noise to break trivial fixed point
        noise_std = 0.05 * x_init.std()  # 5% of signal std
        x_init = x_init + noise_std * torch.randn_like(x_init)
        
        print(f"  x_init shape: {x_init.shape}")
        print(f"  noise_std: {noise_std.item():.6f}")
        print(f"  x_init range: [{x_init.min().item():.4f}, {x_init.max().item():.4f}]")
        print()
        
        # Step 3.3: Define forward function
        print("Step 3.3: Setting up forward function...")
        def forward_fn(x_t):
            """Forward function for SGD."""
            return nn_forward_full_sequence_with_grad(
                x_t, t_full_np, t_y_np, theta_candidate, model, L, device=device
            )
        
        # Step 3.4: Run SGD optimization
        print("Step 3.4: Running SGD optimization...")
        print(f"  num_iters: {num_iters}, lr: {lr}, lam: {lam}")
        
        try:
            result = solve_x_sgd(
                x_init=x_init,
                forward_fn=forward_fn,
                y_obs_t=y_obs_t,
                bounds=bounds,
                lam=lam,
                num_iters=num_iters,
                lr=lr,
                verbose=verbose,
            )
            
            x_star = result['x_star']
            y_hat = result['y_hat']
            losses = result['losses']
            final_loss = result['final_loss']
            initial_loss_computed = result.get('initial_loss', losses[0])
            delta_x_final = result.get('delta_x_final', 0.0)
            
            print(f"  Optimization completed!")
            print(f"  Initial loss (computed): {initial_loss_computed:.8f}")
            print(f"  Initial loss (losses[0]): {losses[0]:.8f}")
            print(f"  Final loss: {final_loss:.8f}")
            print(f"  Loss reduction: {((initial_loss_computed - final_loss) / initial_loss_computed * 100):.2f}%")
            print(f"  Final delta_x (||x_star - x_init|| / ||x_init||): {delta_x_final:.8f}")
            print()
            
            # Check if this is essentially a sanity check (y_hat already matches y_obs and delta_x≈0)
            mse_check = np.mean((y_obs_np_valid - y_hat) ** 2)
            if mse_check < 1e-4 and delta_x_final < 1e-3:
                print(f"  Note: This run appears to be a sanity check:")
                print(f"    MSE between y_hat and y_obs: {mse_check:.8f} (very small)")
                print(f"    delta_x: {delta_x_final:.8f} (x barely changed)")
                print(f"    Proceeding to theta sweep may be appropriate.")
                print()
            
            # Step 3.5: Evaluate predictions
            print("Step 3.5: Evaluating predictions...")
            # y_hat and y_obs_np_valid are both shape (M-1,) after skipping index 0
            mse = np.mean((y_obs_np_valid - y_hat) ** 2)
            mae = np.mean(np.abs(y_obs_np_valid - y_hat))
            ss_res = np.sum((y_obs_np_valid - y_hat) ** 2)
            ss_tot = np.sum((y_obs_np_valid - np.mean(y_obs_np_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")
            print()
            
            # Store results
            results[f"theta_{theta_candidate}"] = {
                "theta_candidate": theta_candidate,
                "final_loss": float(final_loss),
                "initial_loss": float(losses[0]),
                "loss_reduction_pct": float((losses[0] - final_loss) / losses[0] * 100),
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "losses": [float(l) for l in losses],
            }
            
            # Step 3.6: Plot results
            print("Step 3.6: Plotting results...")
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: y_obs vs y_hat (both skip index 0)
            ax1 = axes[0]
            ax1.plot(y_obs_np_valid, label='y_obs (skip idx 0)', linewidth=1.5, alpha=0.8, color='blue')
            ax1.plot(y_hat, label='y_hat (skip idx 0)', linewidth=1.5, alpha=0.8, color='red', linestyle='--')
            ax1.set_xlabel('Sample Index', fontsize=12)
            ax1.set_ylabel('Value (seconds)', fontsize=12)
            ax1.set_title(
                f'Inverse Optimization Result: y_obs vs y_hat\n'
                f'Subject: {subject}, theta_candidate: {theta_candidate}, R²: {r2:.4f}',
                fontsize=14
            )
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Add metrics text
            metrics_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.4f}\nFinal Loss: {final_loss:.6f}'
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 2: Loss curve
            ax2 = axes[1]
            ax2.plot(losses, linewidth=1.5, color='green')
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('SGD Loss Curve', fontsize=14)
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_dir = os.path.join(project_root, "results", "models")
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(
                output_dir, 
                f"inverse_nn_{subject}_theta_{theta_candidate}.png"
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Plot saved to: {plot_path}")
            print()
            
        except Exception as e:
            print(f"  Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 4: Save results to JSON
    print("Step 4: Saving results...")
    output_dir = os.path.join(project_root, "results", "models")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"inverse_nn_{subject}_results.json")
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved to: {json_path}")
    print()
    print("=" * 60)
    print("Phase C Demo Complete!")
    print("=" * 60)
    
    return results


def main():
    """Main function."""
    # Parameters
    subject = "S2"
    theta_gt = 10.0
    candidate_thetas = [10.0]  # Start with single theta (only one model available)
    L = 16
    hidden_dim = 16
    s = 2
    num_iters = 1000  # Increased to allow more learning
    lr = 1e-2  # Increased to overcome numerical precision issues (gradient scaling also helps)
    lam = 0.01  # TV regularization
    bounds = (0.3, 2.0)
    # Auto-detect device: use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"Using CPU with {torch.get_num_threads()} threads")
    verbose = True
    
    # Run inverse optimization
    results = run_inverse_nn_single_theta(
        subject=subject,
        theta_gt=theta_gt,
        candidate_thetas=candidate_thetas,
        L=L,
        hidden_dim=hidden_dim,
        s=s,
        num_iters=num_iters,
        lr=lr,
        lam=lam,
        bounds=bounds,
        device=device,
        verbose=verbose,
    )
    
    return results


if __name__ == "__main__":
    main()

