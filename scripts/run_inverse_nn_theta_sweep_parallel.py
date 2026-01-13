"""
Phase C: Theta sweep for inverse optimization (Parallel Version).

Run inverse optimization for multiple candidate theta values and generate error curve.
This version uses multiprocessing to process multiple theta candidates in parallel.
"""

import sys
import os
import numpy as np
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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


def _process_single_theta_candidate(args):
    """
    Process a single theta candidate. This function is designed to be called
    from a worker process in a process pool.
    
    Args:
        args: tuple containing:
            - theta_candidate: float
            - x_full_np: numpy array
            - t_full_np: numpy array
            - t_y_np: numpy array
            - y_obs_np_valid: numpy array
            - model_path: str
            - L: int
            - hidden_dim: int
            - num_iters: int
            - lr: float
            - lam: float
            - bounds: tuple
            - device: str
    
    Returns:
        tuple: (theta_candidate, result_dict) or (theta_candidate, None) if error
    """
    (theta_candidate, x_full_np, t_full_np, t_y_np, y_obs_np_valid,
     model_path, L, hidden_dim, num_iters, lr, lam, bounds, device) = args
    
    try:
        # Import here to avoid issues with multiprocessing
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import numpy as np
        import torch
        
        # Set threads for this worker process
        # Use multiple threads per worker for matrix operations
        # Let each worker use a reasonable number of threads (e.g., 4-8)
        # The OS will handle thread scheduling across workers
        num_available_cores = os.cpu_count() or 4
        # Use about 4-8 threads per worker (good balance for BLAS operations)
        threads_per_worker = min(8, max(2, num_available_cores // 4))
        torch.set_num_threads(threads_per_worker)
        
        from src.nn_forward_wrapper import load_frozen_window_nn, nn_forward_full_sequence_with_grad
        from src.SGD_solver import solve_x_sgd
        
        # Load model in this worker process
        import os as worker_os
        worker_id = worker_os.getpid()
        print(f"  [Worker PID={worker_id}] Starting theta={theta_candidate}: Loading model...", flush=True)
        model = load_frozen_window_nn(model_path, L=L, hidden_dim=hidden_dim, device=device)
        
        # Initialize x
        x_init = torch.FloatTensor(x_full_np.copy()).to(device)
        noise_std = 0.05 * x_init.std().item()
        x_init = x_init + noise_std * torch.randn_like(x_init)
        y_obs_t = torch.FloatTensor(y_obs_np_valid).to(device)
        
        # Define forward function
        def forward_fn(x_t):
            return nn_forward_full_sequence_with_grad(
                x_t, t_full_np, t_y_np, theta_candidate, model, L, device=device
            )
        
        # Run SGD optimization
        print(f"  [Worker PID={worker_id}] theta={theta_candidate}: Running {num_iters} SGD iterations...", flush=True)
        result = solve_x_sgd(
            x_init=x_init.clone(),
            forward_fn=forward_fn,
            y_obs_t=y_obs_t,
            bounds=bounds,
            lam=lam,
            num_iters=num_iters,
            lr=lr,
            tol=1e-6,
            verbose=False,
        )
        
        # Extract results
        y_hat = result['y_hat']
        final_loss = result['final_loss']
        initial_loss = result.get('initial_loss', final_loss)
        delta_x_final = result.get('delta_x_final', 0.0)
        
        # Compute additional metrics
        mse = np.mean((y_obs_np_valid - y_hat) ** 2)
        mae = np.mean(np.abs(y_obs_np_valid - y_hat))
        
        result_dict = {
            "final_loss": float(final_loss),
            "initial_loss": float(initial_loss),
            "mse": float(mse),
            "mae": float(mae),
            "delta_x": float(delta_x_final),
        }
        
        return (theta_candidate, result_dict)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing theta={theta_candidate}: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        return (theta_candidate, None)


def run_theta_sweep(
    subject="S1",
    theta_gt=10.0,
    theta_candidates=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 29],
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
    Run theta sweep for inverse optimization (parallel version).
    
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
    print(f"Theta Sweep: Inverse Optimization (Subject {subject}) - PARALLEL VERSION")
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
    
    # Convert numpy arrays to ensure they're pickle-able for multiprocessing
    x_full_np = np.array(x_full_np)
    t_full_np = np.array(t_full_np)
    y_obs_np_valid = np.array(y_obs_np_valid)
    t_y_np = np.array(t_y_np)
    
    # Step 3: Sweep over theta candidates (parallel execution)
    results_dir = os.path.join(project_root, "results", "models")
    model_path = os.path.join(results_dir, f"window_nn_theta_{theta_gt}.pt")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Cannot proceed without the trained model.")
        return {}
    
    print("Step 3: Running theta sweep (parallel processing)...")
    num_cores = os.cpu_count() or 4
    # Use all available cores (or number of candidates if less)
    # Each candidate needs one worker, so we can't use more workers than candidates
    num_workers = min(len(theta_candidates), num_cores)
    print(f"  Available CPU cores: {num_cores}")
    print(f"  Number of theta candidates: {len(theta_candidates)}")
    print(f"  Workers to use: {num_workers} (limited by number of candidates)")
    if len(theta_candidates) > num_cores:
        print(f"  Note: {len(theta_candidates) - num_cores} candidates will wait for workers to become available")
    print(f"  Each candidate will run {num_iters} SGD iterations")
    print(f"  Estimated time per candidate: ~{num_iters / 10:.0f} seconds (very rough estimate)")
    print(f"  Model path: {model_path}")
    print()
    
    # Prepare arguments for worker processes
    worker_args = []
    for theta_candidate in theta_candidates:
        args = (
            theta_candidate, x_full_np, t_full_np, t_y_np, y_obs_np_valid,
            model_path, L, hidden_dim, num_iters, lr, lam, bounds, device
        )
        worker_args.append(args)
    
    results = {}
    
    # Use ProcessPoolExecutor for parallel processing
    start_time = time.time()
    
    print(f"  Submitting {len(theta_candidates)} tasks to {num_workers} worker processes...")
    print(f"  Theta candidates: {theta_candidates}")
    print()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks and track them
        future_to_theta = {}
        task_start_times = {}
        all_theta_set = set(theta_candidates)
        completed_tasks = set()
        
        for args in worker_args:
            theta_candidate = args[0]
            future = executor.submit(_process_single_theta_candidate, args)
            future_to_theta[future] = theta_candidate
            task_start_times[theta_candidate] = time.time()
        
        print(f"  All {len(theta_candidates)} tasks submitted. Processing in parallel...")
        print()
        
        # Process completed tasks as they finish
        completed = 0
        
        for future in as_completed(future_to_theta):
            theta_candidate = future_to_theta[future]
            completed_tasks.add(theta_candidate)
            completed += 1
            
            # Calculate time for this task
            task_time = time.time() - task_start_times[theta_candidate]
            
            try:
                theta_candidate, result_dict = future.result()
                if result_dict is not None:
                    results[str(theta_candidate)] = result_dict
                    print(f"  [{completed}/{len(theta_candidates)}] ✓ theta={theta_candidate}: "
                          f"DONE (final_loss={result_dict['final_loss']:.6f}, time={task_time:.1f}s)", flush=True)
                else:
                    print(f"  [{completed}/{len(theta_candidates)}] ✗ theta={theta_candidate}: "
                          f"FAILED (time={task_time:.1f}s)", flush=True)
            except Exception as e:
                print(f"  [{completed}/{len(theta_candidates)}] ✗ theta={theta_candidate}: "
                      f"ERROR: {e} (time={task_time:.1f}s)", flush=True)
            
            # Show current status
            remaining = len(theta_candidates) - completed
            running_tasks = all_theta_set - completed_tasks  # Tasks not yet completed are running or pending
            
            if remaining > 0:
                elapsed = time.time() - start_time
                avg_time_per_task = elapsed / completed if completed > 0 else 0
                estimated_remaining = avg_time_per_task * remaining if avg_time_per_task > 0 else 0
                
                print(f"    Status: {completed} complete, {remaining} remaining (running or pending)")
                if len(running_tasks) > 0 and len(running_tasks) <= 10:
                    running_list = sorted(list(running_tasks))
                    running_str = ", ".join([f"theta={t}" for t in running_list])
                    print(f"    Still processing: {running_str}")
                elif len(running_tasks) > 10:
                    running_list = sorted(list(running_tasks))[:5]
                    running_str = ", ".join([f"theta={t}" for t in running_list])
                    print(f"    Still processing: {running_str}, ... ({len(running_tasks)} total)")
                if estimated_remaining > 0:
                    print(f"    Estimated remaining: ~{estimated_remaining:.0f}s ({estimated_remaining/60:.1f} min)")
                print()
    
    elapsed_time = time.time() - start_time
    print(f"  Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    print()
    
    print()
    print("=" * 60)
    print("Theta sweep complete.")
    print("=" * 60)
    
    return results


def save_results(subject, theta_candidates, results, theta_gt=None):
    """Save results to JSON file."""
    output_dir = os.path.join(project_root, "results", "theta_sweep")
    os.makedirs(output_dir, exist_ok=True)
    
    # Include theta_gt in filename if provided
    if theta_gt is not None:
        json_path = os.path.join(output_dir, f"theta_sweep_results_S{subject}_theta_gt_{theta_gt}.json")
    else:
        json_path = os.path.join(output_dir, f"theta_sweep_results_S{subject}.json")
    
    output_data = {
        "theta_candidates": theta_candidates,
        "results": results
    }
    if theta_gt is not None:
        output_data["theta_gt"] = theta_gt
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return json_path


def plot_error_curve(subject, theta_candidates, results, theta_gt=None):
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
    # Include theta_gt in title if provided
    if theta_gt is not None:
        ax.set_title(f"Theta Sweep Error Curve (Subject S{subject}, theta_gt={theta_gt})", fontsize=14)
    else:
        ax.set_title(f"Theta Sweep Error Curve (Subject S{subject})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(project_root, "results", "theta_sweep")
    os.makedirs(output_dir, exist_ok=True)
    # Include theta_gt in filename if provided
    if theta_gt is not None:
        plot_path = os.path.join(output_dir, f"error_curve_final_loss_S{subject}_theta_gt_{theta_gt}.png")
    else:
        plot_path = os.path.join(output_dir, f"error_curve_final_loss_S{subject}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    """Main function."""
    # Parameters
    subject = "S2"
    theta_gt = 10.0
    theta_candidates = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    L = 32  # Must match the L value used when training the model
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
    
    # Run theta sweep (parallel execution)
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
        verbose=False,
    )
    
    # Save results
    json_path = save_results(subject, theta_candidates, results, theta_gt=theta_gt)
    print(f"Saved results to {json_path}")
    
    # Plot error curve
    plot_path = plot_error_curve(subject, theta_candidates, results, theta_gt=theta_gt)
    if plot_path:
        print(f"Saved plot to {plot_path}")
    
    print()
    print("Theta sweep complete.")
    print(f"Saved results to results/theta_sweep/")
    print()
    
    return results


if __name__ == "__main__":
    main()

