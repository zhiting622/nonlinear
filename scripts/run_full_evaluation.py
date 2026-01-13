"""
Full Evaluation Script for Nonlinear Inverse Optimization Framework

This script implements the complete experimental plan:
- Phase 0: Protocol setup and configuration
- Phase 1.1: 3-fold cross-validation (main evaluation)
- Phase 1.2: Optimization stability (limited Monte Carlo)
- Phase 1.3: Data ablation (optional)

Uses all 56 cores for parallelization and ensures reproducibility with fixed random seeds.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing as mp
from functools import partial
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Optimize PyTorch for CPU performance
torch.set_num_threads(os.cpu_count() or 56)
torch.set_num_interop_threads(os.cpu_count() or 56)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import load_x
from src.y_generator import generate_y
from src.nn_forward import WindowNN
from src.nn_forward_wrapper import (
    load_frozen_window_nn,
    nn_forward_full_sequence_with_grad
)
from src.piece_utils import extract_and_resample_window
from src.SGD_solver import solve_x_sgd

# ============================================================================
# Phase 0: Protocol Configuration
# ============================================================================

def load_protocol(protocol_path=None):
    """Load experimental protocol configuration."""
    if protocol_path is None:
        protocol_path = os.path.join(project_root, "experimental_protocol.json")
    
    with open(protocol_path, 'r') as f:
        protocol = json.load(f)
    
    return protocol


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)


# ============================================================================
# Surrogate Training
# ============================================================================

class WindowDataset(Dataset):
    """Dataset for window-level training."""
    
    def __init__(self, X_train, Y_train):
        self.X = torch.FloatTensor(X_train)
        self.Y = torch.FloatTensor(Y_train)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def prepare_training_data_multi_subjects(subject_names, theta_seconds, L=32, s=2):
    """
    Prepare training data from multiple subjects.
    
    Args:
        subject_names: list of subject names (e.g., ["S1", "S2", ...])
        theta_seconds: window length in seconds
        L: fixed resampled length
        s: sampling interval
    
    Returns:
        X_train: np.ndarray of shape (num_samples, L)
        Y_train: np.ndarray of shape (num_samples, 1)
    """
    X_train_all = []
    Y_train_all = []
    
    for subject_name in subject_names:
        # Load data
        x_full = load_x(subject_name)
        t_full = np.cumsum(x_full)
        
        # Generate y
        y_full = generate_y(x_full, theta_seconds, s=s)
        y_full = np.array(y_full)
        t_y = np.array([i * s for i in range(len(y_full))])
        
        # Extract windows
        for i in range(len(y_full)):
            center_time = t_y[i]
            target_y = y_full[i]
            
            try:
                x_window_L = extract_and_resample_window(
                    x_full, t_full, center_time, theta_seconds, L
                )
                X_train_all.append(x_window_L)
                Y_train_all.append(target_y)
            except ValueError:
                continue
    
    X_train = np.array(X_train_all)
    Y_train = np.array(Y_train_all).reshape(-1, 1)
    
    return X_train, Y_train


def train_surrogate_model(subject_names, theta_seconds, L=32, hidden_dim=16, 
                          num_epochs=100, learning_rate=1e-3, batch_size=32, s=2, verbose=False):
    """
    Train surrogate model on multiple subjects.
    
    Args:
        verbose: whether to print training progress (False to reduce output in parallel training)
    
    Returns:
        model: trained WindowNN model
    """
    if verbose:
        print(f"  Training surrogate on subjects: {subject_names}", flush=True)
        print(f"  Theta: {theta_seconds}, L: {L}, hidden_dim: {hidden_dim}", flush=True)
    
    # Prepare training data
    import threading
    import time as time_module
    thread_id = threading.current_thread().ident
    data_start = time_module.time()
    
    if verbose:
        print(f"  [Thread {thread_id}] Preparing training data...", flush=True)
    else:
        print(f"  [Thread {thread_id}] theta_gt={theta_seconds}: Preparing data...", flush=True)
    
    X_train, Y_train = prepare_training_data_multi_subjects(
        subject_names, theta_seconds, L=L, s=s, verbose=verbose
    )
    
    data_time = time_module.time() - data_start
    print(f"  [Thread {thread_id}] theta_gt={theta_seconds}: Data prepared ({len(X_train)} samples, {format_time(data_time)})", flush=True)
    
    if len(X_train) == 0:
        raise ValueError("No training samples created!")
    
    if verbose:
        print(f"  Created {len(X_train)} training samples", flush=True)
    
    # Create dataset and dataloader
    dataset = WindowDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = WindowNN(input_dim=L, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            y_hat = model(batch_x)
            loss = loss_fn(y_hat, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1):
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}", flush=True)
    
    # Set to eval mode and freeze parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model


# ============================================================================
# Knee Detection (Slope Zero Crossing)
# ============================================================================

def compute_slopes(thetas, losses):
    """
    Compute slopes between consecutive points.
    
    Args:
        thetas: array of theta values
        losses: array of loss values
    
    Returns:
        slopes: array of slope values (len = len(thetas) - 1)
        slope_points: list of dicts with slope information
    """
    if len(losses) < 2:
        return np.array([]), []
    
    slopes = []
    slope_points = []
    
    for i in range(len(losses) - 1):
        theta_from = thetas[i]
        theta_to = thetas[i + 1]
        loss_from = losses[i]
        loss_to = losses[i + 1]
        
        slope = (loss_to - loss_from) / (theta_to - theta_from)
        slopes.append(slope)
        
        slope_points.append({
            "from_theta": float(theta_from),
            "to_theta": float(theta_to),
            "from_loss": float(loss_from),
            "to_loss": float(loss_to),
            "slope": float(slope),
            "from_idx": int(i),
            "to_idx": int(i + 1)
        })
    
    return np.array(slopes), slope_points


def find_knee_slope_zero_crossing(thetas, losses, slopes=None, slope_points=None):
    """
    Find knee point using slope zero crossing method.
    
    Looks for where slopes cross zero (from negative to positive or vice versa).
    If no zero crossing, returns the theta with minimum loss.
    
    Args:
        thetas: array of theta values
        losses: array of loss values
        slopes: precomputed slopes (optional)
        slope_points: precomputed slope points (optional)
    
    Returns:
        theta_hat: estimated theta (value, not index)
        knee_info: dict with knee detection information
    """
    if slopes is None or slope_points is None:
        slopes, slope_points = compute_slopes(thetas, losses)
    
    if len(slopes) == 0:
        # Fallback: return theta with minimum loss
        min_idx = np.argmin(losses)
        return float(thetas[min_idx]), {
            "method": "min_loss_fallback",
            "theta_hat": float(thetas[min_idx]),
            "reason": "insufficient_points"
        }
    
    # Find zero crossings: where slope changes sign
    zero_crossings = []
    for i in range(len(slopes) - 1):
        if slopes[i] * slopes[i + 1] < 0:  # Sign change
            # Use the point after the crossing (where slope becomes positive if going from neg to pos)
            if slopes[i] < 0 and slopes[i + 1] > 0:
                zero_crossings.append({
                    "idx": i + 1,
                    "theta": float(thetas[i + 1]),
                    "type": "neg_to_pos",
                    "slope_before": float(slopes[i]),
                    "slope_after": float(slopes[i + 1])
                })
            elif slopes[i] > 0 and slopes[i + 1] < 0:
                zero_crossings.append({
                    "idx": i + 1,
                    "theta": float(thetas[i + 1]),
                    "type": "pos_to_neg",
                    "slope_before": float(slopes[i]),
                    "slope_after": float(slopes[i + 1])
                })
    
    if len(zero_crossings) > 0:
        # Use first zero crossing (typically the knee)
        knee = zero_crossings[0]
        return knee["theta"], {
            "method": "slope_zero_crossing",
            "theta_hat": knee["theta"],
            "zero_crossings": zero_crossings,
            "all_slopes": [float(s) for s in slopes],
            "slope_points": slope_points
        }
    else:
        # No zero crossing: find minimum loss or first point where slope becomes small
        # Look for first point where slope is close to zero
        abs_slopes = np.abs(slopes)
        min_slope_idx = np.argmin(abs_slopes)
        
        # Use the theta after the minimum slope point
        if min_slope_idx + 1 < len(thetas):
            theta_hat = float(thetas[min_slope_idx + 1])
        else:
            theta_hat = float(thetas[min_slope_idx])
        
        return theta_hat, {
            "method": "min_slope_fallback",
            "theta_hat": theta_hat,
            "min_slope_idx": int(min_slope_idx),
            "min_slope_value": float(abs_slopes[min_slope_idx]),
            "all_slopes": [float(s) for s in slopes],
            "slope_points": slope_points
        }


# ============================================================================
# Theta Sweep for Single Subject
# ============================================================================

def _train_single_model_parallel(args):
    """
    Train a single surrogate model in parallel using ProcessPoolExecutor.
    This function runs in a separate process, so it needs to import everything.
    """
    (theta_gt, train_subjects, L, hidden_dim, num_epochs, learning_rate, batch_size, s) = args
    
    try:
        # Import here to avoid issues with multiprocessing
        import sys
        import os
        import time as time_module
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import numpy as np
        # Set threads BEFORE importing torch (important!)
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        import torch
        # Set threads for this worker process (1 thread per process to avoid oversubscription)
        # Note: set_num_interop_threads must be called before any parallel operations
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # If already set, ignore the error
            pass
        
        from src.data_loader import load_x
        from src.y_generator import generate_y
        from src.nn_forward import WindowNN
        from src.piece_utils import extract_and_resample_window
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        import torch.optim as optim
        
        process_id = os.getpid()
        start_time = time_module.time()
        print(f"  [Process {process_id}] Starting training for theta_gt={theta_gt}...", flush=True)
        
        # Re-implement train_surrogate_model logic here to avoid pickling issues
        # Prepare training data
        X_train_all = []
        Y_train_all = []
        
        for subject_name in train_subjects:
            x_full = load_x(subject_name)
            t_full = np.cumsum(x_full)
            y_full = generate_y(x_full, theta_gt, s=s)
            y_full = np.array(y_full)
            t_y = np.array([i * s for i in range(len(y_full))])
            
            for i in range(len(y_full)):
                center_time = t_y[i]
                target_y = y_full[i]
                try:
                    x_window_L = extract_and_resample_window(
                        x_full, t_full, center_time, theta_gt, L
                    )
                    X_train_all.append(x_window_L)
                    Y_train_all.append(target_y)
                except ValueError:
                    continue
        
        X_train = np.array(X_train_all)
        Y_train = np.array(Y_train_all).reshape(-1, 1)
        
        if len(X_train) == 0:
            raise ValueError(f"No training samples created for theta_gt={theta_gt}!")
        
        print(f"  [Process {process_id}] theta_gt={theta_gt}: Prepared {len(X_train)} samples", flush=True)
        
        # Create dataset and dataloader
        class WindowDataset(Dataset):
            def __init__(self, X_train, Y_train):
                self.X = torch.FloatTensor(X_train)
                self.Y = torch.FloatTensor(Y_train)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]
        
        dataset = WindowDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = WindowNN(input_dim=L, hidden_dim=hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in dataloader:
                y_hat = model(batch_x)
                loss = loss_fn(y_hat, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        # Set to eval mode and freeze parameters
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        elapsed = time_module.time() - start_time
        # Format time manually (format_time function not available in worker process)
        if elapsed < 60:
            time_str = f"{elapsed:.1f} seconds"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f} minutes"
        else:
            time_str = f"{elapsed/3600:.1f} hours"
        print(f"  [Process {process_id}] Completed training for theta_gt={theta_gt} in {time_str}", flush=True)
        
        # Return model state dict instead of model object (easier to pickle)
        return (theta_gt, model.state_dict(), None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error training model for theta_gt={theta_gt}: {e}\n{traceback.format_exc()}"
        print(f"  ERROR in _train_single_model_parallel for theta_gt={theta_gt}: {error_msg}", flush=True)
        return (theta_gt, None, error_msg)


def _process_single_theta_candidate_parallel(args):
    """
    Process a single (subject, theta_gt, theta_candidate) task in parallel.
    This is a helper function for parallel processing using ProcessPoolExecutor.
    Model is loaded from file path in each worker process.
    """
    (subject_name, theta_gt, theta_candidate, x_full_np, t_full_np, t_y_np, y_obs_np_valid,
     model_path, L, hidden_dim, num_iters, lr, lam, bounds, device, random_seed) = args
    
    try:
        # Import here to avoid issues with multiprocessing
        import sys
        import os
        import time as time_module
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Get process ID to track parallel execution
        process_id = os.getpid()
        task_start_time = time_module.time()
        
        import numpy as np
        # Set threads BEFORE importing torch (important for multiprocessing!)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        import torch
        # Set threads for this worker process
        # Use 1 thread per worker to avoid oversubscription
        torch.set_num_threads(1)
        # Don't set interop threads - it causes errors in multiprocessing
        
        from src.nn_forward_wrapper import load_frozen_window_nn, nn_forward_full_sequence_with_grad
        from src.SGD_solver import solve_x_sgd
        
        # Set random seed for this worker
        seed_offset = hash(f"{subject_name}_{theta_gt}_{theta_candidate}") % 10000
        np.random.seed(random_seed + seed_offset)
        torch.manual_seed(random_seed + seed_offset)
        
        # Load model in this worker process
        # Note: This may take a few seconds per worker on first load
        model_load_start = time_module.time()
        model = load_frozen_window_nn(model_path, L=L, hidden_dim=hidden_dim, device=device)
        model_load_time = time_module.time() - model_load_start
        
        # Initialize x (each worker gets its own copy)
        x_init = torch.FloatTensor(x_full_np.copy()).to(device)
        noise_std = 0.05 * x_init.std()
        x_init = x_init + noise_std * torch.randn_like(x_init)
        y_obs_t = torch.FloatTensor(y_obs_np_valid).to(device)
        
        # Define forward function
        def forward_fn(x_t):
            return nn_forward_full_sequence_with_grad(
                x_t, t_full_np, t_y_np, theta_candidate, model, L, device=device
            )
        
        # Run SGD optimization
        sgd_start = time_module.time()
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
        sgd_time = time_module.time() - sgd_start
        total_time = time_module.time() - task_start_time
        
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
            "_debug": {
                "process_id": process_id,
                "model_load_time": float(model_load_time),
                "sgd_time": float(sgd_time),
                "total_time": float(total_time),
            }
        }
        
        return (subject_name, theta_gt, theta_candidate, result_dict, None)
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing subject={subject_name}, theta_gt={theta_gt}, theta_candidate={theta_candidate}: {e}\n{traceback.format_exc()}"
        return (subject_name, theta_gt, theta_candidate, None, error_msg)


def run_theta_sweep_single_subject(
    subject_name,
    theta_gt,
    theta_candidates,
    model,
    L=32,
    hidden_dim=16,
    s=2,
    num_iters=1000,
    lr=1e-2,
    lam=0.01,
    bounds=(0.3, 2.0),
    device="cpu",
    random_seed=42,
    save_intermediate=True,
    output_dir=None,
    use_parallel=True,
    max_workers=None
):
    """
    Run theta sweep for a single subject.
    
    Args:
        use_parallel: whether to use parallel processing for theta candidates
        max_workers: number of parallel workers (None = auto-detect)
    
    Returns:
        dict with results per theta and knee detection info
    """
    # Set random seed for this run
    set_random_seeds(random_seed)
    
    # Load data
    x_full_np = load_x(subject_name)
    t_full_np = np.cumsum(x_full_np)
    
    # Generate teacher y_obs using theta_gt
    y_obs_np = generate_y(x_full_np, theta_gt, s=s)
    y_obs_np = np.array(y_obs_np)
    t_y_np = np.array([i * s for i in range(len(y_obs_np))])
    
    # Skip index 0 for y_obs (boundary effect)
    y_obs_np_valid = y_obs_np[1:]
    t_y_np_valid = t_y_np[1:]
    
    results = {}
    thetas_list = []
    losses_list = []
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(len(theta_candidates), os.cpu_count() or 4)
    
    if use_parallel and len(theta_candidates) > 1:
        # Parallel processing
        print(f"      Processing {len(theta_candidates)} theta candidates in parallel ({max_workers} workers)...")
        sweep_start_time = time.time()
        
        # Prepare arguments for each theta candidate
        args_list = [
            (theta_candidate, x_full_np, t_full_np, t_y_np, y_obs_np_valid,
             model, L, num_iters, lr, lam, bounds, device, random_seed)
            for theta_candidate in theta_candidates
        ]
        
        # Process in parallel
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_theta = {
                executor.submit(_process_single_theta_candidate_parallel, args): args[0]
                for args in args_list
            }
            
            for future in as_completed(future_to_theta):
                theta_candidate = future_to_theta[future]
                try:
                    theta_candidate, result_dict, error_msg = future.result()
                    completed_count += 1
                    
                    if result_dict is not None:
                        theta_key = str(theta_candidate)
                        results[theta_key] = result_dict
                        thetas_list.append(float(theta_candidate))
                        losses_list.append(float(result_dict["final_loss"]))
                        
                        # Save intermediate result if requested
                        if save_intermediate and output_dir:
                            intermediate_path = os.path.join(
                                output_dir,
                                f"intermediate_{subject_name}_theta_gt_{theta_gt}_candidate_{theta_candidate}.json"
                            )
                            with open(intermediate_path, 'w') as f:
                                json.dump({
                                    "subject": subject_name,
                                    "theta_gt": theta_gt,
                                    "theta_candidate": theta_candidate,
                                    "result": result_dict
                                }, f, indent=2)
                        
                        print(f"      Theta candidate {theta_candidate} ({completed_count}/{len(theta_candidates)}) completed")
                    else:
                        print(f"      ERROR processing theta={theta_candidate}: {error_msg}")
                except Exception as e:
                    print(f"      ERROR processing theta={theta_candidate}: {e}")
        
        sweep_time = time.time() - sweep_start_time
        print(f"      All theta candidates completed in {format_time(sweep_time)}")
    else:
        # Sequential processing (original code)
        for idx, theta_candidate in enumerate(theta_candidates):
            try:
                theta_candidate_start_time = time.time()
                
                # Initialize x
                x_init = torch.FloatTensor(x_full_np.copy()).to(device)
                noise_std = 0.05 * x_init.std()
                x_init = x_init + noise_std * torch.randn_like(x_init)
                y_obs_t = torch.FloatTensor(y_obs_np_valid).to(device)
                
                # Define forward function
                def forward_fn(x_t):
                    return nn_forward_full_sequence_with_grad(
                        x_t, t_full_np, t_y_np, theta_candidate, model, L, device=device
                    )
                
                # Run SGD optimization
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
                
                theta_candidate_time = time.time() - theta_candidate_start_time
                
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
                
                thetas_list.append(float(theta_candidate))
                losses_list.append(float(final_loss))
                
                # Print time for this theta candidate
                print(f"      Theta candidate {theta_candidate} ({idx+1}/{len(theta_candidates)}) completed in {format_time(theta_candidate_time)}")
                
                # Save intermediate result if requested
                if save_intermediate and output_dir:
                    intermediate_path = os.path.join(
                        output_dir,
                        f"intermediate_{subject_name}_theta_gt_{theta_gt}_candidate_{theta_candidate}.json"
                    )
                    with open(intermediate_path, 'w') as f:
                        json.dump({
                            "subject": subject_name,
                            "theta_gt": theta_gt,
                            "theta_candidate": theta_candidate,
                            "result": results[theta_key],
                            "time_seconds": theta_candidate_time
                        }, f, indent=2)
            
            except Exception as e:
                print(f"      ERROR processing theta={theta_candidate}: {e}")
                continue
    
    # Compute slopes and find knee
    thetas_array = np.array(thetas_list)
    losses_array = np.array(losses_list)
    
    # Sort by theta
    sort_idx = np.argsort(thetas_array)
    thetas_sorted = thetas_array[sort_idx]
    losses_sorted = losses_array[sort_idx]
    
    slopes, slope_points = compute_slopes(thetas_sorted, losses_sorted)
    theta_hat, knee_info = find_knee_slope_zero_crossing(
        thetas_sorted, losses_sorted, slopes, slope_points
    )
    
    return {
        "results": results,
        "thetas": thetas_sorted.tolist(),
        "losses": losses_sorted.tolist(),
        "slopes": slopes.tolist(),
        "slope_points": slope_points,
        "theta_hat": theta_hat,
        "knee_info": knee_info,
        "theta_gt": theta_gt
    }


# ============================================================================
# Main Evaluation Functions
# ============================================================================

def create_folds(subject_list, n_folds=3, random_seed=42):
    """
    Create n_folds for cross-validation with random subject assignment.
    
    Each fold uses 1 fold (5 subjects) for training and 2 folds (10 subjects) for testing.
    Subjects are randomly shuffled before assignment to ensure random distribution.
    
    Args:
        subject_list: list of subject names
        n_folds: number of folds
        random_seed: random seed for reproducibility
    
    Returns:
        folds: list of dicts, each with 'train' and 'test' subject lists
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle subjects randomly
    shuffled_subjects = subject_list.copy()
    np.random.shuffle(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    subjects_per_fold = n_subjects // n_folds
    
    folds = []
    for i in range(n_folds):
        start_idx = i * subjects_per_fold
        end_idx = (i + 1) * subjects_per_fold if i < n_folds - 1 else n_subjects
        
        # Training: 1 fold (5 subjects)
        train_subjects = shuffled_subjects[start_idx:end_idx]
        # Testing: 2 folds (10 subjects) - all other subjects
        test_subjects = [s for s in shuffled_subjects if s not in train_subjects]
        
        folds.append({
            "fold": i + 1,
            "train_subjects": sorted(train_subjects),  # Sort for readability
            "test_subjects": sorted(test_subjects)  # Sort for readability
        })
    
    return folds


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"


def estimate_phase_time(protocol):
    """
    Estimate total time needed for Phase 1.1 based on protocol configuration.
    
    This is a rough estimate based on typical computation times.
    """
    n_folds = protocol["phase_1"]["cross_validation"]["n_folds"]
    n_theta_gts = len(protocol["phase_0"]["ground_truth_thetas"]["values"])
    n_theta_candidates = len(protocol["phase_0"]["theta_candidates"]["values"])
    n_test_subjects_per_fold = protocol["phase_1"]["cross_validation"]["test_subjects_per_fold"]
    num_epochs = 100
    num_iters = protocol["phase_0"]["optimizer"]["num_iters"]
    
    # Rough estimates (in seconds) - these are conservative estimates
    # Model training: ~2 seconds per epoch per subject, with 5 training subjects
    time_per_model_training = num_epochs * 2 * 5 / 10  # ~100 seconds per model
    
    # Theta sweep per subject per theta_gt: ~30 seconds per theta candidate
    time_per_theta_sweep = n_theta_candidates * 30
    
    # Total time per fold
    time_step1 = n_theta_gts * time_per_model_training
    time_step2 = n_test_subjects_per_fold * n_theta_gts * time_per_theta_sweep
    time_per_fold = time_step1 + time_step2
    
    # Total phase time
    total_time = n_folds * time_per_fold
    
    return total_time, time_step1, time_step2, time_per_fold


def estimate_fold_time(protocol, fold_idx, n_folds):
    """
    Estimate time for a single fold.
    """
    n_theta_gts = len(protocol["phase_0"]["ground_truth_thetas"]["values"])
    n_theta_candidates = len(protocol["phase_0"]["theta_candidates"]["values"])
    n_test_subjects_per_fold = protocol["phase_1"]["cross_validation"]["test_subjects_per_fold"]
    num_epochs = 100
    
    # Rough estimates
    time_per_model_training = num_epochs * 2 * 5 / 10  # ~100 seconds per model
    time_per_theta_sweep = n_theta_candidates * 30
    
    time_step1 = n_theta_gts * time_per_model_training
    time_step2 = n_test_subjects_per_fold * n_theta_gts * time_per_theta_sweep
    time_per_fold = time_step1 + time_step2
    
    return time_per_fold, time_step1, time_step2


def run_phase_1_1_cross_validation(protocol, output_base_dir=None):
    """
    Phase 1.1: 3-fold cross-validation (main evaluation).
    
    For each fold:
    - Train surrogate on 5 subjects
    - Test on 10 subjects
    - For each test subject and each theta_gt, run theta sweep
    """
    if output_base_dir is None:
        output_base_dir = os.path.join(project_root, "results", "full_evaluation")
    os.makedirs(output_base_dir, exist_ok=True)
    
    phase_dir = os.path.join(output_base_dir, "phase_1_1_cross_validation")
    os.makedirs(phase_dir, exist_ok=True)
    
    # Load protocol
    theta_candidates = protocol["phase_0"]["theta_candidates"]["values"]
    theta_gts = protocol["phase_0"]["ground_truth_thetas"]["values"]
    subject_list = protocol["phase_1"]["cross_validation"]["subject_list"]
    n_folds = protocol["phase_1"]["cross_validation"]["n_folds"]
    
    # Model parameters
    L = protocol["phase_0"]["model_parameters"]["L"]
    hidden_dim = protocol["phase_0"]["model_parameters"]["hidden_dim"]
    s = protocol["phase_0"]["model_parameters"]["s"]
    
    # Optimization parameters
    num_iters = protocol["phase_0"]["optimizer"]["num_iters"]
    lr = protocol["phase_0"]["optimizer"]["learning_rate"]
    lam_str = str(protocol["phase_0"]["regularization"]["parameter"])
    # Handle both "lam = 0.01" and "0.01" formats
    if "=" in lam_str:
        lam_value = float(lam_str.split("=")[1].strip())
    else:
        lam_value = float(lam_str.strip())
    bounds = (
        protocol["phase_0"]["bounds"]["lower"],
        protocol["phase_0"]["bounds"]["upper"]
    )
    
    # Random seed
    seed = protocol["phase_0"]["random_seed"]["value"]
    set_random_seeds(seed)
    
    # Estimate total phase time
    estimated_phase_time, estimated_step1_time, estimated_step2_time, estimated_fold_time = estimate_phase_time(protocol)
    
    phase_start_time = time.time()
    print("=" * 80)
    print("Phase 1.1: 3-Fold Cross-Validation (Main Evaluation)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Theta candidates: {theta_candidates}")
    print(f"Ground truth thetas: {theta_gts}")
    print(f"Number of folds: {n_folds}")
    print(f"Total subjects: {len(subject_list)}")
    print()
    print(f"Estimated total phase time: {format_time(estimated_phase_time)}")
    print(f"  Estimated time per fold: {format_time(estimated_fold_time)}")
    print(f"  Estimated Step 1 (training) per fold: {format_time(estimated_step1_time)}")
    print(f"  Estimated Step 2 (testing) per fold: {format_time(estimated_step2_time)}")
    print()
    
    # Create folds with random assignment
    folds = create_folds(subject_list, n_folds, random_seed=seed)
    
    # Save fold information
    folds_info_path = os.path.join(phase_dir, "folds_info.json")
    with open(folds_info_path, 'w') as f:
        json.dump(folds, f, indent=2)
    print(f"Fold information saved to: {folds_info_path}")
    print()
    
    all_results = {}
    fold_results = []
    
    # Process each fold
    for fold_idx, fold in enumerate(folds):
        # Estimate time for this fold
        est_fold_time, est_step1_time, est_step2_time = estimate_fold_time(protocol, fold_idx, n_folds)
        
        fold_start_time = time.time()
        print("=" * 80)
        print(f"Processing Fold {fold['fold']}/{n_folds}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"Training subjects: {fold['train_subjects']}")
        print(f"Test subjects: {fold['test_subjects']}")
        print(f"Estimated fold time: {format_time(est_fold_time)}")
        print(f"  Estimated Step 1 (training): {format_time(est_step1_time)}")
        print(f"  Estimated Step 2 (testing): {format_time(est_step2_time)}")
        print()
        fold_dir = os.path.join(phase_dir, f"fold_{fold['fold']}")
        os.makedirs(fold_dir, exist_ok=True)
        
        fold_result = {
            "fold": fold['fold'],
            "train_subjects": fold['train_subjects'],
            "test_subjects": fold['test_subjects'],
            "subject_results": {}
        }
        
        # Step 1: Train all surrogate models in parallel
        step1_start_time = time.time()
        print("Step 1: Training all surrogate models in parallel...")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Estimated time: {format_time(est_step1_time)}")
        
        # Get max_workers for training
        max_workers_train = protocol.get("parallelization", {}).get("n_cores", None)
        if max_workers_train is None:
            max_workers_train = min(len(theta_gts), os.cpu_count() or 4)
        
        # Prepare training arguments
        train_args_list = [
            (theta_gt, fold['train_subjects'], L, hidden_dim, 100, 1e-3, 32, s)
            for theta_gt in theta_gts
        ]
        
        models = {}
        model_paths = {}  # Store model file paths for ProcessPoolExecutor
        print(f"  Starting training for {len(train_args_list)} models with {max_workers_train} workers (ProcessPoolExecutor)...", flush=True)
        
        with ProcessPoolExecutor(max_workers=max_workers_train) as executor:
            future_to_theta = {
                executor.submit(_train_single_model_parallel, args): args[0]
                for args in train_args_list
            }
            
            print(f"  Submitted {len(future_to_theta)} training tasks", flush=True)
            completed_count = 0
            
            for future in as_completed(future_to_theta):
                theta_gt = future_to_theta[future]
                completed_count += 1
                elapsed = time.time() - step1_start_time
                
                try:
                    theta_gt_result, model_state_dict, error_msg = future.result()
                    if model_state_dict is not None:
                        # Save model to temporary file
                        model_path = os.path.join(fold_dir, f"temp_model_theta_{theta_gt_result}.pt")
                        torch.save(model_state_dict, model_path)
                        model_paths[theta_gt_result] = model_path
                        
                        # Also load model in main process for potential use
                        model = WindowNN(input_dim=L, hidden_dim=hidden_dim)
                        model.load_state_dict(model_state_dict)
                        model.eval()
                        for param in model.parameters():
                            param.requires_grad = False
                        models[theta_gt_result] = model
                        
                        print(f"  [{completed_count}/{len(train_args_list)}] Model for theta_gt={theta_gt_result} training completed and saved (elapsed: {format_time(elapsed)})", flush=True)
                    else:
                        print(f"  [{completed_count}/{len(train_args_list)}] ERROR training model for theta_gt={theta_gt_result}: {error_msg}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"  [{completed_count}/{len(train_args_list)}] ERROR training model for theta_gt={theta_gt}: {e}", flush=True)
                    print(traceback.format_exc(), flush=True)
        
        step1_time = time.time() - step1_start_time
        print(f"Step 1 completed in {format_time(step1_time)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check if all models were trained successfully
        if len(model_paths) == 0:
            raise RuntimeError(f"Step 1 failed: No models were successfully trained! All {len(train_args_list)} models failed.")
        elif len(model_paths) < len(theta_gts):
            print(f"  WARNING: Only {len(model_paths)}/{len(theta_gts)} models trained successfully")
            print(f"  Missing models for: {set(theta_gts) - set(model_paths.keys())}")
            print(f"  Continuing with available models only...")
            print()
        
        # Step 2: Process all (subject, theta_gt, theta_candidate) tasks in parallel
        step2_start_time = time.time()
        print("Step 2: Processing all (subject, theta_gt, theta_candidate) tasks in parallel...")
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Estimated time: {format_time(est_step2_time)}")
        
        # Get max_workers for testing
        max_workers_test = protocol.get("parallelization", {}).get("n_cores", None)
        if max_workers_test is None:
            max_workers_test = os.cpu_count() or 56
        
        # Prepare all tasks: (subject, theta_gt, theta_candidate) combinations
        print("  Preparing tasks (loading data and generating y_obs)...", flush=True)
        all_tasks = []
        n_test_subjects = len(fold['test_subjects'])
        n_theta_gts_available = len([t for t in theta_gts if t in model_paths])
        total_subject_theta_combinations = n_test_subjects * n_theta_gts_available
        print(f"    Will process {n_test_subjects} subjects × {n_theta_gts_available} theta_gts = {total_subject_theta_combinations} (subject, theta_gt) combinations", flush=True)
        print(f"    Each combination will create {len(theta_candidates)} tasks (one per theta_candidate)", flush=True)
        print(f"    Total tasks will be: {total_subject_theta_combinations} × {len(theta_candidates)} = {total_subject_theta_combinations * len(theta_candidates)}", flush=True)
        completed_combinations = 0
        
        for subject_idx, test_subject in enumerate(fold['test_subjects']):
            # Initialize subject_result
            if test_subject not in fold_result["subject_results"]:
                fold_result["subject_results"][test_subject] = {
                    "subject": test_subject,
                    "theta_results": {}
                }
            
            # Load data for this subject once
            print(f"    Loading data for {test_subject} ({subject_idx+1}/{len(fold['test_subjects'])})...", flush=True)
            subject_data_start = time.time()
            x_full_np = load_x(test_subject)
            t_full_np = np.cumsum(x_full_np)
            subject_data_time = time.time() - subject_data_start
            print(f"      Data loaded in {format_time(subject_data_time)}", flush=True)
            
            for theta_gt_idx, theta_gt in enumerate(theta_gts):
                # Skip if model training failed for this theta_gt
                if theta_gt not in model_paths:
                    print(f"      WARNING: Skipping theta_gt={theta_gt} for subject {test_subject} (model training failed)", flush=True)
                    continue
                
                # Generate y_obs for this subject and theta_gt
                y_gen_start = time.time()
                y_obs_np = generate_y(x_full_np, theta_gt, s=s)
                y_obs_np = np.array(y_obs_np)
                t_y_np = np.array([i * s for i in range(len(y_obs_np))])
                y_obs_np_valid = y_obs_np[1:]
                y_gen_time = time.time() - y_gen_start
                
                completed_combinations += 1
                print(f"      Generated y_obs for theta_gt={theta_gt} ({theta_gt_idx+1}/{len(theta_gts)}) in {format_time(y_gen_time)}", flush=True)
                
                # Create task for each theta_candidate
                task_creation_start = time.time()
                for theta_candidate in theta_candidates:
                    all_tasks.append((
                        test_subject, theta_gt, theta_candidate,
                        x_full_np.copy(), t_full_np.copy(), t_y_np.copy(), y_obs_np_valid.copy(),
                        model_paths[theta_gt], L, hidden_dim, num_iters, lr, lam_value, bounds, "cpu", seed
                    ))
                task_creation_time = time.time() - task_creation_start
                print(f"        Created {len(theta_candidates)} tasks in {format_time(task_creation_time)} (progress: {completed_combinations}/{total_subject_theta_combinations} combinations)", flush=True)
        
        total_tasks = len(all_tasks)
        print(f"  Task preparation completed: {total_tasks} tasks created", flush=True)
        print(f"  Total tasks: {total_tasks} (subjects × theta_gts × theta_candidates)")
        print(f"  Using {max_workers_test} workers")
        print(f"  Estimated time per task: ~{est_step2_time / total_tasks:.1f} seconds")
        print()
        
        # Process all tasks in parallel using ProcessPoolExecutor for true parallelism
        print("  Starting ProcessPoolExecutor and submitting tasks...", flush=True)
        submit_start = time.time()
        task_results = {}  # {(subject, theta_gt): {theta_candidate: result}}
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers_test) as executor:
            print(f"    ProcessPoolExecutor created, submitting {total_tasks} tasks...", flush=True)
            future_to_task = {}
            for task_idx, task in enumerate(all_tasks):
                if task_idx % 500 == 0:
                    print(f"      Submitting task {task_idx+1}/{total_tasks}...", flush=True)
                future = executor.submit(_process_single_theta_candidate_parallel, task)
                future_to_task[future] = task[:3]
            
            submit_time = time.time() - submit_start
            print(f"    All {total_tasks} tasks submitted in {format_time(submit_time)}", flush=True)
            print(f"    Waiting for results...", flush=True)
            print(f"    Note: First result may take 30-60 seconds (worker startup + model loading + first SGD run)", flush=True)
            print(f"    Estimated total time: ~{format_time(est_step2_time)} ({total_tasks} tasks / {max_workers_test} workers × ~30 sec/task)", flush=True)
            print()
            
            result_start = time.time()
            first_result_time = None
            unique_process_ids = set()  # Track unique process IDs to verify parallelism
            
            for future in as_completed(future_to_task):
                subject_name, theta_gt, theta_candidate = future_to_task[future]
                completed_count += 1
                
                try:
                    subj, tgt, tc, result_dict, error_msg = future.result()
                    
                    # Track parallel execution by collecting unique process IDs
                    if result_dict is not None and "_debug" in result_dict:
                        proc_id = result_dict["_debug"]["process_id"]
                        unique_process_ids.add(proc_id)
                    
                    # Track first result time
                    if first_result_time is None:
                        first_result_time = time.time()
                        first_result_elapsed = first_result_time - result_start
                        if result_dict is not None and "_debug" in result_dict:
                            proc_id = result_dict["_debug"]["process_id"]
                            print(f"  ✓ First result received after {format_time(first_result_elapsed)}: ({subject_name}, theta_gt={theta_gt}, candidate={theta_candidate}) [PID: {proc_id}]", flush=True)
                        else:
                            print(f"  ✓ First result received after {format_time(first_result_elapsed)}: ({subject_name}, theta_gt={theta_gt}, candidate={theta_candidate})", flush=True)
                    
                    if result_dict is not None:
                        # Store result
                        key = (subj, str(tgt))
                        if key not in task_results:
                            task_results[key] = {}
                        task_results[key][str(tc)] = result_dict
                        
                        # Save intermediate result
                        intermediate_path = os.path.join(
                            fold_dir,
                            f"intermediate_{subj}_theta_gt_{tgt}_candidate_{tc}.json"
                        )
                        with open(intermediate_path, 'w') as f:
                            json.dump({
                                "subject": subj,
                                "theta_gt": tgt,
                                "theta_candidate": tc,
                                "result": result_dict
                            }, f, indent=2)
                        
                        # Print progress more frequently
                        if completed_count == 1:
                            # First result - already printed above
                            pass
                        elif completed_count <= 10:
                            # Print every result for first 10
                            elapsed = time.time() - result_start
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = (total_tasks - completed_count) / rate if rate > 0 else 0
                            debug_info = ""
                            if "_debug" in result_dict:
                                proc_id = result_dict["_debug"]["process_id"]
                                sgd_t = result_dict["_debug"]["sgd_time"]
                                debug_info = f" [PID: {proc_id}, SGD: {sgd_t:.1f}s]"
                            print(f"  Progress: {completed_count}/{total_tasks} completed ({rate:.2f} tasks/sec, ~{format_time(remaining)} remaining) - ({subj}, θ_gt={tgt}, θ_cand={tc}){debug_info}", flush=True)
                        elif completed_count % 5 == 0 or completed_count == total_tasks:
                            # Print every 5 tasks after first 10
                            elapsed = time.time() - result_start
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            remaining = (total_tasks - completed_count) / rate if rate > 0 else 0
                            # Count unique processes that have completed tasks
                            unique_procs = len(set(r.get("_debug", {}).get("process_id") for r in task_results.values() for r2 in r.values() if isinstance(r2, dict) and "_debug" in r2))
                            print(f"  Progress: {completed_count}/{total_tasks} completed ({rate:.2f} tasks/sec, ~{format_time(remaining)} remaining) [~{unique_procs} unique processes]", flush=True)
                    else:
                        print(f"  ✗ ERROR: {error_msg}", flush=True)
                except Exception as e:
                    print(f"  ✗ ERROR processing task ({subject_name}, {theta_gt}, {theta_candidate}): {e}", flush=True)
        
        # Aggregate results by subject and theta_gt
        for (subj, tgt_str), candidate_results in task_results.items():
            if subj not in fold_result["subject_results"]:
                fold_result["subject_results"][subj] = {
                    "subject": subj,
                    "theta_results": {}
                }
            subject_result = fold_result["subject_results"][subj]
            
            # Convert candidate_results to the format expected by knee detection
            thetas_list = []
            losses_list = []
            results_dict = {}
            
            for tc_str, result_dict in candidate_results.items():
                tc = float(tc_str)
                thetas_list.append(tc)
                losses_list.append(result_dict["final_loss"])
                results_dict[tc_str] = result_dict
            
            # Sort by theta
            sort_idx = np.argsort(thetas_list)
            thetas_sorted = np.array(thetas_list)[sort_idx]
            losses_sorted = np.array(losses_list)[sort_idx]
            
            # Compute slopes and find knee
            slopes, slope_points = compute_slopes(thetas_sorted, losses_sorted)
            theta_hat, knee_info = find_knee_slope_zero_crossing(
                thetas_sorted, losses_sorted, slopes, slope_points
            )
            
            # Store sweep result
            subject_result["theta_results"][tgt_str] = {
                "results": results_dict,
                "thetas": thetas_sorted.tolist(),
                "losses": losses_sorted.tolist(),
                "slopes": slopes.tolist(),
                "slope_points": slope_points,
                "theta_hat": theta_hat,
                "knee_info": knee_info,
                "theta_gt": float(tgt_str)
            }
            
            print(f"  Subject {subj}, Theta GT {tgt_str}: Theta_hat = {theta_hat:.2f}")
        
        step2_time = time.time() - step2_start_time
        print(f"Step 2 completed in {format_time(step2_time)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Clean up temporary model files
        for theta_gt, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
            except Exception as e:
                print(f"  Warning: Could not remove temporary model file {model_path}: {e}")
        
        fold_time = time.time() - fold_start_time
        print(f"Fold {fold['fold']} completed in {format_time(fold_time)} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Save fold results
        fold_result_path = os.path.join(fold_dir, "fold_results.json")
        with open(fold_result_path, 'w') as f:
            json.dump(fold_result, f, indent=2)
        print(f"Fold results saved to: {fold_result_path}")
        print()
        
        fold_results.append(fold_result)
        all_results[f"fold_{fold['fold']}"] = fold_result
    
    # Compute aggregate statistics
    print("=" * 80)
    print("Computing Aggregate Statistics...")
    print("=" * 80)
    
    all_errors = []
    all_subject_errors = defaultdict(list)
    
    for fold_result in fold_results:
        for subject, subject_result in fold_result["subject_results"].items():
            for theta_gt_str, theta_result in subject_result["theta_results"].items():
                theta_gt = float(theta_gt_str)
                theta_hat = theta_result["theta_hat"]
                error = abs(theta_hat - theta_gt)
                all_errors.append(error)
                all_subject_errors[subject].append(error)
    
    # Compute statistics
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    median_error = np.median(all_errors)
    within_1s = np.sum(np.array(all_errors) <= 1.0) / len(all_errors) * 100
    within_2s = np.sum(np.array(all_errors) <= 2.0) / len(all_errors) * 100
    
    # Worst subject
    worst_subject = max(all_subject_errors.items(), key=lambda x: np.mean(x[1]))
    
    aggregate_stats = {
        "mean_absolute_error": float(mean_error),
        "std_absolute_error": float(std_error),
        "median_absolute_error": float(median_error),
        "percent_within_1s": float(within_1s),
        "percent_within_2s": float(within_2s),
        "worst_subject": {
            "subject": worst_subject[0],
            "mean_error": float(np.mean(worst_subject[1])),
            "all_errors": [float(e) for e in worst_subject[1]]
        },
        "total_evaluations": len(all_errors)
    }
    
    # Save aggregate statistics
    stats_path = os.path.join(phase_dir, "aggregate_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    print(f"Mean absolute error: {mean_error:.3f} ± {std_error:.3f}")
    print(f"Median absolute error: {median_error:.3f}")
    print(f"Within ±1s: {within_1s:.1f}%")
    print(f"Within ±2s: {within_2s:.1f}%")
    print(f"Worst subject: {worst_subject[0]} (mean error: {np.mean(worst_subject[1]):.3f})")
    print()
    
    # Create summary table
    print("Creating summary table...")
    summary_table = []
    for fold_result in fold_results:
        for subject, subject_result in fold_result["subject_results"].items():
            for theta_gt_str, theta_result in subject_result["theta_results"].items():
                theta_gt = float(theta_gt_str)
                theta_hat = theta_result["theta_hat"]
                error = abs(theta_hat - theta_gt)
                summary_table.append({
                    "fold": fold_result["fold"],
                    "subject": subject,
                    "theta_gt": theta_gt,
                    "theta_hat": theta_hat,
                    "absolute_error": error
                })
    
    summary_path = os.path.join(phase_dir, "summary_table.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_table, f, indent=2)
    print(f"Summary table saved to: {summary_path}")
    print()
    
    # Save all results
    all_results_path = os.path.join(phase_dir, "all_results.json")
    with open(all_results_path, 'w') as f:
        json.dump({
            "aggregate_statistics": aggregate_stats,
            "summary_table": summary_table,
            "folds": all_results
        }, f, indent=2)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(phase_dir, fold_results, aggregate_stats)
    
    phase_time = time.time() - phase_start_time
    print("=" * 80)
    print("Phase 1.1 Complete!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total phase time: {format_time(phase_time)}")
    print("=" * 80)
    print(f"All results saved to: {phase_dir}")
    print()
    
    return all_results, aggregate_stats


def create_visualizations(phase_dir, fold_results, aggregate_stats):
    """Create visualization plots."""
    plots_dir = os.path.join(phase_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Error distribution plot
    all_errors = []
    for fold_result in fold_results:
        for subject, subject_result in fold_result["subject_results"].items():
            for theta_gt_str, theta_result in subject_result["theta_results"].items():
                theta_gt = float(theta_gt_str)
                theta_hat = theta_result["theta_hat"]
                error = abs(theta_hat - theta_gt)
                all_errors.append(error)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_errors):.3f}')
    ax.axvline(np.median(all_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(all_errors):.3f}')
    ax.set_xlabel('Absolute Error (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Absolute Errors Across All Evaluations', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved error distribution plot")
    
    # 2. Box plot of errors by subject
    subject_errors = defaultdict(list)
    for fold_result in fold_results:
        for subject, subject_result in fold_result["subject_results"].items():
            for theta_gt_str, theta_result in subject_result["theta_results"].items():
                theta_gt = float(theta_gt_str)
                theta_hat = theta_result["theta_hat"]
                error = abs(theta_hat - theta_gt)
                subject_errors[subject].append(error)
    
    subjects_sorted = sorted(subject_errors.keys(), key=lambda x: int(x[1:]))
    errors_by_subject = [subject_errors[s] for s in subjects_sorted]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(errors_by_subject, labels=subjects_sorted, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Absolute Error (seconds)', fontsize=12)
    ax.set_title('Error Distribution by Subject', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "error_by_subject_boxplot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved error by subject boxplot")
    
    # 3. Example error curves (good/average/bad)
    # Find examples: good (low error), average (median error), bad (high error)
    subject_mean_errors = {s: np.mean(errors) for s, errors in subject_errors.items()}
    sorted_subjects = sorted(subject_mean_errors.items(), key=lambda x: x[1])
    
    example_subjects = [
        ("good", sorted_subjects[0][0]),
        ("average", sorted_subjects[len(sorted_subjects)//2][0]),
        ("bad", sorted_subjects[-1][0])
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (label, subject) in enumerate(example_subjects):
        # Find a theta_gt result for this subject
        found = False
        for fold_result in fold_results:
            if subject in fold_result["subject_results"]:
                subject_result = fold_result["subject_results"][subject]
                # Use first theta_gt found
                for theta_gt_str, theta_result in subject_result["theta_results"].items():
                    thetas = theta_result["thetas"]
                    losses = theta_result["losses"]
                    theta_gt = theta_result["theta_gt"]
                    theta_hat = theta_result["theta_hat"]
                    
                    axes[idx].plot(thetas, losses, 'o-', linewidth=2, markersize=6)
                    axes[idx].axvline(theta_gt, color='g', linestyle='--', linewidth=2, label=f'Theta GT: {theta_gt}')
                    axes[idx].axvline(theta_hat, color='r', linestyle='--', linewidth=2, label=f'Theta Hat: {theta_hat:.2f}')
                    axes[idx].set_xlabel('Theta (seconds)', fontsize=11)
                    axes[idx].set_ylabel('Final Loss', fontsize=11)
                    axes[idx].set_title(f'{label.capitalize()} Example: {subject}\n(Error: {abs(theta_hat - theta_gt):.2f}s)', fontsize=12)
                    axes[idx].legend(fontsize=10)
                    axes[idx].grid(True, alpha=0.3)
                    found = True
                    break
                if found:
                    break
            if found:
                break
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "example_error_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved example error curves")
    
    print(f"All visualizations saved to: {plots_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function."""
    experiment_start_time = time.time()
    
    print("=" * 80)
    print("Full Evaluation Script for Nonlinear Inverse Optimization Framework")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load protocol
    protocol_path = os.path.join(project_root, "experimental_protocol.json")
    protocol = load_protocol(protocol_path)
    
    print("Protocol loaded:")
    print(f"  Theta candidates: {len(protocol['phase_0']['theta_candidates']['values'])} values")
    print(f"  Ground truth thetas: {len(protocol['phase_0']['ground_truth_thetas']['values'])} values")
    print(f"  Random seed: {protocol['phase_0']['random_seed']['value']}")
    print()
    
    # Estimate total experiment time
    estimated_total_time, _, _, _ = estimate_phase_time(protocol)
    print(f"Estimated total experiment time: {format_time(estimated_total_time)}")
    print("  (Note: This is a rough estimate based on typical computation times)")
    print()
    
    # Set random seeds
    seed = protocol["phase_0"]["random_seed"]["value"]
    set_random_seeds(seed)
    
    # Create output directory
    output_base_dir = os.path.join(project_root, "results", "full_evaluation")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Run Phase 1.1: Cross-validation
    print("Starting Phase 1.1: 3-Fold Cross-Validation...")
    print()
    phase_1_1_results, phase_1_1_stats = run_phase_1_1_cross_validation(protocol, output_base_dir)
    
    experiment_total_time = time.time() - experiment_start_time
    
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiment time: {format_time(experiment_total_time)}")
    print()


if __name__ == "__main__":
    main()

