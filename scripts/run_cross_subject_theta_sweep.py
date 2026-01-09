"""
Cross-subject theta sweep: Train on S1 with theta=15, test on S3 with theta sweep.

This script:
1. Trains a NN forward surrogate on S1 with theta=15
2. Uses the trained model to run inverse optimization on S3
3. Performs theta sweep on S3 with various theta candidates
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.train_nn_forward import train_nn_forward
from scripts.run_inverse_nn_theta_sweep import run_theta_sweep, save_results, plot_error_curve


def infer_L_from_theta(theta):
    """
    Infer L from theta using a fixed time resolution formula.
    This simulates the scenario where we know the model's theta but not the training L.
    
    The formula is based on maintaining similar time resolution:
    - Original setting: theta=10s, L=16 -> resolution = 10/16 = 0.625s per point
    - For any theta: L = theta / 0.625
    
    Note: In practice, L should be saved with the model (as metadata or in filename).
    This function assumes a convention that can be inferred from theta alone.
    """
    time_resolution = 10.0 / 16.0  # seconds per point (convention from original setting)
    L = int(round(theta / time_resolution))
    return L


def main():
    """Main function for cross-subject theta sweep."""
    print("=" * 80)
    print("Cross-Subject Theta Sweep Experiment")
    print("=" * 80)
    print("Training: S1 with theta=15")
    print("Testing: S3 with theta_gt=15, theta' sweep")
    print()
    
    # Training parameters
    train_subject = "S1"
    train_theta = 20.0
    
    # Use fixed L=16 for both training and testing
    # This avoids dimension mismatch and data leakage issues
    L_train = 32
    print(f"Training L: {L_train} (fixed, independent of theta)")
    print()
    
    # ==========================================
    # Step 1: Train model on S1 with theta=15
    # ==========================================
    print("Step 1: Training NN forward surrogate on S1 with theta=15...")
    print()
    
    train_nn_forward(
        subject_name=train_subject,
        theta_seconds=train_theta,
        L=L_train,  # Use calculated L
        num_epochs=20,  # Default from train_nn_forward.py
        learning_rate=1e-3,  # Default
        batch_size=32,  # Default
        hidden_dim=16,  # Default
        s=2  # Default
    )
    
    print()
    print("=" * 80)
    print("Step 1 Complete: Model trained and saved")
    print("=" * 80)
    print()
    
    # ==========================================
    # Step 2: Run theta sweep on S3
    # ==========================================
    print("Step 2: Running theta sweep on S3...")
    print()
    
    # Parameters for theta sweep
    test_subject = "S3"
    theta_gt = 20.0  # Ground truth theta used to generate y_obs (for generating y_obs only)
    theta_candidates = [5, 7, 9, 11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 27, 29]
    
    # SGD parameters
    num_iters = 1000  # As requested
    lr = 1e-2  # Default from run_inverse_nn_theta_sweep.py
    lam = 0.01  # Default
    bounds = (0.3, 2.0)  # Default
    
    # Check if model exists
    # Note: We need to know which model to load. Since we trained with train_theta=15.0,
    # we look for window_nn_theta_15.0.pt. But in a real scenario, we might not know
    # which theta the model was trained with. For now, we assume we know which model to use.
    model_path = os.path.join(project_root, "results", "models", f"window_nn_theta_{train_theta}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")
    
    # For theta sweep: Use fixed L=16 (same as training)
    # This is correct because during theta sweep we don't know the true theta,
    # so we can't infer L from theta_gt. L=16 is a fixed hyperparameter.
    # It must match the training L to avoid dimension mismatch.
    L_test = 32  # Fixed value, must match L_train
    
    print(f"Using fixed L={L_test} for theta sweep (matches training L)")
    print("Note: During theta sweep, we don't know the true theta, so L is fixed")
    print()
    
    hidden_dim = 16  # This is a hyperparameter, typically known
    s = 2  # This is a data parameter, typically known
    
    # Auto-detect device
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"Using CPU with {torch.get_num_threads()} threads")
    print()
    
    # Run theta sweep
    results = run_theta_sweep(
        subject=test_subject,
        theta_gt=theta_gt,
        theta_candidates=theta_candidates,
        L=L_test,  # Use inferred L (not hardcoded)
        hidden_dim=hidden_dim,
        s=s,
        num_iters=num_iters,
        lr=lr,
        lam=lam,
        bounds=bounds,
        device=device,
        verbose=False,  # Set to True if you want per-iteration output
    )
    
    print()
    print("=" * 80)
    print("Step 2 Complete: Theta sweep finished")
    print("=" * 80)
    print()
    
    # ==========================================
    # Step 3: Save results and plot
    # ==========================================
    print("Step 3: Saving results and generating plots...")
    print()
    
    # Save results (using a descriptive name)
    json_path = save_results(test_subject, theta_candidates, results)
    print(f"Saved results to: {json_path}")
    
    # Plot error curve
    plot_path = plot_error_curve(test_subject, theta_candidates, results)
    if plot_path:
        print(f"Saved plot to: {plot_path}")
    
    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Training: S1, theta=15")
    print(f"  - Testing: S3, theta_gt=15")
    print(f"  - Theta candidates: {theta_candidates}")
    print(f"  - Results saved to: {json_path}")
    print(f"  - Plot saved to: {plot_path}")
    print()
    
    return results


if __name__ == "__main__":
    results = main()

