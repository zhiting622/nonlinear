"""
Phase B: Train NN forward surrogate
Train a small NN to learn the mapping from IBI time window (resampled to fixed length L) 
to a single HR value.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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


class WindowDataset(Dataset):
    """Dataset for window-level training."""
    
    def __init__(self, X_train, Y_train):
        """
        Args:
            X_train: np.ndarray of shape (num_samples, L)
            Y_train: np.ndarray of shape (num_samples, 1)
        """
        self.X = torch.FloatTensor(X_train)
        self.Y = torch.FloatTensor(Y_train)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def prepare_training_data(x_full, t_full, y_full, t_y, theta_seconds, L, s=2):
    """
    Prepare training samples: extract windows and resample to fixed length L.
    
    Args:
        x_full: IBI sequence, shape (N,)
        t_full: timestamps in seconds, shape (N,)
        y_full: HR values, shape (M,)
        t_y: HR time points in seconds, shape (M,)
        theta_seconds: window length in seconds
        L: fixed resampled length
        s: sampling interval (for generating t_y if not provided)
    
    Returns:
        X_train: np.ndarray of shape (num_samples, L)
        Y_train: np.ndarray of shape (num_samples, 1)
    """
    X_train = []
    Y_train = []
    
    # If t_y is not provided, generate it from y_full indices
    if t_y is None:
        t_y = np.array([i * s for i in range(len(y_full))])
    
    # For each HR index i
    for i in range(len(y_full)):
        center_time = t_y[i]
        target_y = y_full[i]
        
        try:
            # Extract and resample window
            x_window_L = extract_and_resample_window(
                x_full, t_full, center_time, theta_seconds, L
            )
            
            X_train.append(x_window_L)
            Y_train.append(target_y)
        except ValueError as e:
            # Skip if window is empty
            print(f"  Warning: Skipping sample {i} (center_time={center_time:.2f}): {e}")
            continue
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train).reshape(-1, 1)
    
    return X_train, Y_train


def train_nn_forward(
    subject_name="S1",
    theta_seconds=10.0,
    L=32,
    num_epochs=20,
    learning_rate=1e-3,
    batch_size=32,
    hidden_dim=16,
    s=2
):
    """
    Train NN forward surrogate.
    
    Args:
        subject_name: subject identifier (e.g., "S1")
        theta_seconds: window length in seconds
        L: fixed input length
        num_epochs: number of training epochs
        learning_rate: learning rate
        batch_size: batch size
        hidden_dim: hidden dimension of NN
        s: sampling interval for generating y
    """
    print("=" * 60)
    print("Phase B: Training NN Forward Surrogate")
    print("=" * 60)
    print(f"Subject: {subject_name}")
    print(f"Theta (seconds): {theta_seconds}")
    print(f"Fixed window length L: {L}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
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
    
    # Step 2: Generate HR teacher (ground truth)
    print("Step 2: Generating HR teacher (ground truth)...")
    # Note: theta in generate_y is in seconds, but the function uses it as window size
    # We need to convert theta_seconds to the format expected by generate_y
    # From the code, generate_y expects theta as window size in seconds
    y_full = generate_y(x_full, theta_seconds, s=s)
    y_full = np.array(y_full)
    
    # Generate time points for y: each y_i corresponds to time i * s
    t_y = np.array([i * s for i in range(len(y_full))])
    
    print(f"  Generated y_full with length: {len(y_full)}")
    print(f"  Generated t_y with length: {len(t_y)}")
    print(f"  y range: [{np.min(y_full):.4f}, {np.max(y_full):.4f}]")
    print(f"  y unit: seconds (mean Peak-to-Peak interval from IBI sequence)")
    print()
    
    # Step 3: Construct training samples
    print("Step 3: Constructing training samples...")
    X_train, Y_train = prepare_training_data(
        x_full, t_full, y_full, t_y, theta_seconds, L, s=s
    )
    print(f"  Created {len(X_train)} training samples")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Y_train shape: {Y_train.shape}")
    print()
    
    if len(X_train) == 0:
        raise ValueError("No training samples created! Check window extraction.")
    
    # Step 4: Create dataset and dataloader
    print("Step 4: Creating dataset and dataloader...")
    dataset = WindowDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Number of batches: {len(dataloader)}")
    print()
    
    # Step 5: Initialize model
    print("Step 5: Initializing model...")
    model = WindowNN(input_dim=L, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    print(f"  Model: WindowNN(input_dim={L}, hidden_dim={hidden_dim})")
    print()
    
    # Step 6: Training loop
    print("Step 6: Training...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            # Forward pass
            y_hat = model(batch_x)
            loss = loss_fn(y_hat, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    print()
    
    # Step 7: Save model
    print("Step 7: Saving model...")
    results_dir = os.path.join(project_root, "results", "models")
    os.makedirs(results_dir, exist_ok=True)
    
    model_path = os.path.join(results_dir, f"window_nn_theta_{theta_seconds}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to: {model_path}")
    print()
    
    # Step 8: Quick validation - test on a few random windows
    print("Step 8: Quick validation...")
    model.eval()
    with torch.no_grad():
        # Test on a few random samples
        test_indices = np.random.choice(len(X_train), min(5, len(X_train)), replace=False)
        print(f"  Testing on {len(test_indices)} random samples:")
        
        for idx in test_indices:
            x_test = torch.FloatTensor(X_train[idx:idx+1])
            y_true = Y_train[idx][0]
            y_pred = model(x_test).item()
            error = abs(y_pred - y_true)
            print(f"    Sample {idx}: y_true={y_true:.4f}, y_pred={y_pred:.4f}, error={error:.4f}")
    
    print()
    
    # Step 9: Plot curves (y_true vs y_pred)
    print("Step 9: Generating curve plot (y_true vs y_pred)...")
    model.eval()
    with torch.no_grad():
        # Randomly sample 500 samples (or all if less than 500)
        num_samples_to_plot = min(500, len(X_train))
        plot_indices = np.random.choice(len(X_train), num_samples_to_plot, replace=False)
        # Sort indices to plot in order
        plot_indices = np.sort(plot_indices)
        
        y_true_plot = []
        y_pred_plot = []
        
        for idx in plot_indices:
            x_test = torch.FloatTensor(X_train[idx:idx+1])
            y_true_val = Y_train[idx][0]
            y_pred_val = model(x_test).item()
            y_true_plot.append(y_true_val)
            y_pred_plot.append(y_pred_val)
        
        y_true_plot = np.array(y_true_plot)
        y_pred_plot = np.array(y_pred_plot)
        
        # Create curve plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_true_plot, label='y_true', linewidth=1.5, alpha=0.8)
        ax.plot(y_pred_plot, label='y_pred', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Value (seconds)', fontsize=12)
        ax.set_title(f'NN Prediction vs Ground Truth\n({num_samples_to_plot} samples, theta={theta_seconds}s)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Calculate and display R² (manually to avoid sklearn dependency)
        ss_res = np.sum((y_true_plot - y_pred_plot) ** 2)
        ss_tot = np.sum((y_true_plot - np.mean(y_true_plot)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        results_dir = os.path.join(project_root, "results", "models")
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f"scatter_y_true_vs_pred_theta_{theta_seconds}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Curve plot saved to: {plot_path}")
        print(f"  R² score: {r2:.4f}")
    
    print()
    print("=" * 60)
    print("Phase B Training Complete!")
    print("=" * 60)


def main():
    """Main function."""
    # Parameters
    subject_name = "S1"
    theta_seconds = 10.0  # window length in seconds
    L = 32                # fixed input length
    num_epochs = 20
    learning_rate = 1e-3
    batch_size = 32
    hidden_dim = 16
    s = 2                 # sampling interval for generating y
    
    train_nn_forward(
        subject_name=subject_name,
        theta_seconds=theta_seconds,
        L=L,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        s=s
    )


if __name__ == "__main__":
    main()

