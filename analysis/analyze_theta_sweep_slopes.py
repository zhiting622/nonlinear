"""
Analyze theta sweep error curves: compute slopes from original data.

This script:
1. Loads three error curve JSON files with different true theta values
2. Computes slopes (differences divided by theta differences) between neighboring points
3. Uses original data (before interpolation)
4. Plots and saves the slopes for each curve
"""

import sys
import os
import json
import numpy as np

# Try to import matplotlib with non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Define the three error curve files and their true theta values
ERROR_CURVES = [
    {
        "file": os.path.join(project_root, "results", "theta_sweep", "theta_sweep_results_SS3.json"),
        "true_theta": 20.0,
        "label": "S3, true_theta=20"
    },
    {
        "file": os.path.join(project_root, "results", "theta_sweep_result", "theta_sweep_results_SS2.json"),
        "true_theta": 10.0,
        "label": "S2, true_theta=10"
    },
    {
        "file": os.path.join(project_root, "results", "theta_sweep_result", "theta_sweep_results_SS3.json"),
        "true_theta": 15.0,
        "label": "S3, true_theta=15"
    }
]

# Output directory
output_dir = os.path.join(project_root, "analysis")
os.makedirs(output_dir, exist_ok=True)


def load_error_curve(json_path):
    """Load error curve data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    theta_candidates = data["theta_candidates"]
    results = data["results"]
    
    # Extract theta and final_loss values
    thetas = []
    losses = []
    
    for theta in theta_candidates:
        theta_key = str(theta)
        if theta_key in results:
            thetas.append(float(theta))
            losses.append(results[theta_key]["final_loss"])
    
    # Sort by theta
    sorted_indices = np.argsort(thetas)
    thetas = np.array(thetas)[sorted_indices]
    losses = np.array(losses)[sorted_indices]
    
    return thetas, losses


def compute_slopes(thetas, losses):
    """
    Compute slopes between neighboring points using original data.
    
    Slope = (loss[i+1] - loss[i]) / (theta[i+1] - theta[i])
    
    Args:
        thetas: array of theta values (original, not interpolated)
        losses: array of loss values (original, not interpolated)
    
    Returns:
        theta_slope: array of theta values for slopes (midpoints between consecutive points)
        slopes: array of slope values
    """
    if len(losses) < 2:
        return np.array([]), np.array([])
    
    # Compute slopes: (loss[i+1] - loss[i]) / (theta[i+1] - theta[i])
    loss_diffs = losses[1:] - losses[:-1]
    theta_diffs = thetas[1:] - thetas[:-1]
    
    # Avoid division by zero (shouldn't happen if thetas are unique, but just in case)
    slopes = np.divide(loss_diffs, theta_diffs, 
                       out=np.zeros_like(loss_diffs), 
                       where=theta_diffs != 0)
    
    # Theta values for slopes are midpoints between consecutive points
    theta_slope = (thetas[:-1] + thetas[1:]) / 2.0
    
    return theta_slope, slopes


def plot_slopes(theta_slope, slopes, true_theta, label, output_path):
    """Plot slopes and save to file."""
    if not HAS_MATPLOTLIB:
        print(f"  Skipping plot (matplotlib not available)")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(theta_slope, slopes, marker='o', linewidth=2, markersize=6, label='Slope')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.axvline(x=true_theta, color='g', linestyle='--', linewidth=1, alpha=0.5, 
                   label=f'True theta={true_theta}')
        
        ax.set_xlabel("Theta (seconds)", fontsize=12)
        ax.set_ylabel("Slope (Δloss / Δtheta)", fontsize=12)
        ax.set_title(f"Slopes in Error Curve\n{label}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot to: {output_path}")
    except Exception as e:
        print(f"  ERROR plotting: {e}")


def save_slopes_data(theta_slope, slopes, true_theta, output_path):
    """Save slopes data to JSON file."""
    data = {
        "true_theta": true_theta,
        "theta_slope": theta_slope.tolist(),
        "slopes": slopes.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved slopes data to: {output_path}")


def main():
    """Main function."""
    print("=" * 80)
    print("Analyzing Theta Sweep Error Curves: Computing Slopes")
    print("=" * 80)
    print()
    
    for i, curve_info in enumerate(ERROR_CURVES, 1):
        json_path = curve_info["file"]
        true_theta = curve_info["true_theta"]
        label = curve_info["label"]
        
        print(f"Processing curve {i}/3: {label}")
        print(f"  File: {json_path}")
        
        # Check if file exists
        if not os.path.exists(json_path):
            print(f"  ERROR: File not found: {json_path}")
            print()
            continue
        
        # Load error curve (original data, no interpolation)
        thetas, losses = load_error_curve(json_path)
        print(f"  Loaded {len(thetas)} data points (original data)")
        print(f"  Theta range: [{thetas.min():.1f}, {thetas.max():.1f}]")
        print(f"  Loss range: [{losses.min():.6f}, {losses.max():.6f}]")
        
        # Compute slopes from original data
        theta_slope, slopes = compute_slopes(thetas, losses)
        print(f"  Computed {len(slopes)} slopes")
        if len(slopes) > 0:
            print(f"  Slope range: [{slopes.min():.6f}, {slopes.max():.6f}]")
        
        # Save plot
        plot_filename = f"slopes_true_theta_{int(true_theta)}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plot_slopes(theta_slope, slopes, true_theta, label, plot_path)
        
        # Save slopes data
        data_filename = f"slopes_true_theta_{int(true_theta)}.json"
        data_path = os.path.join(output_dir, data_filename)
        save_slopes_data(theta_slope, slopes, true_theta, data_path)
        
        print()
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()

