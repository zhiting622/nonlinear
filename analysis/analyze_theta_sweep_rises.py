"""
Analyze theta sweep error curves: interpolate and compute rises.

This script:
1. Loads three error curve JSON files with different true theta values
2. Linearly interpolates each curve to have data points at every integer in their domain
3. Computes rises (differences) between neighboring integer points
4. Plots and saves the rises for each curve
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


def interpolate_to_integers(thetas, losses):
    """
    Linearly interpolate the curve to have data points at every integer in the domain.
    
    Args:
        thetas: array of theta values (may not be integers)
        losses: array of corresponding loss values
    
    Returns:
        theta_int: array of integer theta values (from min to max)
        loss_int: array of interpolated loss values at integer thetas
    """
    # Find integer range
    theta_min = int(np.ceil(thetas.min()))
    theta_max = int(np.floor(thetas.max()))
    theta_int = np.arange(theta_min, theta_max + 1, dtype=int)
    
    # Use numpy.interp for linear interpolation
    # numpy.interp handles extrapolation by using the nearest boundary value
    loss_int = np.interp(theta_int, thetas, losses)
    
    return theta_int, loss_int


def compute_rises(theta_int, loss_int):
    """
    Compute rises (differences) between neighboring integer points.
    
    Args:
        theta_int: array of integer theta values
        loss_int: array of loss values at integer thetas
    
    Returns:
        theta_rise: array of theta values for rises (midpoints between consecutive integers)
        rises: array of rise values (loss[i+1] - loss[i])
    """
    if len(loss_int) < 2:
        return np.array([]), np.array([])
    
    # Compute rises: loss[i+1] - loss[i]
    rises = loss_int[1:] - loss_int[:-1]
    
    # Theta values for rises are midpoints between consecutive integers
    theta_rise = theta_int[:-1] + 0.5
    
    return theta_rise, rises


def plot_rises(theta_rise, rises, true_theta, label, output_path):
    """Plot rises and save to file."""
    if not HAS_MATPLOTLIB:
        print(f"  Skipping plot (matplotlib not available)")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(theta_rise, rises, marker='o', linewidth=2, markersize=6, label='Rise')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax.axvline(x=true_theta, color='g', linestyle='--', linewidth=1, alpha=0.5, 
                   label=f'True theta={true_theta}')
        
        ax.set_xlabel("Theta (seconds)", fontsize=12)
        ax.set_ylabel("Rise (loss[i+1] - loss[i])", fontsize=12)
        ax.set_title(f"Rises in Error Curve\n{label}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot to: {output_path}")
    except Exception as e:
        print(f"  ERROR plotting: {e}")


def save_rises_data(theta_rise, rises, true_theta, output_path):
    """Save rises data to JSON file."""
    data = {
        "true_theta": true_theta,
        "theta_rise": theta_rise.tolist(),
        "rises": rises.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved rises data to: {output_path}")


def main():
    """Main function."""
    print("=" * 80)
    print("Analyzing Theta Sweep Error Curves: Computing Rises")
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
        
        # Load error curve
        thetas, losses = load_error_curve(json_path)
        print(f"  Loaded {len(thetas)} data points")
        print(f"  Theta range: [{thetas.min():.1f}, {thetas.max():.1f}]")
        print(f"  Loss range: [{losses.min():.6f}, {losses.max():.6f}]")
        
        # Interpolate to integers
        theta_int, loss_int = interpolate_to_integers(thetas, losses)
        print(f"  Interpolated to {len(theta_int)} integer points: [{theta_int.min()}, {theta_int.max()}]")
        
        # Compute rises
        theta_rise, rises = compute_rises(theta_int, loss_int)
        print(f"  Computed {len(rises)} rises")
        if len(rises) > 0:
            print(f"  Rise range: [{rises.min():.6f}, {rises.max():.6f}]")
        
        # Save plot
        plot_filename = f"rises_true_theta_{int(true_theta)}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plot_rises(theta_rise, rises, true_theta, label, plot_path)
        
        # Save rises data
        data_filename = f"rises_true_theta_{int(true_theta)}.json"
        data_path = os.path.join(output_dir, data_filename)
        save_rises_data(theta_rise, rises, true_theta, data_path)
        
        print()
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()

