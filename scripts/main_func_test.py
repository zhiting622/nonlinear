"""
Extension pipeline for window size estimation:
1. Get data
2. Truncate into pieces
3. Create H, solve with LP, generate error curve
4. Find inflection point
5. Recreate grouped signal by LP
6. Verify with observed y
7. Median aggregation over segments of an individual

This pipeline uses functions from the src/ folder without parallel processing or Monte Carlo.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

from src.data_loader import load_x
from src.y_generator import generate_y
from src.H_constructor import construct_H
from src.LP_solver import solve_constrained_ls as solve_lp
from src.SGD_solver import solve_constrained_ls as solve_sgd
from src.error_calculator import calculate_error
from src.error_matrix_generator import generate_error_matrix
from src.inflection_finder import find_inflection
from src.piece_utils import get_piece_starting_indices


def ungroup_x(x_grouped, g):
    """Convert grouped x to ungrouped by replacing each element with value/g repeated g times."""
    x_ungrouped = []
    for val in x_grouped:
        ungrouped_val = val / g
        x_ungrouped.extend([ungrouped_val] * g)
    return np.array(x_ungrouped)


def calculate_l1_error(y1, y2):
    """Calculate L1 error between two arrays."""
    return np.mean(np.abs(y1 - y2))


def get_x_grouped(y_piece, theta_hat_value, s=2, g=4, bounds_factor_low=0.3, 
                  bounds_factor_high=2.0, lam=0, solver='lp'):
    """
    Get x_grouped for a given piece using specified solver.
    
    Args:
        y_piece: piece of y data
        theta_hat_value: theta value to use for constructing H
        s: sampling interval
        g: grouping factor
        bounds_factor_low: lower bound factor
        bounds_factor_high: upper bound factor
        lam: regularization parameter
        solver: 'lp' or 'sgd'
    
    Returns:
        x_grouped: reconstructed grouped signal
    """
    H = construct_H(y_piece, theta_hat_value, s, g)
    H = np.array(H)
    bounds = (bounds_factor_low * g, bounds_factor_high * g)
    
    if solver.lower() == 'sgd':
        solve_func = solve_sgd
    else:
        solve_func = solve_lp
    
    x_grouped = solve_func(H, y_piece, bounds=bounds, lam=lam)
    return x_grouped


def process_single_piece(y_piece, thetas, s=2, g=4, bounds_factor_low=0.3, bounds_factor_high=2.0, lam=0, solver='lp'):
    """
    Process a single piece: generate error matrix, find inflection, and recreate signal.
    
    Args:
        y_piece: piece of y data to process
        thetas: array of theta values to test
        s: sampling interval
        g: grouping factor
        bounds_factor_low: lower bound factor for solver
        bounds_factor_high: upper bound factor for solver
        lam: regularization parameter
        solver: 'lp' for linear programming or 'sgd' for stochastic gradient descent
    
    Returns:
        tuple: (theta_hat, medians, x_grouped, y_recreated)
            theta_hat: estimated theta (1-based index) or None
            medians: median errors for each theta
            x_grouped: reconstructed grouped signal
            y_recreated: recreated y from the reconstructed signal
    """
    # Generate error matrix (runs=1, no Monte Carlo)
    errs = generate_error_matrix(y_piece, runs=1, thetas=thetas, s=s, g=g, solver=solver)
    
    # Find inflection point (includes median aggregation internally)
    theta_hat, medians, all_min_idxs, kept_min_idxs, froms, tos, rises, slopes = find_inflection(errs, g=g)
    
    # Recreate grouped signal using theta_hat
    if theta_hat is not None:
        # Convert 1-based to 0-based for indexing
        theta_idx = int(theta_hat) - 1
        if 0 <= theta_idx < len(thetas):
            theta_hat_value = thetas[theta_idx]
        else:
            theta_hat_value = thetas[len(thetas)//2]  # fallback to middle
    else:
        theta_hat_value = thetas[len(thetas)//2]  # fallback to middle
    
    # Construct H and solve for grouped signal
    H = construct_H(y_piece, theta_hat_value, s, g)
    H = np.array(H)
    bounds = (bounds_factor_low * g, bounds_factor_high * g)
    
    # Select solver based on solver parameter
    if solver.lower() == 'sgd':
        solve_func = solve_sgd
    else:  # Default to 'lp'
        solve_func = solve_lp
    
    x_grouped = solve_func(H, y_piece, bounds=bounds, lam=lam)
    
    # Create ungrouped solution and recreate y
    x_ungrouped = ungroup_x(x_grouped, g)
    y_recreated = generate_y(x_ungrouped, theta_hat_value, s)
    y_recreated = np.array(y_recreated)
    
    return theta_hat, medians, x_grouped, y_recreated


def pipeline(subject_name, true_theta, piece_len=1000, overlap_ratio=0.0, 
             thetas=None, s=2, g=4, bounds_factor_low=0.3, bounds_factor_high=2.0, lam=0, 
             solver='lp', output_dir=None):
    """
    Complete pipeline for a single subject:
    1. Get data
    2. Truncate into pieces
    3. Process each piece (error curve, inflection, recreate signal)
    4. Verify with observed y
    5. Median aggregation over segments
    
    Args:
        subject_name: subject identifier (e.g., "S1")
        true_theta: true theta value for generating y
        piece_len: length of each piece
        overlap_ratio: overlap ratio between pieces
        thetas: array of theta values to test (default: 1-30)
        s: sampling interval
        g: grouping factor
        bounds_factor_low: lower bound factor for solver
        bounds_factor_high: upper bound factor for solver
        lam: regularization parameter
        solver: 'lp' for linear programming or 'sgd' for stochastic gradient descent (used only for finding theta_hat)
        output_dir: directory to save plots
    
    Returns:
        dict with results including:
        - theta_hats: list of estimated thetas for each piece
        - theta_hat_median: median of all piece estimates
        - pieces_data: data for each piece
    """
    if thetas is None:
        thetas = np.arange(1, 31)
    
    # Set default output directory if not provided
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "results", "plots_comparison")
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing subject {subject_name} with true_theta={true_theta}")
    
    # Step 1: Get data
    print("  Step 1: Loading data...")
    x = load_x(subject_name)
    print(f"    Loaded x with length: {len(x)}")
    
    # Step 2: Generate y and truncate into pieces
    print("  Step 2: Generating y and truncating into pieces...")
    y = generate_y(x, true_theta, s)
    y = np.array(y)
    print(f"    Generated y with length: {len(y)}")
    
    piece_start_indices = get_piece_starting_indices(y, piece_len, overlap_ratio)
    print(f"    Found {len(piece_start_indices)} pieces")
    
    # Step 3: Process each piece - get x from both LP and SGD
    print("  Step 3: Processing pieces with both LP and SGD...")
    theta_hats_lp = []
    theta_hats_sgd = []
    pieces_data = []
    
    for i, start_idx in enumerate(piece_start_indices):
        print(f"    Processing piece {i+1}/{len(piece_start_indices)} (start_idx={start_idx})...")
        y_piece = y[start_idx:start_idx + piece_len]
        
        if len(y_piece) < piece_len:
            print(f"      Skipping piece {i+1}: too short ({len(y_piece)} < {piece_len})")
            continue
        
        try:
            # Generate error matrices for both LP and SGD
            print(f"      Generating error matrix with LP solver...")
            errs_lp = generate_error_matrix(y_piece, runs=1, thetas=thetas, s=s, g=g, solver='lp')
            error_curve_lp = errs_lp[0, :]  # Extract error curve (first run, or use median if multiple runs)
            
            print(f"      Generating error matrix with SGD solver...")
            errs_sgd = generate_error_matrix(y_piece, runs=1, thetas=thetas, s=s, g=g, solver='sgd')
            error_curve_sgd = errs_sgd[0, :]  # Extract error curve
            
            # Find theta_hat for both LP and SGD
            theta_hat_lp, medians_lp, all_min_idxs_lp, kept_min_idxs_lp, froms_lp, tos_lp, rises_lp, slopes_lp = find_inflection(errs_lp, g=g)
            theta_hat_sgd, medians_sgd, all_min_idxs_sgd, kept_min_idxs_sgd, froms_sgd, tos_sgd, rises_sgd, slopes_sgd = find_inflection(errs_sgd, g=g)
            
            # Determine theta_hat_value for LP
            if theta_hat_lp is not None:
                theta_idx_lp = int(theta_hat_lp) - 1
                if 0 <= theta_idx_lp < len(thetas):
                    theta_hat_value_lp = thetas[theta_idx_lp]
                else:
                    theta_hat_value_lp = thetas[len(thetas)//2]
            else:
                theta_hat_value_lp = thetas[len(thetas)//2]
            
            # Determine theta_hat_value for SGD
            if theta_hat_sgd is not None:
                theta_idx_sgd = int(theta_hat_sgd) - 1
                if 0 <= theta_idx_sgd < len(thetas):
                    theta_hat_value_sgd = thetas[theta_idx_sgd]
                else:
                    theta_hat_value_sgd = thetas[len(thetas)//2]
            else:
                theta_hat_value_sgd = thetas[len(thetas)//2]
            
            # Use LP theta_hat_value for x comparison plots (for consistency in visualization)
            theta_hat_value = theta_hat_value_lp
            
            # Plot error curves in two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Helper function to get y-lim (min to 5th largest value)
            def get_ylim_values(values):
                sorted_values = np.sort(values)
                if len(sorted_values) >= 5:
                    y_min = np.min(values)
                    y_max = sorted_values[-5]  # 5th largest (from the end)
                else:
                    y_min = np.min(values)
                    y_max = np.max(values)
                return y_min, y_max
            
            # Plot LP error curve
            y_min_lp, y_max_lp = get_ylim_values(error_curve_lp)
            ax1.plot(thetas, error_curve_lp, 'b-', label='LP', linewidth=2, alpha=0.7, marker='o', markersize=4)
            ax1.set_xlabel('Theta', fontsize=12)
            ax1.set_ylabel('Error', fontsize=12)
            ax1.set_title(f'Piece {i+1} (start_idx={start_idx}): LP Error Curve', fontsize=14)
            ax1.set_ylim(y_min_lp, y_max_lp)
            ax1.legend(loc='best', fontsize=11)
            ax1.grid(True, alpha=0.3)
            
            # Plot SGD error curve
            y_min_sgd, y_max_sgd = get_ylim_values(error_curve_sgd)
            ax2.plot(thetas, error_curve_sgd, 'r-', label='SGD', linewidth=2, alpha=0.7, marker='s', markersize=4)
            ax2.set_xlabel('Theta', fontsize=12)
            ax2.set_ylabel('Error', fontsize=12)
            ax2.set_title(f'Piece {i+1} (start_idx={start_idx}): SGD Error Curve', fontsize=14)
            ax2.set_ylim(y_min_sgd, y_max_sgd)
            ax2.legend(loc='best', fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the error curve plot
            error_plot_filename = os.path.join(output_dir, f'piece_{i+1}_start_{start_idx}_error_curves.png')
            plt.savefig(error_plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      Saved error curve plot to {error_plot_filename}")
            
            # Get x_grouped from both LP and SGD
            print(f"      Getting x from LP solver...")
            x_grouped_lp = get_x_grouped(y_piece, theta_hat_value, s, g, 
                                         bounds_factor_low, bounds_factor_high, lam, solver='lp')
            
            print(f"      Getting x from SGD solver...")
            x_grouped_sgd = get_x_grouped(y_piece, theta_hat_value, s, g, 
                                          bounds_factor_low, bounds_factor_high, lam, solver='sgd')
            
            # Plot both x on the same figure
            fig, ax = plt.subplots(figsize=(12, 6))
            indices = np.arange(len(x_grouped_lp))
            ax.plot(indices, x_grouped_lp, 'b-', label='LP', linewidth=2, alpha=0.7)
            ax.plot(indices, x_grouped_sgd, 'r-', label='SGD', linewidth=2, alpha=0.7)
            ax.set_xlabel('Index', fontsize=12)
            ax.set_ylabel('x_grouped', fontsize=12)
            ax.set_title(f'Piece {i+1} (start_idx={start_idx}): LP vs SGD Comparison\n'
                        f'theta_hat={theta_hat_value:.2f}', fontsize=14)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the x comparison plot
            x_plot_filename = os.path.join(output_dir, f'piece_{i+1}_start_{start_idx}_comparison.png')
            plt.savefig(x_plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"      Saved x comparison plot to {x_plot_filename}")
            
            pieces_data.append({
                'piece_idx': i,
                'start_idx': start_idx,
                'theta_hat_lp': theta_hat_lp,
                'theta_hat_sgd': theta_hat_sgd,
                'theta_hat_value_lp': theta_hat_value_lp,
                'theta_hat_value_sgd': theta_hat_value_sgd,
                'error_curve_lp': error_curve_lp,
                'error_curve_sgd': error_curve_sgd,
                'x_grouped_lp': x_grouped_lp,
                'x_grouped_sgd': x_grouped_sgd,
                'error_plot_filename': error_plot_filename,
                'x_plot_filename': x_plot_filename
            })
            
            # Add theta_hat values to lists
            if theta_hat_lp is not None:
                theta_hats_lp.append(theta_hat_lp)
            if theta_hat_sgd is not None:
                theta_hats_sgd.append(theta_hat_sgd)
                
        except Exception as e:
            print(f"      Error processing piece {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 4: Median aggregation over segments of this individual
    print("  Step 4: Median aggregation...")
    if theta_hats_lp:
        theta_hat_median_lp = np.median(theta_hats_lp)
    else:
        theta_hat_median_lp = None
    
    if theta_hats_sgd:
        theta_hat_median_sgd = np.median(theta_hats_sgd)
    else:
        theta_hat_median_sgd = None
    
    results = {
        'subject_name': subject_name,
        'true_theta': true_theta,
        'theta_hats_lp': theta_hats_lp,
        'theta_hats_sgd': theta_hats_sgd,
        'theta_hat_median_lp': theta_hat_median_lp,
        'theta_hat_median_sgd': theta_hat_median_sgd,
        'pieces_data': pieces_data,
        'num_pieces': len(pieces_data)
    }
    
    return results


def main():
    """Example usage of the pipeline."""
    # Test with a single subject
    subject_name = "S1"
    true_theta = 15
    output_dir = os.path.join(project_root, "results", "plots_comparison")
    
    results = pipeline(
        subject_name=subject_name,
        true_theta=true_theta,
        piece_len=1000,
        overlap_ratio=0.5,
        thetas=np.arange(1, 31),
        s=2,
        g=4,
        solver='lp',  # Only used for finding theta_hat
        output_dir=output_dir
    )
    
    print(f"\n=== Results Summary ===")
    print(f"Subject: {results['subject_name']}")
    print(f"True theta: {results['true_theta']}")
    print(f"\nLP Results:")
    print(f"  Median estimated theta: {results['theta_hat_median_lp']}")
    print(f"  Number of processed pieces: {results['num_pieces']}")
    print(f"  Individual piece estimates: {results['theta_hats_lp']}")
    print(f"\nSGD Results:")
    print(f"  Median estimated theta: {results['theta_hat_median_sgd']}")
    print(f"  Number of processed pieces: {results['num_pieces']}")
    print(f"  Individual piece estimates: {results['theta_hats_sgd']}")
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

