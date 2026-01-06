# pipeline
import sys
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from .data_loader import load_x
from .y_generator import generate_y
from .H_constructor import construct_H
from .LP_solver import solve_constrained_ls as solve_lp
from .SGD_solver import solve_constrained_ls as solve_sgd
from .error_calculator import calculate_error

"""
for each theta:
    construct H
    get x with LP_solver 
    get error value from Hx-y

get err matrix 
"""
def generate_error_matrix(y, runs=100, thetas=np.arange(1, 31), s=2, g=4, solver='lp'):
    """
    Generate error matrix for different theta values.
    
    Args:
        y: input data
        runs: number of runs (Monte Carlo iterations)
        thetas: array of theta values to test
        s: sampling interval
        g: grouping factor
        solver: 'lp' for linear programming or 'sgd' for stochastic gradient descent
    """
    # 根据solver参数选择求解器
    if solver.lower() == 'sgd':
        solve_func = solve_sgd
    else:  # 默认为 'lp'
        solve_func = solve_lp
    
    errs = np.empty((runs, len(thetas)))
    for run in range(runs):
        np.random.seed(run)  # Set seed for each run for reproducibility
        print("run", run)  # Commented out to reduce terminal output
        for theta in thetas:
            H = construct_H(y, theta, s, g)
            x = solve_func(H, y, bounds=(0.3*g, 2.0*g), lam=0)
            err = calculate_error(H, x, y)

            # append this single error
            errs[run, theta - 1] = err
    return errs

