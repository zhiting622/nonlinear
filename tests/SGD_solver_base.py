# -*- coding: utf-8 -*-
import numpy as np


def solve_constrained_ls(H, y, bounds=(0.9, 6.0), lam=0, learning_rate=0.01, 
                         max_iter=1000, batch_size=None, tol=1e-6, verbose=False):
    """
    Solve constrained minimization problem using Stochastic Gradient Descent (SGD):
    min ‖Hx – y‖₁ + λ‖D x‖₁
    s.t.  lower ≤ x ≤ upper
    
    Uses subgradient method and projection to handle L1 norm and bound constraints.
    
    Args:
        H: matrix H (m x n)
        y: target vector y (m,)
        bounds: (lower, upper) bounds tuple
        lam: regularization parameter λ
        learning_rate: learning rate (initial learning rate, adaptive)
        max_iter: maximum number of iterations
        batch_size: batch size (None means use all data)
        tol: convergence tolerance
        verbose: whether to print progress information
    
    Returns:
        x: optimized solution vector (n,)
    """
    H = np.array(H)
    y = np.array(y)
    m, n = H.shape
    
    lower, upper = bounds
    
    # Initialize: random initialization within bounds
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(lower, upper, n)
    
    # Build first-order difference matrix D (for smooth regularization)
    D = np.eye(n, k=0) - np.eye(n, k=1)
    D = D[:-1]  # (n-1) x n
    
    # Adaptive learning rate
    lr = learning_rate
    
    # For convergence detection
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        # Select batch indices
        if batch_size is None or batch_size >= m:
            # Use all data
            batch_idx = np.arange(m)
        else:
            # Randomly select batch
            batch_idx = np.random.choice(m, size=batch_size, replace=False)
        
        H_batch = H[batch_idx, :]
        y_batch = y[batch_idx]
        
        # Compute residual
        residual = np.dot(H_batch, x) - y_batch
        
        # Subgradient of L1 norm: sign(residual) @ H_batch
        # For values close to 0, use smooth approximation
        epsilon = 1e-8
        sign_residual = np.sign(residual)
        # For values very close to 0, use smooth approximation to avoid non-unique subgradient
        sign_residual[np.abs(residual) < epsilon] = residual[np.abs(residual) < epsilon] / epsilon
        
        # Gradient: H_batch^T @ sign_residual
        grad_data = np.dot(H_batch.T, sign_residual)
        
        # If using batch, need to scale gradient
        if batch_size is not None and batch_size < m:
            grad_data = grad_data * (m / batch_size)
        
        # Subgradient of regularization term
        if lam > 0:
            Dx = np.dot(D, x)
            sign_Dx = np.sign(Dx)
            sign_Dx[np.abs(Dx) < epsilon] = Dx[np.abs(Dx) < epsilon] / epsilon
            grad_reg = lam * np.dot(D.T, sign_Dx)
            grad = grad_data + grad_reg
        else:
            grad = grad_data
        
        # Gradient descent step
        x_new = x - lr * grad
        
        # Project to bound constraints: project to [lower, upper]
        x_new = np.clip(x_new, lower, upper)
        
        # Update
        x = x_new
        
        # Compute full loss (for convergence detection)
        if iteration % 100 == 0 or iteration == max_iter - 1:
            residual_full = np.dot(H, x) - y
            loss = np.sum(np.abs(residual_full))
            if lam > 0:
                Dx_full = np.dot(D, x)
                loss += lam * np.sum(np.abs(Dx_full))
            
            # Check convergence
            if abs(prev_loss - loss) < tol:
                if verbose:
                    print(f"SGD converged at iteration {iteration}, loss={loss:.6f}")
                break
            
            prev_loss = loss
            
            # Adaptive learning rate decay
            if iteration > 0 and iteration % 500 == 0:
                lr = lr * 0.9
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, loss={loss:.6f}, lr={lr:.6f}")
    
    return x

