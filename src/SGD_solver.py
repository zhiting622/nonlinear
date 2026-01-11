# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def solve_constrained_ls(H, y, bounds=(0.9, 6.0), lam=0, learning_rate=0.001, 
                         max_iter=5000, batch_size=None, tol=1e-8, verbose=False, 
                         huber_delta=1e-3):
    """
    Solve constrained minimization problem using Stochastic Gradient Descent (SGD):
    min ‖Hx – y‖₁ + λ‖D x‖₁
    s.t.  lower ≤ x ≤ upper
    
    Uses Huber loss to smoothly approximate L1 norm, with improved initialization and learning rate strategy.
    
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
        huber_delta: delta parameter for Huber loss (used for smoothing L1 norm)
    
    Returns:
        x: optimized solution vector (n,)
    """
    H = np.array(H)
    y = np.array(y)
    m, n = H.shape
    
    lower, upper = bounds
    
    # Improved initialization: use least squares solution projected into bounds
    try:
        # Try to use least squares solution
        HTH = np.dot(H.T, H)
        HTy = np.dot(H.T, y)
        # Add small regularization term to avoid singular matrix
        x_ls = np.linalg.solve(HTH + 1e-8 * np.eye(n), HTy)
        # Project into bounds
        x = np.clip(x_ls, lower, upper)
    except:
        # If failed, initialize with midpoint of bounds
        x = np.full(n, (lower + upper) / 2.0)
    
    # Build first-order difference matrix D (for smooth regularization)
    D = np.eye(n, k=0) - np.eye(n, k=1)
    D = D[:-1]  # (n-1) x n
    
    # Adaptive learning rate
    lr = learning_rate
    best_loss = float('inf')
    best_x = x.copy()
    patience = 1000  # Early stop if loss no longer decreases
    no_improve_count = 0
    
    # For convergence detection
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        # Use all data (no batch to improve accuracy)
        H_batch = H
        y_batch = y
        
        # Compute residual
        residual = np.dot(H_batch, x) - y_batch
        
        # Use Huber loss to smoothly approximate L1 norm (smoother, more stable gradients)
        # Huber loss: L_delta(r) = 0.5*r^2/delta if |r| < delta, else |r| - 0.5*delta
        abs_residual = np.abs(residual)
        huber_grad = np.where(
            abs_residual < huber_delta,
            residual / huber_delta,  # When |r| < delta, gradient is r/delta
            np.sign(residual)  # When |r| >= delta, gradient is sign(r)
        )
        
        # Gradient: H_batch^T @ huber_grad
        grad_data = np.dot(H_batch.T, huber_grad)
        
        # Gradient of regularization term (also using Huber smoothing)
        if lam > 0:
            Dx = np.dot(D, x)
            abs_Dx = np.abs(Dx)
            huber_grad_Dx = np.where(
                abs_Dx < huber_delta,
                Dx / huber_delta,
                np.sign(Dx)
            )
            grad_reg = lam * np.dot(D.T, huber_grad_Dx)
            grad = grad_data + grad_reg
        else:
            grad = grad_data
        
        # Gradient descent step
        x_new = x - lr * grad
        
        # Project to bound constraints: project to [lower, upper]
        x_new = np.clip(x_new, lower, upper)
        
        # Update
        x = x_new
        
        # Compute full loss (using true L1 norm, not Huber loss)
        if iteration % 50 == 0 or iteration == max_iter - 1:
            residual_full = np.dot(H, x) - y
            loss = np.sum(np.abs(residual_full))
            if lam > 0:
                Dx_full = np.dot(D, x)
                loss += lam * np.sum(np.abs(Dx_full))
            
            # Record best solution
            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 50
            
            # Check convergence (using relative change)
            if prev_loss > 1e-10 and abs(prev_loss - loss) / prev_loss < tol:
                if verbose:
                    print(f"SGD converged at iteration {iteration}, loss={loss:.6f}")
                break
            
            # Early stop if no improvement for long time
            if no_improve_count >= patience:
                if verbose:
                    print(f"SGD stopped early at iteration {iteration} (no improvement), best_loss={best_loss:.6f}")
                x = best_x
                break
            
            prev_loss = loss
            
            # Adaptive learning rate decay (more conservative strategy)
            if iteration > 0 and iteration % 200 == 0:
                lr = lr * 0.95  # Slower decay
            
            if verbose and iteration % 200 == 0:
                print(f"Iteration {iteration}, loss={loss:.6f}, best_loss={best_loss:.6f}, lr={lr:.6f}")
    
    # Return best solution
    if best_loss < np.sum(np.abs(np.dot(H, x) - y)):
        x = best_x
    
    return x


def solve_x_sgd(
    x_init: torch.Tensor,
    forward_fn,  # callable(x_t) -> y_hat_t (M,)
    y_obs_t: torch.Tensor,  # (M,)
    bounds: tuple = (0.9, 6.0),
    lam: float = 0.0,
    num_iters: int = 1000,
    lr: float = 0.001,
    tol: float = 1e-6,  # Relaxed tolerance to allow more iterations
    verbose: bool = False,
    penalty_fn=None,  # optional additional penalty function
) -> dict:
    """
    Optimize x only using SGD with a generic forward function.
    
    This function is forward-agnostic: it works with both linear (H @ x) 
    and nonlinear (NN) forward models.
    
    Args:
        x_init: initial guess for x, shape (N,)
        forward_fn: callable that takes x_t (N,) and returns y_hat_t (M,)
        y_obs_t: observed y values, shape (M,), torch.Tensor
        bounds: (lower, upper) bounds for x
        lam: regularization parameter for TV penalty
        num_iters: maximum number of iterations
        lr: learning rate
        tol: convergence tolerance
        verbose: whether to print progress
        penalty_fn: optional additional penalty function(x) -> scalar
    
    Returns:
        dict with keys:
            - x_star: optimized x, shape (N,), numpy array
            - y_hat: predicted y from final x, shape (M,), numpy array
            - losses: list of loss values during optimization
            - final_loss: final loss value
    """
    device = x_init.device
    lower, upper = bounds
    
    # Create x as a leaf Parameter to ensure optimizer updates it correctly
    # This is created once outside the loop and never reassigned
    x = torch.nn.Parameter(x_init.clone().detach(), requires_grad=True)
    y_obs_t = y_obs_t.to(device)
    
    # Build first-order difference matrix D for TV penalty (if needed)
    N = len(x_init)
    if lam > 0:
        # D is (N-1) x N: D @ x gives differences
        D = torch.zeros(N - 1, N, device=device)
        for i in range(N - 1):
            D[i, i] = 1.0
            D[i, i + 1] = -1.0
    
    # Optimizer - pass [x] to ensure it updates the Parameter
    optimizer = torch.optim.SGD([x], lr=lr)
    
    # Save initial x for delta_x calculation (detached copy)
    x_init_saved = x.data.clone()
    
    # Compute initial loss using exact same forward path and loss definition
    with torch.no_grad():
        y_hat_init = forward_fn(x)
        residual_init = y_hat_init - y_obs_t
        loss_data_init = torch.mean(residual_init ** 2)  # MSE
        loss_reg_init = torch.tensor(0.0, device=device)
        if lam > 0:
            Dx_init = torch.matmul(D, x)
            loss_reg_init = lam * torch.mean(torch.abs(Dx_init))
        loss_penalty_init = torch.tensor(0.0, device=device)
        if penalty_fn is not None:
            loss_penalty_init = penalty_fn(x)
        initial_loss = (loss_data_init + loss_reg_init + loss_penalty_init).item()
    
    # Track losses
    losses = []
    best_loss = float('inf')
    best_x = None
    patience = 200
    no_improve_count = 0
    prev_loss = float('inf')
    
    for iteration in range(num_iters):
        optimizer.zero_grad()
        
        # Forward pass
        y_hat_t = forward_fn(x)  # (M,)
        
        # Data mismatch loss: MSE (can be changed to L1)
        residual = y_hat_t - y_obs_t
        loss_data = torch.mean(residual ** 2)  # MSE
        
        # TV regularization penalty
        loss_reg = torch.tensor(0.0, device=device)
        if lam > 0:
            Dx = torch.matmul(D, x)  # (N-1,)
            loss_reg = lam * torch.mean(torch.abs(Dx))
        
        # Additional penalty function
        loss_penalty = torch.tensor(0.0, device=device)
        if penalty_fn is not None:
            loss_penalty = penalty_fn(x)
        
        # Total loss
        loss = loss_data + loss_reg + loss_penalty
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm for diagnostic (check if optimization is really moving)
        grad_norm = x.grad.norm().item() if x.grad is not None else 0.0
        
        # Gradient scaling: if grad_norm is too small, scale up the gradients
        # This prevents updates from being smaller than machine precision
        min_grad_norm = 1e-2  # Minimum gradient norm threshold
        if grad_norm > 0 and grad_norm < min_grad_norm:
            scale_factor = min_grad_norm / (grad_norm + 1e-10)
            x.grad.data = x.grad.data * scale_factor
            grad_norm_scaled = x.grad.norm().item()
            if iteration < 5 and verbose:
                print(f"    Gradient scaled: {grad_norm:.6e} -> {grad_norm_scaled:.6e} (scale={scale_factor:.2f})")
        else:
            grad_norm_scaled = grad_norm
        
        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_([x], max_norm=10.0)
        
        # Compute expected update before step (for diagnostic)
        if iteration < 5:
            with torch.no_grad():
                expected_update_norm = (lr * x.grad.data).norm().item() if x.grad is not None else 0.0
                if verbose:
                    print(f"    Expected update norm (lr * grad): {expected_update_norm:.10e}")
        
        # Optimizer step - this should update x in-place
        optimizer.step()
        
        # Compute actual update after step (for diagnostic)
        if iteration < 5:
            with torch.no_grad():
                # Get x before clamping
                x_before_clamp = x.data.clone()
                actual_update_norm = (x_before_clamp - x_init_saved).norm().item()
                if verbose:
                    print(f"    Actual update norm after step: {actual_update_norm:.10e}")
        
        # Project to bounds (in-place update of x.data)
        with torch.no_grad():
            x.data = torch.clamp(x.data, lower, upper)
        
        # Track loss (detach for logging)
        loss_val = loss.item()
        losses.append(loss_val)
        
        # Compute delta_x = ||x - x_init|| / ||x_init|| to verify x actually updates
        with torch.no_grad():
            delta_x = (x.data - x_init_saved).norm().item() / (x_init_saved.norm().item() + 1e-10)
            x_norm_diff = (x.data - x_init_saved).norm().item()
        
        # For first ~5 iterations, print x[0] and ||x - x_init|| to verify updates
        if iteration < 5 and verbose:
            print(f"  Iteration {iteration}:")
            print(f"    x[0].item() = {x.data[0].item():.8f}")
            print(f"    ||x - x_init|| = {x_norm_diff:.8f}")
            print(f"    delta_x = {delta_x:.8f}")
        
        # Log initial loss and gradient norm for diagnostic (especially for degenerate init)
        if iteration == 0 and verbose:
            print(f"  Iteration 0 (detailed):")
            print(f"    initial_loss (pre-computed): {initial_loss:.8f}")
            print(f"    losses[0] (from forward): {loss_val:.8f}")
            print(f"    Difference: {abs(initial_loss - loss_val):.10f}")
            print(f"    grad_norm: {grad_norm:.8f}")
            print(f"    x[0].item() = {x.data[0].item():.8f}")
            print(f"    ||x - x_init|| = {x_norm_diff:.8f}")
            print(f"    delta_x (||x - x_init|| / ||x_init||): {delta_x:.8f}")
        
        # Check for best solution (save detached copy, but don't reassign x)
        if loss_val < best_loss:
            best_loss = loss_val
            best_x = x.data.clone()  # Save best x, but don't reassign the Parameter
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Convergence check (only after minimum iterations to allow proper learning)
        min_iterations_before_convergence_check = 50
        if iteration >= min_iterations_before_convergence_check and prev_loss > 1e-10:
            rel_change = abs(prev_loss - loss_val) / prev_loss
            if rel_change < tol:
                if verbose:
                    print(f"SGD converged at iteration {iteration}, loss={loss_val:.6f}, rel_change={rel_change:.8f}")
                break
        
        # Early stopping (only after minimum iterations)
        # DO NOT reassign x here - we'll use best_x at the end instead
        min_iterations_before_early_stop = 100
        if iteration >= min_iterations_before_early_stop and no_improve_count >= patience:
            if verbose:
                print(f"SGD stopped early at iteration {iteration} (no improvement for {patience} iterations), best_loss={best_loss:.6f}")
                # x is still the current Parameter, we'll use best_x as x_final at the end
            break
        
        prev_loss = loss_val
        
        # Learning rate decay
        if iteration > 0 and iteration % 200 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
        
        if verbose and iteration % 100 == 0 and iteration >= 5:  # Skip if already printed in first 5 iterations
            with torch.no_grad():
                delta_x_current = (x.data - x_init_saved).norm().item() / (x_init_saved.norm().item() + 1e-10)
            print(f"Iteration {iteration}, loss={loss_val:.6f}, best_loss={best_loss:.6f}, grad_norm={grad_norm:.6f}, delta_x={delta_x_current:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Even when verbose=False, print every 200 iterations to show progress
        if not verbose and iteration > 0 and iteration % 200 == 0:
            print(f"  ... iteration {iteration}/{num_iters}, loss={loss_val:.6f}", flush=True)
        
        # Also print at key milestones (50, 100) for faster feedback
        if not verbose and iteration in [50, 100]:
            print(f"  ... iteration {iteration}/{num_iters}, loss={loss_val:.6f}", flush=True)
    
    # Use best solution
    if best_x is not None:
        x_final = best_x
    else:
        x_final = x.detach()
    
    # Final forward pass (no grad)
    with torch.no_grad():
        y_hat_final = forward_fn(x_final)
    
    # Compute final delta_x
    with torch.no_grad():
        delta_x_final = (x_final - x_init_saved).norm().item() / (x_init_saved.norm().item() + 1e-10)
    
    return {
        'x_star': x_final.cpu().numpy(),
        'y_hat': y_hat_final.cpu().numpy(),
        'losses': losses,
        'final_loss': best_loss if best_x is not None else losses[-1],
        'initial_loss': initial_loss,
        'delta_x_final': delta_x_final
    }

