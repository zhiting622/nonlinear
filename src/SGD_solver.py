# -*- coding: utf-8 -*-
import numpy as np


def solve_constrained_ls(H, y, bounds=(0.9, 6.0), lam=0, learning_rate=0.001, 
                         max_iter=5000, batch_size=None, tol=1e-8, verbose=False, 
                         huber_delta=1e-3):
    """
    使用随机梯度下降（SGD）求解约束最小化问题：
    min ‖Hx – y‖₁ + λ‖D x‖₁
    s.t.  lower ≤ x ≤ upper
    
    使用Huber损失平滑近似L1范数，改进初始化和学习率策略。
    
    Args:
        H: 矩阵 H (m x n)
        y: 目标向量 y (m,)
        bounds: (lower, upper) 边界元组
        lam: 正则化参数 λ
        learning_rate: 学习率（初始学习率，可自适应）
        max_iter: 最大迭代次数
        batch_size: 批次大小（None表示使用全部数据）
        tol: 收敛容忍度
        verbose: 是否打印进度信息
        huber_delta: Huber损失的delta参数（用于平滑L1范数）
    
    Returns:
        x: 优化后的解向量 (n,)
    """
    H = np.array(H)
    y = np.array(y)
    m, n = H.shape
    
    lower, upper = bounds
    
    # 改进的初始化：使用最小二乘解投影到边界内
    try:
        # 尝试使用最小二乘解
        HTH = np.dot(H.T, H)
        HTy = np.dot(H.T, y)
        # 添加小的正则化项以避免奇异矩阵
        x_ls = np.linalg.solve(HTH + 1e-8 * np.eye(n), HTy)
        # 投影到边界内
        x = np.clip(x_ls, lower, upper)
    except:
        # 如果失败，使用边界中点初始化
        x = np.full(n, (lower + upper) / 2.0)
    
    # 构建一阶差分矩阵 D（用于平滑正则化）
    D = np.eye(n, k=0) - np.eye(n, k=1)
    D = D[:-1]  # (n-1) x n
    
    # 自适应学习率
    lr = learning_rate
    best_loss = float('inf')
    best_x = x.copy()
    patience = 1000  # 如果loss不再下降，提前停止
    no_improve_count = 0
    
    # 用于收敛检测
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        # 使用全部数据（不使用batch以提高精度）
        H_batch = H
        y_batch = y
        
        # 计算残差
        residual = np.dot(H_batch, x) - y_batch
        
        # 使用Huber损失平滑近似L1范数（更平滑，梯度更稳定）
        # Huber损失: L_delta(r) = 0.5*r^2/delta if |r| < delta, else |r| - 0.5*delta
        abs_residual = np.abs(residual)
        huber_grad = np.where(
            abs_residual < huber_delta,
            residual / huber_delta,  # 当|r| < delta时，梯度为 r/delta
            np.sign(residual)  # 当|r| >= delta时，梯度为 sign(r)
        )
        
        # 梯度：H_batch^T @ huber_grad
        grad_data = np.dot(H_batch.T, huber_grad)
        
        # 正则化项的梯度（也使用Huber平滑）
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
        
        # 梯度下降步骤
        x_new = x - lr * grad
        
        # 投影到边界约束：投影到 [lower, upper]
        x_new = np.clip(x_new, lower, upper)
        
        # 更新
        x = x_new
        
        # 计算完整损失（使用真实L1范数，而不是Huber损失）
        if iteration % 50 == 0 or iteration == max_iter - 1:
            residual_full = np.dot(H, x) - y
            loss = np.sum(np.abs(residual_full))
            if lam > 0:
                Dx_full = np.dot(D, x)
                loss += lam * np.sum(np.abs(Dx_full))
            
            # 记录最佳解
            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()
                no_improve_count = 0
            else:
                no_improve_count += 50
            
            # 检查收敛（使用相对变化）
            if prev_loss > 1e-10 and abs(prev_loss - loss) / prev_loss < tol:
                if verbose:
                    print(f"SGD converged at iteration {iteration}, loss={loss:.6f}")
                break
            
            # 如果长时间没有改进，提前停止
            if no_improve_count >= patience:
                if verbose:
                    print(f"SGD stopped early at iteration {iteration} (no improvement), best_loss={best_loss:.6f}")
                x = best_x
                break
            
            prev_loss = loss
            
            # 自适应学习率衰减（更保守的策略）
            if iteration > 0 and iteration % 200 == 0:
                lr = lr * 0.95  # 更慢的衰减
            
            if verbose and iteration % 200 == 0:
                print(f"Iteration {iteration}, loss={loss:.6f}, best_loss={best_loss:.6f}, lr={lr:.6f}")
    
    # 返回最佳解
    if best_loss < np.sum(np.abs(np.dot(H, x) - y)):
        x = best_x
    
    return x

