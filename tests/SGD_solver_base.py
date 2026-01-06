# -*- coding: utf-8 -*-
import numpy as np


def solve_constrained_ls(H, y, bounds=(0.9, 6.0), lam=0, learning_rate=0.01, 
                         max_iter=1000, batch_size=None, tol=1e-6, verbose=False):
    """
    使用随机梯度下降（SGD）求解约束最小化问题：
    min ‖Hx – y‖₁ + λ‖D x‖₁
    s.t.  lower ≤ x ≤ upper
    
    使用次梯度方法和投影来处理L1范数和边界约束。
    
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
    
    Returns:
        x: 优化后的解向量 (n,)
    """
    H = np.array(H)
    y = np.array(y)
    m, n = H.shape
    
    lower, upper = bounds
    
    # 初始化：在边界内随机初始化
    np.random.seed(42)  # 为了可重复性
    x = np.random.uniform(lower, upper, n)
    
    # 构建一阶差分矩阵 D（用于平滑正则化）
    D = np.eye(n, k=0) - np.eye(n, k=1)
    D = D[:-1]  # (n-1) x n
    
    # 自适应学习率
    lr = learning_rate
    
    # 用于收敛检测
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        # 选择批次索引
        if batch_size is None or batch_size >= m:
            # 使用全部数据
            batch_idx = np.arange(m)
        else:
            # 随机选择批次
            batch_idx = np.random.choice(m, size=batch_size, replace=False)
        
        H_batch = H[batch_idx, :]
        y_batch = y[batch_idx]
        
        # 计算残差
        residual = np.dot(H_batch, x) - y_batch
        
        # L1范数的次梯度：sign(residual) @ H_batch
        # 对于接近0的值，使用平滑近似
        epsilon = 1e-8
        sign_residual = np.sign(residual)
        # 对于非常接近0的值，使用平滑近似避免次梯度不唯一
        sign_residual[np.abs(residual) < epsilon] = residual[np.abs(residual) < epsilon] / epsilon
        
        # 梯度：H_batch^T @ sign_residual
        grad_data = np.dot(H_batch.T, sign_residual)
        
        # 如果使用批次，需要缩放梯度
        if batch_size is not None and batch_size < m:
            grad_data = grad_data * (m / batch_size)
        
        # 正则化项的次梯度
        if lam > 0:
            Dx = np.dot(D, x)
            sign_Dx = np.sign(Dx)
            sign_Dx[np.abs(Dx) < epsilon] = Dx[np.abs(Dx) < epsilon] / epsilon
            grad_reg = lam * np.dot(D.T, sign_Dx)
            grad = grad_data + grad_reg
        else:
            grad = grad_data
        
        # 梯度下降步骤
        x_new = x - lr * grad
        
        # 投影到边界约束：投影到 [lower, upper]
        x_new = np.clip(x_new, lower, upper)
        
        # 更新
        x = x_new
        
        # 计算完整损失（用于收敛检测）
        if iteration % 100 == 0 or iteration == max_iter - 1:
            residual_full = np.dot(H, x) - y
            loss = np.sum(np.abs(residual_full))
            if lam > 0:
                Dx_full = np.dot(D, x)
                loss += lam * np.sum(np.abs(Dx_full))
            
            # 检查收敛
            if abs(prev_loss - loss) < tol:
                if verbose:
                    print(f"SGD converged at iteration {iteration}, loss={loss:.6f}")
                break
            
            prev_loss = loss
            
            # 自适应学习率衰减
            if iteration > 0 and iteration % 500 == 0:
                lr = lr * 0.9
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, loss={loss:.6f}, lr={lr:.6f}")
    
    return x

