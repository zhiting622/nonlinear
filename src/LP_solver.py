import numpy as np
from scipy.optimize import linprog


def solve_constrained_ls(H, y, bounds=(0.9, 6.0), lam=0):
    """
    min ‖Hx – y‖₁ + λ‖D x‖₁
    s.t.  lower ≤ x ≤ upper
    """


    N = np.array(H).shape[1]

    # First-difference matrix D for smoothness
    D = np.eye(N, k=0) - np.eye(N, k=1)
    D = D[:-1]

    A1 = H
    A2 = np.sqrt(lam) * D
    A = np.vstack([A1, A2])
    b = np.hstack([y, np.zeros(D.shape[0])])

    m = A.shape[0]

    # Variables: x (N), u (m)
    # Minimize: sum(u)
    # Subject to:
    #   -u <= Ax - b <= u
    #   lower <= x <= upper

    # Decision vars: [x_0, ..., x_{N-1}, u_0, ..., u_{m-1}]
    c = np.hstack([np.zeros(N), np.ones(m)])

    # Inequality constraints for |Ax - b| <= u → Ax - b - u <= 0 and -Ax + b - u <= 0
    A_ineq = np.block([
        [A, -np.eye(m)],
        [-A, -np.eye(m)]
    ])
    b_ineq = np.hstack([b, -b])

    # Bounds
    x_bounds = [bounds] * N
    u_bounds = [(0, None)] * m
    all_bounds = x_bounds + u_bounds

    # Solve LP
    res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, bounds=all_bounds, method='highs')

    if not res.success:
        raise RuntimeError("L1 optimization failed: " + res.message)

    return res.x[:N]