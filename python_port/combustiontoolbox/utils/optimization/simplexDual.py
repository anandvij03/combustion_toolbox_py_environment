import numpy as np

def simplexDual(A, b):
    """
    Use simplex method to solve the linear programming problem:
        * max(min x) -> max t,
        * A * x = b,
        *    x >= 0
    """
    A_original = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    m, n = A_original.shape

    # Compact max-min formulation with x = z + t, z >= 0, t >= 0
    sum_col = np.sum(A_original, axis=1, keepdims=True)
    A_eq = np.hstack([A_original, sum_col])
    
    c = np.concatenate([np.zeros(n), [-1.0]])

    # Solve equality-form linear program
    from combustiontoolbox.utils.optimization.simplex import simplex
    zt = simplex(A_eq, b, c)

    # Recover solution
    x_min = max(zt[-1], 0.0)
    x = zt[0:n] + x_min
    return x, x_min
