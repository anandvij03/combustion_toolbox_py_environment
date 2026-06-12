import numpy as np
import scipy.linalg

def simplex(A, b, c):
    """
    Use revised simplex method to solve the linear programming problem:
        * min c' * x,
        * A * x = b,
        *    x >= 0
    """
    tol = 1e-4

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()
    c = np.asarray(c, dtype=float).flatten()
    m, n = A.shape

    assert len(b) == m, "simplex:DimensionMismatch: b length must match rows of A."
    assert len(c) == n, "simplex:DimensionMismatch: c length must match columns of A."

    itMax = max(100, 20 * (m + n))

    # Remove zero rows and normalize rows so b is nonnegative
    rowNorm = np.max(np.abs(A), axis=1)
    FLAG_ZERO_ROW = rowNorm <= tol

    if np.any(np.abs(b[FLAG_ZERO_ROW]) > tol):
        raise ValueError("simplex:Infeasible: Zero constraint row has nonzero right-hand side.")

    A = A[~FLAG_ZERO_ROW, :]
    b = b[~FLAG_ZERO_ROW]
    rowNorm = rowNorm[~FLAG_ZERO_ROW]

    FLAG_NEGATIVE_B = b < 0
    A[FLAG_NEGATIVE_B, :] = -A[FLAG_NEGATIVE_B, :]
    b[FLAG_NEGATIVE_B] = -b[FLAG_NEGATIVE_B]

    rowScale = np.maximum(rowNorm, 1.0)
    A = A / rowScale[:, np.newaxis]
    b = b / rowScale

    # Remove linearly dependent rows
    A, b = independentRows(A, b, tol)
    m, n = A.shape

    if m == 0:
        return np.zeros(n)

    # Phase I: add artificial variables to obtain an explicit feasible basis
    A_phase = np.hstack([A, np.eye(m)])
    c_phase = np.concatenate([np.zeros(n), np.ones(m)])
    basis = n + np.arange(m)

    x_phase, basis = solvePhase(A_phase, b, c_phase, basis, tol, itMax)
    phase1Objective = c_phase @ x_phase

    if phase1Objective > max(1.0, np.max(np.abs(b))) * tol:
        raise ValueError("simplex:Infeasible: Phase I could not find a feasible solution.")

    basis = removeArtificialBasis(A_phase, A, basis, n, tol)

    # Phase II: optimize original objective from the feasible basis
    x, _ = solvePhase(A, b, c, basis, tol, itMax)

    x[np.abs(x) <= tol] = 0.0
    return x

def independentRows(A, b, tol):
    m, n = A.shape
    if m == 0:
        return A, b

    # Estimate row rank from column pivoting of A'
    Q, R, P = scipy.linalg.qr(A.T, mode='economic', pivoting=True)
    diagR = np.abs(np.diag(R))

    eps = np.finfo(float).eps
    if len(diagR) > 0:
        rankA = np.sum(diagR > max(A.shape) * eps * np.max(diagR))
    else:
        rankA = 0

    if rankA == m:
        return A, b

    if rankA == 0:
        if np.max(np.abs(b)) > tol:
            raise ValueError("simplex:Infeasible: Rank-deficient constraints are inconsistent.")
        return np.zeros((0, n)), np.zeros(0)

    indexKeep = np.sort(P[:rankA])
    A_keep = A[indexKeep, :]
    b_keep = b[indexKeep]

    A_pinv = np.linalg.pinv(A_keep)
    if np.max(np.abs(A @ (A_pinv @ b_keep) - b)) > max(1.0, np.max(np.abs(b))) * tol:
        raise ValueError("simplex:Infeasible: Rank-deficient constraints are inconsistent.")

    return A_keep, b_keep

def removeArtificialBasis(A_phase, A, basis, numOriginalVariables, tol):
    numRows = len(basis)
    for row in range(numRows):
        if basis[row] < numOriginalVariables:
            continue
        
        B = A_phase[:, basis]
        
        flag_nonbasic_original = np.ones(numOriginalVariables, dtype=bool)
        for b_var in basis:
            if b_var < numOriginalVariables:
                flag_nonbasic_original[b_var] = False
        
        candidates = np.where(flag_nonbasic_original)[0]
        pivotColumn = None
        
        for candidate in candidates:
            direction = np.linalg.solve(B, A[:, candidate])
            if abs(direction[row]) > tol:
                pivotColumn = candidate;
                break
                
        if pivotColumn is None:
            raise ValueError("simplex:DegenerateBasis: Could not remove artificial variable from basis.")
            
        basis[row] = pivotColumn
        
    return basis

def solvePhase(A, b, c, basis, tol, itMax):
    m, n = A.shape
    basis = np.asarray(basis, dtype=int).copy()
    FLAG_BASIC = np.zeros(n, dtype=bool)
    FLAG_BASIC[basis] = True

    Binv = basisInverse(A, basis)
    xBasis = Binv @ b

    it = 0
    refactorEvery = 50

    while True:
        it += 1
        if it > itMax:
            raise ValueError("simplex:IterationLimit: Simplex iteration limit exceeded.")

        if it % refactorEvery == 0:
            Binv = basisInverse(A, basis)
            xBasis = Binv @ b

        xBasis[np.abs(xBasis) <= tol] = 0.0

        if np.any(xBasis < -tol):
            raise ValueError("simplex:InvalidBasis: Current basis is primal infeasible.")

        lambda_val = Binv.T @ c[basis]

        nonBasis = np.where(~FLAG_BASIC)[0]
        reducedCosts = c[nonBasis] - A[:, nonBasis].T @ lambda_val

        if np.all(reducedCosts >= -tol):
            x = np.zeros(n)
            x[basis] = np.maximum(xBasis, 0.0)
            return x, basis

        entering_idx = np.where(reducedCosts < -tol)[0][0]
        entering = nonBasis[entering_idx]

        direction = Binv @ A[:, entering]
        FLAG_POSITIVE = direction > tol

        if not np.any(FLAG_POSITIVE):
            raise ValueError("simplex:Unbounded: Linear program is unbounded.")

        ratios = np.full(m, np.inf)
        ratios[FLAG_POSITIVE] = xBasis[FLAG_POSITIVE] / direction[FLAG_POSITIVE]
        minRatio = np.min(ratios)

        leavingCandidates = np.where(np.abs(ratios - minRatio) <= tol)[0]
        indexTie = np.argmin(basis[leavingCandidates])
        leavingRow = leavingCandidates[indexTie]
        leaving = basis[leavingRow]
        pivot = direction[leavingRow]

        if abs(pivot) <= tol:
            raise ValueError("simplex:NumericalPivot: Pivot is too small.")

        theta = xBasis[leavingRow] / pivot
        xBasis = xBasis - direction * theta
        xBasis[leavingRow] = theta

        pivotRow = Binv[leavingRow, :] / pivot
        Binv = Binv - np.outer(direction, pivotRow)
        Binv[leavingRow, :] = pivotRow

        basis[leavingRow] = entering
        FLAG_BASIC[leaving] = False
        FLAG_BASIC[entering] = True

def basisInverse(A, basis):
    m = len(basis)
    return np.linalg.solve(A[:, basis], np.eye(m))
