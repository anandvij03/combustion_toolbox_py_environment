import numpy as np

def equilibriumDerivatives(J, N, A0, NE, indexGas, indexCondensed, H0RT):
    """
    Obtain thermodynamic derivative of the moles of the species and of the moles of the mixture
    respect to temperature and pressure from a given composition [moles] at equilibrium
    
    Args:
        J (np.ndarray): Matrix J to solve the linear system J*x = b
        N (np.ndarray): Mixture composition [mol]
        A0 (np.ndarray): Stoichiometric matrix
        NE (int): Temporal total number of elements
        indexGas (np.ndarray): Temporal index of gaseous species in the final mixture
        indexCondensed (np.ndarray): Temporal index of condensed species in the final mixture
        H0RT (np.ndarray): Dimensionless enthalpy
        
    Returns:
        dNi_T (np.ndarray): Thermodynamic derivative of the moles of the species respect to temperature
        dN_T (float):  Thermodynamic derivative of the moles of the mixture respect to temperature
        dNi_p (np.ndarray): Thermodynamic derivative of the moles of the species respect to pressure
        dN_p (float):  Thermodynamic derivative of the moles of the mixture respect to pressure
    """
    J = np.asarray(J)
    N = np.asarray(N)
    A0 = np.asarray(A0)
    indexGas = np.asarray(indexGas)
    indexCondensed = np.asarray(indexCondensed)
    H0RT = np.asarray(H0RT)

    # Equilibrium derivative respect temperature
    dNi_T, dN_T = equilibrium_dT(J, N, A0, NE, indexGas, indexCondensed, H0RT)

    # Equilibrium derivative respect pressure
    dNi_p, dN_p = equilibrium_dp(J, N, A0, NE, indexGas, indexCondensed)

    return dNi_T, dN_T, dNi_p, dN_p


def equilibrium_dT(J, N, A0, NE, indexGas, indexCondensed, H0RT):
    # Initialization
    dNi_T = np.zeros(len(N))

    # Construction of vector b
    b = update_vector_b_dT(A0, N, indexGas, indexCondensed, H0RT)

    # Solve of the linear system J*x = b
    try:
        import scipy.linalg
        x = scipy.linalg.solve(J, b, assume_a='sym')
    except (ImportError, AttributeError):
        x = np.linalg.solve(J, b)

    # Extract solution
    dpii_T = x[0:NE]
    dNi_T[indexCondensed] = x[NE:-1]
    dN_T = x[-1]

    # Compute remainder dNi_T (gas)
    dNi_T[indexGas] = H0RT[indexGas] + A0[indexGas, :] @ dpii_T + dN_T

    return dNi_T, dN_T


def update_vector_b_dT(A0, N, indexGas, indexCondensed, H0RT):
    N_gas = N[indexGas]
    H0RT_gas = H0RT[indexGas]
    A0_gas = A0[indexGas, :]
    
    factor = (N_gas * H0RT_gas)[:, np.newaxis]
    sum_part = np.sum(A0_gas * factor, axis=0)
    H0RT_cond = H0RT[indexCondensed]
    sum_gas = np.array([np.sum(N_gas * H0RT_gas)])
    
    b = np.concatenate((sum_part, H0RT_cond, sum_gas))
    return -b


def equilibrium_dp(J, N, A0, NE, indexGas, indexCondensed):
    # Initialization
    dNi_p = np.zeros(len(N))

    # Construction of vector b
    b = update_vector_b_dp(J, N, indexGas)

    # Solve of the linear system J*x = b
    try:
        import scipy.linalg
        x = scipy.linalg.solve(J, b, assume_a='sym')
    except (ImportError, AttributeError):
        x = np.linalg.solve(J, b)

    # Extract solution
    dpii_p = x[0:NE]
    dNi_p[indexCondensed] = x[NE:-1]
    dN_p = x[-1]
    
    # Compute remainder dNi_T (gas)
    dNi_p[indexGas] = -1.0 + A0[indexGas, :] @ dpii_p + dN_p

    return dNi_p, dN_p


def update_vector_b_dp(J, N, ind_nswt):
    b = J[:, -1].copy()
    b[-1] = np.sum(N[ind_nswt])
    return b
