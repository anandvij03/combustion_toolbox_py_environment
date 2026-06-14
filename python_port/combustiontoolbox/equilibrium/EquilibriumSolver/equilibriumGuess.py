import numpy as np
from combustiontoolbox.utils.optimization.simplex import simplex
from combustiontoolbox.utils.optimization.simplexDual import simplexDual

def equilibriumGuess(self, N, NP, A0, muRT, b0, index, indexGas, indexIons, NG, molesGuess):
    """
    Initialize molar composition from a previous calculation or using the simplex algorithm
    
    Args:
        self (EquilibriumSolver): EquilibriumSolver object
        N (np.ndarray): Mixture composition [mol]
        NP (float): Total moles of gaseous species [mol]
        A0 (np.ndarray): Stoichiometric matrix
        muRT (np.ndarray): Dimensionless chemical potentials
        b0 (np.ndarray): Moles of each element
        index (np.ndarray): Index of chemical species
        indexGas (np.ndarray): Index of gaseous species
        indexIons (np.ndarray): Index of ionized species
        NG (int): Number of gaseous species
        molesGuess (np.ndarray): Mixture composition [mol] of a previous computation
        
    Returns:
        N (np.ndarray): Mixture composition [mol]
        NP (float): Total moles of gaseous species [mol]
    """
    N = np.asarray(N, dtype=float).copy()
    A0 = np.asarray(A0, dtype=float)
    muRT = np.asarray(muRT, dtype=float)
    b0 = np.asarray(b0, dtype=float)
    index = np.asarray(index, dtype=int)
    indexGas = np.asarray(indexGas, dtype=int)
    indexIons = np.asarray(indexIons, dtype=int)

    # Get molar composition from a previous calculation
    if molesGuess is not None and len(molesGuess) > 0:
        molesGuess = np.asarray(molesGuess, dtype=float)
        N[indexGas] = molesGuess[indexGas]
        NP = np.sum(molesGuess[indexGas])
        return N, NP

    # Get molar composition using the simplex method
    N = getSimplex(N, A0, muRT, b0, index, indexIons, NG)

    # Recompute mol gaseous species
    NP = np.sum(N[indexGas])
    return N, NP


# SUB-PASS FUNCTIONS
def getSimplex(N, A0, muRT, b0, index, indexIons, NG):
    # Get molar composition using the simplex algorithm
    alpha = 0.01
    FLAG_MINOR = True

    # Initialization
    Nminor = np.zeros_like(N)

    # Get major species
    Nmajor = simplex(A0, b0, muRT)

    # Remove ionized species from Nmajor
    local_index_ions = np.where(np.isin(index, indexIons))[0]
    Nmajor[local_index_ions] = 0.0

    # Get minor species
    if FLAG_MINOR:
        tol = 1e-4
        FLAG_MAXMIN = Nmajor > tol

       # Gaseous columns are 0:NG, condensed columns are NG:NS
        if np.any(FLAG_MAXMIN[NG:]):
            indexPass = np.arange(len(index))
        else:
            cond_indices = NG + np.where(FLAG_MAXMIN[NG:])[0]
            indexPass = np.concatenate((np.arange(NG), cond_indices))

        indexPass = indexPass.astype(int)

        Nminor_indexPass, Nmin = simplexDual(A0[:, indexPass], b0)
        Nminor[index[indexPass]] = Nminor_indexPass
        Nmin = Nmin + 1e-10

    # Merge solutions
    N[index] = (1.0 - alpha) * Nmajor + alpha * Nminor[index]
    
    zero_mask = N[index] == 0.0
    N[index[zero_mask]] = alpha * Nmin

    return N
