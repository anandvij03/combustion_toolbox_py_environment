import numpy as np

def equilibriumCheckCondensed(A0, pi_i, W, indexCondensed, muRT, NC_max, FLAG_ONE, FLAG_RULE):
    """
    Check condensed species that may appear at chemical equilibrium
    
    Args:
        A0 (float): Stoichiometric matrix
        pi_i (float): Dimensionless Lagrange multiplier vector
        W (float): Molecular mass [kg/mol] vector
        indexCondensed (float): Index condensed species to be considered
        muRT (float): List of chemical species indices in gaseous phase
        NC_max (float): Maximum number of condensed species (Gibbs phase rule)
        FLAG_ONE (bool): Flag indicating to include condensed species in the system one by one
        FLAG_RULE (bool): Flag indicating to include condensed species in the system up to the maximum number of condensed species that satisfy the Gibbs phase rule
        
    Returns:
        indexCondensed (np.ndarray): Index condensed species that may appear at chemical equilibrium
        FLAG_CONDENSED (bool): Flag indicating additional condensed species that may appear at chemical equilibrium
        dL_dnj (np.ndarray): Vapor pressure test vector of the species that may appear at chemical equilibrium
    """
    FLAG_CONDENSED = False

    # Checks
    if indexCondensed is None or len(indexCondensed) == 0:
        dL_dnj = np.array([])
        return np.array([], dtype=int), FLAG_CONDENSED, dL_dnj

    A0 = np.asarray(A0)
    pi_i = np.asarray(pi_i)
    W = np.asarray(W)
    indexCondensed = np.asarray(indexCondensed)
    muRT = np.asarray(muRT)

    # Get length condensed species
    NC = len(indexCondensed)
    dL_dnj = np.zeros(NC)

    for i in range(NC - 1, -1, -1):
        idx = indexCondensed[i]
        # Only check if there were atoms of the species in the initial mixture
        if np.sum(A0[idx, :]) == 0:
            continue

        # Calculate dLdnj of the condensed species
        dL_dnj[i] = (muRT[idx] - np.dot(pi_i, A0[idx, :])) / W[i]

    # Get condensed species that may appear at chemical equilibrium
    FLAG = dL_dnj < 0

    # Check if any condensed species have to be considered
    if not np.any(FLAG):
        indexCondensed = np.array([], dtype=int)
        return indexCondensed, FLAG_CONDENSED, dL_dnj

    # Get index of the condensed species to be added to the system
    indexCondensed = indexCondensed[FLAG]
    dL_dnj = dL_dnj[FLAG]

    # Testing
    if FLAG_RULE:
        temp = np.argsort(dL_dnj)
        limit = min(int(NC_max), len(temp))
        indexCondensed = indexCondensed[temp[:limit]]
    elif FLAG_ONE:
        temp = np.argmin(dL_dnj)
        indexCondensed = np.array([indexCondensed[temp]])

    # Update flag
    FLAG_CONDENSED = True

    return indexCondensed, FLAG_CONDENSED, dL_dnj
