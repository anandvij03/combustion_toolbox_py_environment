import numpy as np

def equilibriumCheckIons(self, N, A0, ind_E, indexGas, indexIons):
    """
    Check convergence of ionized species in the mixture
    
    Args:
        self (EquilibriumSolver): EquilibriumSolver object
        N (np.ndarray): Mixture composition [mol]
        A0 (np.ndarray): Stoichiometric matrix
        ind_E (int): Index of electron element
        indexGas (np.ndarray): List of chemical species indices in gaseous phase
        indexIons (np.ndarray): List of ionized chemical species indices 
        
    Returns:
        N (np.ndarray): Mixture composition [mol]
        STOP (float): Relative error in charge balance [-]
        FLAG_ION (bool): Flag indicating if ionized species are present in the mixture
    """
    STOP = 0.0

    # Check if there are ionized species
    if indexIons is None or len(indexIons) == 0:
        FLAG_ION = False
        return N, STOP, FLAG_ION

    FLAG_ION = True

    N = np.asarray(N)
    A0 = np.asarray(A0)
    indexGas = np.asarray(indexGas)
    indexIons = np.asarray(indexIons)

    # Get error in the electro-neutrality of the mixture
    delta_ions, _ = ionsFactor(N, A0, ind_E, indexGas, indexIons)

    # Reestimate composition of ionized species
    if abs(delta_ions) > self.tolMultiplierIons:
        N, STOP = recomputeIons(
            N, A0, ind_E, indexGas, indexIons, delta_ions,
            self.tolMoles, self.tolMultiplierIons, self.itMaxIons
        )

    return N, STOP, FLAG_ION


# SUB-PASS FUNCTIONS
def recomputeIons(N, A0, ind_E, indexGas, indexIons, delta_ions, TOL, TOL_pi, itMax):
    # Reestimate composition of ionized species
    A0_ions = A0[indexIons, ind_E]
    STOP = 1.0
    it = 0

    # Reestimate composition of ionized species
    while STOP > TOL_pi and it < itMax:
        it += 1
        # Apply correction
        N[indexIons] = N[indexIons] * np.exp(A0_ions * delta_ions)
        # Compute correction of the Lagrangian multiplier for ions divided by RT
        delta_ions, _ = ionsFactor(N, A0, ind_E, indexGas, indexIons)
        STOP = abs(delta_ions)

    if STOP > 0.1:
        N[indexIons] = 0.0

    Xi_ions = N[indexIons] / np.sum(N)

    # Set error to zero if molar fraction of ionized species are below tolerance
    if not np.any(Xi_ions > TOL):
        STOP = 0.0

    return N, STOP


def ionsFactor(N, A0, ind_E, indexGas, indexIons):
    # Compute relaxation factor for ionized species
    if indexIons is None or len(indexIons) == 0:
        delta = None
        deltaN3 = 0.0
        return delta, deltaN3

    A0_gas_E = A0[indexGas, ind_E]
    N_gas = N[indexGas]

    numerator = np.sum(A0_gas_E * N_gas)
    denominator = np.sum((A0_gas_E ** 2) * N_gas)

    if denominator == 0:
        delta = 0.0
    else:
        delta = -numerator / denominator

    deltaN3 = abs(np.sum(N_gas * A0_gas_E))
    return delta, deltaN3
