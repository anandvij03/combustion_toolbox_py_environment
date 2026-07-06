import numpy as np
import warnings
from combustiontoolbox.common.Constants import Constants
from combustiontoolbox.core.EquationStateJCZ3.EquationStateJCZ3 import EquationStateJCZ3

def equilibriumHelmholtz(self, system, productSpeciesSet, v, T, mix, molesGuess):
    """
    Obtain equilibrium composition [moles] for the given temperature [K] and volume [m3].
    """
    # Constants
    R0 = Constants.R0

    # Definitions
    N = np.zeros(productSpeciesSet.numSpecies)  # Composition vector [moles_i]
    A0 = np.array(productSpeciesSet.stoichiometricMatrix, dtype=float) # Stoichiometric matrix [a_ij]
    RT = R0 * T                                 # [J/mol]
    tau0 = self.tolTau                           # Tolerance of the slack variables for condensed species
    
    # Initialization
    NatomE = np.asarray(mix.natomElementsReact, dtype=float)
    max_NatomE = np.max(NatomE) if len(NatomE) > 0 else 1.0
    NP = 0.1
    SIZE = -np.log(self.tolMoles)
    FLAG_CONDENSED = False
    STOP_ions = 0.0
    STOP = 1.0

    # Set moles from molesGuess (if it was given) to 1e-6 to avoid singular matrix
    if molesGuess is not None and len(molesGuess) > 0:
        molesGuess = np.asarray(molesGuess, dtype=float).copy()
        molesGuess[molesGuess < self.tolMolesGuess] = self.tolMolesGuess
    else:
        molesGuess = np.array([])
    
    # Find indices of the species/elements that we have to remove from the stoichiometric matrix A0
    # for the sum of elements whose value is <= tolMoles
    A0, indexRemoveSpecies, ind_E, NatomE = self.removeElements(NatomE, A0, system.ind_E, self.tolMoles)
    
    # Check if element E (electron) is present
    FLAG_E = ind_E is not None and ind_E != "" and ind_E != [] and not (isinstance(ind_E, (int, np.integer)) and ind_E < 0)

    # List of indices with nonzero values
    index, indexGas, indexCondensed, indexIons, indexElements, NE, NG, NS = self.tempValues(productSpeciesSet, NatomE)
    
    # Remove elements with zero atoms from the stoichiometric matrix A0
    A0 = A0[:, indexElements]
    A0_T = A0.T

    # Update temp values
    if indexRemoveSpecies is not None and len(indexRemoveSpecies) > 0:
        index, indexCondensed, indexGas, indexIons, NG, NS, _ = self.updateTemp(N, indexRemoveSpecies, indexCondensed, indexGas, indexIons, NP, NG, NS, SIZE)

    # Remove ionized species below the configured temperature threshold
    if FLAG_E and indexIons is not None and len(indexIons) > 0 and T < self.temperatureIons:
        removeIons()

    # Remove condensed species with temperature out of bounds
    indexCondensed, _ = self.filterSpeciesTemperatureRange(productSpeciesSet, T, indexCondensed, NS - NG, False)

    # Remove gas species with temperature out of bounds
    indexGas, NG = self.filterSpeciesTemperatureRange(productSpeciesSet, T, indexGas, NG, self.FLAG_EXTRAPOLATE)

    # First, compute chemical equilibrium with only gaseous species
    indexGas_0 = indexGas.copy()
    indexCondensed_0 = indexCondensed.copy()
    index0 = np.concatenate((indexGas_0, indexCondensed_0))
    indexCondensed = np.array([], dtype=int)
    index = np.concatenate((indexGas, indexCondensed))
    NS = len(index)
    
    # Initialize vectors g0 (molar Gibbs energy) and h0 (molar enthalpy) with zeros
    g0 = np.zeros(productSpeciesSet.numSpecies)
    h0 = np.zeros(productSpeciesSet.numSpecies)
    
    # Molar Gibbs energy [J/mol]
    index0Global = np.asarray(productSpeciesSet.indexGlobal)[np.concatenate((indexGas_0, indexCondensed_0))]
    local_indices = np.concatenate((indexGas_0, indexCondensed_0))
    g0[local_indices] = system.evaluateSpeciesThermoG(T, index0Global).flatten()
    
    # Dimensionless Gibbs energy
    g0RT = g0 / RT

    # Dimensionless chemical potential
    muRT = g0RT.copy()
    
    # Initialize composition matrix N [mol, FLAG_CONDENSED]
    N, NP = self.equilibriumGuess(N, NP, A0_T[:, index0], muRT[index0], NatomE, index0, indexGas_0, indexIons, NG, molesGuess)

    # Initialization 
    psi_j = np.zeros(productSpeciesSet.numSpecies)
    tau = tau0 * np.min(NatomE) if len(NatomE) > 0 else 0.0

    # NESTED FUNCTION: removeIons
    def removeIons():
        nonlocal N, indexGas, indexIons, A0, A0_T, indexElements, NatomE, NE, FLAG_E, index, NG, NS
        if indexIons is not None and len(indexIons) > 0:
            N[indexIons] = 0.0
            indexGas = indexGas[~np.isin(indexGas, indexIons)]
            indexIons = np.array([], dtype=int)

        if FLAG_E:
            A0 = np.delete(A0, ind_E, axis=1)
            A0_T = np.delete(A0_T, ind_E, axis=0)
            indexElements = np.delete(indexElements, ind_E)
            NatomE = np.delete(NatomE, ind_E)
            NE = NE - 1
            FLAG_E = False

        index = np.concatenate((indexGas, indexCondensed))
        NG = len(indexGas)
        NS = len(index)

    # NESTED FUNCTION: equilibriumLoop
    def equilibriumLoop():
        nonlocal indexGas, indexCondensed, indexIons, index, NG, NS, N, NP, psi_j, STOP_ions, FLAG_E, NE, A0, A0_T, indexElements, NatomE, STOP
        
        it = 0
        counter_errors = 0
        itMax = self.itMaxGibbs
        STOP = 1.0

        while STOP > self.tolGibbs and it < itMax:
            it += 1
            # Chemical potential
            # Code for adding the non-ideal chemical potential as well, is given here.
            # 1. Ideal gas mu (baseline)
            muRT[indexGas] = g0RT[indexGas] + np.log(N[indexGas] * RT / v * 1e-5)
            
            # 2. Non-Ideal (Excess) contribution via toggle. This can be switched off for convenience.
            FLAG_JCZ3 = True
            if FLAG_JCZ3 == True:
                # To be Removed: This has to be replaced with instantiation in the example file itself.
                # Get the exact names of the active products in the correct order
                jcz3_names = [system.listSpecies[i] for i in productSpeciesSet.indexGlobal]
                jcz3_model = EquationStateJCZ3(species_names=jcz3_names, index_condensed=indexCondensed)
                # Excess Chemical Potential (mu_excess)
                mu_excess = jcz3_model.getDepartureFunctions(N,v,T)
                muRT[indexGas] += mu_excess[indexGas]/ RT

            # Compute total number of moles
            NP = np.sum(N[indexGas])
            
            # Construction of matrix J
            J = update_matrix_J(A0_T, N, indexGas, indexCondensed, psi_j)
            
            # Construction of vector b      
            b = update_vector_b(A0, N, NatomE, ind_E, index, indexGas, indexCondensed, indexIons, muRT, tau)
            
            # Solve the linear system J*x = b
            try:
                import scipy.linalg
                try:
                    x = scipy.linalg.solve(J, b, assume_a='sym')
                except np.linalg.LinAlgError:
                    x = np.nan * np.ones(b.shape)
            except (ImportError, AttributeError):
                try:
                    x = np.linalg.solve(J, b)
                except np.linalg.LinAlgError:
                    x = np.nan * np.ones(b.shape)

            
            # Check singular matrix
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                if FLAG_E:
                    norm_row = np.sum(np.abs(J[ind_E, :]))
                    norm_col = np.sum(np.abs(J[:, ind_E]))
                    val_b = abs(b[ind_E])
                    if max(norm_row, norm_col, val_b) < self.tolE:
                        removeIons()
                        continue

                # Update temp indices
                indexGas = indexGas_0.copy()
                indexCondensed = indexCondensed_0.copy()
                index = np.concatenate((indexGas, indexCondensed))

                NG = len(indexGas)
                NS = len(index)

                # Reset removed species to tolMolesGuess
                N[index[N[index] < self.tolMoles]] = self.tolMolesGuess
                psi_j[indexCondensed] = self.slackGuess

                if counter_errors > 2:
                    return np.nan * np.ones(NE + NS - NG)

                counter_errors += 1
                continue
            
            # Extract solution
            pi_i = x[0:NE]
            Delta_nj = x[NE:]
            
            # Compute correction moles of gases
            Delta_ln_nj = update_Delta_ln_nj(A0, pi_i, muRT, indexGas)
            
            # Calculate correction factor
            deltaGas = self.relaxFactorGas(NP, N[indexGas], Delta_ln_nj, 0.0)
            
            # Apply correction
            N[indexGas] = N[indexGas] * np.exp(deltaGas * Delta_ln_nj)

            # Calculate and apply correction condensed species
            N, psi_j, _ = self.relaxFactorCondensed(NP, N, psi_j, Delta_nj, indexCondensed, NG, NS, SIZE, tau, RT)
            
            # Compute STOP criteria
            STOP = compute_STOP(N[index], np.concatenate((Delta_ln_nj, Delta_nj)), NG, A0[index, :], NatomE, max_NatomE, self.tolE)

            # Update temp values in order to remove species with moles < tolerance
            index, indexCondensed, indexGas, indexIons, NG, NS, N = self.updateTemp(N, index, indexCondensed, indexGas, indexIons, NP, NG, NS, SIZE)

        # Check convergence of charge balance (ionized species)
        N, STOP_ions, FLAG_ION = self.equilibriumCheckIons(N, A0, ind_E, indexGas, indexIons)
        
        # Additional checks in case there are ions in the mixture
        if not FLAG_ION:
            return x
        
        # Check that there is at least one species with n_i > tolerance 
        if np.any(N[indexIons] > self.tolMoles):
            return x
        
        # Remove ionized species that do not satisfy n_i > tolerance
        index, indexCondensed, indexGas, indexIons, NG, NS, N = self.updateTemp(N, index, indexCondensed, indexGas, indexIons, NP, NG, NS, SIZE)
        
        # If none of the ionized species satisfy n_i > tolerance, remove
        # electron "element" from the stoichiometric matrix
        if indexIons is not None and len(indexIons) > 0:
            return x
        
        # Remove ionized species and element E from matrix
        removeIons()
        
        # Recompute chemical equilibrium without ions
        return equilibriumLoop()

    # NESTED FUNCTION: equilibriumLoopCondensed
    def equilibriumLoopCondensed(x):
        nonlocal indexGas, indexCondensed, indexIons, index, NG, NS, N, psi_j, indexGas_0
        
        if indexCondensed_0 is None or len(indexCondensed_0) == 0:
            return x

        # Update list possible gaseous species (in case singular matrix)
        indexGas_0 = indexGas.copy()
        
        # Set list with indices of the condensed species to be checked
        indexCondensed_check = indexCondensed_0.copy()

        # Get molecular weight species [kg/mol]
        W = productSpeciesSet.molecularWeight.flatten()

        # Definitions
        NC_max = NE - 1
        FLAG_ALL = False  # Include all the condensed species at once
        FLAG_ONE = False  # Include only the condensed species that satisfies the vapour pressure test
        FLAG_RULE = False # Include only up to NC_max condensed species that satisfies the vapour pressure test
        
        # Initialization
        it = 0
        itMaxRecursion = self.itMaxRecursion
        while indexCondensed_check is not None and len(indexCondensed_check) > 0:
            # Update iteration
            it += 1

            # Check Gibbs phase rule
            if len(indexCondensed) > NC_max:
                break

            # Check condensed species
            indexCondensed_add, FLAG_CONDENSED, _ = self.equilibriumCheckCondensed(
                A0, x[0:NE], W[indexCondensed_check], indexCondensed_check, muRT, NC_max, FLAG_ONE, FLAG_RULE
            )
            
            if not FLAG_CONDENSED:
                break

            # Update indices
            if FLAG_ALL:
                indexCondensed = indexCondensed_check.copy()
                indexCondensed_check = np.array([], dtype=int)
            else:
                if FLAG_ONE:
                    indexCondensed_check = indexCondensed_check[indexCondensed_check != indexCondensed_add]
                else:
                    indexCondensed_check = indexCondensed_check[~np.isin(indexCondensed_check, indexCondensed_add)]
                
                indexCondensed = np.unique(np.concatenate((indexCondensed, indexCondensed_add)))

            index = np.concatenate((indexGas, indexCondensed))

            # Update length
            NS = len(index)

            # Save backup
            N_backup = N.copy()
            
            # Check if there are non initialized condensed species
            N[indexCondensed_add[N[indexCondensed_add] == 0]] = self.tolMolesGuess

            # Initialize Lagrange multiplier vector psi
            psi_j[indexCondensed_add] = self.slackGuess

            # Compute chemical equilibrium considering condensed species
            x0 = equilibriumLoop()

            # Update solution vector
            if not np.any(np.isnan(x0)):
                x = x0
                indexGas_0 = indexGas.copy()
                continue

            # Singular matrix: remove last added condensed species
            indexGas = indexGas_0.copy()
            N = N_backup.copy()
            N[indexCondensed[:-1]] = 1.0
            N[indexCondensed[-1]] = -1.0
            _, indexCondensed, indexGas, indexIons, NG, NS, _ = self.updateTemp(
                N, index, indexCondensed, indexGas, indexIons, NP, NG, NS, SIZE
            )
            N[indexCondensed] = 0.0
            indexCondensed_check = indexCondensed.copy()

        # Check recursion limit
        if it > itMaxRecursion:
            warnings.warn(f"equilibriumLoopCondensed: Recursion limit {it}")
            return x

        # Check if there were species not considered
        _, FLAG_CONDENSED, dL_dnj = self.equilibriumCheckCondensed(
            A0, x[0:NE], W[indexCondensed_0], indexCondensed_0, muRT, NC_max, FLAG_ONE, FLAG_RULE
        )
        
        # Recompute if there are condensed species that may appear at chemical equilibrium
        if FLAG_CONDENSED and np.any(np.abs(dL_dnj) > 1e-4):
            x = equilibriumLoopCondensed(x)
            
        return x

    # Solve system
    x = equilibriumLoop()

    # Compute chemical equilibrium with condensed species
    x = equilibriumLoopCondensed(x)

    # Remove guesses for condensed species that do not satisfy the vapor pressure test
    mask = np.isin(indexCondensed_0, indexCondensed, invert=True)
    N[indexCondensed_0[mask]] = 0.0

    # Update matrix J (jacobian) to compute the thermodynamic derivatives
    J = update_matrix_J(A0_T, N, indexGas, indexCondensed, psi_j)
    
    J12_2 = np.concatenate((np.sum(A0_T[:, indexGas] * N[indexGas][np.newaxis, :], axis=1), np.zeros(NS - NG)))
    J = np.vstack((np.hstack((J, J12_2[:, np.newaxis])), np.append(J12_2, 0.0)[np.newaxis, :]))

    # Molar enthalpy [J/mol]
    h0[index] = system.evaluateSpeciesThermoH(T, np.asarray(productSpeciesSet.indexGlobal)[index]).flatten()
    
    # Dimensionless enthalpy
    H0RT = h0 / RT

    # Compute thermodynamic derivatives
    dNi_T, dN_T, dNi_p, dN_p = self.equilibriumDerivatives(J, N, A0, NE, indexGas, indexCondensed, H0RT)

    return N, dNi_T, dN_T, dNi_p, dN_p, index, STOP, STOP_ions, h0


# SUB-PASS FUNCTIONS
def compute_STOP(N_index, deltaN, NG, A0_index, NatomE, max_NatomE, tolE):
    # Compute stop criteria
    NPi = np.sum(N_index)
    if NPi == 0:
        return 0.0
    deltaN1 = N_index * np.abs(deltaN) / NPi
    deltaN1[NG:] = np.abs(deltaN[NG:]) / NPi
    term_sum = np.sum(N_index[:, np.newaxis] * A0_index, axis=0)
    deltab = np.abs(NatomE - term_sum)
    if max_NatomE > 0:
        valid_elements = deltab[NatomE > max_NatomE * tolE]
    else:
        valid_elements = deltab[NatomE > tolE]
    deltab_max = np.max(valid_elements) if len(valid_elements) > 0 else 0.0
    return max(np.max(deltaN1), deltab_max)


def update_matrix_J11(A0_T, N, indexGas):
    # Compute submatrix J11
    A0_T_gas = A0_T[:, indexGas]
    N_gas = N[indexGas]
    temp = A0_T_gas * N_gas[np.newaxis, :]
    J11 = A0_T_gas @ temp.T
    J11 = (J11 + J11.T) / 2.0
    return J11


def update_matrix_J12(A0_T, indexCondensed):
    # Compute submatrix J12
    return A0_T[:, indexCondensed]


def update_matrix_J(A0_T, N, indexGas, indexCondensed, psi_j):
    # Compute matrix J
    J11 = update_matrix_J11(A0_T, N, indexGas)
    J12 = update_matrix_J12(A0_T, indexCondensed)
    NC = len(indexCondensed)
    if NC > 0:
        diag_val = psi_j[indexCondensed] / N[indexCondensed]
        J22 = -np.diag(diag_val)
        row1 = np.hstack((J11, J12))
        row2 = np.hstack((J12.T, J22))
        J = np.vstack((row1, row2))
    else:
        J = J11
    return J


def update_vector_b(A0, N, NatomE, ind_E, index, indexGas, indexCondensed, indexIons, muRT, tau):
    # Compute vector b
    bi = N[index] @ A0[index, :]
    if indexIons is not None and len(indexIons) > 0:
        bi[ind_E] = NatomE[ind_E]
    
    A0_gas = A0[indexGas, :]
    N_gas = N[indexGas]
    muRT_gas = muRT[indexGas]
    factor = (N_gas * muRT_gas)[:, np.newaxis]
    sum_part = np.sum(A0_gas * factor, axis=0)

    b1 = NatomE - bi + sum_part
    b2 = muRT[indexCondensed] - tau / N[indexCondensed]
    
    if len(indexCondensed) > 0:
        return np.concatenate((b1, b2))
    else:
        return b1


def update_Delta_ln_nj(A0, pi_i, muRT, indexGas):
    # Compute correction moles of gases
    return A0[indexGas, :] @ pi_i - muRT[indexGas]
