import numpy as np
from typing import Tuple, Union, Any

from combustiontoolbox.common.constants import R0
from .thermo_helpers import (unpack_NASA_coefficients, set_elements, 
                              set_element_matrix, compute_change_moles_gas_reaction, 
                              compute_interval_NASA)



def species_thermo_NASA(species: str, temperature: Union[float, list, np.ndarray], db: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute thermodynamic function using NASA's 9 polynomials.

    Args:
        species (str): Chemical species.
        temperature (float or array-like): Range of temperatures to evaluate [K].
        db (Any): Database object with custom thermodynamic polynomials functions 
                  generated from NASA's 9 polynomials fits.

    Returns:
        Tuple containing 1D numpy arrays:
        * cP  (ndarray): Specific heat at constant pressure in molar basis [J/(mol-K)]
        * cV  (ndarray): Specific heat at constant volume in molar basis   [J/(mol-K)]
        * h0  (ndarray): Enthalpy in molar basis [J/mol]
        * DhT (ndarray): Thermal enthalpy in molar basis [J/mol]
        * e0  (ndarray): Internal energy in molar basis [J/mol]
        * DeT (ndarray): Thermal internal energy in molar basis [J/mol]
        * s0  (ndarray): Entropy in molar basis [J/(mol-K)]
        * g0  (ndarray): Gibbs energy in molar basis [J/mol]
    """
    # Ensure temperature is a 1D numpy array for consistent vectorized operations
    temp_array = np.atleast_1d(temperature).astype(float)
    n_temps = len(temp_array)

    # Pre-allocate output arrays
    cp0 = np.zeros(n_temps)
    cv0 = np.zeros(n_temps)
    h0  = np.zeros(n_temps)
    DhT = np.zeros(n_temps)
    e0  = np.zeros(n_temps)
    DeT = np.zeros(n_temps)
    s0  = np.zeros(n_temps)
    g0  = np.zeros(n_temps)

    # Extract species data from database
    species_data = getattr(db, species)
    hf0 = species_data.hf  # [J/mol]
    t_ref_base = 298.15    # [K]

    # Unpack NASA's polynomials coefficients (Assuming these helper functions are defined)
    a, b, tRange, tExponents, ctTInt, txFormula, phase = unpack_NASA_coefficients(species, db)
    
    # Get elements and compute change in moles
    elements = set_elements()
    element_matrix = set_element_matrix(txFormula, elements)
    Delta_n = compute_change_moles_gas_reaction(element_matrix, phase)

    # Compute thermodynamic properties
    for i in range(n_temps - 1, -1, -1):
        T = temp_array[i]

        if getattr(species_data, 'ctTInt', 0) > 0:
            # Compute interval temperature (assuming MATLAB function returns a 1-based index)
            t_interval = compute_interval_NASA(species, T, db, tRange, ctTInt)
            t_idx = t_interval - 1  # Convert to Python 0-based indexing

            # Extract arrays for the specific temperature interval
            a_int = np.array(a[t_idx])
            b_int = np.array(b[t_idx])
            t_exp = np.array(tExponents[t_idx])

            # Pre-calculate T^exponents
            T_pow = T ** t_exp

            # Specific Heat (Cp)
            cp0[i] = R0 * np.sum(a_int * T_pow)

            # Enthalpy (H)
            # Slicing multiplier arrays to len(a_int) to prevent dimension mismatch errors 
            h_multipliers = np.array([-1.0, np.log(T), 1.0, 1/2, 1/3, 1/4, 1/5, 0.0])
            h_multipliers = h_multipliers[:len(a_int)]
            
            h0[i] = R0 * T * (np.sum(a_int * T_pow * h_multipliers) + b_int[0] / T)

            # Entropy (S)
            s_multipliers = np.array([-1/2, -1.0, np.log(T), 1.0, 1/2, 1/3, 1/4, 0.0])
            s_multipliers = s_multipliers[:len(a_int)]
            
            s0[i] = R0 * (np.sum(a_int * T_pow * s_multipliers) + b_int[1])

            # Internal Energies and Thermal differentials
            ef0 = hf0 - Delta_n * R0 * t_ref_base
            e0[i] = ef0 + (h0[i] - hf0) - (1 - phase) * R0 * (T - t_ref_base)
            
            cv0[i] = cp0[i] - R0
            DhT[i] = h0[i] - hf0
            DeT[i] = e0[i] - ef0
            
            g0[i] = h0[i] - T * s0[i]

        else:
            t_ref = tRange[0]
            
            cp0[i] = 0.0
            cv0[i] = 0.0
            h0[i]  = hf0
            e0[i]  = hf0 - Delta_n * R0 * t_ref
            s0[i]  = 0.0
            DhT[i] = 0.0
            DeT[i] = 0.0
            
            g0[i]  = getattr(species_data, 'Hf0', hf0) 

    return cp0, cv0, h0, DhT, e0, DeT, s0, g0