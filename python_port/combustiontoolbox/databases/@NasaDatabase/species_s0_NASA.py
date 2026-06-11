import numpy as np
from typing import Union, Any

# Assuming standard constants and helpers are imported from your toolbox:
# from combustiontoolbox.common.constants import R0
# from .thermo_helpers import unpack_NASA_coefficients, compute_interval_NASA

# Fallback definition if R0 is not imported
R0 = 8.31446261815324  # Universal Gas Constant [J/(mol-K)]

def species_s0_NASA(species: str, temperature: Union[float, list, np.ndarray], db: Any) -> np.ndarray:
    """
    Compute entropy [J/(mol-K)] of the species at the given temperature [K]
    using NASA's 9 polynomials.

    Args:
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        s0 (ndarray): Entropy in molar basis [J/(mol-K)]
    """
    # Ensure temperature is a 1D numpy array for consistent iteration
    temp_array = np.atleast_1d(temperature).astype(float)
    n_temps = len(temp_array)

    # Pre-allocate output array
    s0 = np.zeros(n_temps)

    # Extract species data from database
    species_data = getattr(db, species)

    # Unpack NASA's polynomials coefficients
    # Slicing the first 5 elements to match the exact expected outputs of this specific MATLAB call
    a, b, tRange, tExponents, ctTInt = unpack_NASA_coefficients(species, db)[:5]

    # Compute entropy
    for i in range(n_temps):
        T = temp_array[i]

        if getattr(species_data, 'ctTInt', 0) > 0:
            # Compute interval temperature
            t_interval = compute_interval_NASA(species, T, db, tRange, ctTInt)
            t_idx = t_interval - 1  # Convert to Python 0-based indexing

            # Extract arrays for the specific temperature interval
            a_int = np.array(a[t_idx])
            b_int = np.array(b[t_idx])
            t_exp = np.array(tExponents[t_idx])

            # Pre-calculate T^exponents
            T_pow = T ** t_exp
            
            # Entropy (S) multiplier array: [-1/2, -1, ln(T), 1, 1/2, 1/3, 1/4, 0]
            s_multipliers = np.array([-0.5, -1.0, np.log(T), 1.0, 0.5, 1/3, 0.25, 0.0])
            s_multipliers = s_multipliers[:len(a_int)] # Ensure dimensions match

            # Calculate entropy (b_int[1] corresponds to MATLAB's b{tInterval}(2))
            s0[i] = R0 * (np.sum(a_int * T_pow * s_multipliers) + b_int[1])
            
        else:
            # If the species is only a reactant
            s0[i] = 0.0

    return s0