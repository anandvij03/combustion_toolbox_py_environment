import numpy as np
from typing import Union, Any, Tuple
from combustiontoolbox.common.Constants import Constants


def species_h0_NASA(
    species: str, temperature: Union[float, list, np.ndarray], db: Any
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute enthalpy and thermal enthalpy [J/mol] of the species at the
    given temperature [K] using NASA's 9 polynomials

    Args:
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        Tuple containing:
        * h0 (float or ndarray): Enthalpy in molar basis [J/mol]
        * DhT (float or ndarray): Thermal enthalpy in molar basis [J/mol]

    Example:
        h0, DhT = species_h0_NASA('H2O', np.arange(300, 6100, 100), db)
    """
    R0 = Constants.R0
    hf0 = db.species[species].hf

    sp = db.species[species]
    a = sp.a
    b = sp.b
    tExponents = sp.Texponents
    ctTInt = sp.Tintervals

    # Ensure temperature is a 1D numpy array for consistent iteration
    temp_array = np.atleast_1d(temperature).astype(float)
    scalar_input = (np.asarray(temperature).ndim == 0)
    n_temps = len(temp_array)

    h0 = np.zeros(n_temps)
    DhT = np.zeros(n_temps)

    for i in range(n_temps - 1, -1, -1):
        T = temp_array[i]

        if ctTInt > 0:
            # Compute interval temperature
            tInterval = db.getIndexTempereratureInterval(species, T, db.species) - 1 # 0-indexed in Python

            # Compute specific enthalpy from NASA's 9 polynomials
            h_multipliers = np.array([-1.0, np.log(T), 1.0, 1/2, 1/3, 1/4, 1/5, 0.0], dtype=float)
            a_int = a[tInterval]
            h_multipliers = h_multipliers[:len(a_int)]

            h0[i] = R0 * T * (np.sum(a_int * (T ** tExponents[tInterval]) * h_multipliers) + b[tInterval][0] / T)
            DhT[i] = h0[i] - hf0
        else:
            h0[i] = hf0
            DhT[i] = 0.0

    if scalar_input:
        return float(h0[0]), float(DhT[0])
    return h0, DhT
