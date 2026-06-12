import numpy as np
from typing import Union, Any, Tuple
from combustiontoolbox.common.Constants import Constants
from combustiontoolbox.core.Elements.Elements import Elements


def species_e0_NASA(
    species: str, temperature: Union[float, list, np.ndarray], db: Any
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute internal energy and the thermal internal energy [J/mol] of
    the species at the given temperature [K] using NASA's 9 polynomials

    Args:
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        Tuple containing:
        * e0 (float or ndarray): Internal energy in molar basis [J/mol]
        * DeT (float or ndarray): Thermal internal energy in molar basis [J/mol]

    Example:
        e0, DeT = species_e0_NASA('H2O', np.arange(300, 6100, 100), db)
    """
    R0 = Constants.R0
    hf0 = db.species[species].hf
    Tref = 298.15

    sp = db.species[species]
    a = sp.a
    b = sp.b
    tRange = sp.Trange
    tExponents = sp.Texponents
    ctTInt = sp.Tintervals
    swtCondensed = sp.phase

    elements = Elements()
    elementMatrix = sp.getElementMatrix(elements.listElements)
    Delta_n = db.getChangeMolesGasReaction(elements, elementMatrix, swtCondensed)

    # Ensure temperature is a 1D numpy array for consistent iteration
    temp_array = np.atleast_1d(temperature).astype(float)
    scalar_input = (np.asarray(temperature).ndim == 0)
    n_temps = len(temp_array)

    e0 = np.zeros(n_temps)
    DeT = np.zeros(n_temps)

    for i in range(n_temps - 1, -1, -1):
        T = temp_array[i]

        if ctTInt > 0:
            # Compute interval temperature
            tInterval = db.getIndexTempereratureInterval(species, T, db.species) - 1 # 0-indexed in Python

            # Compute specific enthalpy from NASA's 9 polynomials
            h_multipliers = np.array([-1.0, np.log(T), 1.0, 1/2, 1/3, 1/4, 1/5, 0.0], dtype=float)
            a_int = a[tInterval]
            h_multipliers = h_multipliers[:len(a_int)]

            h0 = R0 * T * (np.sum(a_int * (T ** tExponents[tInterval]) * h_multipliers) + b[tInterval][0] / T)
            ef0 = hf0 - Delta_n * R0 * Tref
            e0[i] = ef0 + (h0 - hf0) - (1 - swtCondensed) * R0 * (T - Tref)
            DeT[i] = e0[i] - ef0
        else:
            Tref_interval = tRange[0]
            e0[i] = hf0 - Delta_n * R0 * Tref_interval
            DeT[i] = 0.0

    if scalar_input:
        return float(e0[0]), float(DeT[0])
    return e0, DeT
