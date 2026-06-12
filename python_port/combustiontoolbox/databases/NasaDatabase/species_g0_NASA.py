import numpy as np
from typing import Union, Any
from combustiontoolbox.common.Constants import Constants


def species_g0_NASA(
    obj: Any, species: str, temperature: Union[float, list, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute Gibbs energy [J/mol] of the species at the given
    temperature [K] using NASA's 9 polynomials

    Args:
        obj (NasaDatabase): NasaDatabase object
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]

    Returns:
        g0 (float or ndarray): Gibbs energy in molar basis [J/mol]

    Example:
        g0 = species_g0_NASA(db, 'H2O', np.arange(300, 6100, 100))
    """
    R0 = Constants.R0
    temperature = np.asarray(temperature, dtype=float)
    scalar_input = (temperature.ndim == 0)
    temperature_arr = np.atleast_1d(temperature)
    N = len(temperature_arr)

    # Unpack NASA's polynomials coefficients
    a, b, _, tExponents, ctTInt, _, _, _, _ = obj.getCoefficients(species, obj.species)

    g0 = np.zeros(N)

    for i, T in enumerate(temperature_arr):
        if ctTInt > 0:
            # Compute interval temperature
            tInterval = obj.getIndexTempereratureInterval(species, T, obj.species) - 1 # 0-indexed in Python

            # Compute Gibbs energy
            g_coeffs = np.array([-0.5, 1.0 + np.log(T), 1.0 - np.log(T), -0.5, -1/6, -1/12, -1/20, 0.0], dtype=float)
            a_int = a[tInterval]
            g_coeffs = g_coeffs[:len(a_int)]

            g0[i] = R0 * T * (np.sum(a_int * (T ** tExponents[tInterval]) * g_coeffs) + b[tInterval][0] / T - b[tInterval][1])
        else:
            g0[i] = getattr(obj.species[species], 'Hf0', obj.species[species].hf)

    if scalar_input:
        return float(g0[0])
    return g0
