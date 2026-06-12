import numpy as np
from combustiontoolbox.common.Constants import Constants


def species_cP_NASA(obj, species, temperature):
    """
    Compute specific heats at constant pressure and at constant volume
    [J/(mol-K)] of the species at the given temperature [K] using NASA's
    9 polynomials
    """
    R0 = Constants.R0
    temperature = np.asarray(temperature, dtype=float)
    scalar_input = (temperature.ndim == 0)
    temperature_arr = np.atleast_1d(temperature)
    N = len(temperature_arr)

    # Unpack NASA's polynomials coefficients
    a, _, _, tExponents, ctTInt, _, _, _, _ = obj.getCoefficients(species, obj.species)

    cp = np.zeros(N)
    cv = np.zeros(N)

    for i, T in enumerate(temperature_arr):
        if ctTInt > 0:
            # Compute interval temperature (getIndexTempereratureInterval is 1-based, so subtract 1 for 0-based Python index)
            tInterval = obj.getIndexTempereratureInterval(species, T, obj.species) - 1

            # Compute specific heat at constant pressure and volume
            cp[i] = R0 * np.sum(a[tInterval] * (T ** tExponents[tInterval]))
            cv[i] = cp[i] - R0
        else:
            cp[i] = 0.0
            cv[i] = 0.0

    if scalar_input:
        return float(cp[0]), float(cv[0])
    return cp, cv
