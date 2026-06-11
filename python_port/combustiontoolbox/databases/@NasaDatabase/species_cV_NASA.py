import numpy as np
from typing import Union, Any
from .species_cP_NASA import species_cP_NASA


def species_cV_NASA(
    species: str, temperature: Union[float, list, np.ndarray], db: Any
) -> Union[float, np.ndarray]:
    """
    Compute specific heat at constant volume [J/(mol-K)] of the species
    at the given temperature [K] using NASA's 9 polynomials

    Args:
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        cV (float or ndarray): Specific heat at constant volume in molar basis [J/(mol-K)]

    Example:
        cV = species_cV_NASA('H2O', np.arange(300, 6100, 100), db)
    """
    _, cv = species_cP_NASA(db, species, temperature)
    return cv
