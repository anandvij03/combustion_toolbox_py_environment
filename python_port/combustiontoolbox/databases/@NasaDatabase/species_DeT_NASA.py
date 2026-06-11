import numpy as np
from typing import Union, Any
from .species_e0_NASA import species_e0_NASA


def species_DeT_NASA(
    species: str, temperature: Union[float, list, np.ndarray], db: Any
) -> Union[float, np.ndarray]:
    """
    Compute thermal internal energy [J/mol] of the species at the given
    temperature [K] using NASA's 9 polynomials

    Args:
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        DeT (float or ndarray): Thermal internal energy in molar basis [J/mol]

    Example:
        DeT = species_DeT_NASA('H2O', np.arange(300, 6100, 100), db)
    """
    _, DeT = species_e0_NASA(species, temperature, db)
    return DeT
