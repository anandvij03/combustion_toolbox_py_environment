import numpy as np
from typing import Union, Any
from .species_h0_NASA import species_h0_NASA


def species_DhT_NASA(
    species: str, temperature: Union[float, list, np.ndarray], db: Any
) -> Union[float, np.ndarray]:
    """
    Compute thermal enthalpy [J/mol] of the species at the given
    temperature [K] using NASA's 9 polynomials

    Args:
        species (str): Chemical species
        temperature (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        DhT (float or ndarray): Thermal enthalpy in molar basis [J/mol]

    Example:
        DhT = species_DhT_NASA('H2O', np.arange(300, 6100, 100), db)
    """
    _, DhT = species_h0_NASA(species, temperature, db)
    return DhT
