import numpy as np
from typing import Union, Any
from .species_cP_NASA import species_cP_NASA


def species_gamma_NASA(
    species: str, T: Union[float, list, np.ndarray], db: Any
) -> Union[float, np.ndarray]:
    """
    Compute adiabatic index of the species [-] at the given temperature
    [K] using piecewise cubic Hermite interpolating polynomials and
    linear extrapolation

    Args:
        species (str): Chemical species
        T (float or array-like): Range of temperatures to evaluate [K]
        db (Any): Database with custom thermodynamic polynomials functions generated from NASAs 9 polynomials fits

    Returns:
        gamma (float or ndarray): Adiabatic index [-]

    Example:
        gamma = species_gamma_NASA('H2O', np.arange(300, 6100, 100), db)
    """
    cp, cv = species_cP_NASA(db, species, T)
    gamma = cp / cv

    assert not np.any(np.isnan(gamma)), "Adiabatic index equal NaN"
    return gamma
