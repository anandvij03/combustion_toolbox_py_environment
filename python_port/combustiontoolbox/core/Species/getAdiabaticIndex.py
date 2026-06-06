import numpy as np


def getAdiabaticIndex(self, T):
    """
    Compute adiabatic index of the species [-]
    at the given temperature [K].
    """

    gamma = (
        self.getHeatCapacityPressure(T)
        / self.getHeatCapacityVolume(T)
    )

    assert np.any(
        ~np.isnan(gamma)
    ), "Adiabatic index equal NaN"

    return gamma