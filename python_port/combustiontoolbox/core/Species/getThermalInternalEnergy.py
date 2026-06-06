def getThermalInternalEnergy(self, T):
    """
    Compute thermal internal energy [J/mol] of the species at the given
    temperature [K] using piecewise cubic Hermite interpolating
    polynomials and linear extrapolation.

    Args:
        T (float): Temperature [K]

    Returns:
        float: Thermal internal energy in molar basis [J/mol]

    Example:
        DeT = species.getThermalInternalEnergy(300)
    """

    # Compute internal energy [J/mol]
    e0 = self.getInternalEnergy(T)

    # Compute thermal internal energy [J/mol]
    DeT = e0 - self.ef

    return DeT