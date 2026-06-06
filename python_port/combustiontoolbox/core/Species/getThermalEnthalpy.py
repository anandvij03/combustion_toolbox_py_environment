def getThermalEnthalpy(self, T):
    """
    Compute thermal enthalpy [J/mol] of the species at the given
    temperature [K] using piecewise cubic Hermite interpolating
    polynomials and linear extrapolation.

    Args:
        T (float): Temperature [K]

    Returns:
        float: Thermal enthalpy in molar basis [J/mol]

    Example:
        DhT = species.getThermalEnthalpy(300)
    """

    # Compute enthalpy [J/mol]
    h0 = self.getEnthalpy(T)

    # Compute thermal enthalpy [J/mol]
    DhT = h0 - self.hf

    return DhT