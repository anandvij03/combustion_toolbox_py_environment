from combustiontoolbox.common.Constants import Constants

def getInternalEnergy(self, T):
    """
    Compute internal energy [J/mol] of the species
    at the given temperature [K].
    """

    R0 = Constants.R0

    h0 = self.getEnthalpy(T)

    e0 = h0 - R0 * T

    return e0