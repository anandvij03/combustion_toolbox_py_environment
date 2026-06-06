from combustiontoolbox.common.Constants import Constants


def getHeatCapacityVolume(self, T):
    """
    Compute specific heat at constant volume [J/(mol-K)]
    at the given temperature [K].
    """

    R0 = Constants.R0

    cv = self.getHeatCapacityPressure(T) - R0

    return cv