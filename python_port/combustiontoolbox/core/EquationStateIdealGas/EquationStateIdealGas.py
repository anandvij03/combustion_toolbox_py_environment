from combustiontoolbox.core.EquationState import EquationState
from combustiontoolbox.common.Constants import Constants


class EquationStateIdealGas(EquationState):
    """
    Implements the ideal gas equation of state.
    """

    R0 = Constants.R0

    def getPressure(
        self,
        temperature,
        molarVolume,
        _,
        __,
        *args,
    ):
        """
        Compute pressure [Pa] using the ideal gas law.
        """

        pressure = (self.R0 * temperature) / molarVolume

        return pressure

    def getVolume(
        self,
        temperature,
        pressure,
        _,
        __,
        *args,
    ):
        """
        Compute molar volume [m^3/mol] using the ideal gas law.
        """

        molarVolume = self.R0 * temperature / pressure

        return molarVolume

    def getPressureDerivativesDimensional(
        self,
        temperature,
        _,
        molarVolume,
        __,
        ___,
        *args,
    ):
        """
        Compute dimensional pressure derivatives.
        """

        dPdV_T = -(self.R0 * temperature) / (molarVolume**2)
        dPdT_V = self.R0 / molarVolume

        return dPdV_T, dPdT_V

    def getVolumeDerivatives(
        self,
        _,
        __,
        ___,
        ____,
        _____,
        ______,
        *args,
    ):
        """
        Compute dimensionless volume derivatives.
        """

        dVdT_p = 1
        dVdp_T = -1

        return dVdT_p, dVdp_T

    def getDepartureFunctions(
        self,
        _,
        __,
        ___,
        ____,
        _____,
        ______,
        *args,
    ):
        """
        Compute departure functions.
        """

        heatCapacityPressureDeparture = 0
        enthalpyDeparture = 0
        entropyDeparture = 0

        return (
            heatCapacityPressureDeparture,
            enthalpyDeparture,
            entropyDeparture,
        )

    def getTemperature(
        self,
        pressure,
        molarVolume,
        *args,
    ):
        """
        Compute temperature [K] using the ideal gas law.
        """

        temperature = (pressure * molarVolume) / self.R0

        return temperature