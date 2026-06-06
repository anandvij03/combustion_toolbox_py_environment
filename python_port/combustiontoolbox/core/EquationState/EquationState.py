from abc import ABC, abstractmethod

from scipy.optimize import root_scalar

from combustiontoolbox.common.Constants import Constants


class EquationState(ABC):

    @abstractmethod
    def getPressure(
        self,
        temperature,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        pass

    @abstractmethod
    def getVolume(
        self,
        temperature,
        pressure,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        pass

    @abstractmethod
    def getDepartureFunctions(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        pass

    @abstractmethod
    def getPressureDerivativesDimensional(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        pass

    def getPressureDerivatives(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):

        dPdV_T, dPdT_V = (
            self.getPressureDerivativesDimensional(
                temperature,
                pressure,
                molarVolume,
                molarFractions,
                chemicalSystem,
                *args,
            )
        )

        dPdV_T = (
            molarVolume / pressure
        ) * dPdV_T

        dPdT_V = (
            temperature / pressure
        ) * dPdT_V

        return dPdV_T, dPdT_V

    def getVolumeDerivatives(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):

        dPdV_T, dPdT_V = (
            self.getPressureDerivativesDimensional(
                temperature,
                pressure,
                molarVolume,
                molarFractions,
                chemicalSystem,
                *args,
            )
        )

        dVdT_p = (
            temperature / molarVolume
        ) * (
            -dPdT_V / dPdV_T
        )

        dVdp_T = (
            pressure / molarVolume
        ) * (
            1.0 / dPdV_T
        )

        return dVdT_p, dVdp_T

    def getVolumeDerivativesDimensional(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):

        dPdV_T, dPdT_V = (
            self.getPressureDerivativesDimensional(
                temperature,
                pressure,
                molarVolume,
                molarFractions,
                chemicalSystem,
                *args,
            )
        )

        dVdT_p = -dPdT_V / dPdV_T
        dVdp_T = 1.0 / dPdV_T

        return dVdT_p, dVdp_T

    def getTemperature(
        self,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        temperatureGuess=None,
        *args,
    ):

        if temperatureGuess is None:
            temperatureGuess = 300.0

        def pressure_error(T):

            return (
                self.getPressure(
                    T,
                    molarVolume,
                    molarFractions,
                    chemicalSystem,
                    *args,
                )
                - pressure
            )

        result = root_scalar(
            pressure_error,
            x0=temperatureGuess,
            x1=temperatureGuess + 1.0,
        )

        return result.root

    @staticmethod
    def getCompressibilityFactor(
        temperature,
        pressure,
        molarVolume,
    ):

        R0 = Constants.R0

        Z = (
            pressure * molarVolume
        ) / (
            R0 * temperature
        )

        return Z