import numpy as np

from combustiontoolbox.core.EquationState import EquationState
from combustiontoolbox.common.Constants import Constants


class EquationStatePengRobinson(EquationState):
    """
    The EquationStatePengRobinson class implements the
    Peng-Robinson equation of state for real gases.

    Example:
        eos = EquationStatePengRobinson()

    See also:
        EquationState
    """

    R0 = Constants.R0

    def __init__(self):
        self.tol0 = 1e-8

        self.cachedListSpecies = None

        self.temperatureCritical = None
        self.pressureCritical = None
        self.acentricFactor = None

        self.FLAG_VALID = None

    def getPressure(
        self,
        temperature,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        """
        Compute pressure [Pa] using the Peng-Robinson EOS.
        """

        a_mix, b_mix, _, _ = self.getMixtureParameters(
            temperature,
            molarFractions,
            chemicalSystem,
        )

        pressure = (
            (self.R0 * temperature)
            / (molarVolume - b_mix)
            - a_mix
            / (
                molarVolume**2
                + 2.0 * b_mix * molarVolume
                - b_mix**2
            )
        )

        return pressure

    def getVolume(
        self,
        temperature,
        pressure,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        """
        Compute gas-phase molar volume [m3/mol].
        """

        a_mix, b_mix, _, _ = self.getMixtureParameters(
            temperature,
            molarFractions,
            chemicalSystem,
        )

        A = (
            a_mix * pressure
        ) / (
            self.R0**2 * temperature**2
        )

        B = (
            b_mix * pressure
        ) / (
            self.R0 * temperature
        )

        coeffs = np.array(
            [
                1.0,
                -(1.0 - B),
                (A - 2.0 * B - 3.0 * B**2),
                -(A * B - B**2 - B**3),
            ]
        )

        Z_roots = np.roots(coeffs)

        mask = (
            np.abs(np.imag(Z_roots))
            < self.tol0
        )

        Z_real = np.real(
            Z_roots[mask]
        )

        if Z_real.size == 0:
            raise RuntimeError(
                "EquationStatePengRobinson:getVolume: "
                "No real roots found for Z."
            )

        Z_gas = np.max(Z_real)

        molarVolume = (
            Z_gas
            * self.R0
            * temperature
            / pressure
        )

        return molarVolume

    def getPressureDerivativesDimensional(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        """
        Compute dimensional pressure derivatives.
        """

        (
            a_mix,
            b_mix,
            dadT_mix,
            _,
        ) = self.getMixtureParameters(
            temperature,
            molarFractions,
            chemicalSystem,
        )

        dPdV_T = (
            -(self.R0 * temperature)
            / (molarVolume - b_mix) ** 2
            + (
                2.0
                * a_mix
                * (molarVolume + b_mix)
            )
            / (
                molarVolume**2
                + 2.0 * b_mix * molarVolume
                - b_mix**2
            )
            ** 2
        )

        dPdT_V = (
            self.R0
            / (molarVolume - b_mix)
            - dadT_mix
            / (
                molarVolume**2
                + 2.0 * b_mix * molarVolume
                - b_mix**2
            )
        )

        return dPdV_T, dPdT_V

    def getDepartureFunctions(
        self,
        temperature,
        pressure,
        molarVolume,
        molarFractions,
        chemicalSystem,
        *args,
    ):
        """
        Compute Peng-Robinson departure functions.
        """

        (
            a_mix,
            b_mix,
            dadT_mix,
            d2adT2_mix,
        ) = self.getMixtureParameters(
            temperature,
            molarFractions,
            chemicalSystem,
        )

        if b_mix < 1e-15:
            return 0.0, 0.0, 0.0

        Z = self.getCompressibilityFactor(
            temperature,
            pressure,
            molarVolume,
        )

        B = (
            b_mix * pressure
        ) / (
            self.R0 * temperature
        )

        arg = (
            Z
            + (1.0 + np.sqrt(2.0)) * B
        ) / (
            Z
            + (1.0 - np.sqrt(2.0)) * B
        )

        logTerm = np.log(
            np.maximum(arg, 1e-12)
        )

        denom = (
            2.0
            * np.sqrt(2.0)
            * b_mix
        )

        enthalpyDeparture = (
            self.R0
            * temperature
            * (Z - 1.0)
            + (
                (
                    temperature * dadT_mix
                    - a_mix
                )
                / denom
            )
            * logTerm
        )

        entropyDeparture = (
            self.R0
            * np.log(
                np.maximum(
                    Z - B,
                    1e-12,
                )
            )
            + (
                dadT_mix
                / denom
            )
            * logTerm
        )

        heatCapacityVolumeDeparture = (
            temperature
            * d2adT2_mix
            / denom
        ) * logTerm

        (
            dPdV_T,
            dPdT_V,
        ) = self.getPressureDerivativesDimensional(
            temperature,
            pressure,
            molarVolume,
            molarFractions,
            chemicalSystem,
            *args,
        )

        heatCapacityPressureDeparture = (
            heatCapacityVolumeDeparture
            + (
                -temperature
                * (dPdT_V**2)
                / dPdV_T
            )
            - self.R0
        )

        return (
            heatCapacityPressureDeparture,
            enthalpyDeparture,
            entropyDeparture,
        )