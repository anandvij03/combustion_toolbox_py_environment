import numpy as np

from combustiontoolbox.core.EquationState.EquationState import EquationState
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
    def initializeCache(self, chemicalSystem):
        """
        Cache the database values once to avoid field lookups
        during iterative solver loops.
        """

        listSpecies = chemicalSystem.listSpecies
        numSpecies = chemicalSystem.numSpecies

        Tc = np.zeros(numSpecies)
        Pc = np.zeros(numSpecies)
        omega = np.zeros(numSpecies)

        species = chemicalSystem.species

        for i, name in enumerate(listSpecies):
            sp = getattr(species, name)

            Tc[i] = sp.Tcritical
            Pc[i] = sp.Pcritical
            omega[i] = sp.acentricFactor

        self.temperatureCritical = Tc
        self.pressureCritical = Pc * 1e5
        self.acentricFactor = omega

        self.FLAG_VALID = (
            ~np.isnan(Tc)
            & (Tc > 0.0)
            & ~np.isnan(Pc)
            & (Pc > 0.0)
        )

        self.cachedListSpecies = listSpecies

    def getMixtureParameters(
        self,
        temperature,
        molarFractions,
        chemicalSystem,
    ):
        """
        Compute mixture parameters using
        van der Waals one-fluid mixing rules.
        """

        if (
            self.cachedListSpecies is None
            or self.cachedListSpecies
            != chemicalSystem.listSpecies
        ):
            self.initializeCache(
                chemicalSystem
            )

        FLAG_ACTIVE = (
            np.asarray(
                molarFractions
            ).reshape(-1)
            > 0.0
        ) & self.FLAG_VALID.reshape(-1)

        if not np.any(FLAG_ACTIVE):
            return (
                0.0,
                0.0,
                0.0,
                0.0,
            )

        X_active = np.asarray(
            molarFractions
        ).reshape(-1)[FLAG_ACTIVE]

        Tc = self.temperatureCritical[
            FLAG_ACTIVE
        ]

        Pc = self.pressureCritical[
            FLAG_ACTIVE
        ]

        omega = self.acentricFactor[
            FLAG_ACTIVE
        ]

        kappa = (
            0.37464
            + 1.54226 * omega
            - 0.26992 * omega**2
        )

        Tr = temperature / Tc

        sqrtTr = np.sqrt(Tr)

        alpha = (
            1.0
            + kappa * (1.0 - sqrtTr)
        ) ** 2

        a_i = (
            0.45724
            * (
                self.R0**2
                * Tc**2
                / Pc
            )
            * alpha
        )

        b_i = (
            0.07780
            * (
                self.R0
                * Tc
                / Pc
            )
        )

        dalpha_dT = (
            -kappa
            / (sqrtTr * Tc)
            * (
                1.0
                + kappa
                * (
                    1.0
                    - sqrtTr
                )
            )
        )

        d2alpha_dT2 = (
            kappa
            * (kappa + 1.0)
            / (
                2.0
                * Tc**2
                * Tr ** (3.0 / 2.0)
            )
        )

        a0 = (
            0.45724
            * (
                self.R0**2
                * Tc**2
                / Pc
            )
        )

        da_dT_i = (
            a0 * dalpha_dT
        )

        d2a_dT2_i = (
            a0 * d2alpha_dT2
        )

        b_mix = np.dot(
            X_active,
            b_i,
        )

        sqrt_a_i = np.sqrt(
            np.maximum(
                a_i,
                1e-20,
            )
        )

        S1 = np.dot(
            X_active,
            sqrt_a_i,
        )

        a_mix = S1**2

        S2 = np.dot(
            X_active,
            da_dT_i
            / (2.0 * sqrt_a_i),
        )

        dadT_mix = (
            2.0
            * S1
            * S2
        )

        S3 = np.dot(
            X_active,
            (
                d2a_dT2_i
                / (
                    2.0
                    * sqrt_a_i
                )
            )
            - (
                da_dT_i**2
                / (
                    4.0
                    * sqrt_a_i**3
                )
            ),
        )

        d2adT2_mix = (
            2.0 * S2**2
            + 2.0 * S1 * S3
        )

        return (
            a_mix,
            b_mix,
            dadT_mix,
            d2adT2_mix,
        )

    def getPseudoCriticalProperties(
        self,
        molarFractions,
        chemicalSystem,
    ):
        """
        Compute pseudo-critical properties
        for a multi-component mixture.
        """

        if (
            self.cachedListSpecies is None
            or self.cachedListSpecies
            != chemicalSystem.listSpecies
        ):
            self.initializeCache(
                chemicalSystem
            )

        mask = (
            np.asarray(
                molarFractions
            ).reshape(-1)
            > 0.0
        ) & self.FLAG_VALID.reshape(-1)

        Xi = np.asarray(
            molarFractions
        ).reshape(-1)[mask]

        Tc_i = self.temperatureCritical[
            mask
        ]

        Pc_i = self.pressureCritical[
            mask
        ]

        omega_i = self.acentricFactor[
            mask
        ]

        a_i_tc = (
            0.45724
            * (
                self.R0**2
                * Tc_i**2
                / Pc_i
            )
        )

        b_i = (
            0.07780
            * (
                self.R0
                * Tc_i
                / Pc_i
            )
        )

        a_mix_tc = (
            np.dot(
                Xi,
                np.sqrt(a_i_tc),
            )
        ) ** 2

        b_mix = np.dot(
            Xi,
            b_i,
        )

        temperatureCritical_mix = (
            a_mix_tc
            * 0.07780
        ) / (
            b_mix
            * 0.45724
            * self.R0
        )

        pressureCritical_mix = (
            0.07780
            * self.R0
            * temperatureCritical_mix
        ) / b_mix

        acentricFactor_mix = np.dot(
            Xi,
            omega_i,
        )

        return (
            temperatureCritical_mix,
            pressureCritical_mix,
            acentricFactor_mix,
        )