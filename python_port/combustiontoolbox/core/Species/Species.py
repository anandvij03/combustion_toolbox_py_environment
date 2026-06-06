import numpy as np

from combustiontoolbox.common.Constants import Constants
from combustiontoolbox.utils.generateID import generateID


class Species:
    """
    Store the properties of a chemical species.
    """

    def __init__(self):
        # Public properties
        self.name = None
        self.fullname = None
        self.refCode = None
        self.comments = None
        self.formula = None
        self.W = None
        self.hf = None
        self.hftoh0 = None
        self.ef = None
        self.phase = None
        self.T = None
        self.Tref = None
        self.Trange = None
        self.Tintervals = None
        self.Texponents = None
        self.a = None
        self.b = None
        self.Tcritical = None
        self.Pcritical = None
        self.Vcritical = None
        self.acentricFactor = None
        self.cpcurve = None
        self.h0curve = None
        self.s0curve = None
        self.g0curve = None

        # Internal/cache properties
        self.id_ = None
        self.elementMatrix = None
        self.FLAG_REFERENCE = False

    def getHeatCapacityPressure(self, T):
        """
        Compute specific heat at constant pressure [J/(mol-K)] of the species
        at the given temperature [K] using piecewise cubic Hermite
        interpolating polynomials and linear extrapolation.
        """

        if not hasattr(Species.getHeatCapacityPressure, "_cachedSpecies"):
            Species.getHeatCapacityPressure._cachedSpecies = []
            Species.getHeatCapacityPressure._cachedCPcurves = []

        cachedSpecies = Species.getHeatCapacityPressure._cachedSpecies
        cachedCPcurves = Species.getHeatCapacityPressure._cachedCPcurves

        try:
            index = cachedSpecies.index(self.name)
            cpcurve = cachedCPcurves[index]
        except ValueError:
            cpcurve = self.cpcurve
            cachedSpecies.append(self.name)
            cachedCPcurves.append(cpcurve)

        cp = cpcurve(T)
        return cp

    def getHeatCapacityVolume(self, T):
        """
        Compute specific heat at constant volume [J/(mol-K)] of the species
        at the given temperature [K] using piecewise cubic Hermite
        interpolating polynomials and linear extrapolation.
        """

        cv = self.getHeatCapacityPressure(T) - Constants.R0
        return cv

    def getEnthalpy(self, T):
        """
        Compute enthalpy [J/mol] of the species at the given temperature [K]
        using piecewise cubic Hermite interpolating polynomials and linear extrapolation.
        """

        if not hasattr(Species.getEnthalpy, "_cachedSpecies"):
            Species.getEnthalpy._cachedSpecies = []
            Species.getEnthalpy._cachedH0curves = []

        cachedSpecies = Species.getEnthalpy._cachedSpecies
        cachedH0curves = Species.getEnthalpy._cachedH0curves

        try:
            index = cachedSpecies.index(self.name)
            h0curve = cachedH0curves[index]
        except ValueError:
            h0curve = self.h0curve
            cachedSpecies.append(self.name)
            cachedH0curves.append(h0curve)

        h0 = h0curve(T)
        return h0

    def getEntropy(self, T):
        """
        Compute entropy [J/(mol-K)] of the species at the given temperature [K]
        using piecewise cubic Hermite interpolating polynomials and linear extrapolation.
        """

        if not hasattr(Species.getEntropy, "_cachedSpecies"):
            Species.getEntropy._cachedSpecies = []
            Species.getEntropy._cachedS0curves = []

        cachedSpecies = Species.getEntropy._cachedSpecies
        cachedS0curves = Species.getEntropy._cachedS0curves

        try:
            index = cachedSpecies.index(self.name)
            s0curve = cachedS0curves[index]
        except ValueError:
            s0curve = self.s0curve
            cachedSpecies.append(self.name)
            cachedS0curves.append(s0curve)

        s0 = s0curve(T)
        return s0

    def getGibbsEnergy(self, T):
        """
        Compute Gibbs energy [J/mol] of the species at the given temperature [K]
        using piecewise cubic Hermite interpolating polynomials and linear extrapolation.
        """

        if not hasattr(Species.getGibbsEnergy, "_cachedSpecies"):
            Species.getGibbsEnergy._cachedSpecies = []
            Species.getGibbsEnergy._cachedG0curves = []

        cachedSpecies = Species.getGibbsEnergy._cachedSpecies
        cachedG0curves = Species.getGibbsEnergy._cachedG0curves

        try:
            index = cachedSpecies.index(self.name)
            g0curve = cachedG0curves[index]
        except ValueError:
            g0curve = self.g0curve
            cachedSpecies.append(self.name)
            cachedG0curves.append(g0curve)

        g0 = g0curve(T)
        return g0

    def getInternalEnergy(self, T):
        """
        Compute internal energy [J/mol] of the species at the given
        temperature [K] using piecewise cubic Hermite interpolating
        polynomials and linear extrapolation.
        """

        h0 = self.getEnthalpy(T)
        e0 = h0 - Constants.R0 * T
        return e0

    def getThermalInternalEnergy(self, T):
        """
        Compute thermal internal energy [J/mol] of the species at the given
        temperature [K] using piecewise cubic Hermite interpolating
        polynomials and linear extrapolation.
        """

        e0 = self.getInternalEnergy(T)
        DeT = e0 - self.ef
        return DeT

    def getThermalEnthalpy(self, T):
        """
        Compute thermal enthalpy [J/mol] of the species at the given
        temperature [K] using piecewise cubic Hermite interpolating
        polynomials and linear extrapolation.
        """

        h0 = self.getEnthalpy(T)
        DhT = h0 - self.hf
        return DhT

    def getAdiabaticIndex(self, T):
        """
        Compute adiabatic index of the species [-] at the given temperature
        [K] using piecewise cubic Hermite interpolating polynomials and
        linear extrapolation.
        """

        gamma = self.getHeatCapacityPressure(T) / self.getHeatCapacityVolume(T)
        assert np.any(~np.isnan(gamma)), "Adibatic index equal NaN"
        return gamma

    def getElementMatrix(self, elements):
        """
        Compute element matrix of the given species formula.

        Returns a 2x5 matrix:
            row 1 = element indices
            row 2 = atom counts
        """

        N = 40
        NE = 5
        step = 8

        elementMatrix = np.zeros((2, NE), dtype=int)

        formula = (self.formula or "").ljust(N)

        for i in range(5, 0, -1):
            end0 = N - step * (NE - i)
            start0 = end0 - 5

            # MATLAB: obj.formula(start0 - 2:end0 - 6)
            element_i = formula[start0 - 3 : end0 - 6].strip()

            if element_i == "":
                continue

            if len(element_i) > 1 and element_i[1] == " ":
                element_i = element_i[0]

            idx = None
            for j, el in enumerate(elements):
                if str(el).lower() == element_i.lower():
                    idx = j + 1  # MATLAB-style 1-based index
                    break

            if idx is None:
                continue

            atom_str = formula[start0 - 1 : end0].strip()
            atom_count = int(float(atom_str)) if atom_str else 0

            elementMatrix[0, i - 1] = idx
            elementMatrix[1, i - 1] = atom_count

        return elementMatrix

    def setID(self):
        """
        Set internal id for caching purposes.
        """

        value = (
            str(self.name)
            + str(self.formula)
            + str(self.W)
            + str(self.hf)
            + str(self.ef)
            + str(self.phase)
            + str(self.Tref)
            + str(self.Tintervals)
        )

        self.id_ = generateID(value)
        return self

    def getID(self):
        """
        Get internal id for caching purposes.
        """
        return self.id_