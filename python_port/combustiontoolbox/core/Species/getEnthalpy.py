import numpy as np


class Species:

    _cachedSpecies = []
    _cachedH0curves = []

    ...

    def getEnthalpy(self, T):
        """
        Compute enthalpy [J/mol] of the species
        at the given temperature [K].
        """

        try:
            index = Species._cachedSpecies.index(
                self.name
            )

            h0curve = Species._cachedH0curves[index]

        except ValueError:

            h0curve = self.h0curve

            Species._cachedSpecies.append(
                self.name
            )

            Species._cachedH0curves.append(
                h0curve
            )

        h0 = h0curve(T)

        return h0