def getEntropy(self, T):
    """
    Compute entropy [J/(mol-K)] of the species
    at the given temperature [K].
    """

    try:
        index = Species._cachedSpeciesS.index(
            self.name
        )

        s0curve = Species._cachedS0curves[index]

    except ValueError:

        s0curve = self.s0curve

        Species._cachedSpeciesS.append(
            self.name
        )

        Species._cachedS0curves.append(
            s0curve
        )

    s0 = s0curve(T)

    return s0