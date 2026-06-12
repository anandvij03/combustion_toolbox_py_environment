def getGibbsEnergy(self, T):
    """
    Compute Gibbs energy [J/mol] of the species
    at the given temperature [K].
    """

    if not hasattr(
        self.__class__.getGibbsEnergy,
        "_cachedSpecies",
    ):
        self.__class__.getGibbsEnergy._cachedSpecies = []
        self.__class__.getGibbsEnergy._cachedG0curves = []

    cachedSpecies = (
        self.__class__.getGibbsEnergy._cachedSpecies
    )

    cachedG0curves = (
        self.__class__.getGibbsEnergy._cachedG0curves
    )

    try:
        index = cachedSpecies.index(self.name)
        g0curve = cachedG0curves[index]

    except ValueError:

        g0curve = self.g0curve

        cachedSpecies.append(self.name)
        cachedG0curves.append(g0curve)

    g0 = g0curve(T)

    return g0