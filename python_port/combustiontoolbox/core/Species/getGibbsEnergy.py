def getGibbsEnergy(self, T):
    """
    Compute Gibbs energy [J/mol] of the species
    at the given temperature [K].
    """

    if not hasattr(
        Species.getGibbsEnergy,
        "_cachedSpecies",
    ):
        Species.getGibbsEnergy._cachedSpecies = []
        Species.getGibbsEnergy._cachedG0curves = []

    cachedSpecies = (
        Species.getGibbsEnergy._cachedSpecies
    )

    cachedG0curves = (
        Species.getGibbsEnergy._cachedG0curves
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