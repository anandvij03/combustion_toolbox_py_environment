def getHeatCapacityPressure(self, T):
    """
    Compute specific heat at constant pressure [J/(mol-K)]
    at the given temperature [K].
    """

    if not hasattr(
        self.__class__.getHeatCapacityPressure,
        "_cachedSpecies",
    ):
        self.__class__.getHeatCapacityPressure._cachedSpecies = []
        self.__class__.getHeatCapacityPressure._cachedCPcurves = []

    cachedSpecies = (
        self.__class__.getHeatCapacityPressure._cachedSpecies
    )

    cachedCPcurves = (
        self.__class__.getHeatCapacityPressure._cachedCPcurves
    )

    try:
        index = cachedSpecies.index(self.name)
        cpcurve = cachedCPcurves[index]

    except ValueError:

        cpcurve = self.cpcurve

        cachedSpecies.append(self.name)
        cachedCPcurves.append(cpcurve)

    cp = cpcurve(T)

    return cp