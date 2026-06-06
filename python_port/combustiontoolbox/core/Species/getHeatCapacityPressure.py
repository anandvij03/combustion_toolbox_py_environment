def getHeatCapacityPressure(self, T):
    """
    Compute specific heat at constant pressure [J/(mol-K)]
    at the given temperature [K].
    """

    if not hasattr(
        Species.getHeatCapacityPressure,
        "_cachedSpecies",
    ):
        Species.getHeatCapacityPressure._cachedSpecies = []
        Species.getHeatCapacityPressure._cachedCPcurves = []

    cachedSpecies = (
        Species.getHeatCapacityPressure._cachedSpecies
    )

    cachedCPcurves = (
        Species.getHeatCapacityPressure._cachedCPcurves
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