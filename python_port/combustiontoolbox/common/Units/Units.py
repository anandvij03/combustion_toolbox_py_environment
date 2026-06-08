class Units:
    """
    Class with conversion factors between different units.
    """

    # Pressure conversion factors
    atm2bar = 1.01325
    bar2atm = 1.01325**-1
    atm2Pa = 101325
    Pa2atm = 101325**-1
    bar2Pa = 1e5
    Pa2bar = 1e-5

    # Temperature conversion factors
    K2C = lambda x: x - 273.15
    C2K = lambda x: x + 273.15
    F2C = lambda x: (x - 32) * 5 / 9
    K2F = lambda x: (x - 273.15) * 9 / 5 + 32

    # Mass conversion factors
    kg2lbs = 2.20462
    lbs2kg = 0.453592
    kg2g = 1e3
    g2kg = 1e-3

    # Volume conversion factors
    m32ft3 = 35.3147
    ft32m3 = 35.3147**-1
    m32L = 1e3
    L2m3 = 1e-3
    ft32L = 28.3168
    L2ft3 = 28.3168**-1

    @staticmethod
    def convert(value_in, unit_in, unit_out):
        """
        Convert a value from one unit to another.
        """

        conversion_key = f"{unit_in}2{unit_out}"

        conversion = getattr(Units, conversion_key)

        if not callable(conversion):
            return value_in * conversion

        return conversion(value_in)

    @staticmethod
    def convertWeightPercentage2moles(
        listSpecies,
        weightPercentage,
        database,
    ):
        """
        Convert weight percentage (wt%) to moles.
        """

        if not isinstance(listSpecies, list):
            listSpecies = [listSpecies]

        W = database.getProperty(listSpecies, "W") * 1e3

        moles = weightPercentage / W

        return moles

    @staticmethod
    def convertData2VelocityField(data):

        from combustiontoolbox.turbulence.VelocityField import VelocityField

        if isinstance(data, VelocityField):
            return data

        if (
            isinstance(data, dict)
            and all(field in data for field in ("u", "v", "w"))
        ):
            return VelocityField(
                data["u"],
                data["v"],
                data["w"],
            )

        if (
            hasattr(data, "ndim")
            and data.ndim == 4
            and data.shape[3] == 3
        ):
            return VelocityField(
                data[:, :, :, 0],
                data[:, :, :, 1],
                data[:, :, :, 2],
            )

        raise ValueError(
            "Unsupported velocity input format. "
            "Expected a VelocityField object, "
            "a dict, or a 4D matrix."
        )