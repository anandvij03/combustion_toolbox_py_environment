class DynamicStruct(dict):
    """
    A dictionary subclass that allows dot-notation attribute access.
    Matches MATLAB struct behavior.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'DynamicStruct' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(
                f"'DynamicStruct' object has no attribute '{name}'"
            )


def generate_DB_Theo() -> DynamicStruct:
    """
    Generate database for theoretical computation of the jump conditions
    of a diatomic species only considering dissociation.

    Returns:
        DB_Theo (DynamicStruct): Database with quantum data of several diatomic species
    """
    DB_Theo = DynamicStruct()

    # N2
    DB_Theo.N2 = DynamicStruct()
    DB_Theo.N2.Tr = 2.87
    DB_Theo.N2.Tv = 3390
    DB_Theo.N2.Td = 113000
    DB_Theo.N2.G = 4.0**2
    DB_Theo.N2.m = 2.3259 * 1e-26

    # O2
    DB_Theo.O2 = DynamicStruct()
    DB_Theo.O2.Tr = 2.08
    DB_Theo.O2.Tv = 2270
    DB_Theo.O2.Td = 59500
    DB_Theo.O2.G = (5.0**2) / 3
    DB_Theo.O2.m = 2.6567 * 1e-26

    # H2
    DB_Theo.H2 = DynamicStruct()
    DB_Theo.H2.Tr = 87.53
    DB_Theo.H2.Tv = 6338
    DB_Theo.H2.Td = 51973
    DB_Theo.H2.G = 2.0**2
    DB_Theo.H2.m = 0.16735 * 1e-26

    # I2
    DB_Theo.I2 = DynamicStruct()
    DB_Theo.I2.Tr = 0.0538
    DB_Theo.I2.Tv = 308
    DB_Theo.I2.Td = 17897
    DB_Theo.I2.G = 4.0**2
    DB_Theo.I2.m = 21.072 * 1e-26

    # F2
    DB_Theo.F2 = DynamicStruct()
    DB_Theo.F2.Tr = 1.27
    DB_Theo.F2.Tv = 1320
    DB_Theo.F2.Td = 18633
    DB_Theo.F2.G = 4.0**2
    DB_Theo.F2.m = 3.1548 * 1e-26

    # Cl2
    DB_Theo.Cl2 = DynamicStruct()
    DB_Theo.Cl2.Tr = 0.0346
    DB_Theo.Cl2.Tv = 2270
    DB_Theo.Cl2.Td = 28770
    DB_Theo.Cl2.G = 4.0**2
    DB_Theo.Cl2.m = 5.8871 * 1e-26

    return DB_Theo
