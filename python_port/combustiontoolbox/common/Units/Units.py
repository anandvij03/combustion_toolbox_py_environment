import numpy as np

class Units:
    """
    Class with conversion factors between different units
    """

    # Pressure conversion factors
    atm2bar = 1.01325      # Atmospheres to bar
    bar2atm = 1.01325**-1  # Bar to atmospheres
    atm2Pa  = 101325       # Atmospheres to Pascals
    Pa2atm  = 101325**-1   # Pascals to atmospheres
    bar2Pa  = 1e5          # Bar to Pascals
    Pa2bar  = 1e-5         # Pascals to bar

    # Temperature conversion factors
    K2C     = staticmethod(lambda x: x - 273.15)      # Kelvin to degrees Celsius
    C2K     = staticmethod(lambda x: x + 273.15)      # Degrees Celsius to Kelvin
    F2C     = staticmethod(lambda x: (x - 32) * 5/9)  # Fahrenheit to degrees Celsius
    K2F     = staticmethod(lambda x: (x - 273.15) * 9/5 + 32) # Kelvin to Fahrenheit

    # Mass conversion factors
    kg2lbs  = 2.20462      # Kilograms to pounds
    lbs2kg  = 0.453592     # Pounds to kilograms
    kg2g    = 1e3          # Kilograms to grams
    g2kg    = 1e-3         # Grams to kilograms

    # Volume conversion factors
    m32ft3  = 35.3147      # Cubic meters to cubic feet
    ft32m3  = 35.3147**-1  # Cubic feet to cubic meters
    m32L    = 1e3          # Cubic meters to liters
    L2m3    = 1e-3         # Liters to cubic meters
    ft32L   = 28.3168      # Cubic feet to liters
    L2ft3   = 28.3168**-1  # Liters to cubic feet

    @staticmethod
    def convert(value_in, unit_in: str, unit_out: str):
        """
        Convert a value from one unit to another

        Args:
            value_in (float or array-like): Value to convert
            unit_in (str): Unit of the input value
            unit_out (str): Unit of the output value

        Returns:
            value_out: Value converted to the output unit
        """
        conversion_key = f"{unit_in}2{unit_out}"

        if not hasattr(Units, conversion_key):
            raise ValueError(f"Conversion from {unit_in} to {unit_out} is not defined.")

        conversion = getattr(Units, conversion_key)

        if callable(conversion):
            return conversion(value_in)

        return value_in * conversion

    @staticmethod
    def convertWeightPercentage2moles(listSpecies, weightPercentage, database):
        """
        Convert weight percentage (wt%) to moles

        Args:
            listSpecies (str or list): List of species
            weightPercentage (float or array-like): Weight percentage of the species [%]
            database: Database object with custom thermodynamic polynomials

        Returns:
            moles (ndarray): Number of moles [mol]
        """
        if isinstance(listSpecies, str):
            listSpecies = [listSpecies]
        elif not isinstance(listSpecies, (list, tuple)):
            listSpecies = list(listSpecies)

        # Get molecular weight [g] of the species
        if hasattr(database, "getProperty"):
            W = np.array(database.getProperty(listSpecies, 'W')) * 1e3
        elif hasattr(database, "get_property"):
            W = np.array(database.get_property(listSpecies, 'W')) * 1e3
        else:
            raise AttributeError("Database object does not have a getProperty or get_property method.")

        # Convert weight percentage (wt%) to moles
        weight_pct = np.array(weightPercentage)
        return weight_pct / W

    @staticmethod
    def convertData2VelocityField(data):
        """
        Convert the input to a VelocityField object

        Args:
            data: Either a VelocityField object, a dict, or a 4D numpy array

        Returns:
            VelocityField: VelocityField object
        """
        # Import locally to avoid circular dependencies
        from combustiontoolbox.turbulence import VelocityField

        if isinstance(data, VelocityField):
            return data

        if isinstance(data, dict) and all(k in data for k in ('u', 'v', 'w')):
            return VelocityField(data['u'], data['v'], data['w'])

        if hasattr(data, 'u') and hasattr(data, 'v') and hasattr(data, 'w'):
            return VelocityField(data.u, data.v, data.w)

        if isinstance(data, np.ndarray) and data.ndim == 4 and data.shape[3] == 3:
            return VelocityField(data[:, :, :, 0],
                                 data[:, :, :, 1],
                                 data[:, :, :, 2])

        raise ValueError("Unsupported velocity input format. Expected a VelocityField object, a dict/struct, or a 4D matrix.")
