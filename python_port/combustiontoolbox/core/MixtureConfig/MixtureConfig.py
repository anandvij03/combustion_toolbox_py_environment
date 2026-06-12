class MixtureConfig:
    """
    Stores configuration settings for a Mixture object.
    """
    def __init__(self, **kwargs):
        # Default settings
        self.mintolDisplay = 1e-6
        self.compositionUnits = 'molar fraction'
        self.FLAG_COMPACT = True

        validUnits = {'mol', 'molar fraction', 'mass fraction'}

        # Parse inputs
        for key, value in kwargs.items():
            if key == 'mintolDisplay':
                if not (isinstance(value, (int, float)) and value > 0):
                    raise ValueError("mintolDisplay must be a positive number")
                self.mintolDisplay = value
            elif key == 'compositionUnits':
                if value not in validUnits:
                    raise ValueError(f"compositionUnits must be one of {validUnits}")
                self.compositionUnits = value
            elif key == 'FLAG_COMPACT':
                if not isinstance(value, bool):
                    raise ValueError("FLAG_COMPACT must be a boolean")
                self.FLAG_COMPACT = value
