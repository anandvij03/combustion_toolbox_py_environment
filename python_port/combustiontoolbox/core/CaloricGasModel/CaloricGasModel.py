from enum import IntEnum

class CaloricGasModel(IntEnum):

    PERFECT = 0
    THERMALLY_PERFECT = 1
    IMPERFECT = 2

    def setPerfect(model):
        return CaloricGasModel.PERFECT
    
    def setThermallyPerfect(model):
        return CaloricGasModel.THERMALLY_PERFECT
    
    def setImperfect(model):
        return CaloricGasModel.IMPERFECT
    
    def isPerfect(model) -> bool:
        return model == CaloricGasModel.PERFECT
    
    def isThermallyPerfect(model) -> bool:
        return model == CaloricGasModel.THERMALLY_PERFECT
    
    def isImperfect(model) -> bool:
        return model == CaloricGasModel.IMPERFECT
    
    @classmethod
    def fromFlag(cls, flag_tchem_frozen: bool, flag_frozen: bool):
    
        # Strict type checking to mimic MATLAB's ~islogical validation
        if not isinstance(flag_tchem_frozen, bool) or not isinstance(flag_frozen, bool):
            raise TypeError("Input flags must be boolean values.")

        # Map legacy flags to caloric model
        if flag_tchem_frozen and flag_frozen:
            raise ValueError("Incompatible flags: Both flag_tchem_frozen and flag_frozen cannot be True simultaneously.")
        elif flag_tchem_frozen:
            return cls.PERFECT
        elif flag_frozen:
            return cls.THERMALLY_PERFECT
        else:
            return cls.IMPERFECT