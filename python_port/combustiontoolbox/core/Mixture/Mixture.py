import copy
import numpy as np

class Mixture:
    """
    The Mixture class is used to store the properties of a chemical mixture.
    
    The Mixture object can be initialized as follows:
    
         mix = Mixture(chemicalSystem)
    
    This creates an instance of the Mixture class and initializes it with a predefined chemical system.
    
    See also: ChemicalSystem, Database, NasaDatabase
    """

    def __init__(self, chemicalSystem, T=300, p=1, eos=None, config=None):
        """
        Mixture constructor
        """
        
        if eos is None:
            from combustiontoolbox.core.EquationStateIdealGas import EquationStateIdealGas
            eos = EquationStateIdealGas()
            
        if config is None:
            from combustiontoolbox.core.MixtureConfig import MixtureConfig
            config = MixtureConfig()

        # Public properties
        self.T = T               # Temperature [K]
        self.p = p               # Pressure [bar]
        self.N = None            # Total number of moles [mol]
        self.hf = None           # Enthalpy of formation [J]
        self.ef = None           # Internal energy of formation [J]
        self.h = None            # Enthalpy [J]
        self.e = None            # Internal energy [J]
        self.g = None            # Gibbs energy [J] 
        self.s = None            # Entropy [J/K]
        self.cp = None           # Specific heat at constant pressure [J/K]
        self.cv = None           # Specific heat at constant volume [J/K]
        self.gamma = None        # Adiabatic index [-]
        self.gamma_s = None      # Adiabatic index [-]
        self.sound = None        # Speed of sound [m/s]
        self.s0 = None           # Entropy (frozen) [J/K]
        self.DhT = None          # Thermal enthalpy [J]
        self.DeT = None          # Thermal internal energy [J]
        self.Ds = None           # Entropy of mixing [J/K]
        self.rho = None          # Density [kg/m3]
        self.v = None            # Volume [m3]
        self.vSpecific = None    # Specific volume [m3/kg]
        self.W = None            # Molecular weight [kg/mol]
        self.MW = None           # Mean molecular weight [kg/mol]
        self.mi = None           # Mass mixture [kg]
        self.Xi = None           # Molar fractions [-]
        self.Yi = None           # Mass fractions [-]
        self.phase = None        # Phase vector [-]
        self.dVdT_p = None       # Dimensionless derivative of volume with respect to temperature at constant pressure [-]
        self.dVdp_T = None       # Dimensionless derivative of volume with respect to pressure at constant temperature [-]
        self.equivalenceRatio = None      # Equivalence ratio [-]
        self.equivalenceRatioSoot = None  # Theoretical equivalence ratio at which soot may appear [-]
        self.stoichiometricMoles = None   # Theoretical moles of the oxidizer of reference for a stoichiometric combustion
        self.percentageFuel = None        # Percentage of fuel in the mixture [%]
        self.fuelOxidizerMassRatio = None # Mass ratio of oxidizer to fuel [-]
        self.oxidizerFuelMassRatio = None # Mass ratio of fuel to oxidizer [-]
        self.natomElements = None         # Vector atoms of each element [-]
        self.natomElementsReact = None    # Vector atoms of each element without frozen species [-]
        
        self.chemicalSystem = chemicalSystem # Chemical system object
        self.equationState = eos          # Equation of State object
        
        self.u = None                     # Velocity module relative to the shock front [m/s]
        self.uShock = None                # Velocity module in the shock tube [m/s]
        self.uNormal = None               # Normal component of u [m/s]
        self.cjSpeed = None               # Chapman-Jouguet speed
        self.mach = None                  # Mach number [-]
        self.driveFactor = None           # Overdriven/Underdriven factor (detonations)
        self.beta = None                  # Wave angle [deg]
        self.theta = None                 # Deflection angle [deg]
        self.betaMin = None               # Minimum wave angle [deg]
        self.betaMax = None               # Maximum wave angle [deg]
        self.thetaMin = None              # Minimum eflection angle [deg]
        self.thetaMax = None              # Maximum deflection angle [deg]
        self.betaSonic = None             # Wave angle at the sonic point [deg]
        self.thetaSonic = None            # Deflection angle at the sonic point [deg]
        self.indexMin = None              # Index of the minimum deflection angle
        self.indexMax = None              # Index of the maximum deflection angle
        self.indexSonic = None            # Index of the sonic point
        self.polar = None                 # Properties of the polar solution
        self.areaRatio = None             # Area ratio = area_i / areaThroat
        self.areaRatioChamber = None      # Area ratio = areaChamber / areaThroat
        self.cstar = None                 # Characteristic velocity [m/s]
        self.cf = None                    # Thrust coefficient [-]
        self.I_sp = None                  # Specific impulse [s]
        self.I_vac = None                 # Vacuum impulse [s]
        self.eta = 0                      # Dilatational-to-solenoidal TKE ratio [-]
        self.chi = 0                      # Entropic–vortical correlation parameter [-]
        self.etaVorticity = 0             # Vorticity generated at the shock due to acoustic disturbances normalized by the upstream vorticity [-]
        self.lia = None                   # Properties for Linear Interaction Analysis (LIA)
        self.config = config              # Mixture configuration object

        # Private properties
        self._indexSpecies = None         # Index of the species (initial mixture)
        self._indexGas = None             # Index of the gas species (initial mixture)
        self._Tspecies = None             # Species-specific initial temperatures [K] (initial mixture)
        self._FLAG_TSPECIES = False       # Flag to indicate species-specific initial temperatures are defined (initial mixture)
        self._FLAG_VOLUME = False         # Flag to indicate specific volume is defined (initial mixture)

        # Hidden properties
        self.errorMoles = 0               # Relative error in the moles calculation [-]
        self.errorMolesIons = 0           # Relative error in the moles of ions calculation [-]
        self.errorProblem = 0             # Relative error in the problem [-]
        self.cp_f = None                  # Frozen component of the specific heat at constant pressure
        self.cv_f = None                  # Frozen component of the specific heat at constant volume
        self.dNi_T = None                 # Partial derivative of the number of moles with respect to temperature
        self.dN_T = None                  # Partial derivative of the total number of moles with respect to temperature
        self.dNi_p = None                 # Partial derivative of the number of moles with respect to pressure
        self.dN_p = None                  # Partial derivative of the total number of moles with respect to pressure
        self.problemType = None           # Problem type
        self.rangeName = None             # Parametric property name
        self.quantity = None              # Composition (initial mixture)
        self.numSpecies = None            # Number of species (initial mixture)
        self.listSpecies = None           # List of species (initial mixture)
        self.listSpeciesFuel = None       # List of species fuel (initial mixture)
        self.listSpeciesOxidizer = None   # List of species oxidizer (initial mixture)
        self.listSpeciesInert = None      # List of species inert (initial mixture)
        self.molesFuel = None             # Moles of fuel (initial mixture)
        self.molesOxidizer = None         # Moles of oxidizer (initial mixture)
        self.molesInert = None            # Moles of inert (initial mixture)
        self.ratioOxidizer = None         # Ratio oxidizer relative to the oxidizer of reference (initial mixture)
        self.fuel = None                  # Fuel atoms (initial mixture)
        self.FLAG_FUEL = False            # Flag to indicate fuel species are defined (initial mixture)
        self.FLAG_OXIDIZER = False        # Flag to indicate oxidizer species are defined (initial mixture)
        self.FLAG_INERT = False           # Flag to indicate inert species are defined (initial mixture)
        self.FLAG_REACTION = False        # Flag to indicate chemical reaction is defined

        # SetAccess = private, Hidden properties
        self._productSpeciesSet = None    # Product species set with ChemicalSystem data and solver-local indices

        # Access = private, Hidden properties
        self._systemMoles = None          # Composition in ChemicalSystem species order [mol]
        self._systemMolesFuel = None      # Fuel moles in ChemicalSystem species order [mol]
        self._systemMolesOxidizer = None  # Oxidizer moles in ChemicalSystem species order [mol]
        self._indexProducts = None        # Product species indices in ChemicalSystem species order
        self._equilibriumSolver_ = None   # Equilibrium solver object
        self._listSpecies_ = None         # Original immutable species list (initial mixture)

    @property
    def productSpeciesSet(self):
        return self._productSpeciesSet

    @property
    def equilibriumSolver(self):
        """Get equilibrium solver object"""
        if self._equilibriumSolver_ is None:
            from combustiontoolbox.core.CaloricGasModel import CaloricGasModel
            from combustiontoolbox.equilibrium.EquilibriumSolver import EquilibriumSolver
            
            caloricGasModel = CaloricGasModel.thermallyPerfect
            self._equilibriumSolver_ = EquilibriumSolver(caloricGasModel=caloricGasModel, FLAG_RESULTS=False)
        return self._equilibriumSolver_

    @equilibriumSolver.setter
    def equilibriumSolver(self, value):
        """Set equilibrium solver object"""
        self._equilibriumSolver_ = value

    @property
    def N_gas(self):
        """Get total number of moles of gas species [mol]"""
        if self.N is not None and self.Xi is not None and self.phase is not None:
            return self.N * np.sum(self.Xi[~self.phase])
        return None

    @property
    def cp_r(self):
        """Get reactive component of the specific heat at constant pressure [J/K]"""
        if self.cp is not None and self.cp_f is not None:
            return self.cp - self.cp_f
        return None

    @property
    def cv_r(self):
        """Get reactive component of the specific heat at constant volume [J/K]"""
        if self.cv is not None and self.cv_f is not None:
            return self.cv - self.cv_f
        return None

    @property
    def gamma_f(self):
        """Get frozen specific heat ratio [-]"""
        if self.cp_f is not None and self.cv_f is not None:
            return self.cp_f / self.cv_f
        return None

    @property
    def cpSpecific(self):
        """Get mass specific heat at constant pressure [J/(kg-K)]"""
        if self.cp is not None and self.mi is not None:
            return self.cp / self.mi
        return None

    @property
    def cvSpecific(self):
        """Get mass specific heat at constant volume [J/(kg-K)]"""
        if self.cv is not None and self.mi is not None:
            return self.cv / self.mi
        return None

    @property
    def hSpecific(self):
        """Get mass specific enthalpy [J/kg]"""
        if self.h is not None and self.mi is not None:
            return self.h / self.mi
        return None

    @property
    def eSpecific(self):
        """Get mass specific internal energy [J/kg]"""
        if self.e is not None and self.mi is not None:
            return self.e / self.mi
        return None

    @property
    def gSpecific(self):
        """Get mass specific Gibbs energy [J/kg]"""
        if self.g is not None and self.mi is not None:
            return self.g / self.mi
        return None

    @property
    def sSpecific(self):
        """Get mass specific entropy [J/(kg-K)]"""
        if self.s is not None and self.mi is not None:
            return self.s / self.mi
        return None

    @property
    def dPdV_T(self):
        """Get dimensionless derivative of pressure with respect to volume at constant temperature [-]"""
        if self.equationState is not None and self.N_gas is not None and self.v is not None:
            dPdV_T, _ = self.equationState.getPressureDerivatives(self.T, self.p * 1e5, self.v / self.N_gas, self.Xi, self.chemicalSystem)
            return dPdV_T
        return None

    @property
    def dPdT_V(self):
        """Get dimensionless derivative of pressure with respect to temperature at constant volume [-]"""
        if self.equationState is not None and self.N_gas is not None and self.v is not None:
            _, dPdT_V = self.equationState.getPressureDerivatives(self.T, self.p * 1e5, self.v / self.N_gas, self.Xi, self.chemicalSystem)
            return dPdT_V
        return None

    def setTemperature(self, T, units='K'):
        """
        Set temperature [K] and compute thermodynamic properties
        """
        if units.lower() != 'k':
            from combustiontoolbox.common.Units import Units
            T = Units.convert(T, units, 'K')

        self.T = T
        self.updateThermodynamics()
        return self

    def setPressure(self, p, units='bar'):
        """
        Set pressure [bar] and compute thermodynamic properties
        """
        if units.lower() != 'bar':
            from combustiontoolbox.common.Units import Units
            p = Units.convert(p, units, 'bar')

        self.p = p
        self.updateThermodynamics()
        return self

    def setVolume(self, vSpecific, units='m3/kg'):
        """
        Set specific volume [m3/kg] and compute thermodynamic properties
        """
        if units.lower() != 'm3/kg':
            from combustiontoolbox.common.Units import Units
            vSpecific = Units.convert(vSpecific, units, 'm3/kg')

        self.vSpecific = vSpecific
        self._FLAG_VOLUME = True
        self.updateThermodynamics()
        return self

    def setEntropy(self, entropy, units='J/K'):
        """
        Set entropy [J/K] and compute the corresponding temperature at fixed p, composition
        """
        if units.lower() != 'j/k':
            raise ValueError('Only entropy in [J/K] is currently supported')

        solver = self.equilibriumSolver
        solver.problemType = 'SP'
        
        self.s = entropy
        solver.solve(self)
        return self

    def setEntropySpecific(self, entropySpecific, units='J/kg-K'):
        """
        Set specific entropy [J/kg-K] and compute the corresponding temperature at fixed p, composition
        """
        if units.lower() != 'j/kg-k':
            raise ValueError('Only specific entropy in [J/kg-K] is currently supported')

        entropy = entropySpecific * self.mi
        return self.setEntropy(entropy)

    def setEnthalpy(self, enthalpy, units='J'):
        """
        Set enthalpy [J] and compute the corresponding temperature at fixed p, composition
        """
        if units.lower() != 'j':
            raise ValueError('Only enthalpy in [J] is currently supported')

        solver = self.equilibriumSolver
        solver.problemType = 'HP'

        self.h = enthalpy
        solver.solve(self)
        return self

    def setEnthalpySpecific(self, enthalpySpecific, units='J/kg'):
        """
        Set specific enthalpy [J/kg] and compute the corresponding temperature at fixed p, composition
        """
        if units.lower() != 'j/kg':
            raise ValueError('Only specific enthalpy in [J/kg] is currently supported')

        enthalpy = enthalpySpecific * self.mi
        return self.setEnthalpy(enthalpy)

    def setInternalEnergy(self, internalEnergy, units='J'):
        """
        Set internal energy [J] and compute the corresponding temperature at fixed p, composition
        """
        if units.lower() != 'j':
            raise ValueError('Only internal energy in [J] is currently supported')

        solver = self.equilibriumSolver
        solver.problemType = 'EV'

        self.e = internalEnergy
        solver.solve(self)
        return self

    def setInternalEnergySpecific(self, internalEnergySpecific, units='J/kg'):
        """
        Set specific internal energy [J/kg] and compute the corresponding temperature at fixed p, composition
        """
        if units.lower() != 'j/kg':
            raise ValueError('Only specific internal energy in [J/kg] is currently supported')

        internalEnergy = internalEnergySpecific * self.mi
        return self.setInternalEnergy(internalEnergy)

    def set(self, listSpecies, type_str=None, quantity=1, units='mol'):
        """
        Set species and quantity and compute thermodynamic properties
        """
        if isinstance(listSpecies, str):
            listSpecies = [listSpecies]
            
        if type_str is not None:
            if units.lower() == 'weightpercentage':
                from combustiontoolbox.common.Units import Units
                quantity = Units.convertWeightPercentage2moles(listSpecies, quantity, self.chemicalSystem.database)
                
            type_lower = type_str.lower()
            if type_lower == 'fuel':
                if self.listSpeciesFuel is None: self.listSpeciesFuel = []
                if self.molesFuel is None: self.molesFuel = []
                self.listSpeciesFuel.extend(listSpecies)
                if isinstance(quantity, (list, tuple)):
                    self.molesFuel.extend(quantity)
                else:
                    self.molesFuel.append(quantity)
                self.FLAG_FUEL = True
            elif type_lower == 'oxidizer':
                if self.listSpeciesOxidizer is None: self.listSpeciesOxidizer = []
                if self.molesOxidizer is None: self.molesOxidizer = []
                self.listSpeciesOxidizer.extend(listSpecies)
                if isinstance(quantity, (list, tuple)):
                    self.molesOxidizer.extend(quantity)
                else:
                    self.molesOxidizer.append(quantity)
                if self.ratioOxidizer is None:
                    self.ratioOxidizer = list(self.molesOxidizer)
                self.FLAG_OXIDIZER = True
            elif type_lower == 'inert':
                if self.listSpeciesInert is None: self.listSpeciesInert = []
                if self.molesInert is None: self.molesInert = []
                self.listSpeciesInert.extend(listSpecies)
                if isinstance(quantity, (list, tuple)):
                    self.molesInert.extend(quantity)
                else:
                    self.molesInert.append(quantity)
                self.FLAG_INERT = True

        if self.listSpecies is None: self.listSpecies = []
        if self.quantity is None: self.quantity = []
        
        self.listSpecies.extend(listSpecies)
        if isinstance(quantity, (list, tuple)):
            self.quantity.extend(quantity)
        else:
            self.quantity.append(quantity)
        self.numSpecies = len(self.listSpecies)

        self._listSpecies_ = list(self.listSpecies)

        self.chemicalSystem.checkSpecies(listSpecies)
        
        self.updateIndexSpecies()

        self.chemicalSystem.setReactIndex(self.listSpeciesInert)
        
        self._indexProducts = self.getIndexProducts()
        self._productSpeciesSet = self.getProductSpeciesSet(self._indexProducts)
        
        if getattr(self.chemicalSystem, 'indexGas', None) is not None:
            gas_species = [self.chemicalSystem.listSpecies[i] for i in self.chemicalSystem.indexGas]
            self._indexGas = [i for i, sp in enumerate(self.listSpecies) if sp in gas_species]
        else:
            self._indexGas = []

        self.updateComposition()
        self.updateThermodynamics()
        
        return self

    def setEquivalenceRatio(self, equivalenceRatio):
        """
        Set equivalence ratio and compute thermodynamic properties
        """
        self.equivalenceRatio = equivalenceRatio
        self.updateComposition()
        self.updateThermodynamics()
        return self

    def setTemperatureSpecies(self, speciesTemperatures):
        """
        Set species-specific temperatures and update equilibrium temperature and properties
        """
        if len(speciesTemperatures) != self.numSpecies:
            raise ValueError(f'Temperature input must be either a scalar or a vector of length equal to the number of species ({self.numSpecies}).')

        self._FLAG_TSPECIES = True
        self._Tspecies = speciesTemperatures

        self.T = self.computeEquilibriumTemperature(speciesTemperatures)
        return self

    def computeEquivalenceRatio(self):
        """
        Compute equivalence ratio [-]
        """
        if not self.FLAG_FUEL or not self.FLAG_OXIDIZER:
            return self

        self.chemicalSystem.setOxidizerReference(self.listSpeciesOxidizer)
        
        self.defineF()
        self.defineO()
        
        self.computeRatiosFuelOxidizer(self._systemMolesFuel, self._systemMolesOxidizer)
        return self

    def computeRatiosFuelOxidizer(self, molesFuel, molesOxidizer):
        """
        Compute percentage Fuel, Oxidizer/Fuel ratio and equivalence ratio
        """
        if self.FLAG_FUEL and self.FLAG_OXIDIZER:
            mass_fuel = self.getMass(self.chemicalSystem, molesFuel)
            mass_oxidizer = self.getMass(self.chemicalSystem, molesOxidizer)
            mass_mixture = self.mi
            
            self.percentageFuel = mass_fuel / mass_mixture * 100 if mass_mixture else 0
            
            self.oxidizerFuelMassRatio = mass_oxidizer / mass_fuel if mass_fuel else float('inf')
            self.fuelOxidizerMassRatio = 1.0 / self.oxidizerFuelMassRatio if self.oxidizerFuelMassRatio else float('inf')
            
            FO_moles = np.sum(molesFuel) / np.sum(np.array(molesOxidizer)[self.chemicalSystem.oxidizerReferenceIndex])
            
            fuel_C = getattr(self.fuel, 'C', 0)
            fuel_H = getattr(self.fuel, 'H', 0)
            fuel_O = getattr(self.fuel, 'O', 0)
            fuel_S = getattr(self.fuel, 'S', 0)
            fuel_Si = getattr(self.fuel, 'Si', 0)
            fuel_B = getattr(self.fuel, 'B', 0)
            
            FO_moles_st = abs(np.sum(molesFuel) / (fuel_C + fuel_H / 4 - fuel_O / 2 + fuel_S + fuel_Si + 0.75 * fuel_B) * (0.5 * self.chemicalSystem.oxidizerReferenceAtomsO))
            
            self.equivalenceRatio = FO_moles / FO_moles_st
            self.computeEquivalenceRatioSoot()
            
        elif self.FLAG_FUEL:
            self.percentageFuel = 100
            self.fuelOxidizerMassRatio = float('inf')
            self.oxidizerFuelMassRatio = 0
            self.equivalenceRatio = None
            self.equivalenceRatioSoot = None
        else:
            self.percentageFuel = 0
            self.fuelOxidizerMassRatio = 0
            self.oxidizerFuelMassRatio = float('inf')
            self.equivalenceRatio = None
            self.equivalenceRatioSoot = None

        return self

    def computeEquivalenceRatioSoot(self):
        """
        Compute guess of equivalence ratio in which soot appears considering complete combustion
        """
        fuel_C = getattr(self.fuel, 'C', 0)
        fuel_H = getattr(self.fuel, 'H', 0)
        fuel_O = getattr(self.fuel, 'O', 0)
        
        self.equivalenceRatioSoot = 2 / (fuel_C - fuel_O) * (fuel_C + fuel_H / 4 - fuel_O / 2) if (fuel_C - fuel_O) != 0 else float('inf')
        
        if self.equivalenceRatioSoot <= 1e-5 or np.isnan(self.equivalenceRatioSoot):
            self.equivalenceRatioSoot = float('inf')
            
        return self

    def setProperties(self, property_name, value, *args):
        """
        Obtain properties at equilibrium for the given thermochemical transformation
        """
        import copy
        
        FLAG_MACH = False
        FLAG_ENTROPY = False
        FLAG_ENTHALPY = False
        FLAG_INTERNAL_ENERGY = False

        properties = [property_name]
        values = [value]
        
        for i in range(0, len(args), 2):
            properties.append(args[i])
            values.append(args[i+1])

        # Reorder if equivalence ratio is not the first property
        index = -1
        for i, prop in enumerate(properties):
            if prop.lower() in ['equivalenceratio', 'phi']:
                index = i
                break
                
        if index != -1 and index != 0:
            properties.insert(0, properties.pop(index))
            values.insert(0, values.pop(index))
            
        numProperties = min(len(properties), len(values))

        # Check vectors
        FLAG_VECTOR = [isinstance(v, (list, tuple, np.ndarray)) and len(np.atleast_1d(v)) > 1 for v in values]
        
        if any(FLAG_VECTOR):
            FLAG_VECTOR_FIRST = FLAG_VECTOR.index(True)
            aux = np.ones(len(values[FLAG_VECTOR_FIRST]))
            
            for i, is_vec in enumerate(FLAG_VECTOR):
                if not is_vec:
                    values[i] = np.array(values[i]) * aux
                    
            self.rangeName = properties[FLAG_VECTOR_FIRST]
            
            valid_range_names = ['t', 'p', 'vspecific', 'phi', 'u', 'mach', 'beta', 'theta', 'drive_factor', 'aratio', 'aratio_c', 'compressibility']
            if self.rangeName.lower() not in valid_range_names:
                rn_lower = self.rangeName.lower()
                if rn_lower == 'temperature': self.rangeName = 'T'
                elif rn_lower == 'pressure': self.rangeName = 'p'
                elif rn_lower in ['volume', 'v']: self.rangeName = 'vSpecific'
                elif rn_lower == 'phi': self.rangeName = 'equivalenceRatio'
                elif rn_lower in ['velocity', 'u1']: self.rangeName = 'u'
                elif rn_lower == 'm1': self.rangeName = 'mach'
                elif rn_lower in ['wave angle', 'waveangle', 'wave']: self.rangeName = 'beta'
                elif rn_lower in ['deflection angle', 'deflectionangle', 'deflection']: self.rangeName = 'theta'
                elif rn_lower == 'drive_factor': self.rangeName = 'driveFactor'
                elif rn_lower == 'aratio': self.rangeName = 'areaRatio'
                elif rn_lower == 'aratio_c': self.rangeName = 'areaRatioChamber'
                elif rn_lower == 'compressibility': self.rangeName = 'eta'
        else:
            FLAG_VECTOR_FIRST = 0
            values = [np.atleast_1d(v) for v in values]

        numCases = len(values[FLAG_VECTOR_FIRST])
        
        objArray = []

        for j in range(numCases):
            obj_copy = copy.copy(self)
            
            for i in range(numProperties):
                prop_lower = properties[i].lower()
                val = values[i][j]
                
                if prop_lower in ['temperature', 't']:
                    obj_copy.T = val
                elif prop_lower in ['pressure', 'p']:
                    obj_copy.p = val
                    obj_copy._FLAG_VOLUME = False
                elif prop_lower in ['volume', 'vspecific', 'v']:
                    obj_copy.vSpecific = val
                    obj_copy._FLAG_VOLUME = True
                elif prop_lower in ['entropy', 's', 's0']:
                    obj_copy.s = val
                    FLAG_ENTROPY = True
                elif prop_lower in ['entropyspecific', 'sspecific', 'smass']:
                    obj_copy.s = val * obj_copy.mi
                    FLAG_ENTROPY = True
                elif prop_lower in ['enthalpy', 'h', 'h0']:
                    obj_copy.h = val
                    FLAG_ENTHALPY = True
                elif prop_lower in ['enthalpyspecific', 'hspecific', 'hmass']:
                    obj_copy.h = val * obj_copy.mi
                    FLAG_ENTHALPY = True
                elif prop_lower in ['internalenergy', 'e', 'e0']:
                    obj_copy.e = val
                    FLAG_INTERNAL_ENERGY = True
                elif prop_lower in ['internalenergyspecific', 'especific', 'emass']:
                    obj_copy.e = val * obj_copy.mi
                    FLAG_INTERNAL_ENERGY = True
                elif prop_lower in ['equivalenceratio', 'phi']:
                    obj_copy.equivalenceRatio = val
                    obj_copy.updateComposition()
                elif prop_lower in ['velocity', 'u', 'u1']:
                    obj_copy.u = val
                elif prop_lower in ['mach', 'm1']:
                    obj_copy.mach = val
                    obj_copy.u = None
                    FLAG_MACH = True
                elif prop_lower in ['wave angle', 'waveangle', 'wave', 'beta']:
                    obj_copy.beta = val
                elif prop_lower in ['deflection angle', 'deflectionangle', 'deflection', 'theta']:
                    obj_copy.theta = val
                elif prop_lower in ['drive_factor', 'drivefactor']:
                    obj_copy.driveFactor = val
                elif prop_lower in ['arearatio', 'aratio']:
                    obj_copy.areaRatio = val
                elif prop_lower in ['arearatiochamber', 'aratio_c']:
                    obj_copy.areaRatioChamber = val
                elif prop_lower in ['compressibility', 'eta']:
                    obj_copy.eta = val
                elif prop_lower == 'chi':
                    obj_copy.chi = val
                elif prop_lower == 'etavorticity':
                    obj_copy.etaVorticity = val
                else:
                    raise ValueError(f'Property not found: {properties[i]}')

            if FLAG_ENTROPY:
                obj_copy.setEntropy(obj_copy.s)
            elif FLAG_ENTHALPY:
                obj_copy.setEnthalpy(obj_copy.h)
            elif FLAG_INTERNAL_ENERGY:
                obj_copy.setInternalEnergy(obj_copy.e)
            else:
                obj_copy.updateThermodynamics()

            if FLAG_MACH:
                obj_copy.u = obj_copy.mach * obj_copy.sound

            objArray.append(obj_copy)

        return objArray

    def updateThermodynamics(self):
        """
        Update the thermodynamic state of the mixture
        """
        if not self.T or (not self.p and not self.vSpecific):
            return self

        currentMoles = self._systemMoles

        if currentMoles is not None and np.sum(currentMoles) > 0:
            self._systemMoles = currentMoles
        elif np.sum(self.quantity):
            self._systemMoles = self.buildSystemMoles(self.listSpecies, self.quantity, self._indexSpecies)
        else:
            return self

        self.computeComposition()
        self.computeThermodynamics()
        return self

    def updateComposition(self):
        """
        Update the composition of the mixture
        """
        if not np.sum(self.quantity):
            return self

        if self.FLAG_FUEL and self.FLAG_OXIDIZER:
            self.chemicalSystem.setOxidizerReference(self.listSpeciesOxidizer)
            
            self.defineF()
            
            if self.equivalenceRatio is not None:
                if self.ratioOxidizer is None:
                    self.ratioOxidizer = list(self.molesOxidizer)
                self.molesOxidizer = self.stoichiometricMoles / self.equivalenceRatio * np.array(self.ratioOxidizer)

            self.defineO()
            
            self.quantity = []
            if self.molesFuel is not None: self.quantity.extend(self.molesFuel)
            if self.molesOxidizer is not None: self.quantity.extend(self.molesOxidizer)
            if self.molesInert is not None: self.quantity.extend(self.molesInert)

        self.mergeDuplicateSpecies()

        self._systemMoles = self.buildSystemMoles(self.listSpecies, self.quantity, self._indexSpecies)

        self.computeComposition()
        self.computeEquivalenceRatio()

        self._indexProducts = self.getIndexProducts()
        self._productSpeciesSet = self.getProductSpeciesSet(self._indexProducts)
        
        return self

    def updateIndexSpecies(self):
        """
        Update index species in the mixture
        """
        from combustiontoolbox.utils.findIndex import findIndex
        self._indexSpecies = findIndex(self.chemicalSystem.listSpecies, self.listSpecies)
        return self

    def vSpecific2vMolar(self, vSpecific, moles, molesGas, index=None):
        """
        Compute molar volume [m3/mol] from specific volume [m3/kg]
        """
        if index is None:
            from combustiontoolbox.utils.findIndex import findIndex
            index = findIndex(self.chemicalSystem.listSpecies, self.listSpecies)
            
        MW = self.computeMeanMolecularWeight(moles, index)
        
        W = MW * np.sum(moles) / np.sum(molesGas)
        
        vMolar = vSpecific * W
        return vMolar

    def getTypeSpecies(self):
        """
        Create cell array with the type of species in the mixture
        """
        typeFuel = ['Fuel'] * len(self.listSpeciesFuel) if self.listSpeciesFuel else []
        typeOxidizer = ['Oxidizer'] * len(self.listSpeciesOxidizer) if self.listSpeciesOxidizer else []
        typeInert = ['Inert'] * len(self.listSpeciesInert) if self.listSpeciesInert else []
        
        return typeInert + typeOxidizer + typeFuel

    def getNumberDensity(self):
        """
        Compute total number density of the mixture
        """
        from combustiontoolbox.common.Constants import Constants
        NA = Constants.NA
        KB = Constants.KB
        pressure = self.p

        numberDensity = pressure / (KB * self.T) * 1e5
        return numberDensity

    def getSpeciesNumberDensity(self):
        """
        Compute species number density of the mixture
        """
        Xi = np.array(self.Xi) if self.Xi is not None else np.array([])
        numberDensity = self.getNumberDensity()
        
        speciesNumberDensity = Xi * numberDensity
        return speciesNumberDensity, numberDensity

    def getNeutralNumberDensity(self):
        """
        Compute neutral number density of the mixture
        """
        isIonized = np.array(self.chemicalSystem.isIonized)
        speciesNumberDensity, _ = self.getSpeciesNumberDensity()
        
        if len(speciesNumberDensity) == 0:
            return 0
            
        neutralNumberDensity = np.sum(speciesNumberDensity[~isIonized])
        return neutralNumberDensity

    def getElectronNumberDensity(self):
        """
        Compute electron number density of the mixture
        """
        from combustiontoolbox.utils.findIndex import findIndex
        indexElectron = findIndex(self.chemicalSystem.listSpecies, 'eminus')
        
        if indexElectron is None or (isinstance(indexElectron, list) and len(indexElectron) == 0):
            return 0
            
        speciesNumberDensity, _ = self.getSpeciesNumberDensity()
        idx = indexElectron[0] if isinstance(indexElectron, list) else indexElectron
        
        return speciesNumberDensity[idx]

    def getIonNumberDensity(self):
        """
        Compute ion number density of the mixture of charged heavy species (ions) excluding electrons
        """
        from combustiontoolbox.utils.findIndex import findIndex
        indexElectron = findIndex(self.chemicalSystem.listSpecies, 'eminus')
        
        if indexElectron is None or (isinstance(indexElectron, list) and len(indexElectron) == 0):
            return 0
            
        speciesNumberDensity, _ = self.getSpeciesNumberDensity()
        charges = np.array(self.chemicalSystem.getCharges())
        
        idx = indexElectron[0] if isinstance(indexElectron, list) else indexElectron
        electronNumberDensity = speciesNumberDensity[idx]
        
        ionNumberDensity = abs(np.sum(charges * speciesNumberDensity) - electronNumberDensity)
        return ionNumberDensity

    def getDegreeIonization(self):
        """
        Compute degree of ionization of the mixture
        """
        neutralNumberDensity = self.getNeutralNumberDensity()
        electronNumberDensity = self.getElectronNumberDensity()
        
        if (electronNumberDensity + neutralNumberDensity) == 0:
            return 0
            
        degreeIonization = electronNumberDensity / (electronNumberDensity + neutralNumberDensity)
        return degreeIonization

    def getDebyeLength(self):
        """
        Compute Debye length of the mixture
        """
        from combustiontoolbox.common.Constants import Constants
        import math
        
        epsilon0 = Constants.E0
        kB = Constants.KB
        e = Constants.E
        T = self.T
        electronNumberDensity = self.getElectronNumberDensity()
        
        if electronNumberDensity == 0:
            return float('inf')
            
        debyeLength = math.sqrt(epsilon0 * kB * T / (electronNumberDensity * e**2))
        return debyeLength

    def getElectronsDebyeSphere(self):
        """
        Compute number of electrons in Debye sphere
        """
        import math
        debyeLength = self.getDebyeLength()
        electronNumberDensity = self.getElectronNumberDensity()

        electronsDebyeSphere = 4 / 3 * math.pi * electronNumberDensity * debyeLength**3
        return electronsDebyeSphere

    def getPlasmaCoupling(self):
        """
        Compute the plasma coupling parameter of the mixture
        """
        from combustiontoolbox.common.Constants import Constants
        import math
        
        charges = np.array(self.chemicalSystem.getCharges())
        epsilon0 = Constants.E0
        electronNumberDensity = self.getElectronNumberDensity()
        speciesNumberDensity, _ = self.getSpeciesNumberDensity()
        kb = Constants.KB
        e = Constants.E
        T = self.T

        if electronNumberDensity == 0:
            return 0.0, np.zeros_like(charges)

        # Compute average inter-particle distance [m]
        distance = (3 / (4 * math.pi * electronNumberDensity))**(1/3)
        
        # Compute plasma coupling electron [-]
        plasmaCouplingElectron = e**2 / (4 * math.pi * epsilon0 * distance * kb * T)

        # Compute plasma coupling species [-]
        plasmaCouplingSpecies = plasmaCouplingElectron * np.abs(charges)**(5/3)

        sum_species = np.sum(speciesNumberDensity)
        if sum_species == 0:
            chargesAverage = 0
        else:
            # Compute average charge of ions
            chargesAverage = math.sqrt(np.sum(np.abs(charges)**(5/3) * speciesNumberDensity) / sum_species)

        # Compute plasma coupling [-]
        plasmaCoupling = plasmaCouplingElectron * chargesAverage
        
        return plasmaCoupling, plasmaCouplingSpecies

    def isWeaklyCoupledPlasma(self):
        """
        Check if the mixture is weakly coupled plasma
        """
        plasmaCoupling, _ = self.getPlasmaCoupling()
        value = plasmaCoupling < 0.2
        return value, plasmaCoupling

    # Private/Hidden Methods

    def buildSystemMoles(self, listSpecies, quantity, index=None):
        """
        Build a full composition vector in ChemicalSystem species order
        """
        system = self.chemicalSystem
        moles = np.zeros(system.numSpecies)

        if not listSpecies or not quantity:
            return moles

        if index is None:
            from combustiontoolbox.utils.findIndex import findIndex
            index = findIndex(system.listSpecies, listSpecies)

        quantity = np.array(quantity).flatten()
        index = np.array(index).flatten()
        
        # Adjust to 0-based index if index returned 1-based, assuming Python port has 0-based
        # If it returns 0-based, we can just use it directly
        for i in range(len(index)):
            idx = int(index[i])
            moles[idx] = moles[idx] + quantity[i]
            
        return moles

    def getIndexProducts(self):
        """
        Get product species indices in ChemicalSystem order
        """
        from combustiontoolbox.utils.findIndex import findIndex
        
        system = self.chemicalSystem
        listProducts = system.listProducts

        if system.FLAG_COMPLETE and self.equivalenceRatio is not None:
            equivalenceRatioSoot = self.equivalenceRatioSoot
            if equivalenceRatioSoot is None:
                equivalenceRatioSoot = float('inf')

            if self.equivalenceRatio < 1:
                listProducts = system.listSpeciesLean
            elif self.equivalenceRatio <= equivalenceRatioSoot:
                listProducts = system.listSpeciesRich
            else:
                listProducts = system.listSpeciesSoot

        if not listProducts:
            raise ValueError('Product species list cannot be empty')

        indexProducts = findIndex(system.listSpecies, listProducts)
        if len(indexProducts) != len(listProducts):
            raise ValueError('Product species must be included in ChemicalSystem')

        indexProducts = np.array(indexProducts)
        indexGas = np.array(system.indexGas)
        indexCondensed = np.array(system.indexCondensed)
        
        gas_mask = np.isin(indexGas, indexProducts)
        cond_mask = np.isin(indexCondensed, indexProducts)
        
        return np.concatenate((indexGas[gas_mask], indexCondensed[cond_mask]))

    def getProductSpeciesSet(self, indexProducts):
        """
        Get product species set with ChemicalSystem data and solver-local indices
        """
        system = self.chemicalSystem
        indexProducts = np.array(indexProducts).flatten()
        
        # Using integer arrays to index directly
        phase = np.array(system.phase)[indexProducts]

        productSpeciesSet = type('ProductSpeciesSet', (), {})()
        productSpeciesSet.indexGlobal = indexProducts
        productSpeciesSet.stoichiometricMatrix = np.array(system.stoichiometricMatrix)[indexProducts, :]
        productSpeciesSet.molecularWeight = np.array(system.molecularWeight)[indexProducts]
        productSpeciesSet.phase = phase
        productSpeciesSet.temperatureMin = np.array(system.temperatureMin)[indexProducts]
        productSpeciesSet.temperatureMax = np.array(system.temperatureMax)[indexProducts]
        productSpeciesSet.numSpecies = len(indexProducts)
        
        productSpeciesSet.indexGas = np.where(~phase)[0]
        productSpeciesSet.indexCondensed = np.where(phase)[0]
        
        # Note: system.indexCryogenic may be a list/array
        sys_cryo = np.array(system.indexCryogenic)
        productSpeciesSet.indexCryogenic = np.where(np.isin(indexProducts, sys_cryo))[0]
        
        sys_ions = np.array(system.indexIons)
        productSpeciesSet.indexIons = np.where(np.isin(indexProducts, sys_ions))[0]
        
        productSpeciesSet.indexSpecies = np.concatenate((productSpeciesSet.indexGas, productSpeciesSet.indexCondensed))
        
        return productSpeciesSet

    def computeProperties(self, *args):
        """
        Compute composition and thermodynamic properties of the mixture
        """
        self.computeComposition()
        self.computeThermodynamics(*args)
        return self

    def computeComposition(self):
        """
        Compute the composition of the mixture
        """
        system = self.chemicalSystem
        Ni = np.array(self._systemMoles)
        molecularWeight = np.array(system.molecularWeight)

        self.N = np.sum(Ni)
        self.phase = np.array(system.phase)

        N_gas = np.sum(Ni[~self.phase])
        
        if self.N > 0:
            self.Xi = Ni / self.N
        else:
            self.Xi = np.zeros_like(Ni)

        if N_gas > 0:
            self.W = np.dot(Ni, molecularWeight) / N_gas
        else:
            self.W = 0
            
        if self.N > 0:
            self.MW = np.dot(Ni, molecularWeight) / self.N
        else:
            self.MW = 0

        self.mi = self.MW * self.N

        if self.mi > 0:
            self.Yi = (Ni * molecularWeight) / self.mi
        else:
            self.Yi = np.zeros_like(Ni)

        self.natomElements = np.sum(Ni[:, np.newaxis] * np.array(system.stoichiometricMatrix), axis=0)

        # Compute vector atoms without frozen species
        indexReact = np.array(system.indexReact)
        if len(indexReact) > 0:
            st_mat_react = np.array(system.stoichiometricMatrix)[indexReact, :]
            self.natomElementsReact = np.sum(Ni[indexReact][:, np.newaxis] * st_mat_react, axis=0)
        else:
            self.natomElementsReact = np.zeros(np.array(system.stoichiometricMatrix).shape[1])

    def computeThermodynamics(self, speciesEnthalpy=None, speciesEnthalpyIndex=None):
        """
        Compute thermodynamic properties of the mixture
        """
        from combustiontoolbox.common.Units import Units
        from combustiontoolbox.common.Constants import Constants
        import math
        
        if getattr(self, 'FLAG_TSPECIES', False):
            self.setTemperatureSpecies(self.Tspecies)
            self.Tspecies = []
            self.FLAG_TSPECIES = False

        temperature = self.T
        pressure = self.p
        if pressure is not None:
            pressure_Pa = pressure * Units.bar2Pa
        else:
            pressure_Pa = None
            
        R0 = Constants.R0
        system = self.chemicalSystem
        Ni = np.array(self._systemMoles)

        active = np.where(Ni > 0)[0]
        h0 = np.zeros(system.numSpecies)
        cp0 = np.zeros(system.numSpecies)
        s0 = np.zeros(system.numSpecies)
        
        FLAG_SPECIES_ENTHALPY = speciesEnthalpy is not None
        FLAG_ALL_ACTIVE_ENTHALPY = False

        if FLAG_SPECIES_ENTHALPY:
            speciesEnthalpy = np.array(speciesEnthalpy).flatten()

            if speciesEnthalpyIndex is not None:
                speciesEnthalpyIndex = np.array(speciesEnthalpyIndex).flatten()
                
                if len(speciesEnthalpy) == system.numSpecies:
                    h0[speciesEnthalpyIndex] = speciesEnthalpy[speciesEnthalpyIndex]
                elif len(speciesEnthalpy) == len(speciesEnthalpyIndex):
                    h0[speciesEnthalpyIndex] = speciesEnthalpy
                else:
                    raise ValueError('speciesEnthalpy must match ChemicalSystem species or speciesEnthalpyIndex length.')

                if len(active) == 0 or np.array_equal(np.sort(speciesEnthalpyIndex), np.sort(active)) or np.all(np.isin(active, speciesEnthalpyIndex)):
                    FLAG_ALL_ACTIVE_ENTHALPY = True
            else:
                if len(speciesEnthalpy) != system.numSpecies:
                    raise ValueError('speciesEnthalpy must have one entry per ChemicalSystem species when speciesEnthalpyIndex is omitted.')
                h0 = speciesEnthalpy
                speciesEnthalpyIndex = []
                FLAG_ALL_ACTIVE_ENTHALPY = True

        if len(active) > 0 and FLAG_ALL_ACTIVE_ENTHALPY:
            cpActive, sActive = system.evaluateSpeciesThermoCPS(temperature, active)
            cp0[active] = cpActive[:, 0] if cpActive.ndim > 1 else cpActive
            s0[active] = sActive[:, 0] if sActive.ndim > 1 else sActive
        elif len(active) > 0 and FLAG_SPECIES_ENTHALPY:
            speciesEnthalpyMask = np.zeros(system.numSpecies, dtype=bool)
            if speciesEnthalpyIndex is not None and len(speciesEnthalpyIndex) > 0:
                speciesEnthalpyMask[speciesEnthalpyIndex] = True
            
            cpActive, sActive = system.evaluateSpeciesThermoCPS(temperature, active)
            cp0[active] = cpActive[:, 0] if cpActive.ndim > 1 else cpActive
            s0[active] = sActive[:, 0] if sActive.ndim > 1 else sActive
            
            activePendingEnthalpy = active[~speciesEnthalpyMask[active]]
            if len(activePendingEnthalpy) > 0:
                hActive = system.evaluateSpeciesThermoH(temperature, activePendingEnthalpy)
                h0[activePendingEnthalpy] = hActive[:, 0] if hActive.ndim > 1 else hActive
        elif len(active) > 0:
            hActive, cpActive, sActive = system.evaluateSpeciesThermoHCPS(temperature, active)
            h0[active] = hActive[:, 0] if hActive.ndim > 1 else hActive
            cp0[active] = cpActive[:, 0] if cpActive.ndim > 1 else cpActive
            s0[active] = sActive[:, 0] if sActive.ndim > 1 else sActive

        self.hf = np.dot(system.formationEnthalpy, Ni)
        self.h = np.dot(h0, Ni)
        self.ef = np.dot(system.formationInternalEnergy, Ni)
        self.cp = np.dot(cp0, Ni)
        self.s0_val = np.dot(s0, Ni) # s0 property used as s0_val avoiding conflict with property s

        N_gas = np.sum(Ni[~self.phase])
        FLAG_NONZERO = self.Xi > 0

        if self._FLAG_VOLUME:
            self.v = self.vSpecific * self.mi
            vMolar = self.v / N_gas if N_gas > 0 else 0
            self.p = self.equationState.getPressure(temperature, vMolar, self.Xi, self.chemicalSystem) * Units.Pa2bar
            pressure_Pa = self.p * Units.bar2Pa
        else:
            if pressure_Pa is not None:
                vMolar = self.equationState.getVolume(temperature, pressure_Pa, self.Xi, self.chemicalSystem)
                self.v = vMolar * N_gas
            else:
                self.v = None
                vMolar = None

        if self.mi > 0 and self.v is not None:
            self.vSpecific = self.v / self.mi
            self.rho = 1 / self.vSpecific
        else:
            self.vSpecific = None
            self.rho = None

        if pressure_Pa is not None and vMolar is not None:
            cp_dep_molar, h_dep_molar, s_dep_molar = self.equationState.getDepartureFunctions(
                temperature, pressure_Pa, vMolar, self.Xi, self.chemicalSystem)

            self.cp = self.cp + cp_dep_molar * N_gas
            self.h = self.h + h_dep_molar * N_gas
            self.s0_val = self.s0_val + s_dep_molar * N_gas
            
            self.e = self.h - pressure_Pa * self.v
        else:
            self.e = None

        if self.e is not None:
            self.DeT = self.e - self.ef
        else:
            self.DeT = None
            
        self.DhT = self.h - self.hf

        self.Ds = self.computeEntropyMixing(Ni, N_gas, R0, FLAG_NONZERO)

        self.s = self.s0_val + self.Ds
        
        if self.s is not None and temperature is not None:
            self.g = self.h - temperature * self.s
        else:
            self.g = None

        if pressure_Pa is not None and vMolar is not None:
            dVdT_p_frozen, dVdp_T_frozen = self.equationState.getVolumeDerivatives(
                temperature, pressure_Pa, vMolar, self.Xi, self.chemicalSystem)
        else:
            dVdT_p_frozen, dVdp_T_frozen = 0.0, 0.0

        self.cp_f = self.cp
        if temperature is not None and pressure_Pa is not None and self.v is not None and dVdp_T_frozen != 0:
            self.cv_f = self.cp_f + (pressure_Pa * self.v / temperature) * (dVdT_p_frozen**2) / dVdp_T_frozen
        else:
            self.cv_f = None

        if getattr(self, 'FLAG_REACTION', False):
            self.dVdT_p = getattr(self, 'dN_T', 0) + dVdT_p_frozen
            self.dVdp_T = getattr(self, 'dN_p', 0) + dVdp_T_frozen

            if hasattr(self, 'dNi_T') and self.dNi_T is not None and not np.any(np.isnan(self.dNi_T)) and not np.any(np.isinf(self.dNi_T)):
                delta = ~self.phase
                h0_j = h0
                
                term = h0_j / temperature * (1 + delta * (Ni - 1)) * self.dNi_T
                self.cp = self.cp_f + np.nansum(term)
                
                if self.dVdp_T != 0:
                    self.cv = self.cp + (pressure_Pa * self.v / temperature) * (self.dVdT_p**2) / self.dVdp_T
                else:
                    self.cv = None
                    
                if self.cv and self.cv != 0:
                    self.gamma = self.cp / self.cv
                else:
                    self.gamma = None
                    
                if self.dVdp_T != 0:
                    self.gamma_s = -self.gamma / self.dVdp_T if self.gamma is not None else None
                else:
                    self.gamma_s = None

                if self.gamma_s is not None and self.gamma_s >= 0:
                    self.sound = math.sqrt(self.gamma_s * pressure_Pa * self.vSpecific)
                else:
                    self.sound = None

                if self.u is not None and self.sound is not None and self.sound > 0:
                    self.mach = self.u / self.sound

                return self

            if self.dVdp_T != 0:
                self.gamma_s = -1 / self.dVdp_T
            else:
                self.gamma_s = None
                
            return self

        self.dVdT_p = dVdT_p_frozen
        self.dVdp_T = dVdp_T_frozen
        self.cv = self.cv_f

        if self.cv and self.cv != 0:
            self.gamma = self.cp / self.cv
        else:
            self.gamma = None
            
        if self.dVdp_T != 0 and self.gamma is not None:
            self.gamma_s = -self.gamma / self.dVdp_T
        else:
            self.gamma_s = None

        if self.gamma_s is not None and pressure_Pa is not None and self.vSpecific is not None and self.gamma_s >= 0:
            self.sound = math.sqrt(self.gamma_s * pressure_Pa * self.vSpecific)
        else:
            self.sound = None
            
        if self.u is not None and self.sound is not None and self.sound > 0:
            self.mach = self.u / self.sound

        return self

    def computeMeanMolecularWeight(self, moles, index):
        """
        Compute Mean Molecular Weight [kg/mol]
        """
        moles_arr = np.array(moles).flatten()
        mw_arr = np.array(self.chemicalSystem.molecularWeight)[index]
        return np.dot(moles_arr, mw_arr) / np.sum(moles_arr)

    def computeEntropyMixing(self, Ni, N_gas, R0, FLAG_NONZERO):
        """
        Compute entropy of mixing [J/K]
        """
        if not np.any(FLAG_NONZERO) or N_gas <= 0:
            return 0.0
            
        Ni_nnz = np.array(Ni)[FLAG_NONZERO]
        phase_nnz = np.array(self.phase)[FLAG_NONZERO]
        
        if self.p is None:
            return 0.0
            
        Dsi = Ni_nnz * np.log(Ni_nnz / N_gas * self.p) * (1 - phase_nnz)
        Ds = -R0 * np.sum(Dsi)
        return Ds

    def defineF(self):
        """
        Set Fuel of the mixture
        """
        if self.FLAG_FUEL:
            self.systemMolesFuel = self.buildSystemMoles(self.listSpeciesFuel, self.molesFuel)
            natomElementsFuel = np.sum(self.systemMolesFuel[:, np.newaxis] * np.array(self.chemicalSystem.stoichiometricMatrix), axis=0)
            self.assignAtomElementsFuel(natomElementsFuel)
            
            c = getattr(self.fuel, 'C', 0)
            h = getattr(self.fuel, 'H', 0)
            o = getattr(self.fuel, 'O', 0)
            s = getattr(self.fuel, 'S', 0)
            si = getattr(self.fuel, 'Si', 0)
            b = getattr(self.fuel, 'B', 0)
            
            self.stoichiometricMoles = abs(c + h/4 - o/2 + s + si + 3/4 * b) / (0.5 * self.chemicalSystem.oxidizerReferenceAtomsO)
            return self

        self.systemMolesFuel = np.zeros(self.chemicalSystem.numSpecies)
        self.fuel.C = 0
        self.fuel.H = 0
        self.fuel.O = 0
        self.fuel.N = 0
        self.fuel.S = 0
        self.fuel.Si = 0
        self.fuel.B = 0
        self.stoichiometricMoles = 0
        return self

    def defineO(self):
        """
        Set Oxidizer of the mixture
        """
        if not self.listSpeciesOxidizer:
            self.systemMolesOxidizer = np.zeros(self.chemicalSystem.numSpecies)
            return self

        self.systemMolesOxidizer = self.buildSystemMoles(self.listSpeciesOxidizer, self.molesOxidizer)
        return self

    def assignAtomElementsFuel(self, natomElementsFuel):
        """
        Assign atomic counts of the fuel
        """
        sys = self.chemicalSystem
        
        def assign_if_exists(ind, attr):
            if ind is not None and len(np.atleast_1d(ind)) > 0:
                idx = ind[0] if isinstance(ind, list) else ind
                # Adjust assuming Python 0-based index returned from system
                setattr(self.fuel, attr, natomElementsFuel[idx])
            else:
                setattr(self.fuel, attr, 0)
                
        assign_if_exists(sys.ind_C, 'C')
        assign_if_exists(sys.ind_H, 'H')
        assign_if_exists(sys.ind_O, 'O')
        assign_if_exists(sys.ind_N, 'N')
        assign_if_exists(sys.ind_S, 'S')
        assign_if_exists(sys.ind_Si, 'Si')
        assign_if_exists(sys.ind_B, 'B')
        return self

    def computeEquilibriumTemperature(self, speciesTemperatures):
        """
        Compute the equilibrium temperature [K]
        """
        speciesTemperatures = np.array(speciesTemperatures).flatten()
        tol0 = 1e-3
        itMax = 100
        
        if len(np.unique(speciesTemperatures)) == 1:
            return speciesTemperatures[0]

        FLAG_FIXED = self.checkTemperatureSpecies()
        if FLAG_FIXED:
            return np.max(speciesTemperatures)

        problemType = self.problemType if self.problemType else 'TP'
        
        pt_lower = problemType.lower()
        if pt_lower in ['tv', 'ev', 'sv', 'volume', 'v']:
            funCpOrCv = "getHeatCapacityVolume"
            funHorE   = "getInternalEnergy"
        else:
            funCpOrCv = "getHeatCapacityPressure"
            funHorE   = "getEnthalpy"

        CpOrCv_0 = self.getPropertyListSpecies(funCpOrCv, speciesTemperatures)
        HorE_0   = self.getPropertyListSpecies(funHorE, speciesTemperatures)
        
        qty = np.array(self.quantity)
        T = np.sum(qty * speciesTemperatures * CpOrCv_0) / np.sum(qty * CpOrCv_0)
        
        it = 0
        STOP = 1.0
        
        while STOP > tol0 and it < itMax:
            it += 1
            
            HorE = self.getPropertyListSpecies(funHorE, T)
            
            f = np.sum(qty * HorE_0) - np.sum(qty * HorE)
            f_rel = f / np.sum(qty * HorE)
            
            CpOrCv = self.getPropertyListSpecies(funCpOrCv, T)
            df = -np.sum(qty * CpOrCv)
            
            DeltaT = -f / df
            T = T + DeltaT
            
            STOP = max(abs(DeltaT), abs(f_rel))

        return T

    def getPropertyListSpecies(self, fun_name, temperatures):
        """
        Evaluate a given property function for each species
        """
        numSpecies = self.numSpecies
        if np.isscalar(temperatures):
            temperatures = np.full(numSpecies, temperatures)
        else:
            temperatures = np.array(temperatures)

        values = np.zeros(numSpecies)
        for i in range(numSpecies):
            species = getattr(self.chemicalSystem.database.species, self.listSpecies[i])
            method = getattr(species, fun_name)
            values[i] = method(temperatures[i])

        return values

    def mergeDuplicateSpecies(self):
        """
        Merge quantities for repeated species names
        """
        if not self.listSpecies_:
            return
            
        uniqueSpecies = []
        indices = []
        
        for i, s in enumerate(self.listSpecies_):
            if s not in uniqueSpecies:
                uniqueSpecies.append(s)
                indices.append(len(uniqueSpecies) - 1)
            else:
                indices.append(uniqueSpecies.index(s))
                
        if self.numSpecies == len(uniqueSpecies):
            return
            
        combinedQuantity = np.zeros(len(uniqueSpecies))
        for idx, qty in zip(indices, self.quantity):
            combinedQuantity[idx] += qty
            
        self.listSpecies = uniqueSpecies
        self.quantity = combinedQuantity.tolist()

    @staticmethod
    def getMass(system, moles):
        """
        Compute mass mixture [kg]
        """
        return np.dot(np.array(moles).flatten(), np.array(system.molecularWeight).flatten())

    def setMolesFast(self, moles, *args):
        """
        Set the local composition vector and compute thermodynamic properties
        """
        moles = np.array(moles).flatten()
        if len(moles) != self.chemicalSystem.numSpecies:
            raise ValueError('Moles vector must have one entry per ChemicalSystem species.')

        self._systemMoles = moles.tolist()
        self.computeProperties(*args)
        return self

    def checkTemperatureSpecies(self):
        """
        Check if condensed species can be only evaluated a particular temperature
        """
        FLAG_FIXED = False
        
        if not hasattr(self, 'Tspecies'):
            self.Tspecies = []
            
        for i in range(self.numSpecies):
            species_name = self.listSpecies[i]
            species = getattr(self.chemicalSystem.database.species, species_name)
            
            if np.isscalar(getattr(species, 'T', None)) and species.T is not None:
                # Padding list if needed
                while len(self.Tspecies) <= i:
                    self.Tspecies.append(None)
                self.Tspecies[i] = species.T
                FLAG_FIXED = True
                
        return FLAG_FIXED

