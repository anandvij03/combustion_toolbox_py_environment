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

