import re
import numpy as np
import copy

class ChemicalSystem:
  
    _thermo_cache = {
        'list_species': None,
        'cp': {}, 'h': {}, 's': {}, 'g': {}
    }

    def __init__(self, database, list_species=None, flag_burcat=False, flag_ion=False):
        # Public Properties
        self.species = {}
        self.listSpecies = list_species if list_species else []
        self.listElements = []
        self.stoichiometricMatrix = np.array([])
        self.molecularWeight = np.array([])
        self.phase = np.array([])
        self.formationEnthalpy = np.array([])
        self.formationInternalEnergy = np.array([])
        self.temperatureMin = np.array([])
        self.temperatureMax = np.array([])
        
        # Indice
        self.indexSpecies = []
        self.indexGas = []
        self.indexCondensed = []
        self.indexCryogenic = []
        self.indexIons = []
        self.indexReact = []
        self.indexFrozen = []
        
        # Predefined lists
        self.listSpeciesLean = ['CO2', 'H2O', 'N2', 'Ar', 'O2']
        self.listSpeciesRich = ['CO2', 'H2O', 'N2', 'Ar', 'CO', 'H2']
        self.listSpeciesSoot = ['N2', 'Ar', 'CO', 'H2', 'Cbgrb']
        
        # Flags
        self.FLAG_COMPLETE = False
        self.FLAG_BURCAT = flag_burcat
        self.FLAG_ION = flag_ion
        self.FLAG_CONDENSED = True
        
        # Hidden/Private Properties
        self.database = database
        self.listProducts = []
        self.oxidizerReferenceIndex = None
        self.oxidizerReferenceAtomsO = np.nan
        self.ind_C = self.ind_H = self.ind_O = self.ind_N = None
        self.ind_E = self.ind_S = self.ind_Si = self.ind_B = None
        
        self._listSpeciesFormula = []
        self._FLAG_INITIALIZE = True
        
        if not self.listSpecies:
            self._FLAG_INITIALIZE = False
            return
            
        self.initialization()

    # Dependent Properties
    @property
    def numSpecies(self):
        return len(self.listSpecies)

    @property
    def numSpeciesGas(self):
        return len(self.indexGas)

    @property
    def numElements(self):
        return len(self.listElements)

    @property
    def indexElements(self):
        return list(range(self.numElements))

    # Core Methods
    def initialization(self):

        self.setSpecies(self.database)
        self.setContainedElements()
        self.sortListSpecies()
        self.setStoichiometricMatrix()
        self.setStaticSpeciesProperties()
        self.listProducts = copy.deepcopy(self.listSpecies)
        return self

    def checkSpecies(self, species_list):
        flag_added = False
        
        if not self._FLAG_INITIALIZE:
            for sp in species_list:
                if sp not in self.listSpecies:
                    self.species[sp] = self.database.species[sp]
                    self.listSpecies.append(sp)
            
            self.initialization()
            return self

        for sp in species_list:
            if sp not in self.listSpecies:
                self.species[sp] = self.database.species[sp]
                self.listSpecies.append(sp)
                self._listSpeciesFormula.append(self.species[sp].formula)
                flag_added = True

        if flag_added:
            self.setContainedElements()
            self.sortListSpecies()
            self.setStoichiometricMatrix()
            self.setStaticSpeciesProperties()
        return self

    def setSpecies(self, database):
        for sp in reversed(self.listSpecies):
            self.species[sp] = database.species[sp]
        return self

    # Thermodynamic Evaluators

    def _update_thermo_cache(self):
        if self._thermo_cache['list_species'] != self.listSpecies:
            self._thermo_cache['list_species'] = copy.deepcopy(self.listSpecies)
            for i, sp in enumerate(self.listSpecies):
                obj = self.species[sp]
                self._thermo_cache['cp'][i] = obj.cpcurve
                self._thermo_cache['h'][i] = obj.h0curve
                self._thermo_cache['s'][i] = obj.s0curve
                self._thermo_cache['g'][i] = obj.g0curve

    def evaluateSpeciesThermo(self, T, index=None):
        if index is None:
            index = list(range(self.numSpecies))
        T = np.atleast_1d(T)
        self._update_thermo_cache()
        
        num_T = len(T)
        thermo = {
            'index': index,
            'enthalpy': np.zeros((len(index), num_T)),
            'heatCapacityPressure': np.zeros((len(index), num_T)),
            'entropy': np.zeros((len(index), num_T)),
            'gibbs': np.zeros((len(index), num_T))
        }
        
        for idx, i in enumerate(index):
            thermo['enthalpy'][idx, :] = self._thermo_cache['h'][i](T)
            thermo['heatCapacityPressure'][idx, :] = self._thermo_cache['cp'][i](T)
            thermo['entropy'][idx, :] = self._thermo_cache['s'][i](T)
            thermo['gibbs'][idx, :] = self._thermo_cache['g'][i](T)
        return thermo


    @staticmethod
    def clearThermoCache():
        ChemicalSystem._thermo_cache = {'list_species': None, 'cp': {}, 'h': {}, 's': {}, 'g': {}}

    # Utilities and Setup
    def sortListSpecies(self):
        self.setIndexPhaseSpecies()
        ordered_indices = self.indexGas + self.indexCondensed
        self.listSpecies = [self.listSpecies[i] for i in ordered_indices]
        self.sortIndexPhaseSpecies()
        return self

    def setContainedElements(self):
        elements = set()
        for i in range(self.numSpecies - 1, -1, -1):
            formula = self._listSpeciesFormula[i] if self._listSpeciesFormula else getattr(self.species[self.listSpecies[i]], 'formula', '')
            
            matches = re.findall(r'[A-Z][a-z]*', formula)
            elements.update(matches)
            
        self.listElements = sorted(list(elements))

        def find_idx(el):
            return self.listElements.index(el) if el in self.listElements else None

        self.ind_C = find_idx('C')
        self.ind_H = find_idx('H')
        self.ind_O = find_idx('O')
        self.ind_N = find_idx('N')
        self.ind_E = find_idx('E')
        self.ind_S = find_idx('S')
        self.ind_Si = find_idx('Si')
        self.ind_B = find_idx('B')
        return self

    def isIonized(self, species_list=None):
        sp_list = species_list if species_list else self.listSpecies
        return np.array([('minus' in sp or 'plus' in sp) and 'cyclominus' not in sp for sp in sp_list])

    def getCharges(self):
        if self.ind_E is not None and self.stoichiometricMatrix.size > 0:
            return -self.stoichiometricMatrix[:, self.ind_E]
        return np.zeros(self.numSpecies)

    def setIndexPhaseSpecies(self):
        self.indexGas, self.indexCondensed, self.indexCryogenic = [], [], []
        
        for i, sp in enumerate(self.listSpecies):
            species_obj = self.species[sp]
            if not getattr(species_obj, 'phase', 0):
                self.indexGas.append(i)
            else:
                self.indexCondensed.append(i)
                if not getattr(species_obj, 'Tintervals', True):
                    self.indexCryogenic.append(i)
                    
        self.indexIons = np.where(self.isIonized(self.listSpecies))[0].tolist()
        self.indexSpecies = self.indexGas + self.indexCondensed
        return self

    def sortIndexPhaseSpecies(self):
        self.setIndexPhaseSpecies()
        return self

    def setStoichiometricMatrix(self):
        A0 = np.zeros((self.numSpecies, self.numElements))
        for i, sp in enumerate(self.listSpecies):
            el_mat = self.species[sp].getElementMatrix(self.listElements)
            A0[i, el_mat[0, :].astype(int)] = el_mat[1, :]
        self.stoichiometricMatrix = A0
        return self

    def setStaticSpeciesProperties(self):
        self.formationEnthalpy = np.zeros(self.numSpecies)
        self.formationInternalEnergy = np.zeros(self.numSpecies)
        self.molecularWeight = np.zeros(self.numSpecies)
        self.phase = np.zeros(self.numSpecies)
        self.temperatureMin = np.zeros(self.numSpecies)
        self.temperatureMax = np.zeros(self.numSpecies)
        
        for i, sp in enumerate(self.listSpecies):
            species_obj = self.species[sp]
            self.formationEnthalpy[i] = getattr(species_obj, 'hf', 0)
            self.formationInternalEnergy[i] = getattr(species_obj, 'ef', 0)
            self.molecularWeight[i] = getattr(species_obj, 'W', 0)
            self.phase[i] = getattr(species_obj, 'phase', 0)
            
            T_arr = getattr(species_obj, 'T', [0, 0])
            self.temperatureMin[i] = T_arr[0]
            self.temperatureMax[i] = T_arr[-1]
            
        return self

    def findProducts(self, *args, **kwargs):
        from .findProducts import findProducts
        return findProducts(self, *args, **kwargs)

    def setListSpecies(self, *args, **kwargs):
        from .setListSpecies import setListSpecies
        return setListSpecies(self, *args, **kwargs)

    def _get_list_species(self, *args, **kwargs):
        from .setListSpecies import _get_list_species
        return _get_list_species(self, *args, **kwargs)

    def _get_formula(self, *args, **kwargs):
        from .setListSpecies import _get_formula
        return _get_formula(self, *args, **kwargs)

    def getIndexElements(self, list_species, max_elements=5):
        from combustiontoolbox.core.Elements.Elements import Elements
        elements_list = Elements().getElements()
        num_species = len(list_species)
        index_elements = np.zeros((num_species, max_elements))
        for i, sp in enumerate(list_species):
            species_obj = self.database.species[sp]
            temp = species_obj.getElementMatrix(elements_list)
            length = temp.shape[1]
            index_elements[i, :length] = temp[0, :]
        index_elements = -np.sort(-index_elements, axis=1)
        return index_elements
