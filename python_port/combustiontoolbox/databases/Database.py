import os
import pickle
import copy
import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import PchipInterpolator, interp1d

try:
    from combustiontoolbox.utils.generateID import generateID
except ImportError:
    import hashlib
    def generateID(value: str) -> float:
        # Fallback implementation of MD5-based ID generation
        md5_hash = hashlib.md5(value.encode('utf-8')).digest()
        hash_arr = np.frombuffer(md5_hash, dtype=np.uint32)
        id_val = int(hash_arr[0] ^ hash_arr[1] ^ hash_arr[2] ^ hash_arr[3])
        return float(id_val)


class DynamicStruct(dict):
    """
    A dictionary subclass that allows dot-notation attribute access.
    Matches MATLAB struct behavior.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DynamicStruct' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DynamicStruct' object has no attribute '{name}'")


class GriddedInterpolant:
    """
    Python wrapper matching MATLAB's griddedInterpolant behavior.
    Supports Pchip/linear interpolation and linear extrapolation.
    """
    def __init__(self, x, y, method='pchip', extrapolation='linear'):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.method = method
        self.extrapolation = extrapolation
        
        # Sort values by x
        idx = np.argsort(self.x)
        self.x = self.x[idx]
        self.y = self.y[idx]
        
        if len(self.x) == 1 or np.all(self.y == self.y[0]):
            self.is_constant = True
            self.const_val = self.y[0]
            return
            
        self.is_constant = False
        
        if method == 'pchip':
            self.interpolator = PchipInterpolator(self.x, self.y)
            # Compute slopes at the boundaries for linear extrapolation
            self.m_start = self.interpolator(self.x[0], 1)
            self.m_end = self.interpolator(self.x[-1], 1)
        elif method == 'linear':
            # interp1d with bounds_error=False and fill_value='extrapolate' does linear extrapolation
            self.interpolator = interp1d(self.x, self.y, kind='linear', bounds_error=False, fill_value='extrapolate')
            self.m_start = None
            self.m_end = None
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def __call__(self, T):
        T = np.asarray(T, dtype=float)
        
        if self.is_constant:
            if T.ndim == 0:
                return float(self.const_val)
            return np.full_like(T, self.const_val)
            
        if self.method == 'pchip' and self.extrapolation == 'linear':
            scalar_input = (T.ndim == 0)
            T_arr = np.atleast_1d(T)
            
            res = self.interpolator(T_arr)
            
            # Left extrapolation: y0 + m_start * (T - x0)
            left_mask = T_arr < self.x[0]
            if np.any(left_mask):
                res[left_mask] = self.y[0] + self.m_start * (T_arr[left_mask] - self.x[0])
                
            # Right extrapolation: y_end + m_end * (T - x_end)
            right_mask = T_arr > self.x[-1]
            if np.any(right_mask):
                res[right_mask] = self.y[-1] + self.m_end * (T_arr[right_mask] - self.x[-1])
                
            return float(res[0]) if scalar_input else res
        else:
            res = self.interpolator(T)
            if T.ndim == 0:
                return float(res)
            return res


def resolve_path(filename):
    """
    Helper to resolve database paths relative to package or Repository Root.
    """
    if not filename:
        return filename
    if os.path.isabs(filename):
        return filename
    if os.path.exists(filename):
        return os.path.abspath(filename)
    
    # Try resolving relative to the outer 'databases/' folder in the repository
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".."))
    db_path_outer = os.path.join(repo_root, "databases", filename)
    if os.path.exists(db_path_outer):
        return db_path_outer
        
    return os.path.abspath(filename)


class Database(ABC):
    """
    The Database abstract class contains the common methods between database objects.
    This class is used as a base class for the NasaDatabase and BurcatDatabase classes.
    """

    _cachedDatabase = None

    def __init__(self, **kwargs):
        # Default parameters
        self.name = kwargs.get('name', 'Database')
        species_input = kwargs.get('species', {})
        self.species = DynamicStruct(species_input)
        self.filename = kwargs.get('filename', 'DB.pkl')
        self.interpolationMethod = kwargs.get('interpolationMethod', 'pchip')
        self.extrapolationMethod = kwargs.get('extrapolationMethod', 'linear')
        self.units = kwargs.get('units', 'molar')
        self.pointsTemperature = kwargs.get('pointsTemperature', 200)
        self.temperatureReference = kwargs.get('temperatureReference', 298.15)
        self.thermoFile = kwargs.get('thermoFile', 'thermo_CT.inp')
        self.FLAG_BENCHMARK = kwargs.get('FLAG_BENCHMARK', False)
        
        # Internal properties
        self.id = None
        
        # Set ID (uses numSpecies = 0 initially, matching MATLAB behavior)
        self.setID()

        # Check if database is in cache and the id matches
        if self.FLAG_BENCHMARK:
            self.load()
            return

        # If thermoFile is not default, we generate the database without loading
        if self.thermoFile != 'thermo_CT.inp':
            self.generateDatabase()
            print("OK!")
            Database._cachedDatabase = self
            return

        if Database._cachedDatabase is not None and Database._cachedDatabase.id == self.id:
            self.__dict__.update(Database._cachedDatabase.__dict__)
        else:
            self.load()
            Database._cachedDatabase = self

    @property
    def listSpecies(self):
        return list(self.species.keys())

    @property
    def numSpecies(self):
        return len(self.listSpecies)

    def merge(self, other, conflict_policy=1):
        """
        Merge two databases.
        
        Args:
            other (Database): Second database
            conflict_policy (int): Policy to handle species name conflicts.
                                   1: prefer self, 2: prefer other
                                   
        Returns:
            Database: Merged database
        """
        if not isinstance(other, Database):
            raise TypeError("Can only merge with another Database object.")
            
        listSpecies1 = self.listSpecies
        listSpecies2 = other.listSpecies
        
        if conflict_policy == 1:
            obj = copy.deepcopy(self)
            listSpeciesAdd = [sp for sp in listSpecies2 if sp not in listSpecies1]
            for sp in listSpeciesAdd:
                obj.species[sp] = copy.deepcopy(other.species[sp])
        elif conflict_policy == 2:
            obj = copy.deepcopy(other)
            listSpeciesAdd = [sp for sp in listSpecies1 if sp not in listSpecies2]
            for sp in listSpeciesAdd:
                obj.species[sp] = copy.deepcopy(self.species[sp])
        else:
            raise ValueError("Invalid conflict policy. Options are: 1 (prefer self), 2 (prefer other)")
            
        obj.setID()
        return obj

    def __add__(self, other):
        return self.merge(other, conflict_policy=1)

    def load(self, filename=None):
        """
        Load database from file
        """
        if filename is not None:
            self.filename = filename
            
        resolved_filename = resolve_path(self.filename)
        
        if os.path.exists(resolved_filename):
            print(f"{self.name} database with thermo loaded from the main path ... ", end="")
            try:
                with open(resolved_filename, 'rb') as f:
                    db = pickle.load(f)
                self.__dict__.update(db.__dict__)
                print("OK!")
            except Exception as e:
                print(f"FAILED (error: {e})")
                self.generateDatabase()
                print("OK!")
        else:
            self.generateDatabase()
            print("OK!")
        return self

    def save(self, filename=None, path=''):
        """
        Save database to a *.pkl file
        """
        if filename is None:
            filename = self.filename
            
        if path:
            full_path = os.path.join(path, filename)
        else:
            full_path = resolve_path(filename)
            
        os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)

    def getProperty(self, listSpecies, property_name):
        """
        Gets the vector of the defined property for the given set of species
        """
        return np.array([getattr(self.species[sp], property_name) for sp in listSpecies])

    def setID(self):
        """
        Concatenate input arguments to create a unique identifier string
        """
        value = (
            str(self.name)
            + str(self.numSpecies)
            + str(self.filename)
            + str(self.interpolationMethod)
            + str(self.extrapolationMethod)
            + str(self.units)
            + str(self.pointsTemperature)
            + str(self.temperatureReference)
            + str(self.thermoFile)
        )
        self.id = generateID(value)
        return self

    @abstractmethod
    def generateDatabase(self):
        pass

    @abstractmethod
    def getSpeciesThermo(self, *args, **kwargs):
        pass

    @staticmethod
    def fullname2name(species):
        raise NotImplementedError("fullname2name must be implemented by subclasses.")

    def addSpecies(self, species_name, DB_master):
        """
        Add species to the database
        """
        species_name_clean = self.fullname2name(species_name)
        species = DB_master[species_name_clean]
        
        Tintervals = species.Tintervals
        Trange = species.Trange
        
        if Tintervals == 0:
            species = self.computeConstantTemperatureSpecies(species, species_name_clean, Trange, DB_master)
        else:
            species = self.computeVariableTemperatureSpecies(species, species_name_clean, Trange, Tintervals, DB_master)
            
        species.setID()
        self.species[species_name_clean] = species

    def computeConstantTemperatureSpecies(self, species, species_name, Trange, DB_master):
        Tref = Trange[0]
        
        # Get thermodynamic data at reference temperature
        Cp0, Hf0, H0, Ef0, S0, DfG0 = self.getSpeciesThermo(DB_master, species_name, Tref, self.units)
        
        species.hf = Hf0
        species.ef = Ef0
        species.Tref = Tref
        species.T = Tref
        
        # Generate interpolation curves (constant values)
        species.cpcurve = GriddedInterpolant([Tref, Tref + 1], [Cp0, Cp0], 'linear', 'linear')
        species.h0curve = GriddedInterpolant([Tref, Tref + 1], [H0, H0], 'linear', 'linear')
        species.s0curve = GriddedInterpolant([Tref, Tref + 1], [S0, S0], 'linear', 'linear')
        species.g0curve = GriddedInterpolant([Tref, Tref + 1], [DfG0, DfG0], 'linear', 'linear')
        return species

    def computeVariableTemperatureSpecies(self, species, species_name, Trange, Tintervals, DB_master):
        Tmin = Trange[0][0]
        Tmax = Trange[Tintervals - 1][1]
        T_vector = np.linspace(Tmin, Tmax, self.pointsTemperature)
        
        species.T = T_vector
        _, Hf0, _, Ef0, _, _ = self.getSpeciesThermo(DB_master, species_name, species.Tref, self.units)
        species.hf = Hf0
        species.ef = Ef0
        
        # Get thermodynamic data over the temperature range
        Cp0_vector, _, H0_vector, _, S0_vector, _ = self.getSpeciesThermo(DB_master, species_name, T_vector, self.units)
        DfG0_vector = H0_vector - T_vector * S0_vector
        
        # Generate interpolation curves
        species.cpcurve = GriddedInterpolant(T_vector, Cp0_vector, self.interpolationMethod, self.extrapolationMethod)
        species.h0curve = GriddedInterpolant(T_vector, H0_vector, self.interpolationMethod, self.extrapolationMethod)
        species.s0curve = GriddedInterpolant(T_vector, S0_vector, self.interpolationMethod, self.extrapolationMethod)
        species.g0curve = GriddedInterpolant(T_vector, DfG0_vector, self.interpolationMethod, self.extrapolationMethod)
        
        # Store additional data
        species.Tintervals = Tintervals
        species.Trange = Trange
        species.Texponents = species.Texponents
        species.a = species.a
        species.b = species.b
        return species
