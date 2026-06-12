import os
import re
import numpy as np
from abc import ABC

from combustiontoolbox.databases.database import Database, DynamicStruct
from combustiontoolbox.common.Constants import Constants
from combustiontoolbox.core.Elements.Elements import Elements
from combustiontoolbox.core.Species.Species import Species


class NasaDatabase(Database):
    """
    The NasaDatabase class is used to store thermodynamic data from NASA's database
    using NASA's 9 coefficient polynomial fits.
    """

    def __init__(self, **kwargs):
        # Call superclass constructor with default name and temperatureReference
        kwargs.setdefault('name', 'NASA')
        kwargs.setdefault('temperatureReference', 298.15)
        super().__init__(**kwargs)

    def generateDatabaseMaster(self):
        """
        Generate Master Database (DB_master) with the thermodynamic
        data of the chemical species
        """
        DB_master = self.getDatabaseMaster(self.thermoFile)
        print("OK!")
        return DB_master

    def generateDatabase(self, listSpecies=None):
        """
        Generate Database with thermochemical interpolation curves
        from the data extracted from the thermoFile
        """
        DB_master = self.getDatabaseMaster(self.thermoFile)
        
        # Remove species allocated from cache
        self.species = DynamicStruct()

        if listSpecies is not None:
            # Keep only required species in DB_master
            list_species_set = set(listSpecies)
            keys_to_remove = [k for k in DB_master.keys() if k not in list_species_set]
            for k in keys_to_remove:
                del DB_master[k]
        else:
            listSpecies = list(DB_master.keys())

        numSpecies = len(listSpecies)
        print(f"Generating {self.name} database with thermo ... ", end="")
        
        # Compute interpolation curves for each species
        for sp in listSpecies:
            self.addSpecies(sp, DB_master)
            
        return self

    def getSpeciesThermoFull(self, DB, species, temperature, units):
        """
        Compute thermodynamic function using NASA's 9 polynomials
        """
        R0 = Constants.R0
        temperature = np.asarray(temperature, dtype=float)
        N = len(temperature) if temperature.ndim > 0 else 1
        
        # Compute core thermodynamic functions
        cp, hf, h0, ef, s0, g0 = self.getSpeciesThermo(DB, species, temperature, units)
        
        # Unpack coefficients
        _, _, Trange, _, Tintervals, phase, _, W, _ = self.getCoefficients(species, DB)
        
        # Get elements
        elements = Elements()
        elementMatrix = DB[species].getElementMatrix(elements.listElements)
        Delta_n = self.getChangeMolesGasReaction(elements, elementMatrix, phase)
        
        # Adjust Universal gas constant if units is 'mass'
        R = R0 / W if units.lower() == 'mass' else R0
        
        if Tintervals > 0:
            Tref = DB[species].Tref
            e0 = (ef + (h0 - hf) - (1 - phase) * R * (temperature - Tref))
            cv = cp - R
            DhT = h0 - hf
            DeT = e0 - ef
        else:
            Tref = Trange[0]
            cv = np.zeros(N) if temperature.ndim > 0 else 0.0
            e0 = hf - Delta_n * R * Tref
            if temperature.ndim > 0:
                e0 = np.full(N, e0)
                s0 = np.zeros(N)
                DhT = np.zeros(N)
                DeT = np.zeros(N)
            else:
                s0 = 0.0
                DhT = 0.0
                DeT = 0.0
                
        return cp, cv, h0, DhT, e0, DeT, s0, g0

    def getSpeciesThermo(self, DB, species, temperature, units):
        """
        Calculates the thermodynamic properties of any species included in the NASA database
        """
        R0 = Constants.R0
        temperature = np.asarray(temperature, dtype=float)
        scalar_input = (temperature.ndim == 0)
        temperature_arr = np.atleast_1d(temperature)
        N = len(temperature_arr)

        # Unpack NASA's polynomials coefficients
        a, b, Trange, Texponents, Tintervals, phase, hf, W, FLAG_REFERENCE = self.getCoefficients(species, DB)
        
        # Get elements
        elements = Elements()

        # Get element matrix of the species
        elementMatrix = DB[species].getElementMatrix(elements.listElements)
        
        # Compute change in moles of gases during the formation reaction of a
        # mole of that species starting from the elements in their reference state
        Delta_n = self.getChangeMolesGasReaction(elements, elementMatrix, phase)
        
        # Adjust unit multiplier functions
        def molar2mass(val):
            return val / W

        if Tintervals == 0:
            Tref = Trange[0]
            cp0 = np.zeros(N)
            h0 = np.full(N, hf)
            ef_val = hf - Delta_n * R0 * Tref
            ef = np.full(N, ef_val)
            s0 = np.zeros(N)
            g0 = h0.copy()
            
            if units.lower() == 'mass':
                hf = molar2mass(hf)
                ef = molar2mass(ef)
                cp0 = molar2mass(cp0)
                h0 = molar2mass(h0)
                s0 = molar2mass(s0)
                g0 = molar2mass(g0)
                
            if scalar_input:
                return float(cp0[0]), float(hf[0]) if hasattr(hf, '__len__') else float(hf), float(h0[0]), float(ef[0]) if hasattr(ef, '__len__') else float(ef), float(s0[0]), float(g0[0])
            return cp0, hf, h0, ef, s0, g0

        # Compute thermodynamic properties for variable temperature
        cp0 = np.zeros(N)
        h0 = np.zeros(N)
        s0 = np.zeros(N)
        g0 = np.zeros(N)
        
        Tref = DB[species].Tref
        ef_val = hf - Delta_n * R0 * Tref
        ef = np.full(N, ef_val)

        for i, T in enumerate(temperature_arr):
            # Get temperature interval
            Tinterval = self.getIndexTempereratureInterval(species, T, DB) - 1 # 0-indexed in Python

            # Compute thermodynamic function
            cp0[i] = R0 * np.sum(a[Tinterval] * (T ** Texponents[Tinterval]))
            
            h_coeffs = np.array([-1.0, np.log(T), 1.0, 1/2, 1/3, 1/4, 1/5, 0.0], dtype=float)
            h0[i] = R0 * T * (np.sum(a[Tinterval] * (T ** tExponents[Tinterval]) * h_coeffs) + b[Tinterval][0] / T)
            
            s_coeffs = np.array([-1/2, -1.0, np.log(T), 1.0, 1/2, 1/3, 1/4, 0.0], dtype=float)
            s0[i] = R0 * (np.sum(a[Tinterval] * (T ** tExponents[Tinterval]) * s_coeffs) + b[Tinterval][1])
            
            if not FLAG_REFERENCE:
                g0[i] = h0[i] - T * s0[i]
            else:
                g0[i] = 0.0

        if units.lower() == 'mass':
            cp0 = molar2mass(cp0)
            hf = molar2mass(hf)
            ef = molar2mass(ef)
            h0 = molar2mass(h0)
            s0 = molar2mass(s0)
            g0 = molar2mass(g0)
            
        if scalar_input:
            return float(cp0[0]), float(hf[0]) if hasattr(hf, '__len__') else float(hf), float(h0[0]), float(ef[0]) if hasattr(ef, '__len__') else float(ef), float(s0[0]), float(g0[0])
        return cp0, hf, h0, ef, s0, g0

    def species_g0_NASA(self, species, temperature):
        """
        Compute Gibbs energy [J/mol] of the species at the given temperature [K]
        using NASA's 9 polynomials.
        """
        R0 = Constants.R0
        temperature = np.asarray(temperature, dtype=float)
        scalar_input = (temperature.ndim == 0)
        temperature_arr = np.atleast_1d(temperature)
        N = len(temperature_arr)
        
        # Unpack coefficients
        a, b, _, tExponents, ctTInt, _, _, _, _ = self.getCoefficients(species, self.species)
        
        g0 = np.zeros(N)
        for i, T in enumerate(temperature_arr):
            if ctTInt > 0:
                tInterval = self.getIndexTempereratureInterval(species, T, self.species) - 1 # 0-indexed
                
                g_coeffs = np.array([-1/2, 1.0 + np.log(T), 1.0 - np.log(T), -1/2, -1/6, -1/12, -1/20, 0.0], dtype=float)
                g0[i] = R0 * T * (np.sum(a[tInterval] * (T ** tExponents[tInterval]) * g_coeffs) + b[tInterval][0] / T - b[tInterval][1])
            else:
                g0[i] = self.species[species].hf
                
        return float(g0[0]) if scalar_input else g0


    @staticmethod
    def fullname2name(species):
        if not species:
            return species
            
        FLAG_MILLENIUM = False
        if '_M' in species:
            species = species.replace('_M', '')
            FLAG_MILLENIUM = True
            
        name = species
        if name.endswith('+'):
            name = name[:-1] + 'plus'
        elif name.endswith('-'):
            name = name[:-1] + 'minus'
            
        # replace parentheses with 'b'
        name = re.sub(r'[()]', 'b', name)
        # replace special characters with '_'
        name = re.sub(r'[.,+-]', '_', name)
        
        # check if it starts with a number
        if re.match(r'^[0-9]', name):
            name = 'num_' + name
            
        # replace single quotes with '_'
        name = name.replace("'", "_")
        
        if FLAG_MILLENIUM:
            name += '_M'
            
        return name

    @staticmethod
    def getCoefficients(species, DB):
        sp = DB[species]
        return (
            sp.a,
            sp.b,
            sp.Trange,
            sp.Texponents,
            sp.Tintervals,
            sp.phase,
            sp.hf,
            sp.W,
            sp.FLAG_REFERENCE
        )

    def getDatabaseMaster(self, thermoFile):
        """
        Generate Master Database (DB_master) with the thermodynamic data of the chemical species
        """
        from combustiontoolbox.databases.database import resolve_path
        resolved_thermoFile = resolve_path(thermoFile)
        
        if thermoFile == 'thermo_CT.inp':
            msg = 'Loading NASA database ... '
        else:
            msg = 'Loading an unknown database ... '
            
        print(msg, end="")
        
        DB_master = DynamicStruct()
        line_count = 0
        
        with open(resolved_thermoFile, 'r', errors='ignore') as fid:
            while line_count < 10000:
                tline = fid.readline()
                if not tline:
                    break
                    
                tline_stripped = tline.strip()
                if not tline_stripped:
                    continue
                    
                if tline_stripped.startswith('!'):
                    continue
                    
                if 'thermo' in tline_stripped.lower():
                    # Read the next line to skip header
                    fid.readline()
                    continue
                    
                if 'END' in tline_stripped:
                    continue
                    
                line_count += 1
                
                # Instantiate a new Species
                temp = Species()
                temp.fullname = tline[0:16].strip()
                temp.name = self.fullname2name(temp.fullname)
                temp.comments = tline[18:].strip()
                
                if any(ref in temp.comments for ref in ['Ref-', 'REFERENCE ELEMENT']):
                    temp.FLAG_REFERENCE = True
                    
                if 'HF298=' in temp.comments.upper():
                    temp.Tref = 298.15
                elif 'HF0=' in temp.comments.upper():
                    temp.Tref = 0.0
                else:
                    temp.Tref = self.temperatureReference
                    
                # Read second line
                tline = fid.readline()
                temp.Tintervals = int(tline[0:2].strip())
                temp.refCode = tline[3:9].strip()
                temp.formula = tline[10:50]
                temp.phase = float(int(tline[50:52].strip()) != 0)
                temp.W = float(tline[52:65].strip()) * 1e-3
                temp.hf = float(tline[65:80].strip())
                
                if temp.Tintervals == 0:
                    tline = fid.readline()
                    temp.Trange = np.fromstring(tline[0:22], sep=' ')
                    temp.Texponents = np.fromstring(tline[23:63], sep=' ')
                    temp.hftoh0 = float(tline[65:].strip())
                    
                temp.Trange = [None] * temp.Tintervals
                temp.Texponents = [None] * temp.Tintervals
                temp.hftoh0 = [None] * temp.Tintervals
                temp.a = [None] * temp.Tintervals
                temp.b = [None] * temp.Tintervals
                
                for Tinterval in range(temp.Tintervals):
                    tline = fid.readline()
                    temp.Trange[Tinterval] = np.fromstring(tline[0:22], sep=' ')
                    temp.Texponents[Tinterval] = np.fromstring(tline[23:63], sep=' ')
                    temp.hftoh0[Tinterval] = float(tline[65:].strip())
                    
                    tline = fid.readline()
                    a1 = float(tline[0:16].strip())
                    a2 = float(tline[16:32].strip())
                    a3 = float(tline[32:48].strip())
                    a4 = float(tline[48:64].strip())
                    a5 = float(tline[64:80].strip())
                    
                    tline = fid.readline()
                    a6 = float(tline[0:16].strip())
                    a7 = float(tline[16:32].strip())
                    a8 = 0.0
                    b1 = float(tline[48:64].strip())
                    b2 = float(tline[64:80].strip())
                    
                    temp.a[Tinterval] = [a1, a2, a3, a4, a5, a6, a7, a8]
                    temp.b[Tinterval] = [b1, b2]
                    
                DB_master[temp.name] = temp
                
        return DB_master

    @staticmethod
    def getChangeMolesGasReaction(elements, elementMatrix, phase):
        """
        Compute change in moles of gases during the formation reaction of a mole
        of that species starting from the elements in their reference state.
        """
        diatomic_indices = [elements.indexH, elements.indexN, elements.indexO, elements.indexF, elements.indexCl]
        noble_indices = [elements.indexHe, elements.indexNe, elements.indexAr, elements.indexKr, elements.indexXe, elements.indexRn]
        
        Delta_n_per_mole = np.zeros(elementMatrix.shape[1])
        for idx in range(elementMatrix.shape[1]):
            el_idx = elementMatrix[0, idx]
            if el_idx in diatomic_indices:
                Delta_n_per_mole[idx] = 0.5
            elif el_idx in noble_indices:
                Delta_n_per_mole[idx] = 1.0
                
        Delta_n = 1.0 - phase - np.dot(Delta_n_per_mole, elementMatrix[1, :])
        return Delta_n

    @staticmethod
    def getIndexTempereratureInterval(species, T, DB):
        """
        Get interval of the NASA's polynomials from the Database (DB) for the given species and temperature [K].
        """
        sp = DB[species]
        for idx in range(sp.Tintervals):
            if T >= sp.Trange[idx][0] and T <= sp.Trange[idx][1]:
                return idx + 1 # 1-based to match MATLAB
        return sp.Tintervals
