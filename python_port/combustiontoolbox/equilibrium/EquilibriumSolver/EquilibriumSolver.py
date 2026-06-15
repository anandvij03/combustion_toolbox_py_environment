import time as time_mod
import numpy as np
import warnings
from typing import Any, List, Union

# Import methods from other files in the same directory
from .equilibrate import equilibrate
from .equilibrateT import equilibrateT
from .equilibriumGibbs import equilibriumGibbs
from .equilibriumHelmholtz import equilibriumHelmholtz
from .equilibriumCheckIons import equilibriumCheckIons
from .equilibriumGuess import equilibriumGuess
from .equilibriumCheckCondensed import equilibriumCheckCondensed
from .equilibriumDerivatives import equilibriumDerivatives


class EquilibriumSolver:
    """
    The EquilibriumSolver class is used to compute the composition at the equilibrium
    of multi-component gas mixtures that undergo canonical thermochemical transformations from
    an initial state (reactants), defined by its initial composition, temperature, and pressure,
    to a final state (products), defined by a set of chemical species (in gaseous---included
    ions---or pure condensed phase).
    """

    # Bind imported functions as methods
    equilibrate = equilibrate
    equilibrateT = equilibrateT
    equilibriumGibbs = equilibriumGibbs
    equilibriumHelmholtz = equilibriumHelmholtz
    equilibriumCheckIons = equilibriumCheckIons
    equilibriumGuess = equilibriumGuess
    equilibriumCheckCondensed = staticmethod(equilibriumCheckCondensed)
    equilibriumDerivatives = staticmethod(equilibriumDerivatives)

    def __init__(self, problemType='TP', **kwargs):
        # Default options
        valid_problems = {'TP', 'TV', 'HP', 'EV', 'SP', 'SV'}
        if problemType.upper() not in valid_problems:
            raise ValueError(f"problemType must be one of {valid_problems}")

        self.problemType = problemType.upper()
        self.tolGibbs = kwargs.get('tolGibbs', 1e-6)
        self.tolE = kwargs.get('tolE', 1e-6)
        self.tolMoles = kwargs.get('tolMoles', 1e-14)
        self.tolMolesGuess = kwargs.get('tolMolesGuess', 1e-6)
        self.tolMultiplierIons = kwargs.get('tolMultiplierIons', 1e-4)
        self.tolTau = kwargs.get('tolTau', 1e-25)
        self.itMaxGibbs = kwargs.get('itMaxGibbs', 70)
        self.itMaxIons = kwargs.get('itMaxIons', 30)
        self.itMaxRecursion = kwargs.get('itMaxRecursion', 30)
        self.slackGuess = kwargs.get('slackGuess', 1e-14)
        self.temperatureIons = kwargs.get('temperatureIons', 1500)
        self.tol0 = kwargs.get('tol0', 1e-3)
        self.itMax = kwargs.get('itMax', 30)
        self.rootMethod = kwargs.get('rootMethod', 'newton')
        self.root_T0_l = kwargs.get('root_T0_l', 1000)
        self.root_T0_r = kwargs.get('root_T0_r', 3000)
        self.root_T0 = kwargs.get('root_T0', 3000)
        self.FLAG_EXTRAPOLATE = kwargs.get('FLAG_EXTRAPOLATE', True)
        self.FLAG_FAST = kwargs.get('FLAG_FAST', True)
        self.FLAG_EOS = kwargs.get('FLAG_EOS', False)
        self.FLAG_RESULTS = kwargs.get('FLAG_RESULTS', True)
        self.FLAG_TIME = kwargs.get('FLAG_TIME', True)
        self.FLAG_REPORT = kwargs.get('FLAG_REPORT', False)
        self.FLAG_CACHE = kwargs.get('FLAG_CACHE', True)
        
        # Load helper classes if available
        try:
            from combustiontoolbox.utils.display.PlotConfig import PlotConfig
            self.plotConfig = kwargs.get('plotConfig', PlotConfig())
        except ImportError:
            self.plotConfig = kwargs.get('plotConfig', None)

        try:
            from combustiontoolbox.core.CaloricGasModel.CaloricGasModel import CaloricGasModel
            default_model = CaloricGasModel.IMPERFECT
        except ImportError:
            default_model = None

        self.caloricGasModel = kwargs.get('caloricGasModel', default_model)
        
        # Timer attribute
        self.time = None

        # Handle deprecated warning flags
        if 'FLAG_TCHEM_FROZEN' in kwargs or 'FLAG_FROZEN' in kwargs:
            warnings.warn(
                "The flags 'FLAG_TCHEM_FROZEN' and 'FLAG_FROZEN' are deprecated. "
                "Please use the 'caloricGasModel' parameter with values from the CaloricGasModel enumeration instead."
            )
            flag_tchem = kwargs.get('FLAG_TCHEM_FROZEN', False)
            flag_frozen = kwargs.get('FLAG_FROZEN', False)
            if self.caloricGasModel is not None:
                self.caloricGasModel = self.caloricGasModel.fromFlag(flag_tchem, flag_frozen)

    def set(self, property_name, value, *args):
        """
        Set properties of the EquilibriumSolver object
        """
        varargin = [property_name, value] + list(args)
        for i in range(0, len(varargin), 2):
            prop = varargin[i]
            val = varargin[i + 1]
            if not hasattr(self, prop):
                raise AttributeError(f"Property '{prop}' not found in EquilibriumSolver")
            setattr(self, prop, val)
        return self

    def solve(self, mix, *args):
        """
        Obtain chemical equilibrium composition and thermodynamic properties
        """
        if len(args) > 0:
            self.equilibrate(mix, args[0])
        else:
            self.equilibrate(mix)
        
        mix.problemType = self.problemType
        
        if self.FLAG_RESULTS:
            try:
                from combustiontoolbox.core.Mixture.print import print_mixtures
                print_mixtures(mix)
            except ImportError:
                pass
        return mix

    def solveArray(self, mixArray, *args):
        """
        Obtain chemical equilibrium composition and thermodynamic properties for an array of mixture values
        """
        n = len(mixArray)
        
        # Timer
        start_time = time_mod.perf_counter()

        # Calculations
        self.solve(mixArray[n - 1])
        
        for i in range(n - 2, -1, -1):
            self.solve(mixArray[i], mixArray[i + 1])

        # Timer
        self.time = time_mod.perf_counter() - start_time

        # Print elapsed time
        self.printTime()

        # Postprocess all the results with predefined plots
        if self.FLAG_REPORT:
            self.report(mixArray)

        # Clear cache
        if self.FLAG_CACHE:
            try:
                from combustiontoolbox.utils import clearCache
                clearCache()
            except ImportError:
                pass
        return mixArray

    def printTime(self):
        """
        Print execution time
        """
        if not self.FLAG_TIME or self.time is None:
            return
        print(f"\nElapsed time is {self.time:.5f} seconds")

    def plot(self, mixArray, *args):
        """
        Plot results
        """
        try:
            from combustiontoolbox.utils.display.plotComposition import plotComposition
            from combustiontoolbox.utils.display.plotProperties import plotProperties
        except ImportError:
            plotComposition = None
            plotProperties = None

        # Definitions
        additionalMixtures = len(args)
        numPlotProperties = self.plotConfig.numPlotProperties if self.plotConfig else 0

        # Check if is a scalar value (e.g. single element or empty)
        if not isinstance(mixArray, (list, np.ndarray)) or len(mixArray) <= 1:
            return None

        # Get labels
        labels = []
        if additionalMixtures > 0:
            try:
                flag_complete = mixArray[0].chemicalSystem.FLAG_COMPLETE
                first_arg_complete = args[0][0].chemicalSystem.FLAG_COMPLETE
            except (AttributeError, IndexError):
                flag_complete = True
                first_arg_complete = True

            if flag_complete and not first_arg_complete:
                labels = ['Complete', 'Incomplete']
            else:
                labels = [f"Mixture {i + 1}" for i in range(additionalMixtures + 1)]

        # Plot molar fractions - mixArray
        if plotComposition:
            ax1 = plotComposition(
                mixArray[0],
                mixArray,
                mixArray[0].rangeName,
                'Xi',
                mintol=self.plotConfig.mintolDisplay if self.plotConfig else 1e-6,
                displaySpecies=self.plotConfig.displaySpecies if self.plotConfig else []
            )
        
        # Plot properties - mixArray
        ax2 = None
        if plotProperties and self.plotConfig:
            ax2 = plotProperties(
                [mixArray[0].rangeName] * numPlotProperties,
                mixArray,
                self.plotConfig.plotProperties,
                mixArray,
                basis=self.plotConfig.plotPropertiesBasis,
                config=self.plotConfig
            )
            
        # Check if there are additional mixtures
        if additionalMixtures == 0:
            return ax2

        for i in range(additionalMixtures):
            mixArray_i = args[i]

            # Plot molar fractions - mixArray_i
            if plotComposition:
                ax1 = plotComposition(
                    mixArray_i[0],
                    mixArray_i,
                    mixArray_i[0].rangeName,
                    'Xi',
                    mintol=self.plotConfig.mintolDisplay if self.plotConfig else 1e-6,
                    displaySpecies=self.plotConfig.displaySpecies if self.plotConfig else []
                )
            
            # Plot properties - mixArray_i
            if plotProperties and self.plotConfig:
                ax2 = plotProperties(
                    [mixArray_i[0].rangeName] * numPlotProperties,
                    mixArray_i,
                    self.plotConfig.plotProperties,
                    mixArray_i,
                    basis=self.plotConfig.plotPropertiesBasis,
                    config=self.plotConfig,
                    ax=ax2
                )

        # Set legends
        if ax2 is not None and len(labels) > 0:
            try:
                legend_axes = ax2.axes[-1] if hasattr(ax2, 'axes') else ax2
                if hasattr(legend_axes, 'legend'):
                    legend_axes.legend(labels, fontsize=self.plotConfig.FontSize if self.plotConfig else 10)
            except Exception:
                pass
        
        # Showing the plot
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            pass

        return ax2

    def report(self, mixArray, *args):
        """
        Postprocess all the results with predefined plots
        """
        if len(args) > 0:
            return self.plot(mixArray, *args)
        else:
            return self.plot(mixArray)

    # PRIVATE STATIC METHODS
    @staticmethod
    def removeElements(NatomE, A0, ind_E, tol):
        """
        Find zero sum elements and remove them from stoichiometric matrix
        """
        NatomE = np.asarray(NatomE, dtype=float).copy()
        A0 = np.asarray(A0, dtype=float).copy()
        
        # Define temporal fictitious value if there are ionized species
        temp_NatomE = NatomE.copy()
        
        # Check if electron is present
        FLAG_E = ind_E is not None and ind_E != "" and ind_E != [] and not (isinstance(ind_E, (int, np.integer)) and ind_E < 0)
        
        if FLAG_E:
            temp_NatomE[ind_E] = 1.0
            
        # Get flag of elements to be removed
        FLAG_REMOVE_ELEMENTS = temp_NatomE <= tol
        
        # Get the species to be removed from stoichiometric matrix
        if np.any(FLAG_REMOVE_ELEMENTS):
            remove_sum = np.sum(A0[:, FLAG_REMOVE_ELEMENTS] > 0, axis=1)
            indexRemoveSpecies = np.where(remove_sum > 0)[0]
        else:
            indexRemoveSpecies = np.array([], dtype=int)
            
        # Update stoichiometric matrix by removing columns
        A0 = A0[:, ~FLAG_REMOVE_ELEMENTS]
        
        # Set number of atoms
        NatomE = NatomE[~FLAG_REMOVE_ELEMENTS]
        
        # Check position electron "element"
        if FLAG_E:
            ind_E = ind_E - np.sum(FLAG_REMOVE_ELEMENTS[:ind_E])
        else:
            ind_E = None
            
        return A0, indexRemoveSpecies, ind_E, NatomE

    @staticmethod
    def tempValues(productSpeciesSet, NatomE):
        """
        List of indices with nonzero values and lengths
        """
        indexElements = np.arange(len(NatomE))
        indexGas = np.asarray(productSpeciesSet.indexGas, dtype=int).copy()
        indexCondensed = np.asarray(productSpeciesSet.indexCondensed, dtype=int).copy()
        indexIons = np.asarray(productSpeciesSet.indexIons, dtype=int).copy()
        
        # Handle indexCryogenic if it exists, otherwise empty
        indexCryogenic = getattr(productSpeciesSet, 'indexCryogenic', [])
        if indexCryogenic is None:
            indexCryogenic = []
        indexCryogenic = np.asarray(indexCryogenic, dtype=int)
        
        index = np.concatenate((indexGas, indexCondensed))
        
        # Remove cryogenic species from calculations
        if len(indexCryogenic) > 0:
            index = index[~np.isin(index, indexCryogenic)]
            indexCondensed = indexCondensed[~np.isin(indexCondensed, indexCryogenic)]
            
        NE = len(NatomE)
        NG = len(indexGas)
        NS = len(index)
        
        return index, indexGas, indexCondensed, indexIons, indexElements, NE, NG, NS

    @staticmethod
    def filterSpeciesTemperatureRange(productSpeciesSet, T, index, numSpecies, FLAG_EXTRAPOLATE):
        """
        Remove species indices out of the temperature range if FLAG_EXTRAPOLATE = false
        """
        if FLAG_EXTRAPOLATE:
            return index, numSpecies

        # Convert to numpy arrays if not already
        index = np.asarray(index, dtype=int)
        
        temperatureMin = np.asarray(productSpeciesSet.temperatureMin)[index]
        temperatureMax = np.asarray(productSpeciesSet.temperatureMax)[index]
        
        FLAG_REMOVE = (T < temperatureMin) | (T > temperatureMax)
        index = index[~FLAG_REMOVE]
        numSpecies = len(index)
        
        return index, numSpecies

    @staticmethod
    def updateTemp(N, index, indexCondensed, indexGas, indexIons, NP, NG, NS, SIZE):
        """
        Update temporal values
        """
        N = np.asarray(N, dtype=float).copy()
        index = np.asarray(index, dtype=int)
        indexCondensed = np.asarray(indexCondensed, dtype=int)
        indexGas = np.asarray(indexGas, dtype=int)
        indexIons = np.asarray(indexIons, dtype=int)

        # Get species to be removed
        FLAG_REMOVE = N[index] / NP < np.exp(-SIZE)

        # Check if there are species to be removed
        if not np.any(FLAG_REMOVE):
            return index, indexCondensed, indexGas, indexIons, NG, NS, N

        # Set to zero the moles of the species to be removed
        N[index[FLAG_REMOVE]] = 0.0

        # Get the species to be removed
        removed_species = index[FLAG_REMOVE]

        # Update index lists
        indexGas = indexGas[~np.isin(indexGas, removed_species)]
        indexCondensed = indexCondensed[~np.isin(indexCondensed, removed_species)]
        indexIons = indexIons[~np.isin(indexIons, removed_species)]

        # Update index list and lengths
        index = np.concatenate((indexGas, indexCondensed))
        NG = len(indexGas)
        NS = len(index)

        return index, indexCondensed, indexGas, indexIons, NG, NS, N

    @staticmethod
    def relaxFactorGas(NP, nj_gas, Delta_ln_nj, Delta_ln_NP):
        """
        Compute relaxation factor for gas species
        """
        nj_gas = np.asarray(nj_gas)
        Delta_ln_nj = np.asarray(Delta_ln_nj)
        
        FLAG = Delta_ln_nj > 0
        FLAG_MINOR = (nj_gas / NP <= 1e-8) & FLAG
        
        # Calculate delta1
        delta_vals = [5.0 * Delta_ln_NP]
        if np.any(FLAG):
            delta_vals.extend(Delta_ln_nj[FLAG])
        max_abs = np.max(np.abs(delta_vals))
        delta1 = 2.0 / max_abs if max_abs > 0 else 1.0
        
        # Calculate delta2
        if np.any(FLAG_MINOR):
            numerator = -np.log(nj_gas[FLAG_MINOR] / NP) - 9.2103404
            denominator = Delta_ln_nj[FLAG_MINOR] - Delta_ln_NP
            # Avoid division by zero
            valid_mask = denominator != 0
            if np.any(valid_mask):
                delta2 = np.min(np.abs(numerator[valid_mask] / denominator[valid_mask]))
            else:
                delta2 = 1.0
        else:
            delta2 = 1.0
            
        return min(1.0, delta1, delta2)

    @staticmethod
    def relaxFactorCondensed(NP, N, psi_j, Delta_nj, indexCondensed, NG, NS, SIZE, tau, RT):
        """
        Compute and apply relaxation factor for condensed species
        """
        N = np.asarray(N, dtype=float).copy()
        psi_j = np.asarray(psi_j, dtype=float).copy()
        Delta_nj = np.asarray(Delta_nj, dtype=float)
        indexCondensed = np.asarray(indexCondensed, dtype=int)
        
        delta0 = 0.9999
        FLAG_UNSTABLE = np.array([], dtype=bool)
        
        if NS - NG == 0:
            return N, psi_j, FLAG_UNSTABLE
            
        # First deltaCondensed calculation
        N_cond = N[indexCondensed]
        FLAG_DELTA_1 = N_cond + Delta_nj < 0
        if np.any(FLAG_DELTA_1):
            ratio = -delta0 * N_cond[FLAG_DELTA_1] / Delta_nj[FLAG_DELTA_1]
            deltaCondensed_1 = min(1.0, np.min(ratio))
        else:
            deltaCondensed_1 = 1.0
            
        N[indexCondensed] = N_cond + deltaCondensed_1 * Delta_nj
        
        # Second deltaCondensed calculation
        psi_cond = psi_j[indexCondensed]
        N_cond_updated = N[indexCondensed]
        
        # Avoid division by zero
        non_zero = N_cond_updated != 0
        Delta_psi_j = np.zeros_like(psi_cond)
        Delta_psi_j[non_zero] = (tau - psi_cond[non_zero] * Delta_nj[non_zero]) / N_cond_updated[non_zero] - psi_cond[non_zero]
        
        FLAG_DELTA_2 = psi_cond + Delta_psi_j < 0
        if np.any(FLAG_DELTA_2):
            ratio = -delta0 * psi_cond[FLAG_DELTA_2] / Delta_psi_j[FLAG_DELTA_2]
            deltaCondensed_2 = min(1.0, np.min(ratio))
        else:
            deltaCondensed_2 = 1.0
            
        psi_j[indexCondensed] = psi_cond + deltaCondensed_2 * Delta_psi_j
        
        # Check if there are unstable species
        Omega_pi = np.exp(-psi_j[indexCondensed] / RT)
        # Avoid log10 of 0 or negative
        with np.errstate(divide='ignore', invalid='ignore'):
            log10_Omega = np.log10(Omega_pi)
            log10_Omega[np.isnan(log10_Omega) | np.isinf(log10_Omega)] = 100.0 # Force removal of invalid values
            FLAG_UNSTABLE = (N[indexCondensed] / NP < np.exp(-SIZE)) | (np.abs(log10_Omega) > 1e-2)
            
        N[indexCondensed[FLAG_UNSTABLE]] = 0.0
        
        return N, psi_j, FLAG_UNSTABLE

    @staticmethod
    def getPoint(x_vector, f_vector):
        """
        Get point using the regula falsi method
        """
        x_vector = np.asarray(x_vector)
        f_vector = np.asarray(f_vector)
        return (f_vector[1] * x_vector[0] - f_vector[0] * x_vector[1]) / (f_vector[1] - f_vector[0])

    @staticmethod
    def getPointAitken(x0, g_vector):
        """
        Get fixed point of a function based on the chemical transformation using the Aitken acceleration method
        """
        g_vector = np.asarray(g_vector)
        return x0 - (g_vector[0] - x0) ** 2 / (g_vector[1] - 2 * g_vector[0] + x0)

    # PRIVATE INSTANCE METHODS
    def newton(self, mix1, mix2, attributeName, x0, molesGuess):
        """
        Find the temperature [K] (root) for the set chemical transformation at equilibrium
        using the second-order Newton-Raphson method
        """
        if self.problemType.upper() in {'TP', 'TV'}:
            return x0, 0.0, molesGuess
            
        it = 0
        STOP = 1.0
        
        while STOP > self.tol0 and it < self.itMax:
            it += 1
            
            f0, fprime0, frel, molesGuess = self.getRatioNewton(mix1, mix2, attributeName, x0, molesGuess)
            
            if fprime0 == 0:
                break
                
            x = abs(x0 - f0 / fprime0)
            STOP = max(abs((x - x0) / x), frel)
            x0 = x
            
        if STOP > self.tol0:
            print('\n' + '*' * 59)
            print('Newton method not converged\nCalling Newton-Steffensen root finding algorithm')
            x0 = self.regulaGuess(mix1, mix2, attributeName)
            x, STOP, molesGuess = self.nsteff(mix1, mix2, attributeName, x0, [])
            return x, STOP, molesGuess
            
        self.printError(it, x0, STOP)
        return x0, STOP, molesGuess

    def getRatioNewton(self, mix1, mix2, attributeName, x, molesGuess):
        """
        Get the residual of f, its derivative with temperature, and the
        relative value of the residual
        """
        try:
            self.equilibrateT(mix1, mix2, x, molesGuess)
        except Exception:
            self.equilibrateT(mix1, mix2, x, np.array([]))

        val_mix2 = getattr(mix2, attributeName)
        val_mix1 = getattr(mix1, attributeName)

        # Calculate residual of f = 0
        f = val_mix2 - val_mix1

        # Calculate partial derivative of f with temperature
        fprime = self.getPartialDerivative(mix2)

        # Get relative value of the residual
        frel = abs(f / val_mix2) if val_mix2 != 0 else 0.0

        # Update guess moles
        molesGuess = mix2.N * mix2.Xi

        return f, fprime, frel, molesGuess

    def getAttribute(self):
        """
        Get attribute of the problem type
        """
        prob = self.problemType.upper()
        if prob in {'TP', 'TV'}:
            return 'T'
        elif prob == 'HP':
            return 'hSpecific'
        elif prob == 'EV':
            return 'eSpecific'
        elif prob in {'SP', 'SV'}:
            return 'sSpecific'
        return ''

    def getPartialDerivative(self, mix):
        """
        Get value of the partial derivative for the set problem type [J/kg-K] (HP, EV) or [J/kg-K^2] (SP, SV)
        """
        prob = self.problemType.upper()
        if prob == 'HP':
            return mix.cpSpecific
        elif prob == 'EV':
            return mix.cvSpecific
        elif prob == 'SP':
            return mix.cpSpecific / mix.T
        elif prob == 'SV':
            return mix.cvSpecific / mix.T
        return 0.0

    def nsteff(self, mix1, mix2, attributeName, x0, molesGuess):
        """
        Find the temperature [K] (root) for the set chemical transformation at equilibrium
        using the third-order Newton-Steffensen method
        """
        if self.problemType.upper() in {'TP', 'TV'}:
            return x0, 0.0, molesGuess

        it = 0
        STOP = 1.0
        
        while STOP > self.tol0 and it < self.itMax:
            it += 1
            
            f0, fprime0, frel, molesGuess = self.getRatioNewton(mix1, mix2, attributeName, x0, molesGuess)
            
            # Compute pseudo-solution
            if fprime0 == 0:
                break
            x = abs(x0 - f0 / fprime0)
            
            # Re-estimation of first derivative
            f0_2, _, _, _ = self.getRatioNewton(mix1, mix2, attributeName, x, molesGuess)
            
            # Compute solution
            denominator = fprime0 * (f0 - f0_2)
            if denominator != 0:
                x = abs(x0 - (f0 ** 2) / denominator)
            else:
                x = abs(x0 - f0 / fprime0)
            
            # Compute stop criteria
            STOP = max(abs((x - x0) / x), frel)
            x0 = x
            
        if STOP > self.tol0:
            print('\n' + '*' * 59)
            print('Newton method not converged\nCalling Steffensen-Aitken root finding algorithm')
            x0 = self.regulaGuess(mix1, mix2, attributeName)
            x, STOP, molesGuess = self.steff(mix1, mix2, attributeName, x0, [])
            return x, STOP, molesGuess
            
        self.printError(it, x0, STOP)
        return x0, STOP, molesGuess

    def steff(self, mix1, mix2, attributeName, x0, molesGuess):
        """
        Find the temperature [K] (root) for the set chemical transformation at equilibrium
        using the Steffenson-Aitken method
        """
        if self.problemType.upper() in {'TP', 'TV'}:
            return x0, 0.0, molesGuess

        it = 0
        STOP = 1.0
        
        while STOP > self.tol0 and it < self.itMax:
            it += 1

            # Compute fixed point
            g, g_rel = self.getGpoint(mix1, mix2, attributeName, x0, molesGuess)
            fx = abs(g - x0)

            # Compute auxiliary fixed point
            g_aux, _ = self.getGpoint(mix1, mix2, attributeName, fx, molesGuess)
            fx2 = abs(g_aux - fx)

            # Compute solution
            if abs(fx2 - 2 * fx + x0) > self.tol0:
                x = self.getPointAitken(x0, [fx, fx2])
            else:
                x = fx
            
            # Compute stop criteria
            STOP = max(abs((x - fx) / x), abs(g_rel))
            x0 = x
            
        self.printError(it, x0, STOP)
        return x0, STOP, molesGuess

    def regulaGuess(self, mix1, mix2, attributeName):
        """
        Find an estimate of the temperature for the set chemical equilibrium
        transformation using the regula falsi method
        """
        molesGuess = np.array([])

        # Define branch
        x_l = self.root_T0_l
        x_r = self.root_T0_r
        
        # Compute f(x) = f2(x) - f1 at the branch limits
        g_l, _ = self.getGpoint(mix1, mix2, attributeName, x_l, molesGuess)
        g_r, _ = self.getGpoint(mix1, mix2, attributeName, x_r, molesGuess)
        
        # Update estimate based on the region
        if not (np.isnan(g_l) or np.isnan(g_r)) and g_l * g_r < 0:
            x0 = self.getPoint([x_l, x_r], [g_l, g_r])
        elif not (np.isnan(g_l) or np.isnan(g_r)) and g_l * g_r > 0 and abs(g_l) < abs(g_r):
            x0 = max(x_l - 50.0, 300.0)
        elif (not (np.isnan(g_l) or np.isnan(g_r)) and g_l * g_r > 0) or (np.isnan(g_l) and np.isnan(g_r)):
            x0 = max(x_r + 50.0, 300.0)
        elif np.isnan(g_l) and not np.isnan(g_r):
            x0 = max(x_r - 100.0, 300.0)
        elif not np.isnan(g_l) and np.isnan(g_r):
            x0 = max(x_l + 100.0, 300.0)
        else:
            x0 = self.root_T0
            
        return x0

    def getGpoint(self, mix1, mix2, attributeName, x0, molesGuess):
        """
        Get fixed point of a function based on the chemical transformation
        """
        try:
            # Compute TP problem
            self.equilibrateT(mix1, mix2, x0, molesGuess)

            # Compute f(x) = f2(x) - f1 = 0
            val_mix2 = getattr(mix2, attributeName)
            val_mix1 = getattr(mix1, attributeName)
            gpoint = val_mix2 - val_mix1

            # Compute f(x) / f2(x)
            gpointRelative = gpoint / val_mix2
        except Exception:
            gpoint = np.nan
            gpointRelative = np.nan

        return gpoint, gpointRelative

    def printError(self, it, T, STOP):
        """
        Print error of the method if the number of iterations is greater than maximum iterations allowed
        """
        if it < self.itMax:
            return

        print('***********************************************************')
        print('Root algorithm not converged')
        print(f'   Error       =  {STOP * 100:8.2f} [%]')
        print(f'   Temperature =  {T:8.2f} [K]')
        print(f'   Iterations  =  {it:8d} [it]')
        print('***********************************************************')
