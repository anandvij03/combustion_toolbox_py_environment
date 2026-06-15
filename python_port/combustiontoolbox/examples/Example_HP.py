# -------------------------------------------------------------------------
# EXAMPLE: HP
#
# Compute adiabatic temperature and equilibrium composition at constant
# pressure (p = 1.01325 bar) for lean to rich CH4-air mixtures at T = 300 K,
# and a set of equivalence ratios phi contained in (0.5, 5) [-]
#   
# See wiki or setListspecies method from ChemicalSystem class for predefined
# sets of species
#
# @author: Anand V
# @adapted-from: Alberto Cuadra Lara
#
# Last update June 2026
# -------------------------------------------------------------------------

# Import packages
import numpy as np
from combustiontoolbox.databases.NasaDatabase.NasaDatabase import NasaDatabase
from combustiontoolbox.core import *
from combustiontoolbox.equilibrium import *

# Get Nasa database
DB = NasaDatabase()

# Define chemical system
system = ChemicalSystem(DB)

# Initialize mixture
mix = Mixture(system)

# Define chemical state
mix.set(['CH4'], 'fuel', 1)
mix.set(['N2', 'O2', 'Ar', 'CO2'], 'oxidizer', np.array([78.084, 20.9476, 0.9365, 0.0319]) / 20.9476)

# Define properties
mixArray = mix.setProperties('temperature', 300, 'pressure', 1 * 1.01325, 'equivalenceRatio', np.arange(0.5, 5.01, 0.01))

# Initialize solver
solver = EquilibriumSolver(problemType='HP')

# Solve problem
solver.solveArray(mixArray)

# Generate report
solver.report(mixArray)
