# -------------------------------------------------------------------------
# EXAMPLE: EV
#
# Compute equilibrium composition at adiabatic temperature and defined
# specific volume v = 1 m3/kg for lean to rich CH4-air mixtures at temperature 
# T = 300 K, and a set of equivalence ratios (phi) contained in (0.5, 5) [-]
#   
# See wiki or setListspecies method from ChemicalSystem class for predefined
# sets of species
# @author: Anand V
# @adapted-from: Alberto Cuadra Lara 
#                 
# Last update October 06 2025
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
mix.set({'CH4'}, 'fuel', 1)
mix.set({'N2', 'O2', 'Ar', 'CO2'}, 'oxidizer', np.array([78.084, 20.9476, 0.9365, 0.0319]) / 20.9476)

# Define properties
mixArray = mix.setProperties('temperature', 300, 'volume', 1, 'equivalenceRatio', np.arange(0.5, 5.01, 0.01))

# Initialize solver
solver = EquilibriumSolver('problemType', 'EV')

# Solve the problem
solver.solveArray(mixArray)

# Generate report
report(solver, mixArray)