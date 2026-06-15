# -------------------------------------------------------------------------
# EXAMPLE: SP
# Compute Isentropic compression/expansion and equilibrium composition at 
# a defined set of pressure (p = 1:100 atm) for a rich CH4-air mixture
# at defined specific entropy, and an equivalence ratio phi 1.5 [-]
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
mixArray = mix.setProperties('entropySpecific', mix.sSpecific, 'pressure', 1.01325 * np.logspace(0, 1, 200), 'equivalenceRatio', 1.5)

# Initialize solver
solver = EquilibriumSolver(problemType='SP')

# Solve problem
solver.solveArray(mixArray)

# Generate report
solver.report(mixArray)
