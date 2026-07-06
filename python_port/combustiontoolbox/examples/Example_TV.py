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
mixArray = mix.setProperties('temperature', 10000, 'volume', 0.01, 'equivalenceRatio', np.arange(3.80, 5.01, 0.01))

# Initialize solver
solver = EquilibriumSolver(problemType='TV')

# Solve the problem
solver.solveArray(mixArray)

# Generate report
solver.report(mixArray)