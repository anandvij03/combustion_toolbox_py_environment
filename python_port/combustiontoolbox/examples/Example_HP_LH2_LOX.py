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
mix.set(['H2bLb'], 'fuel', 1)
mix.set(['O2bLb'], 'oxidizer' , 1)

# Define properties
mixArray = mix.setProperties('temperature', 3000, 'pressure', 1.01325, 'equivalenceRatio', np.arange(0.2, 5.05, 0.05))

# Initialize solver
solver = EquilibriumSolver(problemType='HP')

# Solve the problem
solver.solveArray(mixArray)

# Generate report
solver.report(mixArray)