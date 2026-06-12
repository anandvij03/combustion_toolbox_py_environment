import numpy as np
# Assuming your ported classes are structured like this:
from combustiontoolbox.databases import BurcatDatabase
from combustiontoolbox.core.Mixture import Mixture
from combustiontoolbox.core.ChemicalSystem import ChemicalSystem
from combustiontoolbox.equilibrium.EquilibriumSolver import EquilibriumSolver

# ---------------------------------------------------------
# 1. Initialize the Database and Chemical System
# ---------------------------------------------------------
# Load the database you converted earlier
db = BurcatDatabase("databases/thermo_millennium_2_thermoNASA9.inp")

# Define the pool of potential products you want to consider
species_list = ["CH4", "O2", "N2", "CO2", "CO", "H2O", "H2", "OH", "O", "H"]
system = ChemicalSystem(db, species_list)

# ---------------------------------------------------------
# 2. Define the Initial Mixture (Reactants)
# ---------------------------------------------------------
# Create a fresh mixture instance bound to your chemical system
mix = Mixture(system)

# Set up a stoichiometric Methane/Air mixture: CH4 + 2(O2 + 3.76 N2)
initial_mole_fractions = {
    "CH4": 1.0,
    "O2": 2.0,
    "N2": 7.52
}
mix.set_composition(initial_mole_fractions)

# Set state variables (e.g., T = 2500 K, P = 101325 Pa)
mix.T = 2500.0  # Kelvin
mix.P = 101325.0  # Pascals

# ---------------------------------------------------------
# 3. Instantiate the Solver and Equilibrate
# ---------------------------------------------------------
# Configure the solver for a Constant Temperature and Pressure (TP) problem
solver = EquilibriumSolver(problem_type="TP", tol_gibbs=1e-6)

# Run the solver. It returns a new mixture object in chemical equilibrium
eq_mix = solver.equilibrate(mix)

# ---------------------------------------------------------
# 4. Inspect the Results
# ---------------------------------------------------------
print(f"Solver Status/Error Code: {eq_mix.error_problem}")
print(f"Equilibrium Temperature: {eq_mix.T} K")

print("\nEquilibrium Composition (Mole Fractions):")
for species in species_list:
    # Safely get the mole fraction of each species from the mixture array
    x_i = eq_mix.get_mole_fraction(species)
    if x_i > 1e-5:  # Only print major species
        print(f"  {species:<5}: {x_i:.5f}")