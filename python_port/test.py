import numpy as np

# Package imports (Keep whatever import style successfully resolved your last error)
from combustiontoolbox.databases.NasaDatabase.NasaDatabase import NasaDatabase
from combustiontoolbox.core import ChemicalSystem, Mixture
from combustiontoolbox.equilibrium.EquilibriumSolver.EquilibriumSolver import EquilibriumSolver

def run_equilibrium_sweep():
    print("Loading NASA database ...")
    # 1. Get NASA database instance
    db = NasaDatabase()

    # 2. Define the chemical system space using the database
    system = ChemicalSystem(db)

    # 3. Initialize the baseline mixture
    mix = Mixture(system)

    # 4. Define the chemical state
    # Your literal port maps MATLAB's set(mix, ...) directly to the mix.set(...) method
    mix.set(['CH4'], 'fuel', 1)

    # Use a numpy array for the element-wise division of the oxidizer composition
    oxidizer_composition = np.array([78.084, 20.9476, 0.9365, 0.0319]) / 20.9476
    mix.set(['N2', 'O2', 'Ar', 'CO2'], 'oxidizer', oxidizer_composition)

    # 5. Define properties and sweep over equivalence ratios
    # MATLAB '0.5:0.01:5' matches np.arange (we use 5.01 since the upper limit is exclusive)
    equivalence_ratio_range = np.arange(0.5, 5.01, 0.01)
    
    # Retaining your exact camelCase method name and positional argument format
    mix_array = mix.setProperties(
        'temperature', 3000, 
        'pressure', 1 * 1.01325, 
        'equivalenceRatio', equivalence_ratio_range
    )

    # 6. Initialize the solver using your literal configuration style
    solver = EquilibriumSolver(problemType= 'TP')

    # 7. Solve the state array
    solver.solveArray(mix_array)

    # 8. Generate report
    # If your report function is a method inside the solver class:
    solver.report(mix_array)
    
    # Note: If 'report' was ported as a global standalone function instead, 
    # use: report(solver, mix_array)

if __name__ == "__main__":
    run_equilibrium_sweep()