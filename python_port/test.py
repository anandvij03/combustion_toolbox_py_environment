import numpy as np

# Package imports (using the working module structures from your previous run)
from combustiontoolbox.databases.NasaDatabase.NasaDatabase import NasaDatabase
from combustiontoolbox.core import ChemicalSystem, Mixture
from combustiontoolbox.equilibrium.EquilibriumSolver.EquilibriumSolver import EquilibriumSolver

def run_ev_equilibrium_sweep():
    # 1. Get NASA database instance
    db = NasaDatabase()

    # 2. Define chemical system
    system = ChemicalSystem(db)

    # 3. Initialize mixture
    mix = Mixture(system)

    # 4. Define chemical state (Fuel and Oxidizer specs)
    mix.set(['CH4'], 'fuel', 1)
    
    oxidizer_composition = np.array([78.084, 20.9476, 0.9365, 0.0319]) / 20.9476
    mix.set(['N2', 'O2', 'Ar', 'CO2'], 'oxidizer', oxidizer_composition)

    # 5. Define properties
    # Replaces '0.5:0.01:5' with exclusive-bound np.arange to ensure 5.0 is included
    equivalence_ratio_range = np.arange(0.5, 5.01, 0.01)
    
    # Passing 'volume' instead of 'pressure', and setting initial temperature to 300 K
    mix_array = mix.setProperties(
        'temperature', 300,
        'volume', 1,
        'equivalenceRatio', equivalence_ratio_range
    )

    # 6. Initialize solver for Constant Internal Energy and Constant Volume (EV)
    # Applying the keyword-argument fix to prevent the __init__() positional argument error
    solver = EquilibriumSolver(problemType='EV')
    
    # NOTE: If your backend constructor takes zero inputs, use these two lines instead:
    # solver = EquilibriumSolver()
    # solver.problemType = 'EV'

    # 7. Solve problem
    solver.solveArray(mix_array)

    # 8. Generate report
    solver.report(mix_array)

if __name__ == "__main__":
    run_ev_equilibrium_sweep()