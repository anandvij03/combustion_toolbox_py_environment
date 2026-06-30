import numpy as np
from combustiontoolbox.core.EquationState.EquationState import EquationState
from combustiontoolbox.common.Constants import Constants

class EquationStateJCZ3(EquationState):
    """
    Implements the JCZ3 equation of state, for non-ideal situations, 
    and high pressure detonation products. The Exponential-6 soft-wall potential is used here.
    This is particularly useful when modelling detonations of certain special explosives, such as the PBXN family.
    """
    R0 = Constants.R0
    N_A = Constants.NA # Avogadro's Number


# LEVEL 5: THE BASE SPECIES DATA
    # Lookup table for Lennard-Jones/EXP-6 parameters.
    # r_star --> Angstroms, epsilon_k --> Kelvin.
    JCZ3_DATABASE = {
        # Other Radicals from different species must be added here
        'C':    {'r_star': 2.50, 'epsilon_k': 100.0},
        'CH4':  {'r_star': 3.90, 'epsilon_k': 200.0},
        'CHNO': {'r_star': 4.32, 'epsilon_k': 180.0},
        'CO':   {'r_star': 4.10, 'epsilon_k': 30.0},
        'CO2':  {'r_star': 4.30, 'epsilon_k': 240.0},
        'H':    {'r_star': 2.70, 'epsilon_k': 3.0},
        'H2':   {'r_star': 3.75, 'epsilon_k': 4.0},
        'H2O':  {'r_star': 3.85, 'epsilon_k': 50.0},
        'N':    {'r_star': 2.30, 'epsilon_k': 80.0},
        'N2':   {'r_star': 4.11, 'epsilon_k': 103.0},
        'N2H2': {'r_star': 4.26, 'epsilon_k': 150.0},
        'N2H4': {'r_star': 4.75, 'epsilon_k': 205.0},
        'NH3':  {'r_star': 4.10, 'epsilon_k': 70.0},
        'O':    {'r_star': 3.20, 'epsilon_k': 50.0},
        'O2':   {'r_star': 3.83, 'epsilon_k': 130.0},
        'O2-':  {'r_star': 4.00, 'epsilon_k': 125.0},
        'O2+':  {'r_star': 3.00, 'epsilon_k': 125.0},
        'O3':   {'r_star': 4.30, 'epsilon_k': 250.0},
        'OH':   {'r_star': 3.30, 'epsilon_k': 80.0},
    }