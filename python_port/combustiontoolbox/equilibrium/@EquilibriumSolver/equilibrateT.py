import numpy as np
from combustiontoolbox.common.Units import Units

def equilibrateT(self, mix1, mix2, T, molesGuess=None):
    """
    Obtain equilibrium properties and composition for the given
    temperature [K] and pressure [bar] / specific volume [m3/kg]
    
    Args:
        self (EquilibriumSolver): EquilibriumSolver object
        mix1 (Mixture): Properties of the initial mixture
        mix2 (Mixture): Properties of the final mixture
        T (float): Temperature [K]
        molesGuess (float, optional): Mixture composition [mol] of a previous computation
        
    Returns:
        mix2 (Mixture): Properties of the final mixture
    """
    # Check if calculations are for a calorically perfect gas
    if self.caloricGasModel.isPerfect():
        return equilibrateTPerfect(mix1, mix2, T)

    # Check if calculations are for a thermally perfect gas
    if self.caloricGasModel.isThermallyPerfect():
        return equilibrateTFrozen(mix2, T)

    # Definitions
    moles_guess_val = None
    N_mix0 = np.asarray(moles(mix1))  # Get moles of inert species
    system = mix2.chemicalSystem
    productSpeciesSet = mix2.productSpeciesSet

    # Unpack additional inputs
    if molesGuess is not None:
        moles_guess_val = molesGuess
        if system.FLAG_COMPLETE:
            moles_guess_val = None
        elif hasattr(moles_guess_val, "__len__") and len(moles_guess_val) > 0:
            moles_guess_val = np.asarray(moles_guess_val)[productSpeciesSet.indexGlobal]

    # Check flag
    if not self.FLAG_FAST:
        moles_guess_val = None

    # Compute number of moles
    N, dNi_T, dN_T, dNi_p, dN_p, indexProducts, STOP, STOP_ions, h0 = selectEquilibrium(
        self, system, productSpeciesSet, T, mix1, mix2, moles_guess_val
    )

    # Compute mixture state at chemical equilibrium
    def setMixture(mix):
        nonlocal N, dNi_T, dNi_p, h0
        # Scatter only the final active equilibrium species; candidate product slots may contain trial values
        indexActive = np.asarray(productSpeciesSet.indexGlobal)[indexProducts]
        N = productVectorToSystemVector(system.numSpecies, productSpeciesSet, N, indexProducts)
        dNi_T = productVectorToSystemVector(system.numSpecies, productSpeciesSet, dNi_T, indexProducts)
        dNi_p = productVectorToSystemVector(system.numSpecies, productSpeciesSet, dNi_p, indexProducts)
        h0 = productVectorToSystemVector(system.numSpecies, productSpeciesSet, h0, indexProducts)
        N[system.indexFrozen] = N_mix0[system.indexFrozen]

        # Assign values
        mix.T = T
        mix.dNi_T = dNi_T
        mix.dN_T = dN_T
        mix.dNi_p = dNi_p
        mix.dN_p = dN_p
        mix.FLAG_REACTION = True
        mix.errorMoles = STOP
        mix.errorMolesIons = STOP_ions

        # Compute properties of final mixture
        mix.setMolesFast(N, h0, indexActive)

    setMixture(mix2)
    return mix2


# SUB-PASS FUNCTIONS
def moles(mix):
    # Get the moles [mol] of all the species in the mixture
    return mix.Xi * mix.N


def selectEquilibrium(self, system, productSpeciesSet, T, mix1, mix2, molesGuess):
    # Select equilibrium: TP: Gibbs; TV: Helmholtz
    if self.problemType.upper().endswith('P'):
        return self.equilibriumGibbs(system, productSpeciesSet, mix2.p, T, mix1, molesGuess)
    
    return self.equilibriumHelmholtz(system, productSpeciesSet, mix2.v, T, mix1, molesGuess)


def productVectorToSystemVector(numSpecies, productSpeciesSet, vectorProduct, indexProduct):
    # Map product species vector into ChemicalSystem species order
    vector = np.zeros(numSpecies)
    vectorProduct = np.asarray(vectorProduct)
    indexProduct = np.asarray(indexProduct)
    global_indices = np.asarray(productSpeciesSet.indexGlobal)[indexProduct]
    vector[global_indices] = vectorProduct[indexProduct]
    return vector


def equilibrateTPerfect(mix1, mix2, T):
    # Obtain equilibrium properties and composition for the given
    # temperature [K] and pressure [bar] assuming a calorically perfect gas
    Tref = 298.15

    # Recompute properties of mix2
    mix2.setTemperature(T)

    # Change properties that remains thermochemically frozen
    mix2.cp = mix1.cp
    mix2.cv = mix1.cv
    mix2.gamma = mix1.gamma
    mix2.gamma_s = mix1.gamma_s
    mix2.sound = np.sqrt(mix2.gamma * mix2.p * Units.bar2Pa / mix2.rho)

    # Compute enthalpy [J]
    mix2.hf = mix1.hf
    mix2.DhT = mix1.cp * (T - Tref)
    mix2.h = mix2.hf + mix2.DhT

    # Compute internal energy [J]
    mix2.ef = mix1.ef
    mix2.DeT = mix1.cv * (T - Tref)
    mix2.e = mix2.ef + mix2.DeT

    # Compute entropy [J/K]
    mix2.s0 = mix1.s0 + mix1.cp * np.log(T / mix1.T)
    mix2.s = mix2.s0 + mix2.Ds
    
    # Compute Gibbs free energy
    mix2.g = mix2.h - T * mix2.s

    if mix2.u is None or (hasattr(mix2.u, "__len__") and len(mix2.u) == 0):
        return mix2
    
    mix2.uShock = mix2.u
    mix2.mach = mix2.u / mix2.sound
    return mix2


def equilibrateTFrozen(mix, T):
    # Obtain equilibrium properties and composition for the given
    # temperature [K] and pressure [bar] assuming a thermally perfect gas
    mix.FLAG_REACTION = False
    mix.T = T
    mix.quantity = mix.Xi * mix.N
    mix.listSpecies = mix.chemicalSystem.listSpecies
    
    # Update indexSpecies
    mix.updateIndexSpecies()

    # Update thermodynamic properties assuming a thermally perfect gas
    mix.updateThermodynamics()
    return mix
