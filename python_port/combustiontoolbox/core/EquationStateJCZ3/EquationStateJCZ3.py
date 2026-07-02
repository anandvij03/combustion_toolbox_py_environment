from combustiontoolbox.core import MixtureConfig
from combustiontoolbox.core import MixtureConfig
from combustiontoolbox.core.MixtureConfig import MixtureConfig
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

    def __init__(self, species_list):
        
        #Extracts the active mixture's parameters (leaving behind the non-contributing elements) and filters out solid phases.
        
        super().__init__()
        
        # Fixed JCZ3 Physics Parameters
        self.alpha = 13.0  # Repulsive stiffness (Constrained Due to numerical Instabilities at Higher Pressures)
        self.m = 6.0       # Attractive exponent
        self.R0 = Constants.R0
        self.N_A = Constants.NA
        
        self.num_species = len(species_list)
        self.eps_k = np.zeros(self.num_species)
        self.r_star = np.zeros(self.num_species)
        
        for i, species in enumerate(species_list):
            name = species.name
            
            # There is one main objective here:
            # Filter out condensed phases (solids or liquids)
            # The condensed phase elements are assigned a zero volume, so as to not break
            # the solver math.
            if species.phase == 'condensed' or name not in self.JCZ3_DATABASE:
                self.eps_k[i] = 0.0
                self.r_star[i] = 0.0
            else:
                self.eps_k[i] = self.JCZ3_DATABASE[name]['epsilon_k']
                # Convert Angstroms to meters for SI unit calculations
                self.r_star[i] = self.JCZ3_DATABASE[name]['r_star'] * 1e-10
                
        # Precalculating the interaction matrices (nxn, Jacobian)
        self._precompute_interactions()

    def _precompute_interactions(self):
        
        #Building the N x N pairwise interaction matrices using Lorentz-Berthelot mixture rules.
        #These rules define the collision diameter, as well as the energy well for an 'ij'
        #combination, wherein the case is of two different gases reacting.
        
        # e_ij geometric mean
        # np.outer performs rapid vectorized cross-multiplication
        self.e_ij = self.R0 * np.sqrt(np.outer(self.eps_k, self.eps_k))
        
        # r_ij arithmetic mean
        # Broadcasting creates the addition matrix without loops
        r_ij = (self.r_star[:, None] + self.r_star[None, :]) / 2.0
        
        # Calculate the collision volume matrix (v_star_ij)
        self.v_star_ij = (self.N_A / np.sqrt(2.0)) * (r_ij ** 3)

    def _get_mixture_parameters(self, moles_array):
        
        #This function calculates mixture well depth (e_0) and collision volume (V_star)
        #Again, as shown above as well, the reason for vectorizing operations is that this loop
        # is called hundreds of times during the Gordon-McBride approach with the NR solver (cycle).
        # The loop based approach would increase runtime significantly.
        
        total_gas_moles = np.sum(moles_array)
        
        if total_gas_moles <= 1e-16:
            return 0.0, 0.0
            
        # 1. Get current mole fractions
        x_array = moles_array / total_gas_moles
        
        # 2. Build the 2D Probability Matrix (x_i * x_j)
        x_ij_matrix = np.outer(x_array, x_array)
        
        # 3. Vectorized double-summation by multiplying probability by geometry
        e_0 = total_gas_moles * np.sum(x_ij_matrix * self.e_ij) # Multiplying by n gives us the correct extensive form
        V_star = total_gas_moles * np.sum(x_ij_matrix * self.v_star_ij) # Multiplying by n gives us the correct extensive form
        
        return e_0, V_star
    
 
    def _get_composition_derivatives(self, moles_array):
        n_g = np.sum(moles_array)
        x = moles_array / n_g
    
        #e0_bar_k = weighted interaction of species k with the mixture
        e0_bar = self.e_ij @ x          # shape (n_species,)
        vstar_bar = self.v_star_ij @ x
    
        e_0, V_star = self._get_mixture_parameters(moles_array)
    
        d_e0_dnk    = (2.0/n_g) * (e0_bar-e_0)
        d_Vstar_dnk = (2.0/n_g) * (vstar_bar-V_star)
    
        return d_e0_dnk, d_Vstar_dnk

    def _get_df_de0(self, e_0, V_star, n_g, V, T, de=None):
        #∂f/∂e0, holding V*, n_g, V, T fixed
        de = de or max(1e-6 * abs(e_0), 1e-8)
        f_plus  = self._get_f(e_0 + de, V_star, n_g, V, T)
        f_minus = self._get_f(e_0 - de, V_star, n_g, V, T)
        return (f_plus - f_minus) / (2 * de)

    def _get_df_dVstar(self, e_0, V_star, n_g, V, T, dV=None):
        #∂f/∂V*, holding e0, n_g, V, T fixed
        dV = dV or max(1e-6 * abs(V_star), 1e-12)
        f_plus  = self._get_f(e_0, V_star + dV, n_g, V, T)
        f_minus = self._get_f(e_0, V_star - dV, n_g, V, T)
        return (f_plus - f_minus) / (2 * dV)

    def _get_df_dn(self, e_0, V_star, n_g, V, T, dn=None):
        #∂f/∂n (total gas moles), holding e0, V*, V, T fixed.
        #The necessity of this term is not settled yet, due to a change in
        #how the algorithm works, updated in a Sandia JCZ3 paper in 2025. 
        #The function can easily be removed if validation details do not hold up."""
        dn = dn or max(1e-6 * abs(n_g), 1e-10)
        f_plus  = self._get_f(e_0, V_star, n_g + dn, V, T)
        f_minus = self._get_f(e_0, V_star, n_g - dn, V, T)
        return (f_plus - f_minus) / (2 * dn)

    def _get_E0(self, e_0, V_star, V):
        """
        Calculates the baseline Lattice Energy (E0) for the JCZ3 EOS.
        """
        # Fixed JCZ3 parameters
        m = 6.0
        l = 13.0
        B_m = 14.45392
        B_l = 13.99166
        
        # Scaling factor: s = (m*l) / (2*(l-m))
        s = (l*m)/(2*(l-m)) 
        
        # Geometrical volume ratios
        # Note: Repulsion scales with (V/V*), Attraction scales with (V*/V)
        vol_ratio_rep = (V / V_star)**(1.0 / 3.0)
        vol_ratio_att = (V_star / V)**(m/3.0)
        
        # Compute individual branch forces
        r_l = (B_l/l) * np.exp(l * (1.0 - vol_ratio_rep))
        r_m = (B_m/m) * vol_ratio_att
        
        # Assembly
        z = s * (r_l - r_m)
        E_0 = e_0 * z
        return E_0