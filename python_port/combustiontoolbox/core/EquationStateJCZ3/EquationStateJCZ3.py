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

    def __init__(self, species_names, index_condensed):
        
        #Extracts the active mixture's parameters (leaving behind the non-contributing elements) and filters out solid phases.
        
        super().__init__()
        
        # Fixed JCZ3 Physics Parameters
        self.alpha = 13.0  # Repulsive stiffness (Constrained Due to numerical Instabilities at Higher Pressures)
        self.m = 6.0       # Attractive exponent
        self.R0 = Constants.R0
        self.N_A = Constants.NA
        self.l = 13.0
        self.c = 0.577216 # Euler Mascheroni Constant
        self.num_species = len(species_names)
        self.eps_k = np.zeros(self.num_species)
        self.r_star = np.zeros(self.num_species)
        
        for i, name in enumerate(species_names):
            
            # There is one main objective here:
            # Filter out condensed phases (solids or liquids)
            # The condensed phase elements are assigned a zero volume, so as to not break
            # the solver math.
            if i in index_condensed or name not in self.JCZ3_DATABASE:
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
        # 2. Build the 2D Probability Matrix (x_i*x_j)
        x_ij_matrix = np.outer(x_array, x_array)
        # 3. Vectorized double-summation by multiplying probability by geometry
        e_0 = total_gas_moles * np.sum(x_ij_matrix * self.e_ij) # Multiplying by n gives us the correct extensive form
        V_star = total_gas_moles * np.sum(x_ij_matrix * self.v_star_ij) # Multiplying by n gives us the correct extensive form
        
        return e_0, V_star
    
 
    def _get_composition_derivatives(self, moles_array):
        n_g = np.sum(moles_array)
        x = moles_array / n_g
    
        #e0_bar_k = weighted interaction of species k with the mixture (these are intensive (per mole))
        e0_bar = self.e_ij @ x          # shape (n_species,)
        vstar_bar = self.v_star_ij @ x
    
        # e_0 and V_star are extensive (total mixture)
        e_0, V_star = self._get_mixture_parameters(moles_array)
    
        d_e0_dnk    = 2.0 * e0_bar - (e_0/n_g)
        d_Vstar_dnk = 2.0 * vstar_bar - (V_star/n_g)
    
        return d_e0_dnk, d_Vstar_dnk
    
    '''
    # Possible when we implement analytical approaches for this.
    def _get_P0(self, e_0, V_star, V):
        # Calculates the analytical Lattice Pressure P0 = -dE0/dV
        m = 6.0
        l = 13.0
        B_m = 14.45392
        B_l = 13.99166
        s_scale = (l*m)/(2*(l-m))
        
        vol_ratio_rep = (V / V_star)**(1.0 / 3.0)
        vol_ratio_att = (V_star / V)**(m/3.0)
        
        r_l = (B_l/l) * np.exp(l * (1.0 - vol_ratio_rep))
        r_m = (B_m/m) * vol_ratio_att
        
        z_lattice = l * vol_ratio_rep
        
        P0 = e_0 * (s_scale / (3.0 * V)) * (r_l * z_lattice - r_m * m)
        return P0
    '''

    # Update Numerical Differentiation to Analytical Differentiation
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
        #The function can easily be removed if validation details do not hold up.
        #The numerical approach can be updated to an analytical approach to avoid truncation errors.
        dn = dn or max(1e-6 * abs(n_g), 1e-10)
        f_plus  = self._get_f(e_0, V_star, n_g + dn, V, T)
        f_minus = self._get_f(e_0, V_star, n_g - dn, V, T)
        return (f_plus - f_minus) / (2 * dn)

    def _get_E0(self, e_0, V_star, V):
        
        #Calculates the baseline Lattice Energy (E0) for the JCZ3 EOS.
        # Fixed JCZ3 parameters
        m = 6.0
        l = 13.0 # Fixed stiffness due to numerical instabilities.
        B_m = 14.45392 # Repulsive stiffness.
        B_l = 13.99166 # Attractive stiffness.
        
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

    def _get_f(self, e_0, V_star, n_g, V, T):
        #This rests on the principle that f = f_g(y) + f_s(z) (Taken from the Sandia JCZ3 Report)
        #The function itself takes inputs of e_0, V_star, n_g.
        #The other relations are given below, regarding the obtaining of the f parameter. 
        
        #Obtaining f_g: f_g is a low density term and is expressed as a consequence of virial coefficients.
        # f_g = 1 + a1*y + a2*y^2 + a3*y^3
        # The values of a1, a2, a3 have been taken from similar values used in the TIGER code, earlier.
        # y = (V*/V)*(F_th//l)^3, where:
        # F_th = c1 - ln(T*(l-m)/(m*(e_0/(n_g*R0)))).
        # Here, c1 is given as c + l, with c being the Euler Mascheroni Constant (0.577216).
        # To find f_s, we do the following:
        # f_s = 2*((e_0/(n_g*R0*T))*(m/(l-m))*(z/pi)*(z-2)*exp(l-z))^3/2

        a1, a2, a3 = 2.96192, 7.12865, 12.4511
        l, m, R0 = self.l, self.m, self.R0
        z = l * (V_star/V)**(-1.0/3.0)
        c1 = self.c + self.l

        if n_g <= 1e-16 or e_0 <= 0:
            return 1.0  # Ideal-Gas fallback. This implies there will be no excess contribution.
                        # Check for Tuple Related Crash Here

        self.F_therm = c1 - np.log(T * (l - m) / (m * (e_0 / (n_g * R0))))
        y = (V_star/V) * (self.F_therm / l) ** 3
        f_g = 1.0 + a1 * y + a2 * y**2 + a3 * y**3

        g = (e_0/(n_g*R0*T)) * (m/(l-m)) * (z/np.pi) * (z-2.0) * np.exp(l-z)
        #A physical density regime check implies that g >= 0 for f-s to be real. This is implemented below.
        f_s = 2.0 * g ** (3.0/2.0) if g > 0 else 0.0
        f = f_g + f_s

        # Calculating the prime components for further use
        # f_g, f_s, y are to be returned for jacobian calculation
        fg_prime = a1 + 2.0 * a2 * y + 3.0 * a3 * y**2
        fg_double_prime = 2.0 * a2 + 6.0 * a3 * y
        fs_prime = 0.0 # Placeholder
        fs_double_prime = 0.0 # Placeholder

        return f, f_g, f_s, y, fg_prime, fg_double_prime, fs_prime, fs_double_prime


    def _get_P0(self, E_0, e_0, V_star, V):
        # This is a function to get the Lattice Pressure value for the given equation and state parameters. This allows to do two things:
        # 1. The Lattice Pressure is required for calculating the excess chemical potential (due to non-ideal behaviour, modelled by the JCZ3 equation).
        # 2. THe Lattice Pressure is also required for obtaining the final detonation pressure, once the other state values have been obtained.
        
        # The aim is to differentiate E0 and V (P0 is -dE0/dv)
    
        dV = max(1e-6 * abs(V), 1e-12)
        E0_plus  = self._get_E0(e_0, V_star, V + dV)
        E0_minus = self._get_E0(e_0, V_star, V - dV)
        
        return -(E0_plus - E0_minus) / (2.0 * dV)



    def getDepartureFunctions(self, moles_array, V, T):

        # Add comments here
        
        mu_excess = 0
        n_g = np.sum(moles_array)
        e_0, V_star = self._get_mixture_parameters(moles_array)
        E_0_var = self._get_E0(e_0,V_star,V)
        f_var = self._get_f(e_0,V_star,n_g,V,T)
        f_e0 = self._get_df_de0(e_0,V_star,n_g,V,T)
        df_de_0 = self._get_df_de0(e_0,V_star,n_g,V,T)
        df_dV_star = self._get_df_dVstar(e_0,V_star,n_g,V,T)
        df_dn = self._get_df_dn(e_0,V_star,n_g,V,T)
        P_0_var = self._get_P0(E_0_var,e_0,V_star,V)
        d_e0_dnk, d_Vstar_dnk = self._get_composition_derivatives(moles_array)

        mu_excess = ((E_0_var/e_0) + (n_g*self.R0*T)/(f_var)*(df_de_0))*d_e0_dnk + ((V/V_star)*P_0_var + (n_g*self.R0*T*df_dV_star)/(f_var))*d_Vstar_dnk + (self.R0)*T*np.log(f_var) + ((self.R0*T*n_g)/f_var)*df_dn

        return mu_excess

    def getDepartureJacobian(self,moles_array,V,T):
        # Calculate the Jacobian, which is the matrix of interactions across species. This is indexed by the species i and j
        # The Jacobian is the Hessian of the excess free energy, for the non-ideal chemical potential with respect to species moles.
        # This returns a dense NS x NS array

        n_g = np.sum(moles_array)
        num_species = len(moles_array)

        # Initializing the dense matrix
        J_excess = np.zeros((num_species,num_species))

        if n_g <= 1e-16:
            return J_excess
        
        RT = self.R0 * T

        # TODO: Implement the cross derivatives, which involves differentiating the terms in getDepartureFunctions()
        # Getting the mixture parameters first

        '''
        # Possible Code Improvement
        # 2. Get the 1D arrays for the first derivatives of e0 and Vstar (size: num_species)
        # This helper function naturally handles the equation from your image!
        d_e0_dn, d_Vstar_dn = self._get_composition_derivatives(moles_array)
        # 3. Calculate the 1D array of dy_dn for all species at once (size: num_species)
        dy_dn = (y / n_g) * ( (n_g / V_star) * d_Vstar_dn + (3.0 / self.F_therm) * ((n_g / e_0) * d_e0_dn - 1.0) )
        An additional refactor into vectorized operations can be done to improve solver performance.
        '''
        
        e_0, V_star = self._get_mixture_parameters(moles_array)
        f_var, f_g, f_s, y, fg_prime, fg_double_prime, fs_prime, fs_double_prime = self._get_f(e_0, V_star, n_g, V, T)
        d_e0_dn, d_Vstar_dn = self._get_composition_derivatives(moles_array)
        # Derivatives of y, with respect to n_i and n_j

        for i in range(num_species):
            for j in range(num_species):
                dy_dni = (y / n_g) * ( (n_g / V_star) * d_Vstar_dn[i] + (3.0 / self.F_therm) * ((n_g / e_0) * d_e0_dn[i] - 1.0) )
                dy_dnj = (y / n_g) * ( (n_g / V_star) * d_Vstar_dn[j] + (3.0 / self.F_therm) * ((n_g / e_0) * d_e0_dn[j] - 1.0) )        
                
                # Verify
                # Second cross-derivatives for the mixture rules e_0 and V_star
                e_ij = self.e_ij[i, j] 
                term1_e0 = (2.0 * n_g * e_ij) / e_0
                term2_e0 = - (n_g / e_0) * d_e0_dn[i]
                term3_e0 = - (n_g / e_0) * d_e0_dn[j]
                d2e0_dni_dnj = (e_0 / n_g**2) * (term1_e0 + term2_e0 + term3_e0)
                vstar_ij = self.v_star_ij[i, j]
                term1_vstar = (2.0 * n_g * vstar_ij) / V_star
                term2_vstar = - (n_g / V_star) * d_Vstar_dn[i]
                term3_vstar = - (n_g / V_star) * d_Vstar_dn[j]
                d2Vstar_dni_dnj = (V_star / n_g**2) * (term1_vstar + term2_vstar + term3_vstar)
                
                # Verify
                # Equation A29 for the second cross-derivative of y
                term1 = (n_g / y * dy_dni) * (n_g / y * dy_dnj)
                term2 = - (n_g / V_star * d_Vstar_dn[i]) * (n_g / V_star * d_Vstar_dn[j])
                term3 = (n_g**2 / V_star) * d2Vstar_dni_dnj
                term4 = - (3.0 / self.F_therm**2) * ( (n_g / e_0 * d_e0_dn[i]) - 1.0 ) * ( (n_g / e_0 * d_e0_dn[j]) - 1.0 )
                term5 = (3.0 / self.F_therm) * ( (n_g**2 / e_0) * d2e0_dni_dnj - (n_g / e_0 * d_e0_dn[i]) * (n_g / e_0 * d_e0_dn[j]) + 1.0 )
                d2y_dni_dnj = (y / n_g**2) * (term1 + term2 + term3 + term4 + term5)

                z = self.l * (V_star/V)**(-1.0/3.0)
                # Protect against ZeroDivisionError when z = 2.0 (where f_s = 0 anyway)
                if f_s > 1e-32:
                    z1 = (2.0 - z) - (2.0 / (2.0 - z))
                    z2 = (z / 2.0) + (z / (2.0 - z)**2)
                # The vectorized array for dfs_dn (size: num_species)
                    dfs_dn = (f_s / n_g) * ( -(z1 / 2.0) * (n_g / V_star) * d_Vstar_dn + 1.5 * ((n_g / e_0) * d_e0_dn - 1.0) )
                else:
                    z1 = 0.0
                    z2 = 0.0
                    dfs_dn = np.zeros(num_species)

                # Verify
                # Equation for dfg_dni (and by symmetry, dfg_dnj)
                dfg_dni = (1.0 / n_g) * ( y * fg_prime * (n_g / y) * dy_dni )
                dfg_dnj = (1.0 / n_g) * ( y * fg_prime * (n_g / y) * dy_dnj )


            # Verify
                if f_s > 1e-32:
                    term1 = (n_g / f_s * dfs_dn[j]) * (n_g / f_s * dfs_dn[i])
                    term2 = -1.5 * ( (n_g / e_0 * d_e0_dn[i]) * (n_g / e_0 * d_e0_dn[j]) - (n_g**2 / e_0) * d2e0_dni_dnj - 1.0 )
                    term3 = z1 * ( (n_g / V_star * d_Vstar_dn[i]) * (n_g / V_star * d_Vstar_dn[j]) - (n_g**2 / V_star) * d2Vstar_dni_dnj )
                    term4 = - (z2 / 3.0) * ( (n_g / V_star * d_Vstar_dn[i]) * (n_g / V_star * d_Vstar_dn[j]) )
                    
                    d2fs_dni_dnj = (f_s / n_g**2) * (term1 + term2 + term3 + term4)
                else:
                    d2fs_dni_dnj = 0.0

            # Finding the Lattice Terms:
                # Fixed JCZ3 parameters
                m = 6.0
                l = 13.0 # Fixed stiffness due to numerical instabilities.
                B_m = 14.45392 # Repulsive stiffness.
                B_l = 13.99166 # Attractive stiffness.
                
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
                Z = s * (r_l - r_m)

                dZ_dVstar = s/ (3.0 * V_star) * (r_l * z - r_m * m)
                d2Z_dV_star2 = s/ (9.0 * (V_star**2)) * (r_l * z * (z - 4) - r_m * m * (m - 3))
                
                # Verify
                term1 = (e_0 / (n_g * RT)) * (
                    (n_g**2 / e_0) * d2e0_dni_dnj * Z + 
                    (V_star**2) * d2Z_dV_star2 * (n_g / V_star * d_Vstar_dn[i]) * (n_g / V_star * d_Vstar_dn[j])
                )
                term2 = (e_0 * V_star / (n_g * RT)) * dZ_dVstar * (
                    (n_g**2 / V_star) * d2Vstar_dni_dnj + 
                    (n_g / e_0 * d_e0_dn[j]) * (n_g / V_star * d_Vstar_dn[i]) + 
                    (n_g / e_0 * d_e0_dn[i]) * (n_g / V_star * d_Vstar_dn[j])
                )
                
                d2E0_dnj_dni = (term1 + term2)* (RT/ (n_g ** 2))


                d2fg_dni_dnj = (1.0/(n_g**2)) * (
                    (y**2) * fg_double_prime * ((n_g / y) * dy_dni) * ((n_g / y) * dy_dnj) + 
                    y * fg_prime * ((n_g**2) / y) * d2y_dni_dnj
                )

                
                

                # dI_dni, dI_dnj
                dI_dni = (1.0 / f_var) * (dfg_dni + dfs_dn[i])
                dI_dnj = (1.0 / f_var) * (dfg_dnj + dfs_dn[j])

                # d2_I_dni_dnj 
                d2I_dni_dnj = (1.0/(n_g**2))*(-(n_g * dI_dni) * (n_g * dI_dnj) + ((n_g**2) / f_var) * (d2fg_dni_dnj + d2fs_dni_dnj))
                


                # Equation (A68) dGamma_dln_nj
                # Gamma = mu_excess/RT (the imperfection factor)
                # I = ln(f), and f = f_s + f_g
                # Typographical error in the overall structure of the matrix. Apparently it is not dGamma_i_dln_nj, as shown in the 
                # JCZ3 manual, but rather dGamma_i_dnj, which seems to be the partial derivative obtained upon differentiation (checked).
                dGamma_i_dnj = (1.0 / n_g) * (
                    (n_g / RT) * d2E0_dnj_dni + 
                    n_g * dI_dni + 
                    n_g * dI_dnj + 
                    (n_g**2) * d2I_dni_dnj
                )

                J_excess[i, j] = dGamma_i_dnj

        return J_excess


    # Dummy Functions to reduce abstract class errors:
    def getPressure(self, *args, **kwargs):
        raise NotImplementedError("JCZ3 getPressure not yet implemented.")

    def getVolume(self, *args, **kwargs):
        raise NotImplementedError("JCZ3 getVolume not yet implemented.")

    def getPressureDerivativesDimensional(self, *args, **kwargs):
        raise NotImplementedError("JCZ3 getPressureDerivativesDimensional not yet implemented.")
        
    def getVolumeDerivatives(self, *args, **kwargs):
        raise NotImplementedError("JCZ3 getVolumeDerivatives not yet implemented.")
        
    def getTemperature(self, *args, **kwargs):
        raise NotImplementedError("JCZ3 getTemperature not yet implemented.")

'''
Potential Improvement
If you use np.sum(moles_array) without masking, you are dividing the gas-phase interactions by the total moles of the system (gas + condensed). 
As you noted, for PBXN-111, this is a massive error. The equilibrium composition of PBXN-111 detonation products contains substantial amounts 
of solid graphite and solid aluminum oxide (Al2O3). Because the numerator (the interaction matrices e_ij)
correctly evaluates to zero for these solids, but your denominator erroneously includes their moles, you were artificially 
deflating the mole fractions (xi) of the active gases. This would cause the solver to severely underestimate the extreme high-pressure repulsions of the JCZ3 potential.
 '''