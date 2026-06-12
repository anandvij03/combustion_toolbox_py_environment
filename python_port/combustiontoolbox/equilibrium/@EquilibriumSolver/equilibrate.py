import warnings
from typing import Any, Optional, Tuple

class EquilibriumSolver:
    """
    EquilibriumSolver handles the calculation of chemical and thermochemical 
    equilibrium properties for a given mixture.
    """
    
    def __init__(self):
        self.tol_gibbs: float = 1e-6
        self.tol_multiplier_ions: float = 1e-6
        self.problem_type: str = 'TP' 

    def equilibrate(self, mix2: Any, mix_guess: Optional[Any] = None) -> Any:
        """
        Obtain properties at equilibrium for the given thermochemical transformation.
        
        Args:
            mix2 (Mixture): Mixture considering a thermochemical frozen gas.
            mix_guess (Mixture, optional): Mixture object of a previous calculation.
            
        Returns:
            mix2 (Mixture): Mixture at chemical equilibrium.
        """
        # Initialization
        mix1 = mix2.copy()
        
        # Get attribute name of the specified transformations
        attribute_name = self.get_attribute()
        
        # Get temperature and chemical composition guess
        t_guess, moles_guess = self._get_guess(mix1, mix2, mix_guess, attribute_name)
        
        # Root finding: find the value x that satisfies f(x) = mix2.xx(x) - mix1.xx = 0
        t, stop, moles_guess = self._root_finding(mix1, mix2, attribute_name, t_guess, moles_guess)
        
        # Compute properties
        self.equilibrate_t(mix1, mix2, t, moles_guess)
        
        # Check convergence if problemType is TP
        self._check_convergence(
            mix2.error_moles, 
            self.tol_gibbs, 
            mix2.error_moles_ions, 
            self.tol_multiplier_ions, 
            self.problem_type
        )
        
        # Save error from root finding algorithm
        mix2.error_problem = stop
        
        return mix2

    # Helper methods

    def _get_guess(self, mix1: Any, mix2: Any, mix_guess: Optional[Any], attribute_name: str) -> Tuple[float, Any]:
        """Get initial estimates for temperature and molar composition."""
        
        # Standardize string checking using .upper() or .lower()
        if self.problem_type.upper() in ['TP', 'TV']:
            t_guess = mix2.T
            moles_guess = None
            
            if mix_guess is not None:
                moles_guess = mix_guess.Xi * mix_guess.N
                
            return t_guess, moles_guess

        if mix_guess is not None:
            t_guess = mix_guess.T
            moles_guess = mix_guess.Xi * mix_guess.N
        else:
            t_guess = self.regula_guess(mix1, mix2, attribute_name)
            moles_guess = None
            
        return t_guess, moles_guess

    def _root_finding(self, mix1: Any, mix2: Any, attribute_name: str, x0: float, moles_guess: Any) -> Tuple[float, Any, Any]:
        """Calculate the temperature value that satisfies the problem conditions using the root method."""
        return self.root_method(self, mix1, mix2, attribute_name, x0, moles_guess)

    def _check_convergence(self, stop: float, tol: float, stop_ions: float, tol_ions: float, problem_type: str) -> None:
        """Check tolerance error if the convergence criteria was not satisfied."""
        
        if problem_type.upper() != 'TP':
            return

        if stop > tol:
            warnings.warn(f"Convergence error number of moles: {stop:.2e}")

        if stop_ions > tol_ions:
            warnings.warn(f"Convergence error in charge balance: {stop_ions:.2e}")