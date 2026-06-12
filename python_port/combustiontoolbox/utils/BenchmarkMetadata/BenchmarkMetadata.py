import math
import pandas as pd
from typing import Any, List, Union

class BenchmarkMetadata:
    """
    The BenchmarkMetadata class is used to store and manage metadata
    information for benchmark tests in the Combustion Toolbox.

    Example usage:
        metadata = BenchmarkMetadata(solver, mixture_array, filename='run_validation_TP_CEA_1')
    """

    def __init__(self, solver: Any, mixture_array: List[Any], filename: str):
        # Set primary inputs
        self.solver = solver
        self._mixture_array = mixture_array
        self.filename = filename

        # Initialize public attributes
        self.module: str = ""
        self.problem_type: str = ""
        self.num_cases: int = 0
        self.num_species: int = 0
        self.tolerance: Union[float, List[float]] = float('nan')
        self.avg_time: float = 0.0

        # Extract metadata from the solver and mixture array
        self._get_metadata()

    def set_time(self, time: float):
        """Set average execution time. Returns self for method chaining."""
        self.avg_time = time
        return self

    def display(self):
        """Display metadata information."""
        print("Benchmark Metadata:")
        print(f"  Module:            {self.module}")
        print(f"  Problem Type:      {self.problem_type}")
        print(f"  Filename:          {self.filename}")
        print(f"  Number of Cases:   {self.num_cases}")
        print(f"  Number of Species: {self.num_species}")
        
        # Format tolerance gracefully (in case it is an array/list)
        if isinstance(self.tolerance, (list, tuple)):
            tol_str = ", ".join([f"{t:.2e}" for t in self.tolerance])
        else:
            tol_str = f"{self.tolerance:.2e}" if not math.isnan(self.tolerance) else "NaN"
            
        print(f"  Tolerance:         {tol_str}")
        print(f"  Average Time:      {self.avg_time:.6f} seconds")

    def as_dict(self) -> dict:
        """
        Export metadata as a dictionary.
        This is heavily optimized for pandas DataFrame creation.
        """
        solver_class = self.solver.__class__.__name__

        if isinstance(self.tolerance, (list, tuple)):
            tol_str = ", ".join([f"{t:.2e}" for t in self.tolerance])
        else:
            tol_str = f"{self.tolerance:.2e}" if not math.isnan(self.tolerance) else "NaN"

        return {
            'Module': self.module,
            'Solver': solver_class,
            'Problem': self.problem_type,
            'Filename': self.filename,
            'Cases': self.num_cases,
            'Species': self.num_species,
            'Tolerance': tol_str,
            'AvgTime': self.avg_time
        }

    def as_table(self) -> pd.DataFrame:
        """
        Export metadata as a single-row pandas DataFrame.
        Direct equivalent to MATLAB's asTable() method.
        """
        return pd.DataFrame([self.as_dict()])

    # PRIVATE METHODS

    def _get_metadata(self):
        """Extract metadata properties from the solver and mixture_array."""
        solver_class_name = self.solver.__class__.__name__

        # 1. Get module name based on class name
        if 'EquilibriumSolver' in solver_class_name:
            self.module = 'CT-EQUIL'
        elif any(x in solver_class_name for x in ['ShockSolver', 'DetonationSolver', 'JumpConditionsSolver']):
            self.module = 'CT-SD'
        elif 'RocketSolver' in solver_class_name:
            self.module = 'CT-ROCKET'
        elif 'HelmholtzSolver' in solver_class_name:
            self.module = 'CT-TURBULENCE'
        else:
            self.module = 'Unknown'

        # 2. Get problem type
        self.problem_type = getattr(self.solver, 'problemType', 
                            getattr(self.solver, 'problem_type', 'Unknown'))

        # 3. Get number of cases
        self.num_cases = len(self._mixture_array)

        # 4. Get number of species
        try:
            first_mixture = self._mixture_array[0]
            chem_sys = getattr(first_mixture, 'chemicalSystem', 
                       getattr(first_mixture, 'chemical_system', None))
            
            list_species = getattr(chem_sys, 'listSpecies', 
                           getattr(chem_sys, 'list_species', []))
            
            self.num_species = len(list_species)
        except (IndexError, AttributeError):
            self.num_species = 0

        # 5. Get tolerance
        if 'EquilibriumSolver' in solver_class_name and self.problem_type in ['TP', 'TV']:
            self.tolerance = getattr(self.solver, 'tolGibbs', 
                             getattr(self.solver, 'tol_gibbs', float('nan')))
        else:
            self.tolerance = getattr(self.solver, 'tol0', float('nan'))