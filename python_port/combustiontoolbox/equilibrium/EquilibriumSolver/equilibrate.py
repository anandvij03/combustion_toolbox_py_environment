import warnings

def equilibrate(self, mix2, mixGuess=None):
    """
    Obtain properties at equilibrium for the given thermochemical transformation
    
    Args:
        self (EquilibriumSolver): EquilibriumSolver object
        mix2 (Mixture): Mixture considering a thermochemical frozen gas
        mixGuess (Mixture, optional): Mixture object of a previous calculation
        
    Returns:
        mix2 (Mixture): Mixture at chemical equilibrium for the given thermochemical transformation
    """
    # Initialization
    mix1 = mix2.copy()

    # Get attribute xx of the specified transformations
    attributeName = self.getAttribute()

    # Get temperature and chemical composition TGuess
    TGuess, molesGuess = getGuess(self, mix1, mix2, mixGuess, attributeName)

    # Root finding: find the value x that satisfies f(x) = mix2.xx(x) - mix1.xx = 0
    T, STOP, molesGuess = rootFinding(self, mix1, mix2, attributeName, TGuess, molesGuess)

    # Compute properties
    self.equilibrateT(mix1, mix2, T, molesGuess)

    # Check convergence in case the problemType is TP (defined Temperature and Pressure)
    checkConvergence(mix2.errorMoles, self.tolGibbs, mix2.errorMolesIons, self.tolMultiplierIons, self.problemType)

    # Save error from root finding algorithm
    mix2.errorProblem = STOP

    return mix2


# SUB-PASS FUNCTIONS
def getGuess(self, mix1, mix2, mixGuess, attributeName):
    # Get initial estimates for temperature and molar composition

    # Initialization
    if self.problemType.upper() in ['TP', 'TV']:
        TGuess = mix2.T
        molesGuess = None

        if mixGuess is not None:
            molesGuess = mixGuess.Xi * mixGuess.N
        
        return TGuess, molesGuess

    if mixGuess is not None:
        TGuess = mixGuess.T
        molesGuess = mixGuess.Xi * mixGuess.N
    else:
        TGuess = self.regulaGuess(mix1, mix2, attributeName)
        molesGuess = None

    return TGuess, molesGuess


def rootFinding(self, mix1, mix2, attributeName, x0, molesGuess):
    # Calculate the temperature value that satisfied the problem conditions
    # using the @rootMethod
    if isinstance(self.rootMethod, str):
        method = getattr(self, self.rootMethod)
        return method(mix1, mix2, attributeName, x0, molesGuess)
    elif callable(self.rootMethod):
        if hasattr(self.rootMethod, '__self__') and self.rootMethod.__self__ is not None:
            return self.rootMethod(mix1, mix2, attributeName, x0, molesGuess)
        else:
            return self.rootMethod(self, mix1, mix2, attributeName, x0, molesGuess)
    else:
        raise TypeError(f"rootMethod '{self.rootMethod}' is not callable or a valid method name.")


def checkConvergence(STOP, TOL, STOP_ions, TOL_ions, problemType):
    # Check tolerance error if the convergence criteria was not satisfied

    if problemType.upper() != 'TP':
        return

    if STOP > TOL:
        warnings.warn(f"Convergence error number of moles:   {STOP:.2e}")

    if STOP_ions > TOL_ions:
        warnings.warn(f"Convergence error in charge balance: {STOP_ions:.2e}")
