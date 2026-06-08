def setStagnation(mix, FLAG_THERMO_CHEM=None):
    """
    Update mixture properties to stagnation conditions
    
    Args:
        mix (Mixture): Mixture object
        FLAG_THERMO_CHEM (bool, optional): Flag to indicate thermochemically frozen state
        
    Returns:
        mix (Mixture): Mixture object at stagnation conditions
    """
    from combustiontoolbox.equilibrium.EquilibriumSolver import EquilibriumSolver
    import copy

    if FLAG_THERMO_CHEM is None:
        FLAG_THERMO_CHEM = not getattr(mix, 'FLAG_REACTION', False)

    # Shallow copy
    mix = copy.copy(mix)

    # Thermochemical frozen case
    if FLAG_THERMO_CHEM:
        # Additional shallow copy
        mix1 = copy.copy(mix)

        solver = EquilibriumSolver(problemType='TP', FLAG_TCHEM_FROZEN=True, FLAG_RESULTS=False)
        mix.T = mix.T * (1 + 0.5 * (mix.gamma - 1) * mix.mach**2)
        mix.p = mix.p * (1 + 0.5 * (mix.gamma - 1) * mix.mach**2)**(mix.gamma / (mix.gamma - 1))
        solver.equilibrateT(mix1, mix, mix.T)

        # Set flow velocity [m/s]
        mix.u = 0
        mix.uShock = 0
        mix.uNormal = 0
        return mix

    # Definitions
    s_target = mix.s
    h_target = mix.h + 0.5 * mix.u**2 * mix.mi
    solver = EquilibriumSolver(problemType='HP', FLAG_TCHEM_FROZEN=False, FLAG_RESULTS=False)

    # Set initial guess for stagnation pressure (assuming calorically perfect gas) [bar]
    p = mix.p * (1 + 0.5 * (mix.gamma_s - 1) * mix.mach**2)**(mix.gamma_s / (mix.gamma_s - 1))

    # Set flow velocity [m/s]
    mix.u = 0
    mix.uShock = 0
    mix.uNormal = 0

    # Get equilibrium state assuming an isentropic process
    STOP = 1.0
    it = 0
    delta = 1.0
    mix0 = mix

    while STOP > solver.tol0 and it < solver.itMax:
        # Update iteration
        it += 1
        
        # Define state
        mix = copy.copy(mix0)
        mix.p = p
        mix.h = h_target

        # Solve equilibrium state for the current pressure
        solver.solve(mix)
        s, p_curr = _getState(mix)

        # Solve equilibrium state for the perturbed pressure
        p_perturb = mix.p * 1.01
        mix.h = h_target
        mix.p = p_perturb
        solver.solve(mix)
        s_perturb, p_perturb_val = _getState(mix)

        # Compute residual and first derivative
        f0 = s - s_target
        df = (s_perturb - s) / (p_perturb_val - p_curr)
        
        # Update pressure
        p = abs(p_curr - delta * f0 / df)
        
        # Compute stop criteria
        if s_target != 0:
            STOP = max(abs(f0 / s_target), abs(f0))
        else:
            STOP = abs(f0)

    solver.solve(mix)

    # Check if maximum iterations were reached
    if it >= solver.itMax:
        import warnings
        warnings.warn('Maximum iterations reached without full convergence.')

    return mix

def _getState(mix):
    """
    Get state
    """
    entropy = mix.s
    pressure = mix.p
    return entropy, pressure
