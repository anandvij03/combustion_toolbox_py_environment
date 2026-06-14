import sys
import numpy as np

def print_mixtures(mix_main, *args):
    """
    Print properties and composition of the given mixtures in the command window
    
    Args:
        mix_main (Mixture): Mixture object with the properties of the mixture
        *args (Mixture): Additional Mixture objects
        
    Examples:
        print_mixtures(mix1)
        print_mixtures(mix1, mix2, mix3)
    """
    # Group all mixtures into a single list
    mix_list = [mix_main] + list(args)
    num_mixtures = len(mix_list)

    # Definitions
    config = getattr(mix_main, 'config', None)
    if config:
        mintolDisplay = getattr(config, 'mintolDisplay', 1e-6)
        compositionUnits = getattr(config, 'compositionUnits', 'molar fraction')
        FLAG_COMPACT = getattr(config, 'FLAG_COMPACT', True)
    else:
        # Fallbacks if config object doesn't exist directly
        mintolDisplay = 1e-6
        compositionUnits = 'molar fraction'
        FLAG_COMPACT = True

    listSpecies = mix_main.chemicalSystem.listSpecies
    problemType = getattr(mix_list[-1], 'problemType', '')
    if not problemType:
        problemType = ''

    # Start Printing
    print('*' * 108)

    # Print header
    header_composition = _print_header(problemType, num_mixtures, mix_list)

    # Print properties
    _print_properties(problemType, num_mixtures, mix_list)
    
    # Print composition
    _print_composition(mix_list, listSpecies, compositionUnits, header_composition, mintolDisplay, FLAG_COMPACT)

    # End Printing
    print('*' * 108 + '\n\n')

# NESTED/HELPER FUNCTIONS

def _print_composition(mix_list, listSpecies, units, header, mintolDisplay, flag_compact):
    """Router for composition printing based on compactness flag."""
    if flag_compact:
        _print_compact_composition(mix_list, listSpecies, units, mintolDisplay)
        return

    for i, mix in enumerate(mix_list):
        _print_composition_sequential(mix, listSpecies, units, header[i], mintolDisplay)

def _get_properties(prop_name, numberMixtures, mix_list):
    """Fetch property values from a list of mixture objects."""
    vals = []
    for i in range(numberMixtures):
        # Fetch the attribute
        val = getattr(mix_list[i], prop_name, np.nan)
        
        if callable(val):
            val = val()
        vals.append(val)
    return vals

def _get_propertiesLIA(prop_name, numberMixtures, mix_list):
    """Fetch LIA properties from a list of mixture objects."""
    vals = []
    for i in range(numberMixtures):
        lia_obj = getattr(mix_list[i], 'lia', None)
        if lia_obj:
            val = getattr(lia_obj, prop_name, np.nan)
            if callable(val):
                val = val()
            vals.append(val)
        else:
            vals.append(np.nan)
    return vals

def _set_string_value(n_mixtures, fmt='%12.4f', limiter='|'):
    """Generate repeating string formatting templates."""
    if n_mixtures <= 0: return ""
    line_body = f"   {fmt}  {limiter}"
    line_end = f"   {fmt}\n"
    return (line_body * (n_mixtures - 1)) + line_end

def _print_properties(problemType, numberMixtures, mix_list):
    """Massive formatting block for all thermodynamic properties."""
    string_value = _set_string_value(numberMixtures)
    string_value_2 = _set_string_value(numberMixtures - 1)
    
    # Property Map: (Label, attribute_name)
    base_props = [
        ('T [K]', 'T'), ('p [bar]', 'p'), ('r [kg/m3]', 'rho'),
        ('h [kJ/kg]', 'hSpecific'), ('e [kJ/kg]', 'eSpecific'),
        ('g [kJ/kg]', 'gSpecific'), ('s [kJ/(kg-K)]', 'sSpecific'),
        ('W [g/mol]', 'MW'), ('(dlV/dlp)T [-]', 'dVdp_T'),
        ('(dlV/dlT)p [-]', 'dVdT_p'), ('cp [kJ/(kg-K)]', 'cpSpecific'),
        ('gamma [-]', 'gamma'), ('gamma_s [-]', 'gamma_s'),
        ('sound vel [m/s]', 'sound')
    ]
    
    for label, prop in base_props:
        vals = tuple(_get_properties(prop, numberMixtures, mix_list))
        sys.stdout.write((f"{label:<15}|" + string_value) % vals)

    # Problem-Specific Blocks
    if 'SHOCK' in problemType or 'DET' in problemType:
        u1 = _get_properties('velocity_relative', 1, [mix_list[0]])
        u_rest = _get_properties('uShock', numberMixtures - 1, mix_list[1:])
        u_vals = u1 + u_rest
        a_vals = _get_properties('soundspeed', numberMixtures, mix_list)
        mach_vals = [u / a if a else np.nan for u, a in zip(u_vals, a_vals)]
        
        sys.stdout.write((f"{'u [m/s]':<15}|" + string_value) % tuple(u_vals))
        sys.stdout.write((f"{'Mach number [-]':<15}|" + string_value) % tuple(mach_vals))

    if '_OBLIQUE' in problemType or '_POLAR' in problemType:
        print('-' * 108)
        print('PARAMETERS')
        vals_bmin = tuple(_get_properties('betaMin', numberMixtures - 1, mix_list[1:]))
        vals_beta = tuple(_get_properties('beta', numberMixtures - 1, mix_list[1:]))
        vals_theta = tuple(_get_properties('theta', numberMixtures - 1, mix_list[1:]))
        
        sys.stdout.write((f"{'min wave  [deg]':<15}|                 |" + string_value_2) % vals_bmin)
        sys.stdout.write((f"{'wave angle[deg]':<15}|                 |" + string_value_2) % vals_beta)
        sys.stdout.write((f"{'deflection[deg]':<15}|                 |" + string_value_2) % vals_theta)

        if '_POLAR' in problemType:
            vals_tmax = tuple(_get_properties('thetaMax', numberMixtures - 1, mix_list[1:]))
            vals_tsonic = tuple(_get_properties('thetaSonic', numberMixtures - 1, mix_list[1:]))
            sys.stdout.write((f"{'max def.  [deg]':<15}|                 |" + string_value_2) % vals_tmax)
            sys.stdout.write((f"{'sonic def.[deg]':<15}|                 |" + string_value_2) % vals_tsonic)

    elif 'PRANDTL_MEYER' in problemType:
        print('-' * 108)
        print('PARAMETERS')
        vals_theta = tuple(_get_properties('theta', numberMixtures - 1, mix_list[1:]))
        sys.stdout.write((f"{'deflection[deg]':<15}|                 |" + string_value_2) % vals_theta)
        
    elif 'ROCKET' in problemType:
        string_value_3 = _set_string_value(numberMixtures - 2)
        print('-' * 108)
        print('PERFORMANCE PARAMETERS')
        for label, prop in [('A/At [-]', 'areaRatio'), ('CSTAR [m/s]', 'cstar'), 
                            ('CF [-]', 'cf'), ('Ivac [s]', 'I_vac'), ('Isp  [s]', 'I_sp')]:
            vals = tuple(_get_properties(prop, numberMixtures - 2, mix_list[2:]))
            sys.stdout.write((f"{label:<15}|                 |                 |" + string_value_3) % vals)
            
    elif 'SHOCKTURBULENCE' in problemType:
        string_value_1 = _set_string_value(2).replace('|   %12.4f', '|')
        print('-' * 108)
        print('TURBULENCE STATISTICS')
        if 'COMPRESSIBLE' in problemType:
            sys.stdout.write((f"{'eta [-]':<15}|" + string_value_1) % tuple(_get_properties('eta', 1, mix_list)))
            sys.stdout.write((f"{'etaVorticity[-]':<15}|" + string_value_1) % tuple(_get_properties('etaVorticity', 1, mix_list)))
            sys.stdout.write((f"{'chi [-]':<15}|" + string_value_1) % tuple(_get_properties('chi', 1, mix_list)))
        elif 'VORTICAL_ENTROPIC' in problemType:
            sys.stdout.write((f"{'chi [-]':<15}|" + string_value_1) % tuple(_get_properties('chi', 1, mix_list)))

        lia_props = [
            ('TKE [-]', 'K'), ('R11 [-]', 'R11'), ('RTT [-]', 'RTT'),
            ('Enstrophy [-]', 'enstrophy'), ('EnstrophyTT [-]', 'enstrophyTT'),
            ('TKEa [-]', 'Ka'), ('R11a [-]', 'R11a'), ('RTTa [-]', 'RTTa'),
            ('TKEr [-]', 'Kr'), ('R11r [-]', 'R11r'), ('RTTr [-]', 'RTTr'),
            ('Kolmogorov l.r.', 'kolmogorovLengthRatio')
        ]
        for label, prop in lia_props:
            vals = tuple(_get_propertiesLIA(prop, numberMixtures - 1, mix_list[1:]))
            sys.stdout.write((f"{label:<15}|                 |" + string_value_2) % vals)

    print('-' * 108)

def _print_compact_composition(mixCell, listSpecies, units, mintolDisplay):
    numMixtures = len(mixCell)
    nSpecies = len(listSpecies)

    # Build the composition matrix
    comp_matrix = np.zeros((nSpecies, numMixtures))
    for m in range(numMixtures):
        mix_obj = mixCell[m]
        unit_lower = units.lower()
        if unit_lower == 'mol':
            comp_matrix[:, m] = getattr(mix_obj, 'N', np.zeros(nSpecies))
            short_label = 'Ni [mol]'
        elif unit_lower == 'molar fraction':
            comp_matrix[:, m] = getattr(mix_obj, 'Xi', np.zeros(nSpecies))
            short_label = 'Xi [-]'
        elif unit_lower == 'mass fraction':
            comp_matrix[:, m] = getattr(mix_obj, 'Yi', np.zeros(nSpecies))
            short_label = 'Yi [-]'
        else:
            raise ValueError(f"Unsupported composition unit: {units}")

    # Determine major species
    major_mask = np.any(comp_matrix > mintolDisplay, axis=1)
    major_vals = comp_matrix[major_mask, :]
    major_names = np.array(listSpecies)[major_mask]

    # Sort descending based on first mixture
    if major_vals.size > 0:
        idxSort = np.argsort(major_vals[:, 0])[::-1]
        major_vals = major_vals[idxSort, :]
        major_names = major_names[idxSort]

    # Print composition header
    sys.stdout.write(f"{'COMPOSITION':<15}{short_label}")
    for _ in range(numMixtures):
        sys.stdout.write(f"   {short_label:>12}")
    sys.stdout.write('\n')

    # Print major species
    line_fmt = _set_string_value(numMixtures, fmt='%12.4e', limiter=' ')
    for i in range(major_vals.shape[0]):
        sys.stdout.write(f"{major_names[i]:<16}")
        sys.stdout.write(line_fmt % tuple(major_vals[i, :]))

    # Print MINORS
    minor_mask = ~major_mask
    Nminor = np.sum(minor_mask)
    minor_values = np.sum(comp_matrix[minor_mask, :], axis=0)
    sys.stdout.write(f"{f'MINORS[+{Nminor}]':<16}")
    sys.stdout.write(line_fmt % tuple(minor_values))

    # Print TOTAL
    totals = np.sum(comp_matrix, axis=0)
    sys.stdout.write(f"{'TOTAL':<16}")
    sys.stdout.write(line_fmt % tuple(totals))

def _print_composition_sequential(mix_obj, listSpecies, units, header, mintolDisplay):
    unit_lower = units.lower()
    if unit_lower == 'mol':
        variable = np.array(getattr(mix_obj, 'N', []))
        short_label = 'Ni [mol]\n'
    elif unit_lower == 'molar fraction':
        variable = np.array(getattr(mix_obj, 'Xi', []))
        short_label = '  Xi [-]\n'
    elif unit_lower == 'mass fraction':
        variable = np.array(getattr(mix_obj, 'Yi', []))
        short_label = '  Yi [-]\n'

    sys.stdout.write(f"{header}{short_label}")
    
    # Sort descending
    ind_sort = np.argsort(variable)[::-1]
    variable_sorted = variable[ind_sort]
    
    # Mask for displaying variables
    display_mask = variable_sorted > mintolDisplay
    minor_sum = np.sum(variable_sorted[~display_mask])
    
    for i in range(len(variable_sorted)):
        if display_mask[i]:
            sys.stdout.write(f"{listSpecies[ind_sort[i]]:<20} {variable_sorted[i]:1.4e}\n")

    Nminor = len(variable) - np.sum(display_mask)
    spaces = " " * max(0, 4 - len(str(Nminor)))
    
    sys.stdout.write(f"MINORS[+{Nminor}] {spaces}     {minor_sum:12.4e}\n\n")
    sys.stdout.write(f"TOTAL            {np.sum(variable):14.4e}\n")
    print('-' * 108)

def _print_header(problemType, numberMixtures, mix_list):
    """Determine the top header formatting based on Problem Type."""
    FLAG_PHI = getattr(mix_list[0], 'equivalenceRatio', None) is not None
    
    if problemType and FLAG_PHI:
        print('-' * 108)
        print(f"Problem type: {problemType}  | Equivalence ratio = {mix_list[0].equivalenceRatio:4.4f}")
    elif problemType and not FLAG_PHI:
        print('-' * 108)
        print(f"Problem type: {problemType}")
    elif FLAG_PHI:
        print('-' * 108)
        print(f"Equivalence ratio = {mix_list[0].equivalenceRatio:4.4f}")
        
    print('-' * 108)

    header_composition = []

    if '_OBLIQUE' in problemType or '_POLAR' in problemType:
        if numberMixtures == 2:
            header_composition = ['STATE 1               ', 'STATE 2               ']
            print('               |     STATE 1     |     STATE 2')
        elif numberMixtures == 3:
            header_composition = ['STATE 1               ', 'STATE 2-W             ', 'STATE 2-S             ']
            print('               |     STATE 1     |     STATE 2-W   |     STATE 2-S')
        elif numberMixtures == 4:
            header_composition = ['STATE 1               ', 'STATE 2               ', 'STATE 3-W             ', 'STATE 3-S             ']
            print('               |     STATE 1     |     STATE 2     |     STATE 3-W   |     STATE 3-S')
            
    elif '_R' in problemType:
        fmt_str = "               |" + ("    STATE {}    |" * (numberMixtures - 1)) + "    STATE {}\n"
        sys.stdout.write(fmt_str.format(*range(1, numberMixtures + 1)))
        header_composition = [f'STATE {i:<16}' for i in range(1, 9)]

    elif 'ROCKET' in problemType:
        if numberMixtures == 3:
            header_composition = ['INLET CHAMBER         ', 'OUTLET CHAMBER        ', 'THROAT                ']
            print('               |  INLET CHAMBER  | OUTLET CHAMBER  |      THROAT ')
        elif numberMixtures > 3:
            header_exit_prop_last = '|      EXIT\n' if numberMixtures > 4 else ''
            
            if getattr(mix_list[2], 'areaRatio', None) == 1:
                header_exit = ['EXIT                  '] * (numberMixtures - 3)
                header_composition = ['INLET CHAMBER         ', 'OUTLET CHAMBER        ', 'THROAT                '] + header_exit
                if numberMixtures > 4:
                    header_exit_prop = ['|      EXIT       '] * (numberMixtures - 5) + [header_exit_prop_last]
                    sys.stdout.write('               |  INLET CHAMBER  | OUTLET CHAMBER  |     THROAT      ' + "".join(header_exit_prop))
                else:
                    print('               |  INLET CHAMBER  | OUTLET CHAMBER  |     THROAT      |      EXIT')
            else:
                header_exit = ['EXIT                  '] * (numberMixtures - 4)
                header_composition = ['INLET CHAMBER         ', 'INJECTOR              ', 'OUTLET CHAMBER        ', 'THROAT                '] + header_exit
                if numberMixtures > 4:
                    header_exit_prop = ['|      EXIT       '] * (numberMixtures - 5) + [header_exit_prop_last]
                    sys.stdout.write('               |  INLET CHAMBER  |     INJECTOR    | OUTLET CHAMBER  |     THROAT      ' + "".join(header_exit_prop))
                else:
                    print('               |  INLET CHAMBER  |     INJECTOR    | OUTLET CHAMBER  |     THROAT ')

    elif problemType and numberMixtures == 2:
        header_composition = ['REACTANTS             ', 'PRODUCTS              ']
        print('               |    REACTANTS    |      PRODUCTS')
    else:
        fmt_str = "               |" + ("   MIXTURE {}    |" * (numberMixtures - 1)) + "   MIXTURE {}\n"
        sys.stdout.write(fmt_str.format(*range(1, numberMixtures + 1)))
        header_composition = [f'MIXTURE {i:<16}' for i in range(1, 9)]

    return header_composition