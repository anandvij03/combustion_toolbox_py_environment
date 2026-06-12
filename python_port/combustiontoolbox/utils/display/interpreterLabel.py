def check_basis(basis):
    if basis == 'mi':
        return 'kg'
    elif basis == 'mw':
        return 'mol'
    else:
        raise ValueError("Not known basis.")

def property_names(property_name, label_type='medium', flag_basis=True, basis='kg'):
    prop = property_name.lower()
    name, latex, unit = prop, prop, ""

    if prop in ['phi', 'equivalenceratio']:
        name = 'Equivalence ratio'
        latex = '\\phi'
        unit = ''
    elif prop == 'rho':
        name = 'Density'
        latex = '\\rho'
        unit = '[kg/m$^3$]'
    elif prop == 't':
        name = 'Temperature'
        latex = 'T'
        unit = '[K]'
    elif prop == 'p':
        name = 'Pressure'
        latex = 'p'
        unit = '[bar]'
    elif prop == 'h':
        name = 'Enthalpy'
        latex = 'h'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 'e':
        name = 'Internal energy'
        latex = 'e'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 'g':
        name = 'Gibbs energy'
        latex = 'g'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 's':
        name = 'Entropy'
        latex = 's'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 's0':
        name = 'Entropy frozen'
        latex = 's_0'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'ds':
        name = 'Entropy of mixing'
        latex = '\\Delta s'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'w':
        name = 'Molecular weight'
        latex = 'W'
        unit = '[g/mol]'
    elif prop == 'cp':
        name = 'Specific heat pressure'
        latex = 'c_p'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'cp_f':
        name = 'Specific heat pressure frozen'
        latex = 'c_{p,f}'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'cp_r':
        name = 'Specific heat pressure reaction'
        latex = 'c_{p,r}'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'cv':
        name = 'Specific heat volume'
        latex = 'c_v'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'cv_f':
        name = 'Specific heat volume frozen'
        latex = 'c_{v,f}'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'cv_r':
        name = 'Specific heat volume reaction'
        latex = 'c_{v,r}'
        unit = f'[kJ/{basis}-K]' if flag_basis else '[kJ/K]'
    elif prop == 'gamma':
        name = 'Specific heat ratio'
        latex = '\\gamma'
        unit = ''
    elif prop == 'gamma_f':
        name = 'Frozen specific heat ratio'
        latex = '\\gamma_f'
        unit = ''
    elif prop == 'gamma_s':
        name = 'Adiabatic index'
        latex = '\\gamma_s'
        unit = ''
    elif prop in ['sound', 'a']:
        name = 'Sound velocity'
        latex = 'a'
        unit = '[m/s]'
    elif prop == 'u':
        name = 'Velocity'
        latex = 'u'
        unit = '[m/s]'
    elif prop in ['v_shock', 'ushock']:
        name = 'Shock velocity'
        latex = 'v_{\\rm shock}'
        unit = '[m/s]'
    elif prop == 'u_preshock':
        name = 'Pre-shock velocity'
        latex = 'u_{\\rm preshock}'
        unit = '[m/s]'
    elif prop == 'u_postshock':
        name = 'Post-shock velocity'
        latex = 'u_{\\rm postshock}'
        unit = '[m/s]'
    elif prop in ['m', 'mach']:
        name = 'Mach number'
        latex = '\\mathcal{M}'
        unit = ''
    elif prop == 'm1':
        name = 'Pre-shock Mach number'
        latex = '\\mathcal{M}_1'
        unit = ''
    elif prop == 'm2':
        name = 'Post-shock Mach number'
        latex = '\\mathcal{M}_2'
        unit = ''
    elif prop == 'cstar':
        name = 'Characteristic velocity'
        latex = 'C^*'
        unit = '[m/s]'
    elif prop == 'cf':
        name = 'Coefficient of thrust'
        latex = 'C_f'
        unit = ''
    elif prop == 'i_sp':
        name = 'Specific impulse ambient'
        latex = 'I_{sp}'
        unit = '[s]'
    elif prop == 'i_vac':
        name = 'Specific impulse vaccum'
        latex = 'I_{vac}'
        unit = '[s]'
    elif prop == 'n':
        name = 'Total moles'
        latex = 'n'
        unit = '[mol]'
    elif prop == 'v':
        name = 'Volume'
        latex = 'v'
        unit = '[m$^3$]'
    elif prop in ['v_sp', 'vspecific']:
        name = 'Specific volume'
        latex = 'v_{sp}'
        unit = '[m$^3$/kg]'
    elif prop == 'hf':
        name = 'Enthalpy of formation'
        latex = 'h_f'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 'dht':
        name = 'Enthalpy thermal'
        latex = '\\Delta h_T'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 'ef':
        name = 'Internal energy of formation'
        latex = 'e_f'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 'det':
        name = 'Internal energy thermal'
        latex = '\\Delta e_T'
        unit = f'[kJ/{basis}]' if flag_basis else '[kJ]'
    elif prop == 'xi':
        name = 'Molar fractions'
        latex = 'X_j'
        unit = ''
    elif prop == 'yi':
        name = 'Mass fractions'
        latex = 'Y_j'
        unit = ''
    elif prop in ['v_p/v_r', 'v_v', 'vp_vr']:
        name = 'Volume ratio'
        latex = 'v_2/v_1'
        unit = ''
    elif prop == 'error_moles':
        name = 'Relative error moles composition'
        latex = '\\epsilon_{\\rm moles}'
        unit = ''
    elif prop == 'error_moles_ions':
        name = 'Relative error electroneutrality'
        latex = '\\epsilon_{\\rm ions}'
        unit = ''
    elif prop == 'error_problem':
        name = 'Relative error problem'
        latex = '\\epsilon_{\\rm problem}'
        unit = ''
    elif prop in ['dvdtp', 'dvdt_p']:
        name = ''
        latex = '({\\rm d\\,ln}v/{\\rm d\\,ln}T)_p'
        unit = ''
    elif prop in ['dvdpt', 'dvdp_t']:
        name = ''
        latex = '({\\rm d\\,ln}v/{\\rm d\\,ln}p)_T'
        unit = ''
    elif prop in ['dpdvt', 'dpdv_t']:
        name = ''
        latex = '({\\rm d\\,ln}p/{\\rm d\\,ln}v)_T'
        unit = ''
    elif prop in ['dpdtv', 'dpdt_v']:
        name = ''
        latex = '({\\rm d\\,ln}p/{\\rm d\\,ln}T)_V'
        unit = ''
    elif prop == 'theta':
        name = 'Deflection angle'
        latex = '\\theta'
        unit = '[deg]'
    elif prop == 'beta':
        name = 'Wave angle'
        latex = '\\beta'
        unit = '[deg]'
    elif prop in ['drive_factor', 'drivefactor']:
        name = 'Overdriven factor'
        latex = 'u_1/u_{\\rm cj}'
        unit = ''
    elif prop == 'of':
        name = 'Mixture ratio'
        latex = 'O/F'
        unit = ''
    elif prop == 'mi':
        name = 'Mass'
        latex = 'm'
        unit = '[kg]'
    elif prop == 'pv':
        name = 'Pressure $\\times$ Volume'
        latex = 'pv'
        unit = '[bar-m$^3$]'
    elif prop in ['aratio', 'arearatio']:
        name = 'Area exit / throat'
        latex = 'A_{\\rm ratio} = A_e/A_t'
        unit = ''
    elif prop in ['aratio_c', 'arearatiochamber']:
        name = 'Area combustor / throat'
        latex = 'A_{\\rm ratio, c} = A_c/A_t'
        unit = ''
    else:
        unit = ''
        if label_type == 'short':
            name = ''
            latex = property_name
        else:
            name = property_name
            latex = ''

    if latex:
        latex = f"${latex}$"

    if label_type == 'short' and latex:
        name = ''
    elif label_type == 'medium' and name:
        latex = ''
    elif label_type == 'long':
        name = f"{name},"

    return name, latex, unit

def interpreterLabel(property_name, label_type='medium', flag_basis=True, basis='kg'):
    if isinstance(property_name, (list, tuple)):
        if len(property_name) > 1:
            return 'Multiple variables'
        property_name = property_name[0]

    if not property_name:
        return ''

    if basis:
        basis = check_basis(basis)

    name, latex, unit = property_names(property_name, label_type, flag_basis, basis)
    parts = [name, latex, unit]
    return " ".join([p for p in parts if p]).strip()
