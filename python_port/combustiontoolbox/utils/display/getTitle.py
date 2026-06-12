from combustiontoolbox.utils.display.species2latex import species2latex

def cat_mol_species(mol, species):
    if mol == 1:
        value = ""
    else:
        value = f"{mol:.3g}"
    return value + species2latex(species)

def cat_moles_species(moles, species):
    if not species:
        return ""
    cat_text = cat_mol_species(moles[0], species[0])
    for i in range(1, len(species)):
        cat_text += " + " + cat_mol_species(moles[i], species[i])
    return cat_text.replace('_$_{', '$_{')

def getTitle(obj) -> str:
    """
    Get a title based on the problem type and species involved.
    """
    flag_fuel = bool(getattr(obj, 'listSpeciesFuel', []))
    flag_oxidizer = bool(getattr(obj, 'listSpeciesOxidizer', []))
    flag_inert = bool(getattr(obj, 'listSpeciesInert', []))
    flag_ratio_inerts_o2 = bool(getattr(obj, 'ratioOxidizer', []))
    n_oxidizer = len(getattr(obj, 'listSpeciesOxidizer', []))
    
    moles_fuel = getattr(obj, 'molesFuel', [])
    list_species_fuel = getattr(obj, 'listSpeciesFuel', [])
    
    if not flag_fuel and not flag_oxidizer and not flag_inert:
        quantities = getattr(obj, 'quantity', [])
        species_list = getattr(obj, 'listSpecies', [])
        flag_pass = [q > 0 for q in quantities]
        moles_fuel = [m for m, p in zip(quantities, flag_pass) if p]
        list_species_fuel = [s for s, p in zip(species_list, flag_pass) if p]

    label_problemtype = ""
    if getattr(obj, 'problemType', None):
        label_problemtype = obj.problemType.replace('_', ' ') + ": "

    titlename = label_problemtype + cat_moles_species(moles_fuel, list_species_fuel)

    if not flag_oxidizer and not flag_inert:
        return titlename

    if flag_fuel:
        titlename += f" + $\\frac{{{obj.stoichiometricMoles:.3g}}}{{\\phi}}$"

    if flag_oxidizer:
        if n_oxidizer > 1 and flag_fuel:
            titlename += "("

        list_oxidizer = getattr(obj, 'listSpeciesOxidizer', [])
        moles_oxidizer = list(getattr(obj, 'molesOxidizer', []))
        if 'O2' in list_oxidizer:
            ind = list_oxidizer.index('O2')
            o2_moles = moles_oxidizer[ind]
            moles_oxidizer = [m / o2_moles for m in moles_oxidizer]

        titlename += cat_moles_species(getattr(obj, 'ratioOxidizer', []), list_oxidizer)

    if flag_inert and flag_ratio_inerts_o2:
        titlename += " + " + cat_moles_species(getattr(obj, 'molesInert', []), getattr(obj, 'listSpeciesInert', []))

    if flag_oxidizer and n_oxidizer > 1 and flag_fuel:
        titlename += ")"

    if flag_inert and not flag_ratio_inerts_o2:
        titlename += " + " + cat_moles_species(getattr(obj, 'molesInert', []), getattr(obj, 'listSpeciesInert', []))

    return titlename
