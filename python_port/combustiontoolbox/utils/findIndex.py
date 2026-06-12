def findIndex(LS, species):
    """
    Find the index of the species based on the given list (LS).
    
    Args:
        LS (list or str): List of species
        species (list or str): Species to find index values
        
    Returns:
        list: 0-based indices of species in LS
    """
    if LS is None or species is None:
        return []

    # Normalize inputs to lists
    if isinstance(LS, str):
        LS = [LS]
    elif not isinstance(LS, list):
        LS = list(LS)

    if isinstance(species, str):
        species = [species]
    elif not isinstance(species, list):
        species = list(species)

    indices = []
    for sp in species:
        if sp in LS:
            indices.append(LS.index(sp))
            
    return indices
