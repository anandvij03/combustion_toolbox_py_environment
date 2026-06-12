def findIndex(LS, species):
    """
    Find the index of the species based on the given list (LS).
    
    Args:
        LS (list or str): List of species
        species (list or str): Species to find index values
        
    Returns:
        int or list: 0-based index or indices of species in LS
    """
    if LS is None or species is None:
        return None if isinstance(species, str) else []

    # Normalize LS to list
    if isinstance(LS, str):
        LS = [LS]
    elif not isinstance(LS, list):
        LS = list(LS)

    if isinstance(species, str):
        return LS.index(species) if species in LS else None

    # species is a list, set, or other iterable
    indices = []
    for sp in species:
        if sp in LS:
            indices.append(LS.index(sp))
            
    return indices
