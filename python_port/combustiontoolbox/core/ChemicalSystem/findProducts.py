import numpy as np

def getCrossTermms(index_elements, max_elements=5):
    """
    Helper function to generate combinations of element indices.
    """
    temp = []
    
    # Extract unique element IDs and sort them descending
    unique_vals = np.unique(index_elements)
    ind_perm = sorted(unique_vals, reverse=True)
    
    # Pad with zeros to handle elements up to MAX_ELEMENTS - 2
    padding_len = max_elements - 2
    ind_perm.extend([0] * padding_len)
    
    n_elements = len(ind_perm)
    
    for i in range(n_elements):
        if ind_perm[i] == 0:
            continue
        for j in range(i + 1, n_elements):
            for k in range(j + 1, n_elements):
                for l in range(k + 1, n_elements):
                    temp_add = [ind_perm[i], ind_perm[j], ind_perm[k], ind_perm[l], 0]
                    temp.append(temp_add)
                    
    return np.array(temp) if temp else np.empty((0, max_elements))


def findProducts(self, listReactants, **kwargs):
    """
    Find all combinations of species from the database that can appear as
    products for a given list of reactants.
    """
    # Definitions
    MAX_ELEMENTS = 5
    FLAG_BURCAT = kwargs.get('flag_burcat', self.FLAG_BURCAT)
    FLAG_ION = kwargs.get('flag_ion', self.FLAG_ION)
    FLAG_CONDENSED = kwargs.get('flag_condensed', self.FLAG_CONDENSED)
    
    indexElements_DB = None
    FLAG_IND = False
    for key in ['ind', 'ind_elements', 'ind_elements_db', 'ind_db', 'indexelements_db']:
        if key in kwargs:
            indexElements_DB = kwargs[key]
            FLAG_IND = True
            break
            
    listSpecies = []
    listSpecies_DB = list(self.database.listSpecies)
    
    # If FLAG_ION is true, look for ionized species (add free electrons)
    if FLAG_ION:
        if 'eminus' not in listReactants:
            listReactants = list(listReactants) + ['eminus']
            
    # Filter out Burcat database species if flag is False
    if not FLAG_BURCAT:
        listSpecies_DB = [sp for sp in listSpecies_DB if '_M' not in sp]
        if FLAG_IND:
            FLAG_IND = False
            
    # Filter out condensed phase species if flag is False
    if not FLAG_CONDENSED:
        listSpecies_DB = [
            sp for sp in listSpecies_DB 
            if getattr(self.database.species[sp], 'phase', 0) == 0
        ]
        if FLAG_IND:
            FLAG_IND = False
            
    # Get element signatures of the reactants 
    # Implement getIndexElements method on database
    raw_index_elements = self.getIndexElements(listReactants, MAX_ELEMENTS)
    indexElements = np.sort(raw_index_elements, axis=0)
    
    # Get element signatures for the database species if not provided as an argument
    if not FLAG_IND:
        if 'Air' in listSpecies_DB:
            listSpecies_DB.remove('Air')
        indexElements_DB = self.getIndexElements(listSpecies_DB, MAX_ELEMENTS)
        
    # Isolate unique signature rows
    indexElements = np.unique(indexElements, axis=0)
    
    # Generate crossover atomic combinations 
    temp = getCrossTermms(indexElements, MAX_ELEMENTS)
    
    # Join the reactant signatures with cross-term signatures
    if temp.size > 0:
        indexElements = np.vstack([indexElements, temp])
    indexElements = np.unique(indexElements, axis=0)
    
    # Step backward through combinations to populate product lists
    for i in range(indexElements.shape[0] - 1, -1, -1):
        # Find exactly matching row vectors where (DB_row - target_row) == 0
        matches = np.all(indexElements_DB - indexElements[i, :] == 0, axis=1)
        matched_indices = np.where(matches)[0]
        
        for idx in matched_indices:
            listSpecies.append(listSpecies_DB[idx])
            
    return listSpecies, indexElements_DB