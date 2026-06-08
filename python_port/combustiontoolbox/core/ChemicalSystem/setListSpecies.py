def setListSpecies(self, list_species_input=None, phi=None, phi_c=None):
        """
        Set list of species in the mixture (products).
        
        Args:
            list_species_input (list or str): Name of list, or actual list of species
            phi (float, optional): Equivalence ratio
            phi_c (float, optional): Equivalence ratio in which theoretically appears soot
            
        Returns:
            tuple: (self, listSpecies, listSpeciesFormula)
        """
        if not list_species_input:
            self.listSpecies = []
            return self, [], []
            
        # Get list of species and update COMPLETE flag
        raw_list, self.FLAG_COMPLETE = self._get_list_species(self.database, list_species_input, phi, phi_c)
        
        # Remove repeated species while preserving order
        self.listSpecies = list(dict.fromkeys(raw_list))
        
        # Assign formulas
        self._listSpeciesFormula = self._get_formula(self.listSpecies, self.database)
        
        # Check if the list of species contains ions
        if any(self.isIonized(self.listSpecies)):
            self.FLAG_ION = True
            
        return self, self.listSpecies, self._listSpeciesFormula

    # SUB-PASS HELPER FUNCTIONS

def _get_list_species(self, database, list_species_input, phi=None, phi_c=None):
    """Get list of species based on user input or predefined strings."""
    flag_complete = False
        
    # Check if input is already a list or tuple
    if isinstance(list_species_input, (list, tuple)):
        return list(list_species_input), flag_complete
            
    # Check if input is a single string that exists directly in the database
    if isinstance(list_species_input, str) and list_species_input in getattr(database, 'listSpecies', []):
        return [list_species_input], flag_complete
            
    # Check predefined lists
    if isinstance(list_species_input, str):
        keyword = list_species_input.upper()
            
        if keyword in ['COMPLETE', 'COMPLETE REACTION']:
            flag_complete = True
            if phi is not None and phi_c is not None:
                if phi < 1:
                    return self.listSpeciesLean, flag_complete
                elif 1 <= phi < phi_c:
                    return self.listSpeciesRich, flag_complete
                else:
                    return self.listSpeciesSoot, flag_complete
            else:
                return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar', 'Cbgrb'], flag_complete

        elif keyword == 'HC/O2/N2 EXTENDED':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar', 'C2',
                    'CH', 'CH3', 'CH4', 'CN', 'H', 'HCN', 'HCO', 'HO2', 'N', 'N2O',
                    'NH2', 'NH3', 'NO', 'NO2', 'O', 'OH'], flag_complete

        elif keyword == 'HC/O2/N2':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar'], flag_complete

        elif keyword == 'HC/O2/N2 RICH':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar',
                    'C2H4', 'CH', 'CH3', 'CH4', 'CN', 'H', 'HCN', 'HCO',
                    'N', 'NH', 'NH2', 'NH3', 'NO', 'O', 'OH'], flag_complete

        elif keyword == 'SOOT FORMATION':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar', 'Cbgrb',
                    'C2', 'C2H4', 'CH', 'CH3', 'CH4', 'CN', 'H',
                    'HCN', 'HCO', 'N', 'NH', 'NH2', 'NH3', 'NO', 'O', 'OH',
                    'H2ObLb'], flag_complete

        elif keyword == 'SOOT FORMATION EXTENDED':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar', 'Cbgrb',
                    'C2', 'C2H', 'C2H2_acetylene', 'C2H2_vinylidene',
                    'C2H3_vinyl', 'C2H4', 'C2H5', 'C2H5OH', 'C2H6',
                    'C2N2', 'C2O', 'C3', 'C3H3_1_propynl',
                    'C3H3_2_propynl', 'C3H4_allene', 'C3H4_propyne',
                    'C3H5_allyl', 'C3H6O_acetone', 'C3H6_propylene',
                    'C3H8', 'C4', 'C4H2_butadiyne', 'C5', 'C6H2', 'C6H6',
                    'C8H18_isooctane', 'CH', 'CH2', 'CH2CO_ketene',
                    'CH2OH', 'CH3', 'CH3CHO_ethanal', 'CH3CN',
                    'CH3COOH', 'CH3O', 'CH3OH', 'CH4', 'CN', 'COOH', 'H',
                    'H2O2', 'HCCO', 'HCHO_formaldehy', 'HCN', 'HCO', 'HNO2',
                    'HCOOH', 'HNC', 'HNCO', 'HNO', 'HO2', 'N', 'N2O',
                    'NCO', 'NH', 'NH2', 'NH2OH', 'NH3', 'NO', 'NO2',
                    'O', 'OCCN', 'OH', 'C3O2', 'C4N2', 'CH3CO_acetyl',
                    'C4H6_butadiene', 'C4H6_1butyne', 'C4H6_2butyne',
                    'C2H4O_ethylen_o', 'CH3OCH3', 'C4H8_1_butene',
                    'C4H8_cis2_buten', 'C4H8_isobutene',
                    'C4H8_tr2_butene', 'C4H9_i_butyl', 'C4H9_n_butyl',
                    'C4H9_s_butyl', 'C4H9_t_butyl', 'C6H5OH_phenol',
                    'C6H5O_phenoxy', 'C6H5_phenyl', 'C7H7_benzyl',
                    'C7H8', 'C8H8_styrene', 'C10H8_naphthale', 'H2ObLb'], flag_complete

        elif keyword in ['AIR', 'DISSOCIATED AIR']:
            return ['CO2', 'CO', 'O2', 'N2', 'Ar', 'O', 'O3',
                    'N', 'NO', 'NO2', 'NO3', 'N2O', 'N2O3',
                    'N2O4', 'N3', 'C'], flag_complete

        elif keyword in ['AIR_IONS', 'AIR IONS']:
            return ['eminus', 'Ar', 'Arplus', 'C', 'Cplus', 'Cminus',
                    'CN', 'CNplus', 'CNminus', 'CNN', 'CO', 'COplus',
                    'CO2', 'CO2plus', 'C2', 'C2plus', 'C2minus', 'CCN',
                    'CNC', 'OCCN', 'C2N2', 'C2O', 'C3', 'C3O2', 'N',
                    'Nplus', 'Nminus', 'NCO', 'NO', 'NOplus', 'NO2',
                    'NO2minus', 'NO3', 'NO3minus', 'N2', 'N2plus',
                    'N2minus', 'NCN', 'N2O', 'N2Oplus', 'N2O3', 'N2O4',
                    'N2O5', 'N3', 'O', 'Oplus', 'Ominus', 'O2', 'O2plus',
                    'O2minus', 'O3'], flag_complete

        elif keyword in ['IDEAL_AIR', 'AIR_IDEAL']:
            return ['O2', 'N2', 'O', 'O3', 'N', 'NO', 'NO2', 'NO3', 'N2O',
                    'N2O3', 'N2O4', 'N3'], flag_complete

        elif keyword == 'HYDROGEN':
            return ['H2O', 'H2', 'O2', 'N2', 'Ar', 'H', 'HNO',
                    'HNO3', 'NH', 'NH2OH', 'NO3', 'N2H2', 'N2O3', 'N3', 'OH',
                    'HNO2', 'N', 'NH3', 'NO2', 'N2O', 'N2H4', 'N2O5', 'O', 'O3',
                    'HO2', 'NH2', 'H2O2', 'N3H', 'NH2NO2', 'H2ObLb'], flag_complete

        elif keyword in ['HYDROGEN_IONS', 'HYDROGEN IONS']:
            return ['H2O', 'H2', 'O2', 'N2', 'H', 'OH', 'H2O2', 'H2Oplus',
                    'H2minus', 'H2plus', 'H3Oplus', 'HNO', 'HNO2', 'HNO3', 'HO2',
                    'HO2minus', 'Hminus', 'Hplus', 'N', 'N2H2', 'N2H4', 'N2O', 'N2O3',
                    'N2O5', 'N2Oplus', 'N2minus', 'N2plus', 'N3', 'N3H', 'NH', 'NH2',
                    'NH2NO2', 'NH2OH', 'NH3', 'NO2', 'NO2minus', 'NO3', 'NO3minus',
                    'NOplus', 'Nminus', 'Nplus', 'O', 'O2minus', 'O2plus', 'O3',
                    'Ominus', 'Oplus', 'eminus'], flag_complete

        elif keyword in ['HYDROGEN_L', 'HYDROGEN (L)']:
            return ['H2O', 'H2', 'O2', 'H', 'OH', 'O', 'O3', 'HO2',
                        'H2O2', 'H2bLb', 'O2bLb', 'H2ObLb'], flag_complete

        elif keyword == 'HC/O2/N2 PROPELLANTS':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar', 'Cbgrb',
                    'C2', 'C2H', 'C2H2_acetylene', 'C2H2_vinylidene',
                    'C2H3_vinyl', 'C2H4', 'C2H5', 'C2H5OH', 'C2H6',
                    'C2N2', 'C2O', 'C3', 'C3H3_1_propynl',
                    'C3H3_2_propynl', 'C3H4_allene', 'C3H4_propyne',
                    'C3H5_allyl', 'C3H6O_acetone', 'C3H6_propylene',
                    'C3H8', 'C4', 'C4H2_butadiyne', 'C5', 'C6H2', 'C6H6',
                    'C8H18_isooctane', 'CH', 'CH2', 'CH2CO_ketene',
                    'CH2OH', 'CH3', 'CH3CHO_ethanal', 'CH3CN',
                    'CH3COOH', 'CH3O', 'CH3OH', 'CH4', 'CN', 'COOH', 'H',
                    'H2O2', 'HCCO', 'HCHO_formaldehy', 'HCN', 'HCO',
                    'HCOOH', 'HNC', 'HNCO', 'HNO', 'HO2', 'N', 'N2O',
                    'NCO', 'NH', 'NH2', 'NH2OH', 'NH3', 'NO', 'NO2',
                    'O', 'OCCN', 'OH', 'C3O2', 'C4N2', 'RP_1', 'H2bLb',
                    'O2bLb', 'H2ObLb'], flag_complete

        elif keyword == 'SI/HC/O2/N2 PROPELLANTS':
            return ['CO2', 'CO', 'H2O', 'H2', 'O2', 'N2', 'Ar', 'Cbgrb',
                    'C2', 'C2H4', 'CH', 'CH3', 'CH4', 'CN', 'H',
                    'H2O2', 'HCN', 'HCO', 'N', 'NH', 'NH2', 'NH3', 'NO', 'O', 'OH',
                    'O2bLb', 'Si', 'SiH', 'SiH2', 'SiH3', 'SiH4', 'SiO2', 'SiO',
                    'SibLb', 'SiO2bLb', 'Si2', 'H2ObLb'], flag_complete

    # Default fallback
    return [], flag_complete

def _get_formula(self, list_species, database):
    """Get chemical formula from the database (DB)."""
    # Retrieves the formula attribute safely. If it doesn't exist, returns an empty string.
    return [getattr(database.species[sp], 'formula', '') for sp in list_species]