import os
import re
from combustiontoolbox.databases.Database import Database

# Assuming Database is imported from your core modules
# from combustiontoolbox.databases import Database

class BurcatDatabase(Database):
    """
    The BurcatDatabase class is used to store thermodynamic data from Burcat's database
    using NASA's 9 coefficient polynomial fits.
    """

    def __init__(self, **kwargs):
        """
        Constructor. Initializes the BurcatDatabase with the chemical species 
        contained in Burcat's database.
        """
        # Call superclass constructor with default name and temperature reference
        super().__init__(name='Burcat', temperatureReference=298.15, **kwargs)

    @staticmethod
    def thermoMillennium2thermoNASA9(
        filename_input: str = r"C:\Users\sache\Downloads\CombustionToolbox-combustion_toolbox-1.2.9.0\databases\thermo_millennium_2_thermoNASA9.inp", 
        filename_output: str = 'thermo_millennium_2_thermoNASA9.inp', 
        out_dir: str = 'databases', 
        suffix: str = '_M'
    ):
        """
        Read Extended Third Millennium Thermodynamic Database of New NASA
        Polynomials with Active Thermochemical Tables update and write a new
        file compatible with thermo NASA 9 format.
        
        Args:
            filename_input (str): Filename of the thermoMillennium data (relative path).
            filename_output (str): Filename of the output data.
            out_dir (str): Output directory (relative path). Default is 'databases'.
            suffix (str): Suffix to add to species names. Default is '_M'.
        """
        # Define output path using relative directories
        out_path = os.path.join(out_dir, filename_output)

        # Ensure output directory exists (relative to the current working directory)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        flag_new_species = True
        species_map = {}

        try:
            with open(filename_input, 'r', encoding='utf-8') as fid_in, \
                 open(out_path, 'w', encoding='utf-8') as fid_out:
                
                for line in fid_in:
                    line_stripped = line.strip()
                    
                    # Check for blank lines, comments, or lines containing 'see'
                    if not line_stripped:
                        flag_new_species = True
                        continue
                    if line.startswith('!'):
                        continue
                    if 'see' in line:
                        continue
                    
                    # Detect start of a new species block (starts with letter, followed by space)
                    if re.match(r'^[A-Za-z][^\s]*\s+', line):
                        flag_new_species = True

                    if flag_new_species:
                        # Find the first space to isolate the species name
                        match_space = re.search(r'\s', line)
                        if not match_space:
                            fid_out.write(line)
                            continue
                            
                        ind_space = match_space.start()
                        species = line[:ind_space]

                        # Find where the actual thermodynamic data starts after the space
                        rest_of_line = line[ind_space:]
                        match_next_text = re.search(r'\S', rest_of_line)
                        next_text_start = ind_space + match_next_text.start() if match_next_text else len(line)

                        # Calculate spacing to maintain column alignment
                        n_spaces = 18 - ind_space - len(suffix)
                        white_spaces = ' ' * max(0, n_spaces)

                        # Check for phase and electron state keywords
                        line_lower = line.lower()
                        if ' cr ' in line_lower:
                            species += '(cr)'
                        elif ' liq ' in line_lower and '(l)' not in species.lower():
                            species += '(L)'

                        if 'excited' in line_lower:
                            species += '(exc)'
                        elif 'singlet' in line_lower:
                            species += '(slet)'
                        elif 'doublet' in line_lower:
                            species += '(dlet)'
                        elif 'triplet' in line_lower:
                            species += '(tlet)'
                        elif 'quartet' in line_lower:
                            species += '(qtet)'

                        # String formatting replacements
                        species = species.replace('*', ' ')
                        species = species.replace('=', '_')
                        species = species.replace('Al', 'AL')
                        species = species.replace('Cl', 'CL')
                        species = species.replace('Tl', 'TL')
                        species = species.replace('Fl', 'FL')
                        
                        # Fix phase notation consistency
                        species = species.replace('(liq)', 'liq')
                        species = species.replace('liq', '(liq)')
                        species = species.replace('(l)', '(L)')

                        # Handle duplicate species names
                        if species in species_map:
                            species_map[species] += 1
                            species = f"{species}_num{species_map[species]}"
                        else:
                            species_map[species] = 1

                        # Append suffix
                        species += suffix

                        # Write the newly formatted species header line
                        fid_out.write(f"{species}{white_spaces}{line[next_text_start:]}")
                        flag_new_species = False
                    else:
                        # Write the numerical coefficient lines exactly as they are
                        fid_out.write(line)

        except FileNotFoundError:
            raise FileNotFoundError(f"Could not open input file: {filename_input}. Ensure the path is correct.")
        except IOError as e:
            raise IOError(f"An error occurred during file operations: {e}")