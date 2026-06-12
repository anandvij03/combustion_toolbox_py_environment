import re

def species2latex(species: str, flag_burcat: bool = True) -> str:
    """
    Convert species name into LaTeX format.
    """
    if not isinstance(species, str):
        return str(species)

    # Remove database Millennium prefix '_M'
    if not flag_burcat:
        species = species.replace('_M', '')
    else:
        species = species.replace('_M', '$_{\\rm M}$')

    # Check numbers
    digits_indices = [i for i, char in enumerate(species) if char.isdigit()]
    
    if not digits_indices:
        species_latex = species
    else:
        species_latex = ""
        pos1 = 0
        for idx in digits_indices:
            species_latex += species[pos1:idx] + '$_{' + species[idx] + '}$'
            pos1 = idx + 1
        if pos1 < len(species):
            species_latex += species[pos1:]

    # Check if ions
    species_latex = species_latex.replace('plus', '$^+$')
    species_latex = species_latex.replace('minus', '$^-$')

    # Check parentheses
    species_latex = re.sub(r'b([a-zA-Z]+)b', r'(\1)', species_latex)

    # Check concatenate $$
    species_latex = species_latex.replace('$$', '')

    # Check suffix
    species_latex = re.sub(r'_([a-zA-Z])', r'\1', species_latex)

    # Joining the Burcat subscript with the previous subscript
    if '_{\\rm M}' in species_latex and species_latex.count('_{') > 1:
        species_latex = species_latex.replace('_{\\rm M}', '')
        last_brace = species_latex.rfind('}')
        if last_brace != -1:
            species_latex = species_latex[:last_brace] + ', \\rm{M}' + species_latex[last_brace:]

    # Remove consecutive subscripts
    species_latex = species_latex.replace('}_{', '')

    return species_latex
