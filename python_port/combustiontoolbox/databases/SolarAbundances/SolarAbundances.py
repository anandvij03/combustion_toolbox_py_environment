import os
import numpy as np
from typing import Union, List, Tuple, Any
from combustiontoolbox.utils.findIndex import findIndex


def resolve_path(filename: str) -> str:
    """
    Helper to resolve paths relative to the repository databases folder.
    """
    if not filename:
        return filename
    if os.path.isabs(filename):
        return filename
    if os.path.exists(filename):
        return os.path.abspath(filename)

    # Resolve relative to the repository databases folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, "..", "..", "..", ".."))
    db_path_outer = os.path.join(repo_root, "databases", filename)
    if os.path.exists(db_path_outer):
        return db_path_outer

    return os.path.abspath(filename)


class SolarAbundances:
    """
    The SolarAbundances class is used to read solar abundances and compute the
    initial molar composition of the mixture.
    """

    def __init__(self, filename: str = "abundances.txt", elementRefence: str = "H", **kwargs):
        self.elementRefence = elementRefence
        self.logAbundances, self.elements = self.read_abundances(filename)
        self.indexElementRefence = findIndex(self.elements, self.elementRefence)

    def abundances2moles(
        self, elements: Union[str, List[str]], metallicity: float = None
    ) -> Union[float, np.ndarray]:
        """
        Read solar abundances in log 10 scale and compute the initial molar
        fractions in the mixture [-]

        Args:
            elements (str or list): List with the given elements
            metallicity (float, optional): Metallicity

        Returns:
            moles (float or ndarray): moles relative to H of the remaining elements in the mixture
        """
        # Get abundances assuming unity metallicity
        logAbundances = np.copy(self.logAbundances)
        elementsDB = self.elements

        # Recompute with metallicity. NOTE: H and He do not change their abundances
        if metallicity is not None:
            index_H = findIndex(elementsDB, "H")  # 0-indexed in Python
            index_He = findIndex(elementsDB, "He")  # 0-indexed in Python

            index_change = list(range(len(elementsDB)))
            if index_He in index_change:
                index_change.remove(index_He)
            if index_H in index_change:
                index_change.remove(index_H)

            logAbundances[index_change] = logAbundances[index_change] + np.log10(metallicity)

        # Reorganize abundances as the given element list
        is_scalar = isinstance(elements, str)
        elements_list = [elements] if is_scalar else list(elements)

        indices = []
        for el in elements_list:
            idx = findIndex(elementsDB, el)  # 0-indexed
            indices.append(idx)

        # Compute moles relative to H of the remaining elements in the mixture
        ref_idx = self.indexElementRefence  # 0-indexed
        moles = 10.0 ** (logAbundances[indices] - logAbundances[ref_idx])

        if is_scalar:
            return float(moles[0])
        return moles

    @staticmethod
    def read_abundances(filename: str) -> Tuple[np.ndarray, List[str]]:
        """
        Read solar abundances file

        Format: [number element, element, abundance, name, molar mass (g/mol)]

        Args:
            filename (str): Filename with the data

        Returns:
            Tuple containing:
            * abundances (ndarray): Vector with the logarithmic base 10 solar abundances
            * elements (list): List with the given elements
        """
        resolved_filename = resolve_path(filename)
        abundances = []
        elements = []

        with open(resolved_filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    elements.append(parts[1])
                    abundances.append(float(parts[2]))

        return np.array(abundances, dtype=float), elements
