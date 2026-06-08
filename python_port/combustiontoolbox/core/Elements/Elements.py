from combustiontoolbox.utils.findIndex import findIndex


class Elements:
    """
    The Elements class is used to store the list of elements that may appear in the database.

    The Elements object can be initialized as follows:

          elements = Elements()

    This creates an instance of the `Elements` class and initializes it with a predefined list of elements
    and their indices.

    See also: Species, ChemicalSystem, Database, NasaDatabase
    """

    def __init__(self):
        # Properties
        self.listElements = None
        self.indexH = None
        self.indexN = None
        self.indexO = None
        self.indexF = None
        self.indexCl = None
        self.indexHe = None
        self.indexNe = None
        self.indexAr = None
        self.indexKr = None
        self.indexXe = None
        self.indexRn = None

        # Initialization
        elements, _ = self._setElements()
        self.listElements = elements
        self._setIndexStableElements()

    def getElements(self):
        return self.listElements

    def _setIndexStableElements(self):
        """
        The only elements that are stable as diatomic gases are elements
        1 (H), 9 (N), 10 (O), 11 (F), and 19 (Cl). The remaining elements that
        are stable as (monoatomic) gases are the noble gases He (4), Ne (12),
        Ar (20), Kr (38), Xe (56), and Rn (88), which do not form any compound.
        """
        self.indexH = findIndex(self.listElements, "H")
        self.indexN = findIndex(self.listElements, "N")
        self.indexO = findIndex(self.listElements, "O")
        self.indexF = findIndex(self.listElements, "F")
        self.indexCl = findIndex(self.listElements, "Cl")
        self.indexHe = findIndex(self.listElements, "He")
        self.indexNe = findIndex(self.listElements, "Ne")
        self.indexAr = findIndex(self.listElements, "Ar")
        self.indexKr = findIndex(self.listElements, "Kr")
        self.indexXe = findIndex(self.listElements, "Xe")
        self.indexRn = findIndex(self.listElements, "Rn")

    @staticmethod
    def _setElements():
        """
        Set list with elements name.

        Returns:
            Tuple containing:
                * elements (list): Elements
                * ne (int):        Number of elements
        """
        elements = [
            "H",   #  1
            "D",   #  1 - Deuterium - Heavy hydrogen (^2H)
            "T",   #  1 - Tritium   - Heavy hydrogen (^3H)
            "He",  #  2
            "Li",  #  3
            "Be",  #  4
            "B",   #  5
            "C",   #  6
            "N",   #  7
            "O",   #  8
            "F",   #  9
            "Ne",  # 10
            "Na",  # 11
            "Mg",  # 12
            "Al",  # 13
            "Si",  # 14
            "P",   # 15
            "S",   # 16
            "Cl",  # 17
            "Ar",  # 18
            "K",   # 19
            "Ca",  # 20
            "Sc",  # 21
            "Ti",  # 22
            "V",   # 23
            "Cr",  # 24
            "Mn",  # 25
            "Fe",  # 26
            "Co",  # 27
            "Ni",  # 28
            "Cu",  # 29
            "Zn",  # 30
            "Ga",  # 31
            "Ge",  # 32
            "As",  # 33
            "Se",  # 34
            "Br",  # 35
            "Kr",  # 36
            "Rb",  # 37
            "Sr",  # 38
            "Y",   # 39
            "Zr",  # 40
            "Nb",  # 41
            "Mo",  # 42
            "Tc",  # 43
            "Ru",  # 44
            "Rh",  # 45
            "Pd",  # 46
            "Ag",  # 47
            "Cd",  # 48
            "In",  # 49
            "Sn",  # 50
            "Sb",  # 51
            "Te",  # 52
            "I",   # 53
            "Xe",  # 54
            "Cs",  # 55
            "Ba",  # 56
            "La",  # 57
            "Ce",  # 58
            "Pr",  # 59
            "Nd",  # 60
            "Pm",  # 61
            "Sm",  # 62
            "Eu",  # 63
            "Gd",  # 64
            "Tb",  # 65
            "Dy",  # 66
            "Ho",  # 67
            "Er",  # 68
            "Tm",  # 69
            "Yb",  # 70
            "Lu",  # 71
            "Hf",  # 72
            "Ta",  # 73
            "W",   # 74
            "Re",  # 75
            "Os",  # 76
            "Ir",  # 77
            "Pt",  # 78
            "Au",  # 79
            "Hg",  # 80
            "Tl",  # 81
            "Pb",  # 82
            "Bi",  # 83
            "Po",  # 84
            "At",  # 85
            "Rn",  # 86
            "Fr",  # 87
            "Ra",  # 88
            "Ac",  # 89
            "Th",  # 90
            "Pa",  # 91
            "U",   # 92
            "Np",  # 93
            "Pu",  # 94
            "Am",  # 95
            "Cm",  # 96
            "Bk",  # 97
            "Cf",  # 98
            "Es",  # 99
            "Fm",  # 100
            "Md",  # 101
            "No",  # 102
            "Lr",  # 103
            "Rf",  # 104
            "Db",  # 105
            "Sg",  # 106
            "Bh",  # 107
            "Hs",  # 108
            "Mt",  # 109
            "Ds",  # 110
            "Rg",  # 111
            "Cn",  # 112
            "Nh",  # 113
            "Fl",  # 114
            "Mc",  # 115
            "Lv",  # 116
            "Ts",  # 117
            "Og",  # 118
            "E",   # 119 - Electron
        ]
        ne = len(elements)
        return elements, ne