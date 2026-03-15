import numpy as np
import sys

SYMBOL_MAP = {
    "X": 0, "H": 1, "He": 2,  # "X": ghost atom
    "Li": 3, "Be": 4,
    "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38,
    "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}

class Atom:
    def __init__(self, symbol, Z, r):
        self.symbol = symbol
        self.Z = Z
        self.r = np.asarray(r, dtype=float).reshape(3)



def parse_atom(line):
    """Parse a single line [A] [x] [y] [z] into an Atom."""
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 4:
        raise ValueError("Atom Entry Syntax is Incorrect: [A] [x] [y] [z]")
    symbol = parts[0]
    Z = SYMBOL_MAP[symbol]
    x, y, z = map(float, parts[1:])
    return Atom(symbol, Z, np.array([x, y, z]))



def parse(fname):
    """Parse input file: %config section into a dict, %molecule section into atoms via parse_atom."""
    config = {}
    atoms = []
    with open(fname, "r") as f:
        section = None
        for line in f:
            stripped = line.strip()
            if stripped == r"$config":
                section = "config"
                continue
            if stripped == r"$molecule":
                section = "molecule"
                continue
            if stripped == r"$end":
                section = None
                continue

            if section == "config":
                parts = stripped.split()
                key, value = parts[0][:-1], parts[1]
                config[key] = value
            elif section == "molecule":
                if not stripped:
                    continue
                parts = stripped.split()

                if len(parts) == 2: # ignoring charge and multiplicity for now
                    charge = parts[0]
                    multiplicity = parts[1]
                else:
                    atom = parse_atom(line)
                    atoms.append(atom)
    return config, atoms



def print_atoms(atoms):
    string = ""
    for atom in atoms:
        string += f'    {atom.symbol} {atom.r[0]} {atom.r[1]} {atom.r[2]}' + "\n"
    return string

