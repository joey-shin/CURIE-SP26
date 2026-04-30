import os
import numpy as np

SYMBOL_MAP = {
    "X": 0, "H": 1, "He": 2,
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
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}

A_TO_BOHR = 1.889726


class AO:
    """
    Atomic Orbitals (Contracted Gaussians)

    Attributes
    ----------
    shell_type : str
        Orbital Type (S, SP, P, ...)
    alpha : np.array
        Primitive Gaussain Exponents
    d : np.array
        Contraction Coefficients for Primitive Guassians
    """
    def __init__(self, shell_type, alpha, d):
        self.shell_type = shell_type.upper()
        self.alpha = np.array(alpha)
        self.d = np.array(d)

    def __repr__(self):
        return f"AO(shell_type={self.shell_type!r}, n_prim={len(self.alpha)})"


class Atom:
    """
    An atom with its Contracted Gaussian basis (AOs)

    Attributes
    ----------
    symbol : str
    Z : int
        Atomic number.
    r : ndarray, shape (3,)
        Cartesian coordinates in bohr.
    AOs : list of AO
        Contracted shells assigned from the basis set.
    """

    def __init__(self, symbol, Z, r, AOs=None):
        self.symbol = symbol
        self.Z = SYMBOL_MAP[symbol]
        self.r = np.asarray(r).reshape(3)
        self.AOs = AOs

    def __repr__(self):
        return f"Atom(symbol={self.symbol!r}, Z={self.Z}, r={self.r}, n_aos={len(self.AOs)})"



def parse_basis(fname):
    """
    Parse a Gaussian-format basis set file.

    Returns
    -------
    dict mapping element symbol -> list of AO
    """
    basis = {}
    with open(fname, "r") as f:
        lines = [l.rstrip() for l in f if not l.startswith("!")]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "****":
            i += 1
            continue

        parts = line.split()
        # Element header: "H     0"
        if len(parts) == 2 and parts[1] == "0":
            symbol = parts[0]
            basis[symbol] = []
            i += 1
            # Read shells until the next **** separator
            while i < len(lines) and lines[i].strip() != "****":
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                # Shell header: e.g. "S    3   1.00"
                shell_parts = line.split()
                shell_type = shell_parts[0].upper()
                n_prim = int(shell_parts[1])
                i += 1
                exponents = []
                coeffs = []
                for _ in range(n_prim):
                    prim = lines[i].strip().replace("D", "E").replace("d", "e")
                    vals = prim.split()
                    exponents.append(float(vals[0]))
                    if shell_type == "SP":
                        coeffs.append([float(vals[1]), float(vals[2])])
                    else:
                        coeffs.append(float(vals[1]))
                    i += 1
                basis[symbol].append(AO(shell_type, exponents, coeffs))
        else:
            i += 1

    return basis



def parse_atom(line):
    """Parse a single '  H  x  y  z' line into an Atom (no AOs assigned)."""
    parts = line.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Atom entry must be '[symbol] x y z', got: {line!r}")
    symbol = parts[0]
    Z = SYMBOL_MAP[symbol]
    coords = np.array(list(map(float, parts[1:]))) * A_TO_BOHR
    return Atom(symbol, Z, coords)


def parse(fname, basis_file=None):
    """
    Parse an HF input file.

    Parameters
    ----------
    fname : str
        Path to the .in file (uses %config / %molecule / %end sections).
    basis_file : str, optional
        Explicit path to a Gaussian-format basis set file.  If omitted, the
        basis name from the %config section is used to locate a .bas file in
        the same directory as the input file.

    Returns
    -------
    config : dict
    atoms  : list of Atom
    """
    config = {}
    atoms = []

    with open(fname, "r") as f:
        section = None
        for line in f:
            stripped = line.strip()
            if stripped in ("%config", "$config"):
                section = "config"
                continue
            if stripped in ("%molecule", "$molecule"):
                section = "molecule"
                continue
            if stripped in ("%end", "$end"):
                section = None
                continue

            if section == "config":
                if not stripped:
                    continue
                parts = stripped.split()
                key = parts[0].rstrip(":")
                value = parts[1]
                config[key] = value
            elif section == "molecule":
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) == 2:
                    config["charge"] = int(parts[0])
                    config["multiplicity"] = int(parts[1])
                else:
                    atoms.append(parse_atom(line))

    if basis_file is None and "basis" in config:
        input_dir = os.path.dirname(os.path.abspath(fname))
        basis_name = config["basis"].lower()
        search_dirs = [input_dir, os.path.join(input_dir, "..", "basis"), os.path.join(input_dir, "..", "..", "basis")]
        for d in search_dirs:
            candidate = os.path.join(d, basis_name + ".bas")
            if os.path.isfile(candidate):
                basis_file = candidate
                break

    if basis_file is not None:
        basis = parse_basis(basis_file)
        for atom in atoms:
            atom.AOs = list(basis.get(atom.symbol, []))

    return config, atoms


def print_atoms(atoms):
    lines = [f"    {a.symbol}  {a.r[0]:.6f}  {a.r[1]:.6f}  {a.r[2]:.6f}" for a in atoms]
    return "\n".join(lines) + "\n"
