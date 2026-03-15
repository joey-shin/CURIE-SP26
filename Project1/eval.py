import numpy as np
from atom import Atom, print_atoms


R_AR = 3.4 # Angstroms
E_AR = 0.998 # kJ/mol
GOLDEN_RATIO = (np.sqrt(5) - 1) / 2


def atoms_to_coord(atoms):
    """Return coordinates as array of shape (3, n_atoms)."""
    n = len(atoms)
    coord = np.zeros((3, n), dtype=float)
    for i, atom in enumerate(atoms):
        coord[:, i] = atom.r
    return coord.T


def coord_to_atoms(coord, atoms):
    """Update atoms in place from a (3, n_atoms) coordinate array."""
    n = len(atoms)
    for i in range(n):
        atoms[i].r = coord[:, i].copy()
    return atoms


def Energy(atoms):
    """Lennard-Jones energy from Atom list"""
    E = 0.0
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            diff = atoms[i].r - atoms[j].r
            R2_ij = float(np.dot(diff, diff))
            frac_2 = (R_AR * R_AR) / R2_ij
            frac_6 = frac_2**3
            frac_12 = frac_6**2
            E += 4.0 * E_AR * (frac_12 - frac_6)
    return E


def Force(atoms):
    """Analytical force from Atom list"""
    n = len(atoms)
    F = np.zeros((3, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = atoms[i].r - atoms[j].r
            R2_ij = float(np.dot(diff, diff))
            R6_ij = R2_ij**3
            R8_ij = R6_ij * R2_ij
            R14_ij = R8_ij * R6_ij
            factor = 24.0 * E_AR * (
                (2.0 * R_AR**12) / (R14_ij) - (R_AR**6) / (R8_ij)
            )
            F[:, i] += factor * diff
    return F


def Energy_coords(coords):
    """Lennard-Jones energy from (3, n) coordinate matrix"""
    E = 0.0
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            R2_ij = float(np.dot(diff, diff))
            frac_2 = (R_AR * R_AR) / R2_ij
            frac_6 = frac_2**3
            frac_12 = frac_6**2
            E += 4.0 * E_AR * (frac_12 - frac_6)
    return E


def Force_coords(coords):
    """Analytical Force from (3, n) coordinate matrix"""
    n = len(coords)
    F = np.zeros((3, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = coords[i] - coords[j]
            R2_ij = float(np.dot(diff, diff))
            R6_ij = R2_ij**3
            R8_ij = R6_ij * R2_ij
            R14_ij = R8_ij * R6_ij
            factor = 24.0 * E_AR * (
                (2.0 * R_AR**12) / (R14_ij) - (R_AR**6) / (R8_ij)
            )
            F[:, i] += factor * diff
    return F



def GD(atoms, stepsize, epsilon, fname=None, verbose=False, keep_trajectory=False, maxiter=500):
    """Gradient Descent with Analytical Force"""
    trajectory = []
    atoms_opt = atoms.copy()
    coord_old = atoms_to_coord(atoms_opt)
    E_old = Energy_coords(coord_old)
    F = Force_coords(coord_old).T
    iterations = 0

    header = f"""
Initial coordinates: \n{print_atoms(atoms)} 
Initial energy: {E_old:.3e} kJ/mol
Initial Force: \n{F}
Initial stepsize: {stepsize} 
Threshold for convergence: {epsilon} \n
    """
    f = open(fname, "w") if fname is not None else None
    if fname is None:
        print(header)
    else:
        f.write(header)

    while np.linalg.norm(F, ord="fro") > epsilon and iterations < maxiter:
        if verbose:
            log = f"Iteration {iterations}: \n"

        direction = F / np.linalg.norm(F, ord="fro")
        coord_new = coord_old + (stepsize * direction)
        E_new = Energy_coords(coord_new)
        if E_new < E_old:
            coord_old = coord_new
            E_old = E_new
            F = Force_coords(coord_new).T
            stepsize *= 1.2
            if verbose:
                log += "New Step Accepted."
            if keep_trajectory:
                trajectory.append(coord_new)
        else:
            stepsize /= 2.0
            if verbose:
                log += "New Step Rejected, reducing stepsize... "

        log += f"""
Current stepsize: {stepsize}
New coordinates: \n{print_atoms(coord_to_atoms(coord_new.T, atoms_opt))}
New Force: \n{F}
New energy: {E_old} \n
        """

        if verbose:
            if fname is None:
                print(log)
            elif fname is not None:
                f.write(f"{log}\n")
        iterations += 1

    if f is not None:
        f.close()

    atoms_opt = coord_to_atoms(coord_new.T, atoms_opt)
    if keep_trajectory:
        return trajectory
    return atoms_opt, E_new, iterations



def GDLS(atoms, stepsize, epsilon, LS, fname=None,verbose=False, maxiter=500):
    """Gradient Descent with Analytical Force and Line Search"""
    atoms_opt = atoms.copy()
    coord_old = atoms_to_coord(atoms_opt)
    E_old = Energy_coords(coord_old)
    F = Force_coords(coord_old).T
    iterations = 0

    header = f"""
Initial coordinates: \n{print_atoms(atoms)} 
Initial energy: {E_old:.3e} kJ/mol
Initial Force: \n{F}
Threshold for convergence: {epsilon}\n\n
    """

    f = open(fname, "w") if fname is not None else None

    if fname is None:
        print(header)
    else:
        f.write(header)

    while np.linalg.norm(F, ord="fro") > epsilon and iterations < maxiter:
        if verbose:
            log = f"Iteration {iterations}:"
        
        direction = F / np.linalg.norm(F, ord="fro")
        trial_stepsize = stepsize


        step_opt = LS(coord_old, F, stepsize)
        coord_new = coord_old + (step_opt * direction)
        E_new = Energy_coords(coord_new)

        if E_new < E_old:
            coord_old = coord_new
            E_old = E_new
            F = Force_coords(coord_old).T
            if verbose:
                log += f"""
Line search step accepted with step = {step_opt:.3e}
New coordinates: \n{print_atoms(coord_to_atoms(coord_old.T, atoms_opt))}
New Force: \n{F}
New energy: {E_old} \n
                """
            iterations += 1
        else:
            stepsize /= 2.0
            if verbose:
                log += f"""
Line search with initial stepsize = {stepsize} Rejected
Reducing initial stepsize...\n
                """
        if verbose:
            if fname is None:
                print(log)
            else:
                f.write(f"{log}\n")

    if f is not None:
        f.close()

    atoms_opt = coord_to_atoms(coord_old.T, atoms_opt)
    return atoms_opt, E_old, iterations



def BFGS(atoms, stepsize, epsilon, fname=None, verbose=False, maxiter=500):
    """BFGS Quasi-Newton Optimization with Line Search"""
    atoms_opt = atoms.copy()
    coord_old = atoms_to_coord(atoms_opt)
    E_old = Energy_coords(coord_old)
    F = Force_coords(coord_old).T 
    
    n = coord_old.size
    H_inv = np.eye(n) # Create Identity Matrix, Initial Guess for Inverse Hessian
    iterations = 0

    header = f"""
Initial coordinates: \n{print_atoms(atoms)} 
Initial energy: {E_old:.3e} kJ/mol
Initial Force: \n{F}
Initial stepsize: {stepsize} 
Threshold for convergence: {epsilon} \n
    """
    f = open(fname, "w") if fname is not None else None
    if fname is None:
        print(header)
    else:
        f.write(header)

    while np.linalg.norm(F, ord="fro") > epsilon and iterations < maxiter:
        if verbose:
            log = f"Iteration {iterations}: \n"

        g_old = -F.flatten()

        direction = -H_inv @ g_old
        direction_3n = direction.reshape(coord_old.shape)
        
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction_3n = direction_3n / dir_norm

        coord_new = coord_old + (stepsize * direction_3n)
        E_new = Energy_coords(coord_new)

        if E_new < E_old:
            F_new = Force_coords(coord_new).T
            g_new = -F_new.flatten()

            s = (coord_new - coord_old).flatten()   # shape (n,)
            y = g_new - g_old                  # shape (n,)

            sy = s @ y  # scalar curvature
            if sy > 1e-10:  # only update if curvature condition holds
                # rho = 1.0 / sy
                # I = np.eye(n)
                # A = I - rho * np.outer(s, y)
                # B = I - rho * np.outer(y, s)
                # H_inv = A @ H_inv @ B + rho * np.outer(s, s)

                Hy  = H_inv @ y
                yHy = y @ Hy
                sty = sy
                H_inv += (((sty + yHy) / (sty ** 2)) * np.outer(s, s)) - ((np.outer(Hy, s) + np.outer(s, Hy)) / sty)

            coord_old = coord_new
            E_old = E_new
            F = F_new
            stepsize *= 1.2
            if verbose:
                log += "New BFGS Step Accepted"
        else:
            # Step rejected: reduce stepsize, reset inverse Hessian
            stepsize /= 2.0
            H_inv = np.eye(n)
            if verbose:
                log += "New BFGS Step Rejected, reducing stepsize and recomputing inverse Hessian... "

        log += f"""
Current stepsize: {stepsize}
New coordinates: \n{print_atoms(coord_to_atoms(coord_new.T, atoms_opt))}
New Force: \n{F}
New energy: {E_old} \n
        """

        if verbose:
            if fname is None:
                print(log)
            elif fname is not None:
                f.write(f"{log}\n")
        iterations += 1

    if f is not None:
        f.close()

    atoms_opt = coord_to_atoms(coord_new.T, atoms_opt)
    return atoms_opt, E_new, iterations



def GoldenSection(coord, direction, init_step, tol=1e-4, maxiter=100):
    a = 0.0
    b = a + init_step
    A = coord + (a * direction)
    B = coord + (b * direction)
    Ea = Energy_coords(A)
    Eb = Energy_coords(B)

    while Eb < Ea:
        b *= 1.2
        B = coord + (b * direction)
        Eb = Energy_coords(B)

    L = abs(a - b)

    i = 0
    while L > tol:
        c = a + GOLDEN_RATIO**2 * L
        d = b - GOLDEN_RATIO**2 * L
        C = coord + (c * direction)
        D = coord + (d * direction)

        Ec, Ed = Energy_coords(C), Energy_coords(D)
        if Ec > Ed:
            a = c
        elif Ec < Ed:
            b = d

        L = abs(a - b)
        c = a + GOLDEN_RATIO**2 * L
        d = b - GOLDEN_RATIO**2 * L
        A = coord + (a * direction)
        B = coord + (b * direction)
        i += 1
    
    return 0.5 * (a + b)


