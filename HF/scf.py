import numpy as np
from integrals import build_S, build_T, build_V, build_ERI


def scf(atoms, max_iter=100, eps=1e-8, verbose=False):
     # Integrals are often the bottle neck of HF and DFT, in practice the 
     # integrals are computed on the fly with many many other speed up tricks 
    print('Building Integrals...\n')
    S = build_S(atoms)
    T = build_T(atoms)
    V = build_V(atoms)
    ERI = build_ERI(atoms)
    print('Integrals Built!\n')
    
    H = T + V # core Hamiltonian (1e Hamiltonian), this matrix stays constant throughout the SCF proceedure
    n_occ = len([ao for atom in atoms for ao in atom.AOs]) // 2

    vals, vecs = np.linalg.eigh(S)
    X = vecs @ np.diag(1.0 / np.sqrt(vals)) # Orthogonalizer

    print('Initial Guess Method: Hcore \n')
    Fp = X.T @ H @ X
    _, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    print('Initial Guess Made\n')
    if verbose:
        print(f'C: \n{C}')
        print(f'P: \n{P}')

    n = S.shape[0]
    F = H.copy()
    F_new = F
    print('Beginning SCF Proceedure...\n')
    for i in range(max_iter):
        G = np.zeros((n, n)) # mean field contribution from two electron integrals 
        for u in range(n):
            for v in range(n):
                for s in range(n):
                    for l in range(n):
                        G[u, v] += P[s, l] * (ERI[u, v, s, l] - 0.5 * ERI[u, s, v, l]) # tensor contraction 
        F_new = H + G
        if np.linalg.norm(F_new - F, 'fro') < eps: # check for convergence by checking Fock matrix
            print(f'\nSCF Converged!\n')
            break
        F = F_new
        Fp = X.T @ F @ X
        E, Cp = np.linalg.eigh(Fp)
        C = X @ Cp 
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        print(f'    iteration: {i} \t Energy: {E[0]}')
        if verbose:
            print(f'F: \n{C}')
            print(f'C: \n{C}')
            print(f'P: \n{P}')

    return C, P, F_new