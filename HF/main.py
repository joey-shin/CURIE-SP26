import sys
import numpy as np
from atom import parse
from scf import scf
from integrals import build_T, build_V, build_ERI

def main():
    config, atoms = parse(sys.argv[1])
    print('Spawning Job...\n')
    
    if config['job'] == 'HF':
        C, P, F = scf(atoms)

        T = build_T(atoms)
        V = build_V(atoms)
        ERI = build_ERI(atoms)

        nbasis = P.shape[0]
        J = np.zeros((nbasis, nbasis))
        K = np.zeros((nbasis, nbasis))
        for m in range(nbasis):
            for n in range(nbasis):
                for l in range(nbasis):
                    for s in range(nbasis):
                        J[m, n] += P[l, s] * ERI[m, n, l, s]
                        K[m, n] += P[l, s] * ERI[m, l, n, s]

        E_T = np.trace(P @ T)
        E_V = np.trace(P @ V)
        E_J = 0.5  * np.trace(P @ J)
        E_K = -0.25 * np.trace(P @ K)
        E_elec = E_T + E_V + E_J + E_K

        E_nuc = 0.0
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                E_nuc += atoms[i].Z * atoms[j].Z / np.linalg.norm(atoms[i].r - atoms[j].r)

        print(f"E_kinetic = {E_T:.10f} Ha")
        print(f"E_nuclear = {E_V:.10f} Ha")
        print(f"E_coulomb = {E_J:.10f} Ha")
        print(f"E_exchange = {E_K:.10f} Ha")
        print(f"E_nuc_rep = {E_nuc:.10f} Ha")
        print(f"E_tot = {E_elec + E_nuc:.10f} Ha")
    # elif:  you can imagine a long list of conditional statments tailored for different methods in a Quantum Chemistry Program 


if __name__ == "__main__":
    main()
