import numpy as np
from scipy.special import erf


def F0(t):
    """Boys Function evaluation"""
    if t < 1e-8:
        return 1 - t/3
    return 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t))



def norm_1s(alpha):
    """Normalization constant for a 1s primitive GTO (Szabo Eq. 3.203)"""
    return (2.0 * alpha / np.pi) ** 0.75


def S_pq(alpha, Ra, beta, Rb):
    """Primitive Overlap Integral over primitive index p, q (Szabo Eq. A.9)"""
    p = alpha + beta
    mu = alpha * beta / p
    R2 = np.dot(Ra - Rb, Ra - Rb)
    # return norm_1s(alpha) * norm_1s(beta) * (np.pi / p) ** 1.5 * np.exp(-mu * R2)
    return norm_1s(alpha) * norm_1s(beta) * (np.pi / p) ** 1.5 * np.exp(-mu * R2)


def T_pq(alpha, Ra, beta, Rb):
    """Primitive Kinetic Integral T_pq (Szabo Eq. A.11)"""
    p = alpha + beta
    mu = alpha * beta / p
    R2 = np.dot(Ra - Rb, Ra - Rb)
    S = norm_1s(alpha) * norm_1s(beta) * (np.pi / p) ** 1.5 * np.exp(-mu * R2)
    return mu * (3.0 - 2.0 * mu * R2) * S


def V_pq(alpha, Ra, beta, Rb, Zc, Rc):
    """1e nuclear attraction integral V_pq (Szabo Eq. A.33)"""
    p = alpha + beta
    mu = alpha * beta / p
    R2 = np.dot(Ra - Rb, Ra - Rb)
    P = (alpha * Ra + beta * Rb) / p
    Rpc2 = np.dot(P - Rc, P - Rc)
    return (-Zc
            * norm_1s(alpha) * norm_1s(beta)
            * (2.0 * np.pi / p)
            * np.exp(-mu * R2)
            * F0(p * Rpc2))


def I_pqrs(alpha, Ra, beta, Rb, gamma, Rc, delta, Rd):
    """2e repulsion integral over primitive index p, q, r, s (Szabo & Ostlund Eq. A.41)"""
    p = alpha + beta
    q = gamma + delta
    mu_ab = alpha * beta  / p
    mu_cd = gamma * delta / q
    P = (alpha * Ra + beta  * Rb) / p
    Q = (gamma * Rc + delta * Rd) / q
    Rab2 = np.dot(Ra - Rb, Ra - Rb)
    Rcd2 = np.dot(Rc - Rd, Rc - Rd)
    Rpq2 = np.dot(P  - Q,  P  - Q)
    delta_pq = p * q / (p + q)
    return (norm_1s(alpha) * norm_1s(beta) * norm_1s(gamma) * norm_1s(delta)
            * 2.0 * np.pi ** 2.5
            / (p * q * np.sqrt(p + q))
            * np.exp(-mu_ab * Rab2 - mu_cd * Rcd2)
            * F0(delta_pq * Rpq2))




def S_uv(ao_a, Ra, ao_b, Rb):
    """Overlap matrix element over CGF over indicies mu, nu"""
    s = 0.0
    for da, aa in zip(ao_a.d, ao_a.alpha):
        for db, ab in zip(ao_b.d, ao_b.alpha):
            s += da * db * S_pq(aa, Ra, ab, Rb)
    return s


def T_uv(ao_a, Ra, ao_b, Rb):
    """Kinetic matrix element over indicies mu, nu"""
    t = 0.0
    for da, aa in zip(ao_a.d, ao_a.alpha):
        for db, ab in zip(ao_b.d, ao_b.alpha):
            t += da * db * T_pq(aa, Ra, ab, Rb)
    return t


def V_uv(ao_a, Ra, ao_b, Rb, atoms):
    """1e nuclear attraction integral matrix element over indicies mu, nu and over each nuclei"""
    v = 0.0
    for da, aa in zip(ao_a.d, ao_a.alpha):
        for db, ab in zip(ao_b.d, ao_b.alpha):
            for atom in atoms:
                v += da * db * V_pq(aa, Ra, ab, Rb, atom.Z, atom.r)
    return v


def I_uvls(ao_a, Ra, ao_b, Rb, ao_c, Rc, ao_d, Rd):
    """2e ERI rank 4 tensor element over indicies mu, nu, lambda, sigma"""
    g = 0.0
    for da, aa in zip(ao_a.d, ao_a.alpha):
        for db, ab in zip(ao_b.d, ao_b.alpha):
            for dc, ac in zip(ao_c.d, ao_c.alpha):
                for dd, ad in zip(ao_d.d, ao_d.alpha):
                    g += da * db * dc * dd * I_pqrs(aa, Ra, ab, Rb, ac, Rc, ad, Rd)
    return g




def _basis_list(atoms):
    """Flatten all AOs across all atoms into a list of (AO, center) pairs."""
    return [(ao, atom.r) for atom in atoms for ao in atom.AOs]


def build_S(atoms):
    basis = _basis_list(atoms)
    n = len(basis)
    S = np.zeros((n, n))
    for i, (ao_i, Ri) in enumerate(basis):
        for j, (ao_j, Rj) in enumerate(basis):
            S[i, j] = S_uv(ao_i, Ri, ao_j, Rj)
    return S


def build_T(atoms):
    basis = _basis_list(atoms)
    n = len(basis)
    T = np.zeros((n, n))
    for i, (ao_i, Ri) in enumerate(basis):
        for j, (ao_j, Rj) in enumerate(basis):
            T[i, j] = T_uv(ao_i, Ri, ao_j, Rj)
    return T


def build_V(atoms):
    basis = _basis_list(atoms)
    n = len(basis)
    V = np.zeros((n, n))
    for i, (ao_i, Ri) in enumerate(basis):
        for j, (ao_j, Rj) in enumerate(basis):
            V[i, j] = V_uv(ao_i, Ri, ao_j, Rj, atoms)
    return V


def build_ERI(atoms):
    basis = _basis_list(atoms)
    n = len(basis)
    ERI = np.zeros((n, n, n, n))
    for i, (ao_i, Ri) in enumerate(basis):
        for j, (ao_j, Rj) in enumerate(basis):
            for k, (ao_k, Rk) in enumerate(basis):
                for l, (ao_l, Rl) in enumerate(basis):
                    ERI[i, j, k, l] = I_uvls(
                        ao_i, Ri, ao_j, Rj, ao_k, Rk, ao_l, Rl
                    )
    return ERI

