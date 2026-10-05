#!/usr/bin/env python3
"""Magnetic field effect with explicitly stated exchange/dipolar conventions.

Conventions
-----------
Exchange.  H_J = J_gap S1·S2, so E_T − E_S = J_gap: J_gap is the full
singlet–triplet splitting in the absence of Zeeman and hyperfine terms.
Efimova & Hore (Biophys. J. 94, 1565 (2008)) write <S|H_J|S> = J, <T|H_J|T> = −J,
i.e. a splitting 2J, so J_gap = −2 J_EH.

Dipolar.  Point dipoles along unit vector n:
    H_D = 2 D (S1·n)(S2·n) − (2D/3) S1·S2 ,   D = −(3/2) μ0 g² μB² / (4π h r³)
This is the full (non-secular) tensor. With the field along n, the T± – T0
splitting is D, which is Efimova & Hore's D:  D/MHz = −77.9 / (r/nm)³
(their D/μT = −2.78e3 / (r/nm)³).

Frequencies are ordinary (MHz); the Hamiltonian is multiplied by 2π before
propagation. Rates k are 1/µs, not angular.
"""

import numpy as np

from qbscreen.spin_dynamics import SX, SY, SZ, spin_op, singlet_projector, G_E, MU_B, HBAR
from qbscreen.master_equation import singlet_yield_master, electron_dephasing_ops

TWO_PI = 2 * np.pi
D_EH_MHZ_NM3 = -2.78e3 * 1e-6 * G_E * MU_B / (HBAR * TWO_PI) / 1e6   # −77.9


def dipolar_D_MHz(r_nm):
    """Efimova–Hore D (T± − T0 splitting with B ∥ n), MHz."""
    return D_EH_MHZ_NM3 / r_nm ** 3


def zeeman_MHz(B_T):
    return G_E * MU_B * B_T / (HBAR * TWO_PI * 1e6)


def build_H(B_T, theta, hfc_e1, hfc_e2, J_gap, D):
    """Two electrons + spin-½ nuclei; returns H in MHz and the spin count.

    hfc_e1/hfc_e2: isotropic couplings (MHz) of the nuclei on radical 1/2.
    Dipolar axis n = z; the field lies in the xz plane at angle theta to n.
    """
    nuc = list(hfc_e1) + list(hfc_e2)
    n = 2 + len(nuc)
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    s = {(a, i): spin_op(op, i, n) for i in range(n) for a, op in (("x", SX), ("y", SY), ("z", SZ))}

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w = zeeman_MHz(B_T)
        for e in (0, 1):
            H += w * (np.sin(theta) * s[("x", e)] + np.cos(theta) * s[("z", e)])
        for k, a in enumerate(nuc):
            e = 0 if k < len(hfc_e1) else 1
            for ax in "xyz":
                H += a * s[(ax, e)] @ s[(ax, 2 + k)]
        S1S2 = sum(s[(ax, 0)] @ s[(ax, 1)] for ax in "xyz")
        H += J_gap * S1S2
        H += 2 * D * s[("z", 0)] @ s[("z", 1)] - (2 * D / 3) * S1S2
    return H, n


def mfe(B_T, theta, hfc_e1, hfc_e2, J_gap, D, k_per_us, T2e_ns=None):
    """MFE(B) in % relative to zero field, symmetric recombination k_S = k_T = k."""
    def phi(B):
        H, n = build_H(B, theta, hfc_e1, hfc_e2, J_gap, D)
        P_S = singlet_projector(0, 1, n)
        P_T = np.eye(2 ** n) - P_S
        rho0 = P_S / np.trace(P_S)
        T2e_us = None if T2e_ns is None else T2e_ns * 1e-3
        return singlet_yield_master(TWO_PI * H, P_S, P_T, k_per_us, k_per_us, rho0,
                                    electron_dephasing_ops(n, T2e_us))
    p0 = phi(0.0)
    return (phi(B_T) - p0) / p0 * 100


def _spin_matrices(I):
    m = np.arange(I, -I - 1, -1)
    d = len(m)
    Sz = np.diag(m).astype(complex)
    Sp = np.zeros((d, d), dtype=complex)
    for k in range(1, d):
        Sp[k - 1, k] = np.sqrt(I * (I + 1) - m[k] * (m[k] + 1))
    return (Sp + Sp.T.conj()) / 2, (Sp - Sp.T.conj()) / 2j, Sz


def _embed(op, k, dims):
    out = np.array([[1.0 + 0j]])
    for j, d in enumerate(dims):
        out = np.kron(out, op if j == k else np.eye(d))
    return out


def mfe_general(B_T, theta, nuclei, J_gap, D, k_per_us, T2e_ns=None):
    """MFE (%) with arbitrary nuclear spins. nuclei: list of (electron 0|1, I, a_iso_MHz)."""
    dims = [2, 2] + [int(round(2 * I + 1)) for _, I, _ in nuclei]
    S = [_spin_matrices(0.5)] * 2 + [_spin_matrices(I) for _, I, _ in nuclei]
    ops = [[_embed(S[k][a], k, dims) for a in range(3)] for k in range(len(dims))]
    dim = int(np.prod(dims))
    S1S2 = sum(ops[0][a] @ ops[1][a] for a in range(3))
    P_S = 0.25 * np.eye(dim) - S1S2
    P_T = np.eye(dim) - P_S
    rho0 = P_S / np.trace(P_S)

    def phi(B):
        w = zeeman_MHz(B)
        H = sum(w * (np.sin(theta) * ops[e][0] + np.cos(theta) * ops[e][2]) for e in (0, 1))
        for k, (e, _, a) in enumerate(nuclei):
            H = H + a * sum(ops[e][x] @ ops[2 + k][x] for x in range(3))
        H = H + J_gap * S1S2 + 2 * D * ops[0][2] @ ops[1][2] - (2 * D / 3) * S1S2
        deph = [] if T2e_ns is None else [(2.0 / (T2e_ns * 1e-3), ops[e][2]) for e in (0, 1)]
        return singlet_yield_master(TWO_PI * H, P_S, P_T, k_per_us, k_per_us, rho0, deph)

    p0 = phi(0.0)
    return (phi(B_T) - p0) / p0 * 100


if __name__ == "__main__":
    # the general-spin builder must reproduce the spin-½ builder exactly
    a = mfe(50e-6, 0.3, [14.0, 11.0], [44.0], 12.0, -11.0, 1.0, 1000.0)
    b = mfe_general(50e-6, 0.3, [(0, 0.5, 14.0), (0, 0.5, 11.0), (1, 0.5, 44.0)], 12.0, -11.0, 1.0, 1000.0)
    assert np.isclose(a, b, rtol=1e-8), (a, b)
    # convention checks: with B = 0 and no hyperfine, the triplet sublevels split
    # by D (T± vs T0) and the singlet sits J_gap below the triplet centroid.
    H, n = build_H(0.0, 0.0, [], [], J_gap=100.0, D=-10.0)
    ev = np.sort(np.linalg.eigvalsh(H))
    assert np.isclose(ev[0], -75.0), ev                       # singlet: −3J/4
    trip = ev[1:]
    D = -10.0                                                  # <T_m|H_D|T_m> = D(m² − 2/3)
    assert np.allclose(sorted(trip - 25.0), sorted([D / 3, D / 3, -2 * D / 3])), trip
    assert np.isclose(abs(dipolar_D_MHz(1.0)), 77.9, rtol=2e-3)
    print("calibrated_mfe convention checks passed; D(1 nm) =", round(dipolar_D_MHz(1.0), 2), "MHz")
