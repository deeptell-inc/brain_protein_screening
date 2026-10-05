#!/usr/bin/env python3
"""Magnetic field effect on the product yield of a flavin–amine radical pair.

Reaction scheme (rp_theory for the rates):

    closed shell  ⇄  ¹RP        formation and singlet back transfer k_S
    ¹RP, ³RP      →  product    spin-independent forward chemistry k_P
                                (e.g. α-C–H deprotonation of the aminium)

The pair is born singlet. The observable is the product yield
Φ_P = k_P ∫ Tr ρ dt, which is what a change in catalytic flux would require;
the back-transfer yield is Φ_BET = 1 − Φ_P.

Spin Hamiltonian (MHz):
    H = ω_B b̂·(S1 + S2) + Σ_k a_k S_e(k)·I_k + J_gap S1·S2 + S1·T·S2
where J_gap = J_reactive + J_direct and T is the full traceless dipolar tensor.
Recombination is Haberkorn: −½{k_S P_S + k_P 1, ρ}.

Two solvers:
  * `yields_hilbert`: exact, coherent (no electron relaxation), via the
    non-Hermitian H_eff = H − (i/2π)(k_S P_S + k_P)/2 in rad/µs units. Cheap enough
    for many nuclei.
  * `yields_liouville`: exact with single-electron dephasing T2e; small systems.
"""

import numpy as np
from scipy.linalg import solve_sylvester

from qbscreen.calibrated_mfe import _spin_matrices, _embed, zeeman_MHz
from qbscreen.master_equation import build_liouvillian

TWO_PI = 2 * np.pi


def build_system(nuclei):
    """nuclei: list of (electron 0|1, I, a) with a an isotropic coupling (MHz) or a
    3×3 hyperfine tensor (MHz, S·A·I). Returns operator table and dims."""
    dims = [2, 2] + [int(round(2 * I + 1)) for _, I, _ in nuclei]
    S = [_spin_matrices(0.5)] * 2 + [_spin_matrices(I) for _, I, _ in nuclei]
    ops = [[_embed(S[k][a], k, dims) for a in range(3)] for k in range(len(dims))]
    dim = int(np.prod(dims))
    S1S2 = sum(ops[0][a] @ ops[1][a] for a in range(3))
    P_S = 0.25 * np.eye(dim) - S1S2
    Hhf = np.zeros((dim, dim), dtype=complex)
    for k, (e, _, a) in enumerate(nuclei):
        A = np.asarray(a, float) * (np.eye(3) if np.ndim(a) == 0 else 1.0)
        Hhf += sum(A[x, y] * ops[e][x] @ ops[2 + k][y] for x in range(3) for y in range(3) if A[x, y])
    return dict(ops=ops, dim=dim, S1S2=S1S2, P_S=P_S, Hhf=Hhf)


def hamiltonian(sysd, B_T, b_hat, J_gap, T_dip):
    ops = sysd["ops"]
    w = zeeman_MHz(B_T)
    H = sysd["Hhf"] + J_gap * sysd["S1S2"]
    for e in (0, 1):
        H = H + w * sum(b_hat[a] * ops[e][a] for a in range(3))
    for a in range(3):
        for b in range(3):
            if T_dip[a, b]:
                H = H + T_dip[a, b] * ops[0][a] @ ops[1][b]
    return H


def axial_tensor(D, axis=(0.0, 0.0, 1.0)):
    """Traceless tensor whose S1·T·S2 equals 2D[(S1·n)(S2·n) − S1·S2/3]."""
    n = np.asarray(axis, float)
    n /= np.linalg.norm(n)
    return 2 * D * (np.outer(n, n) - np.eye(3) / 3)


def yields_hilbert(sysd, H_MHz, k_S, k_P):
    """(Φ_P, Φ_BET) for a singlet-born pair, no electron relaxation. Rates in 1/µs."""
    P_S = sysd["P_S"]
    rho0 = P_S / np.trace(P_S)
    Heff = TWO_PI * H_MHz - 0.5j * (k_S * P_S + k_P * np.eye(sysd["dim"]))
    # X = ∫ρ dt solves H_eff X − X H_eff† = −iρ0 (Bartels–Stewart; an eigendecomposition
    # of the non-normal H_eff breaks down at near-degenerate eigenvalues)
    X = solve_sylvester(Heff, -Heff.conj().T, -1j * rho0)
    return float(np.real(k_P * np.trace(X))), float(np.real(k_S * np.trace(P_S @ X)))


def yields_liouville(sysd, H_MHz, k_S, k_P, T2e_ns=None, k_d=0.0):
    """Exact yields with single-electron dephasing and the reactive singlet–triplet
    dephasing −k_d(P_S ρ P_T + P_T ρ P_S) that appears at fourth order in the
    electronic coupling (Fay, Lindoy & Manolopoulos 2018, eq. 75a). Rates in 1/µs."""
    P_S = sysd["P_S"]
    dim = sysd["dim"]
    P_T = np.eye(dim) - P_S
    rho0 = P_S / np.trace(P_S)
    deph = [] if T2e_ns is None else [(2.0 / (T2e_ns * 1e-3), sysd["ops"][e][2]) for e in (0, 1)]
    L = build_liouvillian(TWO_PI * H_MHz, P_S, P_T, k_S + k_P, k_P, deph)
    if k_d:
        # vec(A ρ B) = (Bᵀ ⊗ A) vec(ρ), column stacking
        L = L - k_d * (np.kron(P_T.T, P_S) + np.kron(P_S.T, P_T))
    x = np.linalg.solve(-L, rho0.flatten(order="F"))
    X = x.reshape((dim, dim), order="F")
    return float(np.real(k_P * np.trace(X))), float(np.real(k_S * np.trace(P_S @ X)))


def yields_t2e(sysd, H_MHz, k_S, k_P, T2e_ns, tol=1e-12):
    """Same yields as yields_liouville with single-electron dephasing, without
    building the Liouvillian (for systems too large for it).

    With γ = 2/T2e, the dephasing γ Σ_e (S_ez ρ S_ez − ρ/4) splits into a decay
    −(γ/2)ρ, absorbed into K = 2πH − (i/2)(k_S P_S + (k_P + γ/2) I), and a
    recycling term γ R(ρ), R(ρ) = Σ_e S_ez ρ S_ez. X = ∫ρ dt then solves
        K X − X K† = −i ρ0 − i γ R(X),
    which is iterated by GMRES with the left side inverted by a triangular
    Sylvester solve in the Schur basis of K (computed once)."""
    from scipy.linalg import schur
    from scipy.linalg.lapack import ztrsyl
    from scipy.sparse.linalg import LinearOperator, gmres
    dim, P_S = sysd["dim"], sysd["P_S"]
    gam = 2.0 / (T2e_ns * 1e-3)
    Sz = [sysd["ops"][e][2] for e in (0, 1)]
    K = TWO_PI * H_MHz - 0.5j * (k_S * P_S + (k_P + gam / 2) * np.eye(dim))
    T, U = schur(K.astype(complex), output="complex")
    Uh = U.conj().T

    def A(C):                                   # solve K X − X K† = C
        Y, scale, info = ztrsyl(T, T, Uh @ C @ U, trana="N", tranb="C", isgn=-1)
        if info < 0:
            raise RuntimeError(f"trsyl failed ({info})")
        return U @ (Y / scale) @ Uh

    rho0 = P_S / np.trace(P_S)
    b = A(-1j * rho0).reshape(-1)
    op = lambda x: x - A(-1j * gam * sum(S @ x.reshape(dim, dim) @ S for S in Sz)).reshape(-1)
    x, info = gmres(LinearOperator((dim * dim, dim * dim), matvec=op, dtype=complex), b,
                    x0=b, rtol=tol, atol=0.0, restart=60, maxiter=400)
    if info != 0:
        raise RuntimeError(f"GMRES did not converge ({info})")
    X = x.reshape(dim, dim)
    return float(np.real(k_P * np.trace(X))), float(np.real(k_S * np.trace(P_S @ X)))


def mfe_product(nuclei, J_gap, T_dip, k_S, k_P, b_hat=(0.0, 0.0, 1.0), B_T=50e-6, T2e_ns=None, k_d=0.0):
    """Relative change of the product yield at B_T, in %."""
    sysd = build_system(nuclei)
    solve = (lambda H: yields_hilbert(sysd, H, k_S, k_P)) if (T2e_ns is None and not k_d) else \
            (lambda H: yields_liouville(sysd, H, k_S, k_P, T2e_ns, k_d))
    b = np.asarray(b_hat, float)
    b /= np.linalg.norm(b)
    p0 = solve(hamiltonian(sysd, 0.0, b, J_gap, T_dip))[0]
    pB = solve(hamiltonian(sysd, B_T, b, J_gap, T_dip))[0]
    return (pB - p0) / p0 * 100


if __name__ == "__main__":
    nuc = [(0, 1.0, 14.66), (1, 0.5, 44.0)]
    sysd = build_system(nuc)
    T = axial_tensor(-50.0, (0.3, 0.2, 0.9))
    H = hamiltonian(sysd, 1e-3, np.array([0.0, 0.0, 1.0]), 12.0, T)
    a = yields_hilbert(sysd, H, 2.0, 0.5)
    b = yields_liouville(sysd, H, 2.0, 0.5)
    assert np.allclose(a, b, rtol=1e-8, atol=1e-12), (a, b)
    assert np.isclose(sum(a), 1.0, atol=1e-10), a                   # every pair ends somewhere
    d = yields_liouville(sysd, H, 2.0, 0.5, T2e_ns=50.0)
    assert np.allclose(d, yields_t2e(sysd, H, 2.0, 0.5, 50.0), rtol=1e-9, atol=1e-12)  # matrix-free = dense
    c = yields_liouville(sysd, H, 2.0, 0.5, k_d=0.7)
    assert np.isclose(sum(c), 1.0, atol=1e-10), c                   # dephasing conserves population
    # axial tensor reproduces the Efimova–Hore level pattern: T± at D/3, T0 at −2D/3
    s2 = build_system([])
    ev = np.sort(np.linalg.eigvalsh(hamiltonian(s2, 0.0, np.array([0, 0, 1.0]), 0.0, axial_tensor(-10.0))))
    assert np.allclose(ev, sorted([0.0, -10 / 3, -10 / 3, 20 / 3])), ev
    print("product_yield self-checks passed")
