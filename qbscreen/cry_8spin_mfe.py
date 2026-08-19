#!/usr/bin/env python3
"""
cry_8spin_mfe.py — Cryptochrome FAD•-/Trp•+ radical pair: exact 8-spin
magnetic field effect (MFE) on the singlet yield.

WHY THIS MODULE EXISTS
----------------------
An adversarial review panel (2026-05-14) established that no code in this
repository implements the 8-spin cryptochrome model that the manuscripts
claim, and that the previously reported value MFE = -0.55% at 50 uT
(a) came from a stored curve whose generating script is absent,
(b) corresponds to the 35.714 uT grid point, not 50 uT, and
(c) cannot be reproduced from any parameter set in the repository.

The panel further identified a physics error in the earlier treatment:
1/T2* computed as sqrt(sum_i A_i^2 I(I+1)/3) is the *static inhomogeneous*
HFC linewidth.  In a radical-pair calculation that carries the nuclear spins
explicitly in the Hamiltonian, that quantity IS the coherent S-T mixing
drive.  Re-injecting it as an irreversible Lindblad/exponential dephasing
double-counts the same hyperfine coupling in both H and the dissipator, and
structurally forces MFE -> 0 independently of the physics.

This module therefore:
  * carries all 8 spins explicitly in H (14N as I=1, Hilbert dim = 384),
  * treats the irreversible electron dephasing rate Lambda = 1/T2 as an
    explicit FREE parameter that is swept and reported in full -- never
    silently set from 1/T2*,
  * includes the full electron-electron dipolar tensor with its angular
    dependence and performs an orientation average over the field direction,
  * puts 50.000 uT exactly on the field grid,
  * reports a convergence series over nuclear-spin truncation.

METHOD
------
Singlet-born radical pair, spin-independent (Haberkorn) recombination at rate
k.  In the eigenbasis H|m> = w_m|m>, with rho0 = P_S/M (M = nuclear
multiplicity), the singlet yield is the standard exponential-model result
(Timmel, Till, Brocklehurst, Hore, Mol. Phys. 95, 71 (1998)):

    Phi_S = (1/M) [ sum_m |<m|P_S|m>|^2
                    + sum_{m/=n} |<m|P_S|n>|^2 * k(k+L)/((k+L)^2 + w_mn^2) ]

The m = n terms are populations and do not dephase; the m /= n terms are
coherences and are broadened by the irreversible rate L = 1/T2.  Setting
L = 0 recovers the coherent limit exactly.

    MFE(B) = 100 * [Phi_S(B) - Phi_S(0)] / Phi_S(0)     [percent]

CAVEAT (stated, not hidden): dephasing in the H eigenbasis is the standard
phenomenological exponential model.  It is not identical to an Sz-type
Lindblad dissipator, and it does not describe a rotationally modulated
anisotropic bath.  A calculation that claims a specific T2 from molecular
tumbling must derive it from the *traceless* HFC / g / dipolar tensors and
must check the Redfield validity condition X*tau_c << 1 for every channel.
This module does not do that; it reports MFE as a function of an assumed
Lambda so the reader can see exactly how the answer depends on it.

Deterministic: exact diagonalization, no random numbers, no seeds required.

Usage
-----
    python3 cry_8spin_mfe.py                 # full run, writes JSON
    python3 cry_8spin_mfe.py --quick         # coarse grids, ~1 min
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Physical constants (SI, then converted to MHz angular units)
# ----------------------------------------------------------------------
MU_B = 9.2740100783e-24      # J/T   Bohr magneton
H_PLANCK = 6.62607015e-34    # J s
G_E = 2.0023                 # free-electron g factor
# gamma_e / 2pi  [MHz/mT]
GE_MHZ_PER_MT = G_E * MU_B / H_PLANCK * 1e-3 / 1e6   # ~28.03 MHz/mT

# nuclear gyromagnetic ratios, MHz/T  (gamma/2pi)
GAMMA_N_MHZ_PER_T = {"1H": 42.577, "31P": 17.235, "14N": 3.0777}

# point-dipole electron-electron coupling prefactor:
#   D/h [MHz] = 52.04 / r[nm]^3
DIP_PREFACTOR_MHZ_NM3 = 52.04


# ----------------------------------------------------------------------
# Spin operator machinery
# ----------------------------------------------------------------------
def spin_matrices(I: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ix, Iy, Iz) for spin quantum number I."""
    dim = int(round(2 * I + 1))
    m = np.array([I - k for k in range(dim)])
    Iz = np.diag(m).astype(complex)
    # raising operator <m+1|I+|m> = sqrt(I(I+1) - m(m+1))
    Ip = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):
        mm = m[k]
        Ip[k - 1, k] = np.sqrt(I * (I + 1) - mm * (mm + 1))
    Im = Ip.conj().T
    Ix = 0.5 * (Ip + Im)
    Iy = -0.5j * (Ip - Im)
    return Ix, Iy, Iz


def embed(op: np.ndarray, site: int, dims: list[int]) -> np.ndarray:
    """Embed single-site operator into the full product space."""
    mats = []
    for i, d in enumerate(dims):
        mats.append(op if i == site else np.eye(d, dtype=complex))
    out = mats[0]
    for mm in mats[1:]:
        out = np.kron(out, mm)
    return out


@dataclass
class Nucleus:
    label: str
    isotope: str          # "1H", "31P", "14N"
    I: float
    A_iso: float          # isotropic HFC, MHz
    A_ax: float = 0.0     # axial (traceless) HFC anisotropy, MHz
    radical: int = 0      # 0 -> electron 1 (FAD), 1 -> electron 2 (Trp)


@dataclass
class RPModel:
    """Cryptochrome FAD-/Trp+ radical pair."""
    nuclei: list[Nucleus]
    J: float = 0.0                 # isotropic exchange, MHz
    r_nm: float = 1.75             # inter-radical distance, nm (17.5 A)
    include_dipolar: bool = True
    include_nuclear_zeeman: bool = True
    dg: float = 0.0                # g2 - g1 (Delta g)

    dims: list[int] = field(init=False)
    _cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.dims = [2, 2] + [int(round(2 * n.I + 1)) for n in self.nuclei]

    @property
    def dim(self) -> int:
        return int(np.prod(self.dims))

    @property
    def M(self) -> int:
        """Nuclear multiplicity."""
        return int(np.prod(self.dims[2:]))

    @property
    def D_MHz(self) -> float:
        """Point-dipole coupling constant, MHz."""
        return DIP_PREFACTOR_MHZ_NM3 / self.r_nm ** 3

    # ---- operator builders (cached) ----
    def electron_ops(self):
        if "eops" not in self._cache:
            sx, sy, sz = spin_matrices(0.5)
            S1 = [embed(o, 0, self.dims) for o in (sx, sy, sz)]
            S2 = [embed(o, 1, self.dims) for o in (sx, sy, sz)]
            self._cache["eops"] = (S1, S2)
        return self._cache["eops"]

    def singlet_projector(self) -> np.ndarray:
        """P_S = 1/4 - S1.S2  (acts as identity on nuclei)."""
        if "PS" not in self._cache:
            S1, S2 = self.electron_ops()
            S1S2 = sum(S1[a] @ S2[a] for a in range(3))
            self._cache["PS"] = 0.25 * np.eye(self.dim, dtype=complex) - S1S2
        return self._cache["PS"]

    def H_static(self) -> np.ndarray:
        """Field-independent part: HFC + exchange (dipolar handled per-orientation)."""
        if "Hstat" in self._cache:
            return self._cache["Hstat"]
        S1, S2 = self.electron_ops()
        H = np.zeros((self.dim, self.dim), dtype=complex)
        # hyperfine: isotropic part
        for idx, nuc in enumerate(self.nuclei):
            ix, iy, iz = spin_matrices(nuc.I)
            Iops = [embed(o, 2 + idx, self.dims) for o in (ix, iy, iz)]
            S = S1 if nuc.radical == 0 else S2
            if nuc.A_iso != 0.0:
                H += nuc.A_iso * sum(S[a] @ Iops[a] for a in range(3))
        # isotropic exchange  J * S1.S2
        if self.J != 0.0:
            H += self.J * sum(S1[a] @ S2[a] for a in range(3))
        self._cache["Hstat"] = H
        return H

    def H_orient(self, B_mT: float, theta: float, phi: float) -> np.ndarray:
        """Orientation-dependent part: Zeeman + dipolar + axial HFC.

        The molecular frame has the inter-radical vector along z.
        (theta, phi) is the direction of B in that frame.
        """
        S1, S2 = self.electron_ops()
        n = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # --- electron Zeeman along B ---
        w1 = GE_MHZ_PER_MT * B_mT
        w2 = GE_MHZ_PER_MT * B_mT * (1.0 + self.dg / G_E)
        SB1 = sum(n[a] * S1[a] for a in range(3))
        SB2 = sum(n[a] * S2[a] for a in range(3))
        H += w1 * SB1 + w2 * SB2

        # --- nuclear Zeeman ---
        if self.include_nuclear_zeeman and B_mT != 0.0:
            for idx, nuc in enumerate(self.nuclei):
                g = GAMMA_N_MHZ_PER_T[nuc.isotope] * 1e-3   # MHz/mT
                ix, iy, iz = spin_matrices(nuc.I)
                Iops = [embed(o, 2 + idx, self.dims) for o in (ix, iy, iz)]
                H -= g * B_mT * sum(n[a] * Iops[a] for a in range(3))

        # --- electron-electron dipolar, full angular dependence ---
        # H_dd = D * [ S1.S2 - 3 (S1.u)(S2.u) ],  u = inter-radical unit vector = z
        if self.include_dipolar:
            D = self.D_MHz
            S1S2 = sum(S1[a] @ S2[a] for a in range(3))
            H += D * (S1S2 - 3.0 * (S1[2] @ S2[2]))

        # --- axial (traceless) hyperfine anisotropy, symmetry axis = z ---
        for idx, nuc in enumerate(self.nuclei):
            if nuc.A_ax == 0.0:
                continue
            ix, iy, iz = spin_matrices(nuc.I)
            Iops = [embed(o, 2 + idx, self.dims) for o in (ix, iy, iz)]
            S = S1 if nuc.radical == 0 else S2
            # T = A_ax * (3 Sz Iz - S.I) / 2   (axially symmetric, traceless)
            H += 0.5 * nuc.A_ax * (3.0 * (S[2] @ Iops[2])
                                   - sum(S[a] @ Iops[a] for a in range(3)))
        return H


# ----------------------------------------------------------------------
# Singlet yield
# ----------------------------------------------------------------------
def singlet_yield(model: RPModel, B_mT: float, k_MHz: float,
                  Lambda_MHz: float, theta: float, phi: float) -> float:
    """Exponential-model singlet yield for one field magnitude/orientation.

    All rates in MHz *angular* units are handled consistently: the
    Hamiltonian is built in MHz (frequency), so we convert to angular
    frequency by 2*pi for the Lorentzian denominators.
    """
    H = model.H_static() + model.H_orient(B_mT, theta, phi)
    w, V = np.linalg.eigh(H)                       # w in MHz (frequency)
    PS = model.singlet_projector()
    PSe = V.conj().T @ PS @ V
    P2 = np.abs(PSe) ** 2

    # angular frequencies
    wr = 2.0 * np.pi * w
    dw = wr[:, None] - wr[None, :]
    kk = 2.0 * np.pi * k_MHz
    LL = 2.0 * np.pi * Lambda_MHz

    # populations (m == n): no dephasing, weight 1
    diag = np.trace(P2).real
    # coherences (m != n): Lorentzian of width (k + Lambda)
    wid = kk + LL
    lor = kk * wid / (wid ** 2 + dw ** 2)
    np.fill_diagonal(lor, 0.0)
    off = float((P2 * lor).sum().real)

    return (diag + off) / model.M


def orientation_average(model: RPModel, B_mT: float, k_MHz: float,
                        Lambda_MHz: float, n_theta: int, n_phi: int) -> float:
    """Average Phi_S over field directions with proper sin(theta) weight."""
    if B_mT == 0.0 and not model.include_dipolar:
        return singlet_yield(model, 0.0, k_MHz, Lambda_MHz, 0.0, 0.0)
    # Gauss-Legendre in cos(theta), uniform in phi
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    thetas = np.arccos(x)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tot = 0.0
    wsum = 0.0
    for th, wt in zip(thetas, wx):
        for ph in phis:
            tot += wt * singlet_yield(model, B_mT, k_MHz, Lambda_MHz, th, ph)
            wsum += wt
    return tot / wsum


def mfe_curve(model: RPModel, B_grid_mT, k_MHz: float, Lambda_MHz: float,
              n_theta: int = 6, n_phi: int = 6) -> tuple[np.ndarray, float]:
    """Return (MFE percent array, Phi_S at zero field)."""
    phi0 = orientation_average(model, 0.0, k_MHz, Lambda_MHz, n_theta, n_phi)
    out = np.empty(len(B_grid_mT))
    for i, B in enumerate(B_grid_mT):
        ps = orientation_average(model, float(B), k_MHz, Lambda_MHz,
                                 n_theta, n_phi)
        out[i] = 100.0 * (ps - phi0) / phi0
    return out, phi0


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
def cry_nuclei(A_P31: float = 200.0) -> list[Nucleus]:
    """The 8-spin set the manuscripts claim: 2e + 31P + 2 1H_FAD
    + 2 1H_Trp-beta + 14N_Trp   (14N carried as I = 1)."""
    return [
        Nucleus("31P_FMN",  "31P", 0.5, A_P31, A_ax=0.1 * A_P31, radical=0),
        Nucleus("1H_FAD_a", "1H",  0.5, 40.0, A_ax=4.0,  radical=0),
        Nucleus("1H_FAD_b", "1H",  0.5,  8.0, A_ax=1.0,  radical=0),
        Nucleus("1H_Trp_a", "1H",  0.5, 20.0, A_ax=2.0,  radical=1),
        Nucleus("1H_Trp_b", "1H",  0.5, 15.0, A_ax=1.5,  radical=1),
        Nucleus("14N_Trp",  "14N", 1.0, 10.0, A_ax=2.0,  radical=1),
    ]


def build_model(n_nuc: int = 6, A_P31: float = 200.0, J: float = 0.0,
                r_nm: float = 1.75, dipolar: bool = True) -> RPModel:
    nuc = cry_nuclei(A_P31)[:n_nuc]
    return RPModel(nuclei=nuc, J=J, r_nm=r_nm, include_dipolar=dipolar)


# ----------------------------------------------------------------------
# Field grid: 50.000 uT is EXACTLY on it
# ----------------------------------------------------------------------
def field_grid(quick: bool = False) -> np.ndarray:
    """mT.  Explicitly contains 0.050 mT (= 50 uT, Earth) and 0.290 mT."""
    anchors = [0.0, 0.010, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200,
               0.250, 0.290, 0.350, 0.500, 0.750, 1.000, 2.000, 5.000]
    if quick:
        return np.array([0.0, 0.050, 0.290, 1.000])
    return np.array(anchors)


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def run(quick: bool = False) -> dict:
    t0 = time.time()
    B = field_grid(quick)
    i50 = int(np.argmin(np.abs(B - 0.050)))
    assert abs(B[i50] - 0.050) < 1e-12, "50 uT must be exactly on the grid"

    k_MHz = 1.0            # CRY recombination ~1e6 s^-1 (tau_RP ~ 1 us)
    nth, nph = (4, 4) if quick else (6, 6)

    results: dict = {
        "_meta": {
            "module": "qbscreen/cry_8spin_mfe.py",
            "method": ("exact 8-spin diagonalization, exponential (Haberkorn) "
                       "recombination model, orientation-averaged"),
            "hilbert_dim": build_model().dim,
            "nuclear_multiplicity": build_model().M,
            "N14_treated_as": "I = 1 (3 levels)",
            "k_MHz": k_MHz,
            "r_nm": 1.75,
            "D_MHz_point_dipole": build_model().D_MHz,
            "orientation_grid": {"n_theta_gauss": nth, "n_phi": nph},
            "B_grid_mT": B.tolist(),
            "earth_field_index": i50,
            "dephasing_note": ("Lambda = 1/T2 is an explicit free parameter. "
                               "It is NEVER set from 1/T2* = sqrt(sum A^2 I(I+1)/3), "
                               "because that static inhomogeneous HFC width is "
                               "already carried coherently in H; using it as an "
                               "irreversible rate double-counts the hyperfine "
                               "coupling."),
        }
    }

    # ---------- (1) T2 sweep: the headline table ----------
    print("=" * 72)
    print("(1) T2e sweep at 50.000 uT  [8-spin, dim=384, orientation-averaged]")
    print("=" * 72)
    T2_list = [None, 1000.0, 100.0, 10.0, 1.54, 1.10, 0.346]
    m = build_model()
    t2_table = []
    for T2 in T2_list:
        L = 0.0 if T2 is None else 1e3 / T2      # ns -> MHz
        curve, phi0 = mfe_curve(m, B, k_MHz, L, nth, nph)
        row = {
            "T2e_ns": T2, "Lambda_MHz": L,
            "MFE_50uT_pct": float(curve[i50]),
            "MFE_peak_pct": float(curve[np.argmax(np.abs(curve))]),
            "B_peak_mT": float(B[np.argmax(np.abs(curve))]),
            "Phi_S_zero_field": float(phi0),
        }
        t2_table.append(row)
        lab = "coherent" if T2 is None else f"{T2:g} ns"
        print(f"  T2e = {lab:>10s} | Lambda = {L:10.3f} MHz | "
              f"MFE(50uT) = {row['MFE_50uT_pct']:+9.5f} % | "
              f"peak {row['MFE_peak_pct']:+8.3f} % @ {row['B_peak_mT']:.3f} mT")
    results["T2e_sweep_at_earth_field"] = t2_table

    # ---------- (2) full MFE curves for 3 representative T2 ----------
    print()
    print("(2) Full field curves")
    curves = {}
    for T2 in (None, 100.0, 1.10):
        L = 0.0 if T2 is None else 1e3 / T2
        curve, phi0 = mfe_curve(m, B, k_MHz, L, nth, nph)
        key = "coherent" if T2 is None else f"T2e_{T2:g}ns"
        curves[key] = {"B_mT": B.tolist(), "MFE_pct": curve.tolist(),
                       "Phi_S_zero": float(phi0)}
        print(f"  {key:16s} MFE(50uT) = {curve[i50]:+9.5f} %")
    results["mfe_curves"] = curves

    # ---------- (3) nuclear truncation convergence ----------
    print()
    print("(3) Nuclear truncation convergence (coherent limit, 50.000 uT)")
    conv = []
    for n in ([2, 4, 6] if quick else [1, 2, 3, 4, 5, 6]):
        mm = build_model(n_nuc=n)
        c, _ = mfe_curve(mm, np.array([0.0, 0.050]), k_MHz, 0.0, nth, nph)
        conv.append({"n_nuclei": n, "n_spins": 2 + n, "dim": mm.dim,
                     "MFE_50uT_pct": float(c[1])})
        print(f"  n_nuc={n} ({2+n} spins, dim={mm.dim:4d}) : "
              f"MFE(50uT) = {c[1]:+9.5f} %")
    results["truncation_convergence"] = conv

    # ---------- (4) A(31P) sensitivity: 200 MHz (claimed) vs 14 MHz (code) ----------
    print()
    print("(4) A(31P) sensitivity  [coherent limit, 50.000 uT]")
    a_scan = []
    for A in ([14.0, 200.0] if quick else [2.0, 14.0, 50.0, 100.0, 200.0]):
        mm = build_model(A_P31=A)
        c, _ = mfe_curve(mm, np.array([0.0, 0.050]), k_MHz, 0.0, nth, nph)
        a_scan.append({"A_31P_MHz": A, "MFE_50uT_pct": float(c[1])})
        print(f"  A(31P) = {A:6.1f} MHz : MFE(50uT) = {c[1]:+9.5f} %")
    results["A31P_sensitivity"] = a_scan

    # ---------- (5) exchange + dipolar ----------
    print()
    print("(5) Exchange J and dipolar D  [coherent limit, 50.000 uT]")
    jd = []
    for J in ([0.0, 1.0] if quick else [0.0, 0.1, 1.0, 5.0, 10.0]):
        mm = build_model(J=J)
        c, _ = mfe_curve(mm, np.array([0.0, 0.050]), k_MHz, 0.0, nth, nph)
        jd.append({"J_MHz": J, "dipolar": True, "MFE_50uT_pct": float(c[1])})
        print(f"  J = {J:5.2f} MHz (D = {mm.D_MHz:.2f} MHz on) : "
              f"MFE(50uT) = {c[1]:+9.5f} %")
    mm = build_model(dipolar=False)
    c, _ = mfe_curve(mm, np.array([0.0, 0.050]), k_MHz, 0.0, nth, nph)
    jd.append({"J_MHz": 0.0, "dipolar": False, "MFE_50uT_pct": float(c[1])})
    print(f"  J = 0, dipolar OFF                : MFE(50uT) = {c[1]:+9.5f} %")
    results["J_and_dipolar"] = jd

    results["_meta"]["runtime_s"] = round(time.time() - t0, 1)
    print()
    print(f"Total runtime: {results['_meta']['runtime_s']} s")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse grids")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    res = run(quick=args.quick)

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "results_cry_8spin"
        / "cry_8spin_mfe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
