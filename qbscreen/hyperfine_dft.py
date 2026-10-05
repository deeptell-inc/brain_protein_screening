#!/usr/bin/env python3
"""Isotropic (Fermi-contact) hyperfine couplings from a UKS spin density.

    a_iso = (2/3) μ0 g_e μB g_N μN ρ_s(R_N) / h

ρ_s is the α−β density evaluated at the nucleus. Geometries are the GFN2-xTB
optima in calibration_results/lambda; UKS B3LYP single points. The flavin
radical anion is the validation case (Lee et al., J. R. Soc. Interface 11,
20131063 (2014): N5 523 µT, N10 189 µT).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent / "calibration_results" / "lambda"
MU0 = 4e-7 * np.pi
G_E = 2.00231930436
MU_B = 9.2740100783e-24
MU_N = 5.0507837461e-27
H_PLANCK = 6.62607015e-34
A0 = 5.29177210903e-11
G_N = {"H": 5.585694702, "N": 0.40376100, "C": 1.4048236}   # ¹H, ¹⁴N, ¹³C
MHZ_PER_MT = 28.0249514   # g = 2.0023


def a_iso_MHz(rho_bohr3, element):
    return (2 / 3) * MU0 * G_E * MU_B * G_N[element] * MU_N * (rho_bohr3 / A0 ** 3) / H_PLANCK / 1e6


def hfc(xyz_path, charge, spin, basis="def2-tzvp", xc="b3lyp", tensors=False):
    from pyscf import gto, dft
    lines = [l for l in Path(xyz_path).read_text().splitlines()[2:] if l.strip()]
    atoms = [(l.split()[0], tuple(float(v) for v in l.split()[1:4])) for l in lines]
    m = gto.Mole(atom=atoms, basis=basis, charge=charge, spin=spin, verbose=0, max_memory=6000).build()
    mf = dft.UKS(m).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    dm = mf.make_rdm1()
    P_s = dm[0] - dm[1]
    ao = m.eval_gto("GTOval", m.atom_coords())            # (natm, nao), bohr
    rho = np.einsum("ai,ij,aj->a", ao, P_s, ao)
    out = []
    for k, (el, _) in enumerate(atoms):
        if el in G_N and el != "C":
            a = a_iso_MHz(rho[k], el)
            row = dict(index=k, element=el, a_MHz=float(a), a_mT=float(a / MHZ_PER_MT))
            if tensors:
                row["A_dip_MHz"] = dipolar_tensor(m, P_s, k, el, rho[k]).tolist()
            out.append(row)
    return out


def dipolar_tensor(m, P_s, k, element, rho_k):
    """Traceless spin-dipolar hyperfine tensor (MHz) of nucleus k.

    M_ab = ⟨∂_a∂_b (1/|r − R_k|)⟩ over the spin density, by parts on the AO pair;
    its trace is −4π ρ_s(R_k), which is checked, and its traceless part is the
    dipolar kernel (3 r_a r_b − r² δ_ab)/r⁵."""
    with m.with_rinv_origin(m.atom_coord(k)):
        ipip = m.intor("int1e_ipiprinv", comp=9).reshape(3, 3, m.nao, m.nao)
        ipr = m.intor("int1e_iprinvip", comp=9).reshape(3, 3, m.nao, m.nao)
    M = 2 * np.einsum("abij,ij->ab", ipip, P_s) + np.einsum("abij,ij->ab", ipr, P_s) \
        + np.einsum("baij,ij->ab", ipr, P_s)
    M = 0.5 * (M + M.T)
    assert abs(np.trace(M) + 4 * np.pi * rho_k) < 1e-3 * max(1.0, abs(4 * np.pi * rho_k)), (np.trace(M), rho_k)
    pref = MU0 / (4 * np.pi) * G_E * MU_B * G_N[element] * MU_N / A0 ** 3 / H_PLANCK / 1e6
    return pref * (M - np.trace(M) / 3 * np.eye(3))


if __name__ == "__main__":
    # sanity: hydrogen atom 1s, ρ(0) = 1/π → 1422.8 MHz for an infinitely heavy
    # nucleus; the reduced-mass factor (1 + m_e/m_p)^−3 brings it to the measured 1420.4.
    a_H = a_iso_MHz(1 / np.pi, "H")
    assert abs(a_H * (1 + 1 / 1836.15267) ** -3 - 1420.4) < 0.5, a_H
    basis = sys.argv[1] if len(sys.argv) > 1 else "def2-tzvp"
    res = {}
    for tag, q in (("lumiflavin_q-1", -1), ("methylamine_q+1", +1),
                   ("benzylamine_q+1", +1), ("alanine_q+1", +1), ("methylindole_q+1", +1)):
        r = hfc(HERE / f"opt_{tag}" / "xtbopt.xyz", q, 1, basis)
        res[tag] = r
        big = sorted(r, key=lambda d: -abs(d["a_MHz"]))[:6]
        print(tag, " ".join(f"{d['element']}{d['index']}:{d['a_MHz']:+.1f}" for d in big), flush=True)
    (HERE / f"hfc_{basis}.json").write_text(json.dumps(res, indent=2))
