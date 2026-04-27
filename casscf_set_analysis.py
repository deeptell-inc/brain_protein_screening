#!/usr/bin/env python3
"""CASSCF/NEVPT2 analysis of SET vs polar mechanism in MAO-A model.

Multi-reference calculation to detect diradical character at the
H-transfer transition state geometry.

Model: Lumiflavin + methylamine (simplified MAO-A active site)
Method: CASSCF(2,2)/def2-SVP → NEVPT2 correction
Active space: HOMO, LUMO (the frontier orbitals involved in SET)

For larger active spaces, CASSCF(4,4) or (6,6) would be needed
but are computationally demanding for 30+ atom systems.
"""

import numpy as np
import json
import time
from pathlib import Path

OUT_DIR = Path("casscf_results")
OUT_DIR.mkdir(exist_ok=True)


def run_casscf_scan():
    from pyscf import gto, scf, mcscf, lib
    try:
        from pyscf import nevpt2 as nevpt2_mod
        HAS_NEVPT2 = True
    except ImportError:
        from pyscf.mrpt import nevpt2 as nevpt2_mod
        HAS_NEVPT2 = True

    print("=" * 70)
    print("CASSCF/NEVPT2 SET Analysis: Lumiflavin + Methylamine")
    print("Active space: CAS(2,2) — HOMO/LUMO frontier orbitals")
    print("Basis: def2-SVP")
    print("=" * 70)

    # Load optimised geometries from xTB scan
    scan_json = Path("qmmm_reaction_scan/reaction_scan_results.json")
    if scan_json.exists():
        with open(scan_json) as f:
            xtb_results = json.load(f)
    else:
        xtb_results = None

    # Key distances to probe (subset for computational cost)
    distances = [1.0, 1.4, 1.8, 2.5, 3.5]

    results = []

    for d in distances:
        xyz_file = f"qmmm_reaction_scan/complex_d{d:.2f}.xyz"
        if not Path(xyz_file).exists():
            print(f"\n  Skipping d={d:.1f} A (XYZ not found)")
            continue

        print(f"\n{'='*60}")
        print(f"  d(N5...H) = {d:.1f} A")
        print(f"{'='*60}")

        # Read XYZ
        with open(xyz_file) as f:
            lines = f.readlines()
        natom = int(lines[0])
        atom_str = "; ".join(lines[2:2+natom])

        # Build molecule
        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = "def2-svp"
        mol.charge = 0
        mol.spin = 0
        mol.verbose = 3
        mol.max_memory = 32000
        mol.build()

        print(f"  Atoms: {mol.natm}, Basis: {mol.nao}")

        t0 = time.time()

        # Step 1: RHF
        mf = scf.RHF(mol)
        mf.max_cycle = 200
        mf.conv_tol = 1e-8
        e_rhf = mf.kernel()
        print(f"  RHF energy: {e_rhf:.6f} Eh")

        # Step 2: CASSCF(2,2) — 2 electrons in 2 orbitals (HOMO, LUMO)
        ncas = 2  # 2 active orbitals
        nelecas = 2  # 2 active electrons
        mc = mcscf.CASSCF(mf, ncas, nelecas)
        mc.max_cycle_macro = 100
        mc.conv_tol = 1e-7

        try:
            e_casscf = mc.kernel()[0]
            print(f"  CASSCF(2,2) energy: {e_casscf:.6f} Eh")
            converged_casscf = mc.converged
        except Exception as ex:
            print(f"  CASSCF failed: {ex}")
            e_casscf = None
            converged_casscf = False

        # Step 3: CI coefficients → diradical character
        diradical_char = 0.0
        ci_coeffs = {}
        if e_casscf is not None and mc.converged:
            # CI vector: |20⟩ (closed-shell) and |02⟩ (doubly excited)
            # and |αβ⟩, |βα⟩ (open-shell singlet)
            ci = mc.ci
            if ci is not None and hasattr(ci, 'shape'):
                # For CAS(2,2): ci is a 2x2 matrix
                # ci[0,0] = coefficient of |20⟩ (both in HOMO)
                # ci[1,1] = coefficient of |02⟩ (both in LUMO)
                # ci[0,1], ci[1,0] = |αβ⟩, |βα⟩
                if ci.ndim == 2 and ci.shape == (2, 2):
                    c_20 = ci[0, 0]
                    c_02 = ci[1, 1]
                    c_ab = ci[0, 1]
                    c_ba = ci[1, 0]
                    diradical_char = 2 * c_02**2  # y₀ = 2|c₀₂|²
                    ci_coeffs = {
                        'c_20': float(c_20),
                        'c_02': float(c_02),
                        'c_ab': float(c_ab),
                        'c_ba': float(c_ba),
                        'diradical_y0': float(diradical_char),
                    }
                    print(f"  CI: |20⟩={c_20:.4f}, |02⟩={c_02:.4f}, "
                          f"|αβ⟩={c_ab:.4f}, |βα⟩={c_ba:.4f}")
                    print(f"  Diradical character y₀ = 2|c₀₂|² = {diradical_char:.4f}")
                else:
                    print(f"  CI shape: {ci.shape} (unexpected)")
                    # Try to extract occupation numbers
                    occ = mc.make_rdm1()
                    if occ is not None:
                        eigvals = np.linalg.eigvalsh(occ)
                        print(f"  Natural orbital occupations: {eigvals}")
                        # Diradical from NO occupations: y = 1 - |n_HOMO - 1|
                        if len(eigvals) >= 2:
                            n_homo = max(eigvals)
                            n_lumo = min(eigvals)
                            diradical_char = 1.0 - abs(n_homo - 1.0)
                            ci_coeffs['n_HOMO'] = float(n_homo)
                            ci_coeffs['n_LUMO'] = float(n_lumo)
                            ci_coeffs['diradical_y0'] = float(diradical_char)
                            print(f"  n(HOMO)={n_homo:.4f}, n(LUMO)={n_lumo:.4f}")
                            print(f"  Diradical y₀ = {diradical_char:.4f}")

        # Step 4: NEVPT2 correction
        e_nevpt2 = None
        if e_casscf is not None and mc.converged:
            try:
                e_nevpt2 = nevpt2_mod.NEVPT(mc).kernel()
                e_total = e_casscf + e_nevpt2
                print(f"  NEVPT2 correction: {e_nevpt2:.6f} Eh")
                print(f"  CASSCF+NEVPT2 total: {e_total:.6f} Eh")
            except Exception as ex:
                print(f"  NEVPT2 failed: {ex}")
                e_total = e_casscf

        # Step 5: Triplet CASSCF for S-T gap
        e_triplet = None
        mol_t = mol.copy()
        mol_t.spin = 2
        mol_t.build()
        mf_t = scf.UHF(mol_t)
        mf_t.max_cycle = 200
        try:
            e_uhf_t = mf_t.kernel()
            mc_t = mcscf.CASSCF(mf_t, ncas, nelecas)
            mc_t.max_cycle_macro = 100
            e_triplet = mc_t.kernel()[0]
            print(f"  Triplet CASSCF: {e_triplet:.6f} Eh")
            if e_casscf is not None:
                gap = (e_triplet - e_casscf) * 27.2114
                print(f"  S-T gap (CASSCF): {gap:.4f} eV")
        except Exception as ex:
            print(f"  Triplet failed: {ex}")

        elapsed = time.time() - t0
        print(f"  Wall time: {elapsed:.0f} s")

        entry = {
            'd_NH': float(d),
            'E_RHF': float(e_rhf) if e_rhf else None,
            'E_CASSCF': float(e_casscf) if e_casscf else None,
            'E_NEVPT2_corr': float(e_nevpt2) if e_nevpt2 else None,
            'E_triplet_CASSCF': float(e_triplet) if e_triplet else None,
            'diradical_character': float(diradical_char),
            'ci_coefficients': ci_coeffs,
            'converged': converged_casscf,
            'wall_time_s': float(elapsed),
        }

        if e_casscf and e_triplet:
            entry['ST_gap_CASSCF_eV'] = float((e_triplet - e_casscf) * 27.2114)

        results.append(entry)

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY: Diradical Character along Reaction Coordinate")
    print("=" * 70)
    print(f"{'d(N5-H)':<10} {'y₀ (dirad)':<12} {'S-T gap(eV)':<12} "
          f"{'E_CAS(Eh)':<15} {'Interpretation':<30}")
    print("-" * 80)

    for r in results:
        y = r['diradical_character']
        gap = r.get('ST_gap_CASSCF_eV', 0)
        e_cas = r['E_CASSCF']
        e_str = f"{e_cas:.6f}" if e_cas else "—"
        gap_str = f"{gap:.4f}" if gap else "—"

        if y > 0.3:
            interp = "STRONG diradical → SET"
        elif y > 0.1:
            interp = "MODERATE diradical → SET/polar"
        elif y > 0.02:
            interp = "WEAK diradical → mostly polar"
        else:
            interp = "CLOSED-SHELL → polar mechanism"

        print(f"{r['d_NH']:<10.1f} {y:<12.4f} {gap_str:<12} {e_str:<15} {interp:<30}")

    # === Plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    d_arr = [r['d_NH'] for r in results]
    y_arr = [r['diradical_character'] for r in results]
    gap_arr = [r.get('ST_gap_CASSCF_eV', 0) for r in results]

    ax = axes[0]
    ax.plot(d_arr, y_arr, 'ro-', lw=2, ms=8)
    ax.axhline(0.1, color='orange', ls='--', alpha=0.5, label='y=0.1 (moderate)')
    ax.axhline(0.3, color='red', ls='--', alpha=0.5, label='y=0.3 (strong)')
    ax.set_xlabel('d(N5...H) (A)', fontsize=12)
    ax.set_ylabel('Diradical character y0', fontsize=12)
    ax.set_title('CASSCF(2,2) Diradical Character\nMAO-A model (lumiflavin + methylamine)',
                 fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, max(y_arr) * 1.3 + 0.05)

    ax = axes[1]
    ax.plot(d_arr, gap_arr, 'bs-', lw=2, ms=8)
    ax.set_xlabel('d(N5...H) (A)', fontsize=12)
    ax.set_ylabel('S-T gap (eV)', fontsize=12)
    ax.set_title('CASSCF Singlet-Triplet Gap\n(smaller = more diradical)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "casscf_diradical.png", dpi=150)
    plt.savefig("manuscript/figures/fig_casscf_diradical.pdf")
    plt.close()

    # Save
    with open(OUT_DIR / "casscf_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/")
    return results


if __name__ == "__main__":
    run_casscf_scan()
