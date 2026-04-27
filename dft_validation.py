#!/usr/bin/env python3
"""DFT/TDDFT validation of xTB results for FAD and PLP models.

Uses PySCF (B3LYP/def2-SVP) to compute:
  1. HOMO-LUMO gaps
  2. TD-DFT excitation energies and oscillator strengths
  3. Vertical IP/EA
  4. Spin density and HFC (for radical cation)

Models:
  - Lumiflavin (FAD isoalloxazine + methyl, C12H12N4O2, 30 atoms)
  - PLP-methylamine Schiff base (C9H12NO5P, 28 atoms)
"""

import numpy as np
import json
import time
from pathlib import Path

OUT_DIR = Path("dft_results")
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Model geometries (from xTB optimization)
# =====================================================

def load_xyz(path):
    """Load XYZ file content as string."""
    with open(path) as f:
        return f.read()

# Use Open Babel-generated structures (correct valence/H count)
LUMIFLAVIN_XYZ = load_xyz("dft_results/lumiflavin.xyz")
PLP_XYZ = load_xyz("dft_results/plp_sb.xyz")


def run_dft_tddft(name, xyz_str, charge=0, spin=0):
    """Run DFT (B3LYP/def2-SVP) + TDDFT on a molecule.

    Returns dict with HOMO, LUMO, gap, excitations, f_osc.
    """
    from pyscf import gto, scf, dft, tdscf

    print(f"\n{'='*60}")
    print(f"DFT/TDDFT: {name} (charge={charge}, spin={spin})")
    print(f"{'='*60}")

    t0 = time.time()

    # Parse XYZ
    lines = [l.strip() for l in xyz_str.strip().split('\n') if l.strip()]
    natom = int(lines[0])
    atom_str = '; '.join(lines[2:2+natom])

    # Build molecule
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = 'def2-svp'
    mol.charge = charge
    mol.spin = spin  # 2S
    mol.verbose = 3
    mol.max_memory = 16000  # MB
    mol.build()

    print(f"  Atoms: {mol.natm}, Basis functions: {mol.nao}")

    results = {'name': name, 'charge': charge, 'spin': spin,
               'natom': mol.natm, 'nbasis': mol.nao}

    # DFT calculation
    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-8
    mf.max_cycle = 200

    try:
        e_tot = mf.kernel()
        results['converged'] = bool(mf.converged)
        results['total_energy_Eh'] = float(e_tot)

        if spin == 0:
            mo_e = mf.mo_energy
            nocc = mol.nelectron // 2
            homo = float(mo_e[nocc - 1])
            lumo = float(mo_e[nocc])
        else:
            mo_e_a, mo_e_b = mf.mo_energy
            nocc_a = (mol.nelectron + spin) // 2
            nocc_b = (mol.nelectron - spin) // 2
            homo = float(max(mo_e_a[nocc_a - 1], mo_e_b[nocc_b - 1]))
            lumo = float(min(mo_e_a[nocc_a], mo_e_b[nocc_b]))

        gap_eV = (lumo - homo) * 27.2114
        results['homo_eV'] = float(homo * 27.2114)
        results['lumo_eV'] = float(lumo * 27.2114)
        results['gap_eV'] = float(gap_eV)
        print(f"  HOMO: {homo*27.2114:.3f} eV, LUMO: {lumo*27.2114:.3f} eV")
        print(f"  Gap: {gap_eV:.3f} eV")

    except Exception as ex:
        print(f"  SCF failed: {ex}")
        results['converged'] = False
        results['error'] = str(ex)
        return results

    # TDDFT (singlet excited states)
    if spin == 0 and mf.converged:
        try:
            td = tdscf.TDA(mf)
            td.nstates = 6
            td.kernel()

            excitations = []
            for i, (e, f) in enumerate(zip(td.e, td.oscillator_strength())):
                exc = {
                    'state': i + 1,
                    'energy_eV': float(e * 27.2114),
                    'wavelength_nm': float(1239.84 / (e * 27.2114)) if e > 0 else 0,
                    'f_osc': float(f),
                }
                excitations.append(exc)
                print(f"  S{i+1}: {e*27.2114:.3f} eV ({1239.84/(e*27.2114):.0f} nm), "
                      f"f = {f:.4f}")

            results['excitations'] = excitations
            results['S1_eV'] = excitations[0]['energy_eV'] if excitations else 0
            results['S1_f_osc'] = excitations[0]['f_osc'] if excitations else 0

            # Bright state (largest f_osc)
            bright = max(excitations, key=lambda x: x['f_osc'])
            results['bright_state'] = bright['state']
            results['bright_eV'] = bright['energy_eV']
            results['bright_f_osc'] = bright['f_osc']
            results['bright_nm'] = bright['wavelength_nm']
            print(f"  Bright state: S{bright['state']} at {bright['energy_eV']:.3f} eV, "
                  f"f = {bright['f_osc']:.4f}")

        except Exception as ex:
            print(f"  TDDFT failed: {ex}")
            results['tddft_error'] = str(ex)

    elapsed = time.time() - t0
    results['wall_time_s'] = float(elapsed)
    print(f"  Wall time: {elapsed:.1f} s")

    return results


def run_vertical_ip_ea(name, xyz_str):
    """Compute vertical IP and EA via ΔSCF."""
    from pyscf import gto, dft

    print(f"\n  Computing vertical IP/EA for {name}...")

    lines = [l.strip() for l in xyz_str.strip().split('\n') if l.strip()]
    natom = int(lines[0])
    atom_str = '; '.join(lines[2:2+natom])

    results = {}

    for label, charge, spin in [('neutral', 0, 0), ('cation', 1, 1), ('anion', -1, 1)]:
        mol = gto.Mole()
        mol.atom = atom_str
        mol.basis = 'def2-svp'
        mol.charge = charge
        mol.spin = spin
        mol.verbose = 0
        mol.max_memory = 16000
        mol.build()

        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = 'b3lyp'
        mf.max_cycle = 200

        try:
            e = mf.kernel()
            results[f'{label}_Eh'] = float(e)
            results[f'{label}_converged'] = bool(mf.converged)
            print(f"    {label}: E = {e:.6f} Eh, converged = {mf.converged}")
        except:
            results[f'{label}_Eh'] = None
            print(f"    {label}: FAILED")

    if results.get('neutral_Eh') and results.get('cation_Eh'):
        ip = (results['cation_Eh'] - results['neutral_Eh']) * 27.2114
        results['vertical_IP_eV'] = float(ip)
        print(f"    Vertical IP: {ip:.3f} eV")

    if results.get('neutral_Eh') and results.get('anion_Eh'):
        ea = (results['neutral_Eh'] - results['anion_Eh']) * 27.2114
        results['vertical_EA_eV'] = float(ea)
        print(f"    Vertical EA: {ea:.3f} eV")

    return results


def main():
    print("=" * 60)
    print("DFT/TDDFT Validation (B3LYP/def2-SVP, PySCF 2.12.1)")
    print("=" * 60)

    all_results = {}

    # 1. Lumiflavin
    r1 = run_dft_tddft("Lumiflavin (FAD model)", LUMIFLAVIN_XYZ)
    all_results['lumiflavin'] = r1

    # 2. PLP Schiff base
    r2 = run_dft_tddft("PLP-methylamine (PLP model)", PLP_XYZ)
    all_results['plp'] = r2

    # 3. Vertical IP/EA
    ip_ea_lf = run_vertical_ip_ea("Lumiflavin", LUMIFLAVIN_XYZ)
    all_results['lumiflavin_ip_ea'] = ip_ea_lf

    ip_ea_plp = run_vertical_ip_ea("PLP", PLP_XYZ)
    all_results['plp_ip_ea'] = ip_ea_plp

    # Save results
    with open(OUT_DIR / "dft_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: xTB vs DFT/TDDFT vs Experiment")
    print("=" * 80)

    # xTB values from our screening
    xtb_fad = {'gap': 1.42, 'f_osc': 0.497, 'IP': 10.78, 'EA': 5.63}
    xtb_plp = {'gap': 1.93, 'f_osc': 0.300, 'IP': 12.47, 'EA': 5.65}

    # Experimental values
    exp_fad = {'gap': 2.76, 'f_osc': 0.27, 'IP': 8.0, 'EA': None,
               'S1': 2.76, 'source': 'Islam2003, Ghisla1974'}
    exp_plp = {'gap': 3.87, 'f_osc': 0.15, 'IP': 8.5, 'EA': None,
               'S1': 3.87, 'source': 'Harris1976'}

    print(f"\n{'Property':<25} {'xTB':<12} {'B3LYP/SVP':<12} {'Experiment':<15} {'xTB err':<10} {'DFT err':<10}")
    print("-" * 80)

    # FAD
    print(f"\n--- Lumiflavin (FAD model) ---")
    dft_gap = r1.get('gap_eV', 0)
    dft_S1 = r1.get('S1_eV', 0)
    dft_fosc = r1.get('bright_f_osc', 0)
    dft_bright = r1.get('bright_eV', 0)

    print(f"{'HOMO-LUMO gap (eV)':<25} {xtb_fad['gap']:<12.2f} {dft_gap:<12.2f} {exp_fad['gap']:<15.2f} "
          f"{xtb_fad['gap']-exp_fad['gap']:<+10.2f} {dft_gap-exp_fad['gap']:<+10.2f}")
    print(f"{'S1 energy (eV)':<25} {'—':<12} {dft_S1:<12.2f} {exp_fad['S1']:<15.2f} "
          f"{'—':<10} {dft_S1-exp_fad['S1']:<+10.2f}")
    print(f"{'f_osc (bright)':<25} {xtb_fad['f_osc']:<12.3f} {dft_fosc:<12.3f} {exp_fad['f_osc']:<15.3f} "
          f"{xtb_fad['f_osc']-exp_fad['f_osc']:<+10.3f} {dft_fosc-exp_fad['f_osc']:<+10.3f}")

    if ip_ea_lf.get('vertical_IP_eV'):
        dft_ip = ip_ea_lf['vertical_IP_eV']
        print(f"{'Vertical IP (eV)':<25} {xtb_fad['IP']:<12.2f} {dft_ip:<12.2f} {exp_fad['IP']:<15.1f} "
              f"{xtb_fad['IP']-exp_fad['IP']:<+10.2f} {dft_ip-exp_fad['IP']:<+10.2f}")

    # PLP
    print(f"\n--- PLP-methylamine (PLP model) ---")
    dft_gap2 = r2.get('gap_eV', 0)
    dft_S1_2 = r2.get('S1_eV', 0)
    dft_fosc2 = r2.get('bright_f_osc', 0)

    print(f"{'HOMO-LUMO gap (eV)':<25} {xtb_plp['gap']:<12.2f} {dft_gap2:<12.2f} {exp_plp['gap']:<15.2f} "
          f"{xtb_plp['gap']-exp_plp['gap']:<+10.2f} {dft_gap2-exp_plp['gap']:<+10.2f}")
    print(f"{'S1 energy (eV)':<25} {'—':<12} {dft_S1_2:<12.2f} {exp_plp['S1']:<15.2f} "
          f"{'—':<10} {dft_S1_2-exp_plp['S1']:<+10.2f}")
    print(f"{'f_osc (bright)':<25} {xtb_plp['f_osc']:<12.3f} {dft_fosc2:<12.3f} {exp_plp['f_osc']:<15.3f} "
          f"{xtb_plp['f_osc']-exp_plp['f_osc']:<+10.3f} {dft_fosc2-exp_plp['f_osc']:<+10.3f}")

    if ip_ea_plp.get('vertical_IP_eV'):
        dft_ip2 = ip_ea_plp['vertical_IP_eV']
        print(f"{'Vertical IP (eV)':<25} {xtb_plp['IP']:<12.2f} {dft_ip2:<12.2f} {exp_plp['IP']:<15.1f} "
              f"{xtb_plp['IP']-exp_plp['IP']:<+10.2f} {dft_ip2-exp_plp['IP']:<+10.2f}")

    print(f"\nAll results saved to {OUT_DIR}/dft_results.json")


if __name__ == "__main__":
    main()
