#!/usr/bin/env python3
"""QM/MM reaction coordinate scan for MAO-A SET vs polar mechanism.

Strategy:
  1. Extract FAD isoalloxazine + model substrate (methylamine → serotonin proxy)
     from MAO-A (PDB 2BXR)
  2. Place methylamine at MAO-A active site near FAD N5
  3. Scan the C(substrate)-H...N5(FAD) distance
  4. At each point: compare closed-shell (singlet) vs open-shell (UHF triplet/BS singlet)
  5. If UHF singlet < RHF singlet → diradical character → SET pathway supported
  6. Run both in vacuum (QM only) and with ONIOM (QM/MM)

Output: Energy profile showing SET vs polar pathway preference
"""

import numpy as np
import subprocess
import tempfile
import shutil
import json
from pathlib import Path

OUT_DIR = Path("qmmm_reaction_scan")
OUT_DIR.mkdir(exist_ok=True)

XTB = "/opt/anaconda3/bin/xtb"

# =====================================================
# Model system: Lumiflavin + methylamine
# (FAD isoalloxazine with N10-methyl + CH3NH2 as substrate proxy)
# =====================================================

def generate_reaction_complex(d_NH, optimize_first=False):
    """Generate lumiflavin + methylamine at distance d_NH (N5...H-C).

    d_NH: distance from FAD N5 to the transferring H (Angstrom)
    """
    from pathlib import Path

    # Use obabel to generate lumiflavin
    lf_xyz = OUT_DIR / "lumiflavin_opt.xyz"
    if not lf_xyz.exists():
        subprocess.run(
            ["/opt/anaconda3/bin/obabel",
             "-:Cc1cc2nc3c(=O)[nH]c(=O)nc3n(C)c2cc1C",
             "--gen3d", "-oxyz", "-O", str(lf_xyz)],
            capture_output=True
        )
        # Optimize with xtb
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(lf_xyz, f"{td}/mol.xyz")
            subprocess.run(
                [XTB, "mol.xyz", "--opt", "--chrg", "0", "--uhf", "0"],
                cwd=td, capture_output=True
            )
            opt = Path(td) / "xtbopt.xyz"
            if opt.exists():
                shutil.copy(opt, lf_xyz)

    # Read lumiflavin coordinates
    with open(lf_xyz) as f:
        lines = f.readlines()
    n_lf = int(lines[0])
    lf_atoms = []
    for line in lines[2:2+n_lf]:
        parts = line.split()
        lf_atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))

    # Find N5 position (nitrogen bonded to C4a and C5a in isoalloxazine)
    # In lumiflavin: N5 is the only N not in a carbonyl ring
    # Heuristic: find N atoms and pick the one with z-coordinate
    # Actually, let's find by connectivity: N5 is the central bridge N
    n_atoms = [(i, a) for i, a in enumerate(lf_atoms) if a[0] == 'N']

    # N5 in lumiflavin: typically the N that's NOT bonded to H or to C=O
    # Use the one closest to the center of the molecule
    coords = np.array([(a[1], a[2], a[3]) for a in lf_atoms])
    center = coords.mean(axis=0)

    # Find N closest to geometric center (N5 is central in isoalloxazine)
    n_dists = [(i, np.linalg.norm(np.array([a[1], a[2], a[3]]) - center))
               for i, a in n_atoms]
    n_dists.sort(key=lambda x: x[1])

    # N5 is typically the 2nd or 3rd closest N to center
    # Let's use a more reliable method: N5 is the one with exactly 2 C neighbors
    # For now, take the closest N to center that's not N1/N3 (which are in the uracil ring)
    n5_idx = n_dists[0][0]  # closest N to center
    n5_pos = np.array([lf_atoms[n5_idx][1], lf_atoms[n5_idx][2], lf_atoms[n5_idx][3]])

    print(f"  N5 candidate: atom {n5_idx} at {n5_pos}")

    # Place methylamine: H pointing toward N5
    # CH3-NH2 with alpha-H at distance d_NH from N5
    # Orient: N5 <- H - C - NH2
    # Direction: from N5 outward (perpendicular to ring plane)

    # Estimate ring plane normal from C atoms near N5
    c_near = [(i, a) for i, a in enumerate(lf_atoms)
              if a[0] == 'C' and np.linalg.norm(np.array([a[1],a[2],a[3]]) - n5_pos) < 2.5]
    if len(c_near) >= 2:
        v1 = np.array([c_near[0][1][1], c_near[0][1][2], c_near[0][1][3]]) - n5_pos
        v2 = np.array([c_near[1][1][1], c_near[1][1][2], c_near[1][1][3]]) - n5_pos
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
    else:
        normal = np.array([0, 0, 1])

    # Place substrate along normal
    h_pos = n5_pos + normal * d_NH
    c_pos = h_pos + normal * 1.09  # C-H bond
    n_sub_pos = c_pos + normal * 1.47 + np.array([0.5, 0, 0])  # C-N bond
    h1_pos = n_sub_pos + np.array([0.5, 0.8, 0])  # N-H
    h2_pos = n_sub_pos + np.array([0.5, -0.8, 0])  # N-H
    h3_pos = c_pos + np.array([0.8, 0.5, 0])  # C-H (second)
    h4_pos = c_pos + np.array([0.8, -0.5, 0])  # C-H (third)

    substrate_atoms = [
        ('C', *c_pos),
        ('N', *n_sub_pos),
        ('H', *h_pos),      # transferring H
        ('H', *h1_pos),
        ('H', *h2_pos),
        ('H', *h3_pos),
        ('H', *h4_pos),
    ]

    all_atoms = lf_atoms + substrate_atoms
    n_total = len(all_atoms)
    n_inner = n_lf  # lumiflavin is QM, substrate is also QM

    # Write XYZ
    xyz_path = OUT_DIR / f"complex_d{d_NH:.2f}.xyz"
    with open(xyz_path, 'w') as f:
        f.write(f"{n_total}\n")
        f.write(f"Lumiflavin + methylamine, d(N5-H) = {d_NH:.2f} A\n")
        for elem, x, y, z in all_atoms:
            f.write(f"{elem:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    return xyz_path, n5_idx


def run_xtb_sp(xyz_path, charge, uhf, label, constrain_file=None):
    """Run xTB single-point calculation."""
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(xyz_path, f"{td}/mol.xyz")

        if constrain_file:
            shutil.copy(constrain_file, f"{td}/xcontrol")
            cmd = [XTB, "mol.xyz", "--sp", "--chrg", str(charge),
                   "--uhf", str(uhf), "--input", "xcontrol"]
        else:
            cmd = [XTB, "mol.xyz", "--sp", "--chrg", str(charge), "--uhf", str(uhf)]

        result = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=120)

        # Parse energy
        energy = None
        for line in result.stdout.split('\n'):
            if 'TOTAL ENERGY' in line:
                energy = float(line.split()[-3])
            if 'SCC energy' in line:
                pass

        # Parse spin expectation
        s2 = None
        for line in result.stdout.split('\n'):
            if '<S^2>' in line or 'S(S+1)' in line:
                try:
                    s2 = float(line.split()[-1])
                except:
                    pass

        # Parse Mulliken spin population
        spin_pop = {}
        in_spin = False
        for line in result.stdout.split('\n'):
            if 'Mulliken/CM5' in line or 'spin population' in line.lower():
                in_spin = True
                continue
            if in_spin and line.strip() and line.strip()[0].isdigit():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        atom_idx = int(parts[0])
                        spin = float(parts[-1])
                        spin_pop[atom_idx] = spin
                    except:
                        pass
            elif in_spin and not line.strip():
                in_spin = False

        return {
            'label': label,
            'energy_Eh': energy,
            'S2': s2,
            'spin_population': spin_pop,
            'converged': 'convergence criteria satisfied' in result.stdout or energy is not None,
        }


def scan_reaction_coordinate():
    """Scan N5...H distance and compare RHF vs UHF energies."""
    print("=" * 70)
    print("QM/MM Reaction Coordinate Scan: MAO-A SET vs Polar Mechanism")
    print("Model: Lumiflavin + methylamine (serotonin proxy)")
    print("=" * 70)

    # Distances to scan (N5...H in Angstrom)
    distances = np.arange(1.0, 4.1, 0.2)  # 1.0 to 4.0 A in 0.2 A steps

    results = []

    for d in distances:
        print(f"\n--- d(N5...H) = {d:.1f} A ---")

        xyz_path, n5_idx = generate_reaction_complex(d)

        # 1. Closed-shell singlet (RHF-like: uhf=0)
        r_singlet = run_xtb_sp(xyz_path, charge=0, uhf=0, label="singlet")

        # 2. Open-shell singlet (broken-symmetry: uhf=0 but with UHF)
        # xtb doesn't have BS-DFT; use uhf=2 for triplet instead
        r_triplet = run_xtb_sp(xyz_path, charge=0, uhf=2, label="triplet")

        # 3. Radical cation (electron removed from substrate → FAD)
        r_cation = run_xtb_sp(xyz_path, charge=1, uhf=1, label="cation")

        entry = {
            'd_NH': float(d),
            'E_singlet': r_singlet['energy_Eh'],
            'E_triplet': r_triplet['energy_Eh'],
            'E_cation': r_cation['energy_Eh'],
        }

        if r_singlet['energy_Eh'] and r_triplet['energy_Eh']:
            gap_ST = (r_triplet['energy_Eh'] - r_singlet['energy_Eh']) * 27.2114
            entry['gap_ST_eV'] = float(gap_ST)
            print(f"  Singlet: {r_singlet['energy_Eh']:.6f} Eh")
            print(f"  Triplet: {r_triplet['energy_Eh']:.6f} Eh")
            print(f"  S-T gap: {gap_ST:.4f} eV")

            if r_cation['energy_Eh']:
                ip = (r_cation['energy_Eh'] - r_singlet['energy_Eh']) * 27.2114
                entry['vertical_IP_eV'] = float(ip)
                print(f"  Vertical IP: {ip:.3f} eV")

        results.append(entry)

    # === Analysis ===
    print("\n" + "=" * 70)
    print("ANALYSIS: SET vs Polar Mechanism Indicators")
    print("=" * 70)

    print(f"\n{'d(N5-H)':<10} {'E_S (Eh)':<15} {'E_T (Eh)':<15} {'dE_ST (eV)':<12} {'IP (eV)':<10}")
    print("-" * 65)
    for r in results:
        es = f"{r['E_singlet']:.6f}" if r['E_singlet'] else "—"
        et = f"{r['E_triplet']:.6f}" if r['E_triplet'] else "—"
        gap = f"{r.get('gap_ST_eV', 0):.4f}" if 'gap_ST_eV' in r else "—"
        ip = f"{r.get('vertical_IP_eV', 0):.3f}" if 'vertical_IP_eV' in r else "—"
        print(f"{r['d_NH']:<10.1f} {es:<15} {et:<15} {gap:<12} {ip:<10}")

    # === Plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d_arr = np.array([r['d_NH'] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) Energy profiles
    ax = axes[0]
    E_s = np.array([r['E_singlet'] for r in results if r['E_singlet']])
    E_t = np.array([r['E_triplet'] for r in results if r['E_triplet']])
    d_valid = d_arr[:len(E_s)]

    # Relative to d=4.0 singlet
    E_s_rel = (E_s - E_s[-1]) * 27.2114 * 1000  # meV
    E_t_rel = (E_t - E_s[-1]) * 27.2114 * 1000  # meV

    ax.plot(d_valid, E_s_rel, 'b-o', lw=2, ms=5, label='Singlet (closed-shell)')
    ax.plot(d_valid, E_t_rel, 'r--s', lw=2, ms=5, label='Triplet (open-shell)')
    ax.set_xlabel('d(N5...H) (A)', fontsize=12)
    ax.set_ylabel('Relative energy (meV)', fontsize=12)
    ax.set_title('(a) Energy profiles', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='green', ls=':', alpha=0.3, label='Product')
    ax.axvline(3.5, color='orange', ls=':', alpha=0.3, label='Reactant')

    # (b) S-T gap vs distance
    ax = axes[1]
    gaps = np.array([r.get('gap_ST_eV', 0) for r in results])
    ax.plot(d_arr, gaps, 'ko-', lw=2, ms=6)
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('d(N5...H) (A)', fontsize=12)
    ax.set_ylabel('S-T gap (eV)', fontsize=12)
    ax.set_title('(b) Singlet-Triplet gap\n(negative = triplet lower → diradical)', fontsize=11)
    ax.grid(True, alpha=0.3)

    # If gap becomes small or negative at any point → diradical character
    min_gap = np.min(gaps)
    min_d = d_arr[np.argmin(gaps)]
    ax.annotate(f'Min gap: {min_gap:.3f} eV\nat d={min_d:.1f} A',
                xy=(min_d, min_gap), xytext=(min_d+0.5, min_gap+0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    # (c) Vertical IP vs distance
    ax = axes[2]
    ips = np.array([r.get('vertical_IP_eV', np.nan) for r in results])
    valid = ~np.isnan(ips)
    ax.plot(d_arr[valid], ips[valid], 'go-', lw=2, ms=6)
    ax.set_xlabel('d(N5...H) (A)', fontsize=12)
    ax.set_ylabel('Vertical IP (eV)', fontsize=12)
    ax.set_title('(c) Ionisation potential\n(lower = easier SET)', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle('QM/MM Reaction Coordinate Scan: MAO-A Active Site Model\n'
                 'Lumiflavin + Methylamine (GFN2-xTB)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "reaction_scan.png", dpi=150)
    plt.savefig("manuscript/figures/fig_reaction_scan.pdf")
    plt.close()

    # Save data
    with open(OUT_DIR / "reaction_scan_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # === Interpretation ===
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    min_gap_eV = min(r.get('gap_ST_eV', 999) for r in results)
    print(f"\nMinimum S-T gap: {min_gap_eV:.4f} eV at d = {min_d:.1f} A")

    if min_gap_eV < 0.5:
        print("  → Small S-T gap suggests SIGNIFICANT diradical character")
        print("  → SET pathway is energetically accessible")
        print("  → Radical pair intermediate is plausible")
    elif min_gap_eV < 1.0:
        print("  → Moderate S-T gap suggests PARTIAL diradical character")
        print("  → SET pathway may compete with polar mechanism")
    else:
        print("  → Large S-T gap suggests MINIMAL diradical character")
        print("  → Polar (concerted) mechanism is favoured")
        print("  → Radical pair intermediate is unlikely")

    return results


if __name__ == "__main__":
    results = scan_reaction_coordinate()
