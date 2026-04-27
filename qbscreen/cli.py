"""Command-line interface for qbscreen."""

import sys


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1]

    if command == "screen":
        from qbscreen.screener import main as screen_main
        # Remove 'screen' from argv so argparse sees the right args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        screen_main()

    elif command == "simulate":
        from qbscreen.spin_dynamics import main as sim_main
        sim_main()

    elif command == "dft-validate":
        from qbscreen.dft_validation import main as dft_main
        dft_main()

    elif command == "casscf":
        from qbscreen.casscf_analysis import run_casscf_scan
        run_casscf_scan()

    elif command == "reaction-scan":
        from qbscreen.reaction_scan import scan_reaction_coordinate
        scan_reaction_coordinate()

    elif command == "version":
        from qbscreen import __version__
        print(f"qbscreen {__version__}")

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


def print_usage():
    """Print usage information."""
    from qbscreen import __version__
    print(f"""qbscreen v{__version__} — Quantum-active site screening in brain proteins

Usage:
  qbscreen screen <PDB_file_or_ID> [options]    Screen quantum-active sites
  qbscreen simulate                              Run RPM spin dynamics
  qbscreen dft-validate                          DFT/TDDFT validation
  qbscreen casscf                                CASSCF/NEVPT2 mechanism analysis
  qbscreen reaction-scan                         QM/MM reaction coordinate scan
  qbscreen version                               Show version

Examples:
  qbscreen screen 4I6G -o results_cry
  qbscreen screen pdb_files/pdb2bxr.ent --embedding oniom -o results_maoa
  qbscreen simulate
  qbscreen dft-validate

For screening options:
  qbscreen screen --help
""")


if __name__ == "__main__":
    main()
