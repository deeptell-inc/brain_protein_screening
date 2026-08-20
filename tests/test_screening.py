"""Test screening pipeline reproduces paper results.

Requires: xtb, openbabel, biopython
"""

import pytest
import subprocess
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

XTB = "/opt/anaconda3/bin/xtb"
OBABEL = "/opt/anaconda3/bin/obabel"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def has_xtb():
    try:
        r = subprocess.run([XTB, "--version"], capture_output=True, timeout=5)
        return r.returncode == 0 or b"xtb version" in r.stdout + r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not has_xtb(), reason="xtb not installed")
class TestScreeningPipeline:
    """Test the full screening pipeline on CRY (4I6G)."""

    @pytest.fixture(scope="class")
    def cry_results(self, tmp_path_factory):
        """Run screening on 4I6G and return results."""
        outdir = tmp_path_factory.mktemp("cry_test")
        pdb_file = os.path.join(BASE_DIR, "pdb_files", "pdb4i6g.ent")

        if not os.path.exists(pdb_file):
            pytest.skip("PDB file pdb4i6g.ent not found")

        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "screen_quantum_sites.py"),
            pdb_file,
            "-o", str(outdir),
            "--no-hess",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        json_file = outdir / "screening_results.json"
        if not json_file.exists():
            pytest.fail(f"Screening failed:\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")

        with open(json_file) as f:
            return json.load(f)

    def test_sites_detected(self, cry_results):
        """Should detect at least 2 quantum-active sites (FAD chromophores)."""
        assert len(cry_results) >= 2, f"Found {len(cry_results)} sites, expected >=2"

    def test_fad_tdm(self, cry_results):
        """FAD TDM should be ~9-10 D."""
        fad_site = cry_results[0]
        tdm = fad_site.get("estimated_electronic_tdm_D", 0)
        assert 5 < tdm < 15, f"FAD TDM = {tdm:.1f} D, expected ~9-10 D"

    def test_fad_soc(self, cry_results):
        """FAD SOC should be ~60-70 cm^-1."""
        soc = cry_results[0].get("effective_soc_cm", 0)
        assert 50 < soc < 90, f"SOC = {soc:.1f} cm^-1, expected ~67"

    def test_p_atoms_detected(self, cry_results):
        """Should detect 2 P atoms in FAD."""
        n_p = cry_results[0].get("n_P_atoms", 0)
        assert n_p == 2, f"n_P = {n_p}, expected 2"

    def test_t2_star_spin(self, cry_results):
        """T2*(HFC) should be ~1-2 ns."""
        t2 = cry_results[0].get("T2_star_spin_ns", 0)
        assert 0.3 < t2 < 5, f"T2* = {t2:.2f} ns, expected ~1 ns"

    def test_t1e(self, cry_results):
        """T1e should be ~0.5-200 us."""
        t1e = cry_results[0].get("T1e_spin_lattice_us", 0)
        assert 0.1 < t1e < 500, f"T1e = {t1e:.2f} us"

    def test_calc_success(self, cry_results):
        """All sites should have calc_success=True."""
        for site in cry_results:
            assert site.get("calc_success", False), f"Site {site.get('site_id')} failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
