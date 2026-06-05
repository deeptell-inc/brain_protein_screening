"""Tests for the corrected, decoherence-included RPM physics.

These replace the earlier assertion of an inflated coherent-limit MFE (-4.4%),
which peer review correctly identified as an artifact of omitting electronic
decoherence and using an unphysical (CRY-like) lifetime for MAO.
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestMasterEquationConsistency:
    """The master equation must reproduce the coherent limit exactly."""

    def test_reproduces_coherent_greens_function(self):
        from qbscreen.spin_dynamics import spin_op, singlet_projector, SX, SY, SZ, G_E, MU_B, HBAR
        from qbscreen.master_equation import singlet_yield_master, singlet_yield_exponential

        n = 3
        dim = 2 ** n
        P_S = singlet_projector(0, 1, n)
        P_T = np.eye(dim) - P_S
        rho0 = P_S / np.trace(P_S)
        A_P = 200.0
        B = 0.9e-3

        H_mhz = np.zeros((dim, dim), dtype=complex)
        omega_e = G_E * MU_B * B / (HBAR * 2 * np.pi * 1e6)
        H_mhz += omega_e * (spin_op(SZ, 0, n) + spin_op(SZ, 1, n))
        for op in (SX, SY, SZ):
            H_mhz += A_P * spin_op(op, 0, n) @ spin_op(op, 2, n)

        k = 1.0
        phi_master = singlet_yield_master(
            2 * np.pi * H_mhz, P_S, P_T, 2 * np.pi * k, 2 * np.pi * k, rho0, deph_ops=[])
        phi_exp = singlet_yield_exponential(H_mhz, P_S, rho0, k_mhz=k, T2e_ns=None)
        assert abs(phi_master - phi_exp) < 1e-9, "master eqn must equal coherent Green's fn"


class TestDecoherenceSuppression:
    """Electronic T2e must suppress the weak-field MFE (the central correction)."""

    def test_t2e_suppresses_mfe(self):
        from qbscreen.honest_mfe import mfe_master
        # Separated geometry, J~0: coherent MFE is large; T2e=1ns must shrink it.
        mfeE_coh, _ = mfe_master(200, 14.9, 44.0, 0.5, k_MHz=1.0, T2e_ns=None)
        mfeE_real, _ = mfe_master(200, 14.9, 44.0, 0.5, k_MHz=1.0, T2e_ns=1.0)
        assert abs(mfeE_real) < abs(mfeE_coh), "T2e must reduce the MFE"
        # 1 ns coherence essentially kills the weak-field effect
        assert abs(mfeE_real) < 0.05, f"MFE with T2e=1ns should be tiny, got {mfeE_real:.4f}%"


class TestNeurotransmitterEnzymesNullResult:
    """The honest negative result: MAO-family weak-field MFE is negligible."""

    @pytest.mark.parametrize("J,k", [(750.0, 1e5), (400.0, 1e5)])
    def test_active_site_rp_negligible(self, J, k):
        from qbscreen.honest_mfe import mfe_master
        # Active-site RP: large J + picosecond lifetime → MFE ≈ 0, with OR without T2e
        mfeE_coh, _ = mfe_master(200, 14.9, 44.0, J, k_MHz=k, T2e_ns=None)
        mfeE_real, _ = mfe_master(200, 14.9, 44.0, J, k_MHz=k, T2e_ns=1.0)
        assert abs(mfeE_coh) < 0.01, f"active-site coherent MFE should be <0.01%, got {mfeE_coh:.4f}%"
        assert abs(mfeE_real) < 0.01, f"active-site realistic MFE should be <0.01%, got {mfeE_real:.4f}%"

    def test_lifetime_threshold(self):
        """Weak-field sensitivity requires lifetime >> ps (MAO regime gives nothing)."""
        from qbscreen.honest_mfe import mfe_master
        # MAO lifetime ~10 ps (k=1e5 MHz)
        mfe_ps, _ = mfe_master(200, 14.9, 44.0, 0.5, k_MHz=1e5, T2e_ns=None)
        # CRY-like lifetime ~1 µs (k=1 MHz)
        mfe_us, _ = mfe_master(200, 14.9, 44.0, 0.5, k_MHz=1.0, T2e_ns=None)
        assert abs(mfe_ps) < 1e-3, "ps lifetime → no weak-field MFE"
        assert abs(mfe_us) > 10 * abs(mfe_ps), "µs lifetime must give much larger MFE"


class TestCoherentLimitWasArtifact:
    """Document that the old percent-level number was the coherent-limit artifact."""

    def test_coherent_limit_large_then_collapses(self):
        from qbscreen.honest_mfe import mfe_master
        # Coherent, long-lived, J~0: large MFE (the old regime)
        mfe_coh, _ = mfe_master(200, 14.9, 44.0, 0.5, k_MHz=1.0, T2e_ns=None)
        # Same but with active-site T2e=1ns: collapses
        mfe_dec, _ = mfe_master(200, 14.9, 44.0, 0.5, k_MHz=1.0, T2e_ns=1.0)
        assert abs(mfe_coh) > 0.5, "coherent long-lived J~0 should show a sizeable MFE"
        suppression = abs(mfe_coh) / max(abs(mfe_dec), 1e-9)
        assert suppression > 1.3, "decoherence must measurably suppress the coherent MFE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
