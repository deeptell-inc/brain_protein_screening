"""Checks for the calibrated radical-pair model: conventions, the J–τ link, and
agreement with the original solver."""

import numpy as np
import pytest

from qbscreen.calibrated_mfe import build_H, mfe, mfe_general, dipolar_D_MHz
from qbscreen.rp_calibration import (J_superexchange_MHz, k_marcus, k_interpolated, J_tau,
                                     delta_max_for_turnover, EV_TO_MHZ)
from qbscreen.hyperfine_dft import a_iso_MHz


def test_exchange_and_dipolar_conventions():
    """J_gap = E_T − E_S; <T_m|H_D|T_m> = D(m² − 2/3) (Efimova & Hore)."""
    H, _ = build_H(0.0, 0.0, [], [], J_gap=100.0, D=-10.0)
    ev = np.sort(np.linalg.eigvalsh(H))
    assert ev[0] == pytest.approx(-75.0)
    assert sorted(ev[1:] - 25.0) == pytest.approx(sorted([-10 / 3, -10 / 3, 20 / 3]))
    assert dipolar_D_MHz(1.0) == pytest.approx(-77.9, rel=2e-3)


def test_coupling_cancels_from_J_tau():
    for t in (1.0, 22.0, 160.0):
        J_ang = J_superexchange_MHz(t, 1.5) * 2 * np.pi * 1e6
        # exact identity; 1e-8 absorbs the 10-digit h and ħ constants
        assert J_ang / k_marcus(t, -1.5, 0.9) == pytest.approx(J_tau(1.5, 0.9), rel=1e-8)


def test_superexchange_value():
    assert J_superexchange_MHz(100.0, 1.0) == pytest.approx(0.02 * EV_TO_MHZ)


def test_interpolation_caps_both_limits():
    for t in (1.0, 50.0, 300.0):
        k = k_interpolated(t, -1.0, 1.0)
        assert k <= k_marcus(t, -1.0, 1.0) and k <= 1e13


def test_turnover_bound_is_monotonic_in_kcat():
    d = [delta_max_for_turnover(22.0, 1.0, kc) for kc in (1, 10, 100)]
    assert d[0] > d[1] > d[2] > 0


def test_general_spin_matches_spin_half_builder():
    a = mfe(50e-6, 0.3, [14.0, 11.0], [44.0], 12.0, -11.0, 1.0, 1000.0)
    b = mfe_general(50e-6, 0.3, [(0, 0.5, 14.0), (0, 0.5, 11.0), (1, 0.5, 44.0)], 12.0, -11.0, 1.0, 1000.0)
    assert b == pytest.approx(a, rel=1e-8)


def test_reproduces_original_solver_under_convention_map():
    """The original code's D term, D_code(S1zS2z − S1·S2/3), is 2D in this module."""
    from qbscreen.honest_mfe import mfe_master, D_point_dipole
    D_code = D_point_dipole(0.35)
    old, _ = mfe_master(14, 11.0, 44.0, 750.0, 1e5, None, D_MHz=D_code)
    new = mfe(50e-6, 0.0, [14, 11.0], [44.0], 750.0, D_code / 2, 1e5)
    assert new == pytest.approx(old, rel=1e-10)


def test_fermi_contact_constant():
    a_H = a_iso_MHz(1 / np.pi, "H") * (1 + 1 / 1836.15267) ** -3
    assert a_H == pytest.approx(1420.4, abs=0.5)
