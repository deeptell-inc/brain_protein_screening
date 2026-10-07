"""Central claims of the v5 manuscript, checked through the modules' own self-checks."""

import runpy

import pytest


@pytest.mark.parametrize("module", ["qbscreen.rp_theory", "qbscreen.product_yield"])
def test_module_selfchecks(module):
    # conventions, Fay et al. (2018) eqs 81/82, detailed balance, solver agreement
    runpy.run_module(module, run_name="__main__")


def test_competence_bounds():
    # flux identity, Φ_P ≤ 4 k_P/k_S with spin dynamics, conservative G on the grid
    from qbscreen.competence_sensitivity import _selfcheck
    _selfcheck()


def test_mao_estimate_inputs():
    # v6 MAO estimate: dialkylamine proxy 1.30 ± 0.05 V, MAO ox/sq midpoints, contact Coulomb in water
    from qbscreen import make_numbers_v6 as mn
    m = mn.macros()
    assert (m["DeltaMAOlo"], m["DeltaMAOhi"], m["DeltaMAOsub"]) == ("1.41", "1.44", "0.91")
    assert (m["DeltaMAOsubLo"], m["DeltaMAOsubHi"]) == ("0.86", "0.99")
    assert (m["DeltaMaxHi"], m["BornRadius"], m["DeltaSiteLo"]) == ("0.82", "2.81", "0.49")


def test_dead_end_resolvent_and_population_bound():
    import numpy as np
    from qbscreen.product_yield import build_system, hamiltonian, yields_hilbert
    from qbscreen.master_equation import build_liouvillian
    s = build_system([])
    H = hamiltonian(s, 0.0, np.array([0, 0, 1.0]), 0.0, np.zeros((3, 3)))
    P_S = s["P_S"]
    L = build_liouvillian(2 * np.pi * H, P_S, np.eye(4) - P_S, 1.0, 0.0, [])     # k_S = 1, k_P = 0
    assert np.linalg.matrix_rank(L) == 7                                           # no resolvent at k_P = 0
    s2 = build_system([(0, 1.0, 14.66), (1, 0.5, 44.0)])
    H2 = hamiltonian(s2, 50e-6, np.array([0.3, 0.1, 0.95]), 3.0, np.zeros((3, 3)))
    for kS in (0.1, 10.0, 1e3):
        kP = 1e-9                                                                  # dead-end limit
        P = yields_hilbert(s2, H2, kS, kP)[0]
        assert P / kP <= 4 / kS * (1 + 1e-6)                                       # ∫Tr ρ ≤ 4/k_S


def test_G_offgrid_check():
    # G rounds the required rates down; off-grid permitted k_S on the threshold row stay below it
    import json
    import numpy as np
    from pathlib import Path
    from qbscreen.competence_sensitivity import G
    from qbscreen.worst_case_map import Cell, NUCLEI, directions
    from qbscreen.product_yield import build_system
    ex = json.loads(Path("calibration_results/g_offgrid_check.json").read_text())
    m = json.loads(Path("calibration_results/worst_case_v6.json").read_text())
    assert np.isclose(G(m["rows"], ex["delta_eV"], 1.0), ex["G_percent"], rtol=1e-12)
    p = ex["points"][0]
    c = Cell(build_system(NUCLEI), np.array(m["T_contact_MHz"]), p["k_S"], ex["k_P"])
    v = max(c.mfe(J, 0.0, th, ph) for J in (0.0, 0.5, -0.5) for th, ph in directions())
    assert np.isclose(v, p["mfe_percent"], rtol=1e-9)
    assert ex["max_offgrid_percent"] < ex["G_percent"]


def test_phi_bound_assumptions():
    # the factor 4 is approached under depolarizing relaxation; without trace preservation Φ_P > 1
    from fractions import Fraction
    from qbscreen.competence_sensitivity import sharpness_example, trace_counterexample
    assert max(r["ratio"] for r in sharpness_example(out="/dev/null")) > 0.9999
    phi = trace_counterexample()
    assert Fraction(phi).limit_denominator(1000) == Fraction(150, 101) and phi <= 4 * 1.0 / 0.01


def test_stored_cells_reproduce():
    import json
    import numpy as np
    from pathlib import Path
    from qbscreen.worst_case_map import Cell, NUCLEI
    from qbscreen.product_yield import build_system
    m = json.loads(Path("calibration_results/worst_case_v6.json").read_text())
    s = build_system(NUCLEI)
    for kp in (10 ** 1.5, 100.0):                                                  # threshold-setting rows
        r = max((x for x in m["rows"] if np.isclose(x["k_P"], kp)), key=lambda x: x["max_mfe"])
        v = Cell(s, np.array(m["T_contact_MHz"]), r["k_S"], r["k_P"]).mfe(r["J_at_max"], r["s"], r["theta"], r["phi"])
        assert np.isclose(v, r["max_mfe"], rtol=1e-6), (kp, v, r["max_mfe"])


def test_G_not_reported_below_rate_grid():
    # panel counterexample: k_S = 0.01, k_P = 0.001 /µs gives 14.7 % > the grid maximum,
    # so G must be undefined wherever k_min lies below the grid
    from qbscreen.competence_sensitivity import G
    rows = [dict(k_S=s, k_P=p, max_mfe=1.0) for s in (0.01, 1.0) for p in (0.01, 1.0)]
    assert G(rows, 0.15, 1.0) is None


def test_bound_test_stored():
    import json
    from pathlib import Path
    d = json.loads(Path("calibration_results/phi_bound_test.json").read_text())
    assert d["n"] == 300 and max(r["ratio"] for r in d["rows"]) == d["max_ratio"] <= 1.0


def test_numbers_match_results():
    from pathlib import Path
    from qbscreen import make_numbers_v6 as mn
    p = Path("manuscript/magnetobio_jcp/numbers.tex")
    if not p.exists():
        pytest.skip("LaTeX sources are not distributed; run qbscreen.make_numbers_v6 to generate numbers.tex")
    m = mn.macros()
    text = p.read_text()
    for k, v in m.items():
        assert f"\\newcommand{{\\{k}}}{{{v}}}" in text, k
