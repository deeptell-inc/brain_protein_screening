#!/usr/bin/env python3
"""Largest field effect a catalytically competent, thermally formed pair can show.

Scheme: ES → ¹RP (k_f, singlet only), ¹RP → ES (k_S), RP → EP (k_P, any spin).
Δ is the free energy of the pair summed over its four spin states, so (spin
energies negligible against k_BT) the singlet lies Δ_S = Δ + k_BT ln 4 above ES and detailed balance reads
k_f = k_S e^{−Δ_S/k_BT}. The turnover through the pair is k_f Φ_P, and

  * Φ_P ≤ 1                  ⇒  k_S ≥ k_cat e^{Δ_S/k_BT} ≥ k_cat e^{Δ/k_BT};
  * Φ_P ≤ 4 k_P / k_S        ⇒  k_P ≥ k_cat e^{Δ/k_BT}.

The second inequality holds for any spin Hamiltonian and any unital,
positivity-preserving relaxation: with R = (−L)⁻¹ (k_P > 0), −L(I) = k_S P_S + k_P I
gives k_S R(P_S) = I − k_P R(I) ⪯ I. A dead-end pair (k_P = 0) is bounded instead
through k_S ∫ e^{L₀u}(P_S) du = I − e^{L₀t}(I) ⪯ I; the rate bound does not apply
to it. Without spin dynamics the
turnover is exactly e^{−Δ_S/k_BT}/(1/k_S + 1/k_P).

Competence (turnover ≥ k_cat) therefore requires BOTH k_S and k_P ≥
k_min = k_cat e^{Δ/k_BT}. No coupling, reorganisation energy or mechanism for
k_P enters.

With the searched map F(k_S, k_P) (worst_case_map; the largest |MFE| FOUND by a
finite search in one spin model, not a proven supremum),

    G(Δ) = max { F(k_S, k_P) : k_S ≥ k_min(Δ), k_P ≥ k_min(Δ) }

is evaluated by a procedure on the rate grid; it is not a bound on the
continuous region (interval_example stores a permitted point above it).
k_min is rounded down to the rate grid in both directions; above the largest
k_P the edge value is used because max_kS F decreases with k_P on the grid;
above the largest k_S only the k_P constraint is imposed. Where k_min lies
below the grid (slow pairs) G is undefined (None): the grid does not cover
the permitted region there.
"""

import json
import sys
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5


def k_min_per_us(delta, k_cat, T=310.0):
    return k_cat * np.exp(delta / (KB_EV * T)) * 1e-6


def G(rows, delta, k_cat, T=310.0, factor=None):
    """Largest F on the rate grid with k_S, k_P ≥ k_min(Δ); None below the grid.
    factor: optional {k_P: X} multiplying F row by row (model-sensitivity estimates)."""
    km = k_min_per_us(delta, k_cat, T)
    kS_grid = sorted({r["k_S"] for r in rows})
    kP_grid = sorted({r["k_P"] for r in rows})
    if km < min(kS_grid[0], kP_grid[0]) * (1 - 1e-9):
        return None
    down = lambda grid: max(g for g in grid if g <= km * (1 + 1e-9))
    # F is not shown to decrease with k_S, so beyond the k_S grid only the k_P constraint is used
    kS0 = down(kS_grid) if km <= kS_grid[-1] * (1 + 1e-9) else kS_grid[0]
    kP0 = down(kP_grid)
    f = (lambda r: r["max_mfe"] * factor.get(r["k_P"], 1.0)) if factor else (lambda r: r["max_mfe"])
    return max(f(r) for r in rows if r["k_S"] >= kS0 * (1 - 1e-9) and r["k_P"] >= kP0 * (1 - 1e-9))


def non_increasing(g):
    """The permitted region shrinks as Δ grows, so G cannot increase with Δ; the grid
    rules can make it do so, and the running maximum from the right removes that
    conservatively."""
    run = None
    for x in reversed(g):
        if x["G_percent"] is not None:
            run = x["G_percent"] if run is None else max(run, x["G_percent"])
            x["G_percent"] = run
    return g


def threshold(g, level):
    """Smallest Δ with G < level for all larger Δ on the grid (None if never)."""
    ok = [x["delta_eV"] for x in g if x["G_percent"] is not None and x["G_percent"] < level]
    bad = [x["delta_eV"] for x in g if x["G_percent"] is not None and x["G_percent"] >= level]
    return min(d for d in ok if d > max(bad, default=-1)) if ok else None


def monotone_in_kP(rows):
    kPs = sorted({r["k_P"] for r in rows})
    env = [max(r["max_mfe"] for r in rows if r["k_P"] == k) for k in kPs]
    return all(b <= a * (1 + 1e-6) for a, b in zip(env, env[1:])), dict(zip(kPs, env))


def with_relaxation(rows, t2e_rows):
    """Coherent map with every cell replaced by the largest value found with or
    without single-electron dephasing (cells searched with dephasing only)."""
    best = {}
    for r in t2e_rows:
        key = (round(np.log10(r["k_S"]), 3), round(np.log10(r["k_P"]), 3))
        best[key] = max(best.get(key, 0.0), r["max_mfe"])
    out = []
    for r in rows:
        key = (round(np.log10(r["k_S"]), 3), round(np.log10(r["k_P"]), 3))
        out.append(dict(r, max_mfe=max(r["max_mfe"], best.get(key, 0.0))))
    return out


def main(maps=("worst_case_v6.json",), k_cats=(1.0, 10.0, 100.0, 1000.0)):
    deltas = np.linspace(0.0, 2.0, 201)
    out = {}
    for name in maps:
        rows = json.loads((Path("calibration_results") / name).read_text())["rows"]
        t2e = Path("calibration_results/worst_case_v6_t2e.json")
        if name == "worst_case_v6.json" and t2e.exists():
            relaxed = with_relaxation(rows, json.loads(t2e.read_text())["rows"])
            out["with_relaxation"] = {str(kc): non_increasing([dict(delta_eV=float(d), G_percent=G(relaxed, d, kc))
                                                               for d in deltas]) for kc in k_cats}
        mono, env = monotone_in_kP(rows)
        out[name] = dict(monotone_in_kP=mono, envelope_by_kP=env, k_cat={
            str(kc): non_increasing([dict(delta_eV=float(d), k_min_per_us=float(k_min_per_us(d, kc)),
                                          G_percent=G(rows, d, kc)) for d in deltas]) for kc in k_cats})
        print(f"{name}: max_kS F non-increasing in k_P: {mono}")
        for kc in k_cats:
            g = out[name]["k_cat"][str(kc)]
            lo = min(x["delta_eV"] for x in g if x["G_percent"] is not None)
            print(f"  k_cat={kc:g}/s: G defined for Δ≥{lo:.2f}, G<0.1 % for Δ≥{threshold(g, 0.1)}, "
                  f"G<0.01 % for Δ≥{threshold(g, 0.01)} eV")
    Path("calibration_results/competence_sensitivity.json").write_text(json.dumps(out, indent=2))


def interval_example(out="calibration_results/g_interval_example.json"):
    """A permitted off-grid point whose field effect exceeds the gridded G: G is the
    value of a procedure on the rate grid, not a bound on the continuous region."""
    from qbscreen.worst_case_map import Cell, NUCLEI
    from qbscreen.product_yield import build_system
    m = json.loads(Path("calibration_results/worst_case_v6.json").read_text())
    kP, kS = 10 ** 1.5, 76.4499169285051
    delta = KB_EV * 310.0 * np.log(kP * 1e6)                    # k_min = k_P exactly, k_cat = 1 s⁻¹
    v = Cell(build_system(NUCLEI), np.array(m["T_contact_MHz"]), kS, kP).mfe(0.0, 0.0, 0.0, 0.0)
    res = dict(delta_eV=delta, k_cat=1.0, k_S=kS, k_P=kP, J_MHz=0.0, s=0.0, direction="z",
               mfe_percent=v, G_percent=G(m["rows"], delta, 1.0))
    Path(out).write_text(json.dumps(res, indent=2))
    return res


def random_bound_test(n=300, seed=20261001, out="calibration_results/phi_bound_test.json"):
    """Φ_P k_S / (4 k_P) on random spin systems; stored with its inputs for reproduction."""
    from qbscreen.product_yield import build_system, hamiltonian, axial_tensor, yields_hilbert, yields_liouville
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        nuc = [(int(rng.integers(0, 2)), float(rng.choice([0.5, 1.0])), float(10 ** rng.uniform(0, 2.5)))
               for _ in range(int(rng.integers(0, 4)))]
        s = build_system(nuc)
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        ax = rng.normal(size=3)
        B, J, D = 10 ** rng.uniform(-6, -1), float(rng.choice([-1, 1]) * 10 ** rng.uniform(-1, 4)), -10 ** rng.uniform(-1, 3)
        kS, kP = 10 ** rng.uniform(-2, 5), 10 ** rng.uniform(-3, 4)
        T2e = 10 ** rng.uniform(0, 4) if (rng.random() < 0.3 and s["dim"] <= 72) else None
        H = hamiltonian(s, B, b, J, axial_tensor(D, ax))
        P = (yields_liouville(s, H, kS, kP, T2e_ns=T2e) if T2e else yields_hilbert(s, H, kS, kP))[0]
        rows.append(dict(nuclei=nuc, B_T=B, b=b.tolist(), J_MHz=J, D_MHz=D, axis=ax.tolist(), k_S=kS, k_P=kP,
                         T2e_ns=T2e, Phi_P=P, ratio=P * kS / (4 * kP)))
    Path(out).write_text(json.dumps(dict(seed=seed, n=n, max_ratio=max(r["ratio"] for r in rows), rows=rows), indent=2))
    return max(r["ratio"] for r in rows)


def _selfcheck():
    # the flux identity against the explicit steady state of ES ⇄ RP → EP (ES held at 1)
    rng = np.random.default_rng(1)
    for _ in range(20):
        kS, kP, d = 10 ** rng.uniform(0, 9), 10 ** rng.uniform(0, 9), rng.uniform(0, 0.8)
        kf = kS * np.exp(-d / (KB_EV * 310.0))
        rp = kf / (kS + kP)                   # d[RP]/dt = kf − (kS + kP)[RP] = 0
        assert np.isclose(kP * rp, np.exp(-d / (KB_EV * 310.0)) / (1 / kS + 1 / kP), rtol=1e-12)
    # Φ_P ≤ 4 k_P / k_S with spin dynamics (random nuclei, exchange, dipolar, field, T2e)
    from qbscreen.product_yield import build_system, hamiltonian, axial_tensor, yields_hilbert, yields_liouville
    for _ in range(40):
        nuc = [(int(rng.integers(0, 2)), float(rng.choice([0.5, 1.0])), float(10 ** rng.uniform(0, 2.5)))
               for _ in range(rng.integers(0, 3))]
        s = build_system(nuc)
        b = rng.normal(size=3)
        H = hamiltonian(s, 10 ** rng.uniform(-6, -1), b / np.linalg.norm(b), rng.choice([-1, 1]) * 10 ** rng.uniform(-1, 4),
                        axial_tensor(-10 ** rng.uniform(-1, 3), rng.normal(size=3)))
        kS, kP = 10 ** rng.uniform(-2, 5), 10 ** rng.uniform(-3, 4)
        P = yields_liouville(s, H, kS, kP, T2e_ns=10 ** rng.uniform(0, 4))[0] if rng.random() < 0.5 \
            else yields_hilbert(s, H, kS, kP)[0]
        assert P <= 4 * kP / kS * (1 + 1e-9), (P, kS, kP)
    rows =[dict(k_S=s, k_P=p, max_mfe=1.0 / p) for s in (1.0, 10.0) for p in (1.0, 10.0)]
    assert G(rows, 0.0, 1e6) == 1.0 and G(rows, 0.0, 1e7) == 0.1   # k_min = 1 and 10 /µs
    assert G(rows, 0.0, 5e6) == 1.0                                 # off grid: rounded down
    assert G(rows, 1.0, 1e6) == 0.1                                 # beyond the grid: edge value
    assert G(rows, 0.0, 1e5) is None                                # below the grid: undefined
    assert np.isclose(G(rows, 0.0, 1e7, factor={10.0: 3.0}), 0.3)   # row factor
    g = non_increasing([dict(delta_eV=d, G_percent=v) for d, v in ((0, None), (1, 0.1), (2, 0.5), (3, 0.2))])
    assert [x["G_percent"] for x in g] == [None, 0.5, 0.5, 0.2]   # running maximum from the right
    print("competence_sensitivity self-checks passed")


if __name__ == "__main__":
    _selfcheck() if sys.argv[1:] == ["check"] else main()
