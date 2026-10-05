"""Does new_Q_brain's cryptochrome-point capacity survive the calibrated J and D?

Reads new_Q_brain's own code (PYTHONPATH) and changes only the Hamiltonian:
adds the Efimova–Hore dipolar tensor at r = 1.90 nm and exchange from the
Efimova–Hore J0 range. Protocol, estimator and seeds are theirs (f5_cry).
"""
import sys
import numpy as np
import qbscreen
assert "new_Q_brain" in qbscreen.__file__, qbscreen.__file__
from qbscreen.final_numbers import CRY, run_corr_obs, _observable_set
from qbscreen.reservoir import build_reservoir_H, memory_and_ipc, N_SPINS
from qbscreen.spin_dynamics import SX, SY, SZ, spin_op

D = -77.91 / 1.90 ** 3               # Efimova–Hore T± – T0 splitting, MHz
ops = {a: (spin_op(o, 0, N_SPINS), spin_op(o, 1, N_SPINS)) for a, o in (("x", SX), ("y", SY), ("z", SZ))}
S1S2 = sum(ops[a][0] @ ops[a][1] for a in "xyz")


def H_with(J_gap, D_MHz, axis):
    H = build_reservoir_H(**{**CRY, "J": J_gap})
    if D_MHz:
        H = H + 2 * D_MHz * ops[axis][0] @ ops[axis][1] - (2 * D_MHz / 3) * S1S2
    return H


def ipc(H, n_seeds=3, L=900, T2e=1000.0):
    out = []
    for sd in range(n_seeds):
        s = np.random.default_rng(800 + sd).uniform(0, 1, L + 100)
        X = run_corr_obs(s, H, 1.0, T2e, _observable_set("full"))
        out.append(memory_and_ipc(X, s[100:])["IPC_total"])
    return np.mean(out), np.std(out)


cases = [("published: J=0, D=0", 0.0, 0.0, "z"),
         ("D only, B || dipolar axis", 0.0, D, "z"),
         ("D only, B perp dipolar axis", 0.0, D, "x"),
         ("D + J_gap=+12.6 (J0 mid), B ||", +12.6, D, "z"),
         ("D + J_gap=-12.6 (J0 mid), B ||", -12.6, D, "z"),
         ("D + J_gap=+126 (J0 high), B ||", +126.0, D, "z")]
print(f"CRY point (A = {CRY['A_e1_a']}, {CRY['A_e1_b']} | {CRY['A_e2_a']} MHz, tau = 1 us, T2e = 1 us), D = {D:.2f} MHz")
for lab, J, Dm, ax in cases:
    m, sd = ipc(H_with(J, Dm, ax))
    print(f"  {lab:34s} IPC = {m:.3f} +/- {sd:.3f}", flush=True)
