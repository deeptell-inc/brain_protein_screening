"""new_Q_brain routes table (readout_routes.survey) with the calibrated dipolar term."""
import json, numpy as np
import qbscreen; assert "new_Q_brain" in qbscreen.__file__
from qbscreen.readout_routes import CRY, run_routes
from qbscreen.reservoir import build_reservoir_H, memory_and_ipc, N_SPINS
from qbscreen.spin_dynamics import SX, SY, SZ, spin_op
ops = {a: (spin_op(o, 0, N_SPINS), spin_op(o, 1, N_SPINS)) for a, o in (("x", SX), ("y", SY), ("z", SZ))}
S1S2 = sum(ops[a][0] @ ops[a][1] for a in "xyz")
D = -77.91 / 1.90 ** 3
pub = json.load(open("/Users/deeptell01/Documents/alterego/personal/new_Q_brain/simulation_results/readout_routes.json"))
keys = ("YS_end", "YS_t", "SandT_end", "SandT_t", "cidnp")
for lab, J in (("D(1.90 nm), J=0", 0.0), ("D(1.90 nm), J_gap=-12.6", -12.6)):
    H = build_reservoir_H(**{**CRY, "J": J}) + 2 * D * ops["z"][0] @ ops["z"][1] - (2 * D / 3) * S1S2
    acc = {k: [] for k in keys}
    for sd in range(3):
        s = np.random.default_rng(sd).uniform(0, 1, 700 + 80); sp = s[80:]
        R = run_routes(s, H)
        for k in keys:
            acc[k].append(memory_and_ipc(R[k], sp)["IPC_total"])
    print(f"== {lab}")
    for k in keys:
        print(f"   {k:10s} published {pub[k]['IPC']:.3f}   with D {np.mean(acc[k]):.3f} +/- {np.std(acc[k]):.3f}", flush=True)
