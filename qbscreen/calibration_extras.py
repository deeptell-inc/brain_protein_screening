#!/usr/bin/env python3
"""Supporting numbers: the photolyase-anchored coupling and partner-hyperfine
sensitivity of the contact-pair bound. Writes calibration_results/extras.json."""

import json
from pathlib import Path

import numpy as np

from qbscreen.calibrated_mfe import mfe, dipolar_D_MHz
from qbscreen.rp_calibration import fc_weighted_dos, HBAR_EVS

OUT = Path("calibration_results")
FL = [14.66, 5.30]


def anchor_t(tau_s=70e-12, delta=1.78, lam=0.97, T=310.0):
    """Coupling that reproduces a measured non-adiabatic recombination time (Liu et al. 2013)."""
    k = 1 / tau_s
    t_eV = np.sqrt(k / (2 * np.pi / HBAR_EVS * fc_weighted_dos(-delta, lam, T)))
    return float(t_eV * 1e3)


def partner_sensitivity(r_nm=0.35, partners=(5.0, 44.0, 100.0, 257.0, 363.0), tau_us=(1.0, 100.0), D=None):
    D = dipolar_D_MHz(r_nm) if D is None else D
    rows = []
    for a in partners:
        for tau in tau_us:
            rows.append(dict(partner_MHz=a, tau_us=tau, **{
                f"mfe_th{th}": mfe(50e-6, np.radians(th), FL, [a], 0.0, D, 1 / tau) for th in (0, 45, 90)}))
    return dict(r_nm=r_nm, D_MHz=D, J_gap_MHz=0.0, rows=rows)


if __name__ == "__main__":
    dsd = json.loads((OUT / "dipolar_spin_density.json").read_text())
    res = dict(t_anchor_meV=anchor_t(), t_anchor_W384_meV=anchor_t(120e-12, 1.88, 1.23),
               partner_sensitivity=partner_sensitivity(),
               # the same bound with D summed over the DFT spin distributions
               MAO_spinD=partner_sensitivity(0.35, D=dsd["3.50"]["D_MHz"], tau_us=(1.0, 10.0, 100.0)),
               DAO_spinD=partner_sensitivity(0.40, partners=(5.0, 58.0, 100.0), D=dsd["4.00"]["D_MHz"],
                                             tau_us=(1.0, 10.0, 100.0)))
    (OUT / "extras.json").write_text(json.dumps(res, indent=2))
    print(f"t from 70 ps (W382): {res['t_anchor_meV']:.1f} meV; from 120 ps (W384): {res['t_anchor_W384_meV']:.1f} meV")
    m = max(abs(v) for r in res["partner_sensitivity"]["rows"] for k, v in r.items() if k.startswith("mfe"))
    print(f"max |MFE| over partner 5–363 MHz, J=0, tau 1–100 us: {m:.2e} %")
