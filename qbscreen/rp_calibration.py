#!/usr/bin/env python3
"""Exchange coupling and lifetime of a charge-transfer radical pair from one coupling.

A flavin–amine radical pair Fl•−/amine•+ lies an energy Δ above the closed-shell
ground state Fl/amine and decays to it by back electron transfer (BET). A single
one-electron coupling t links the two configurations, so it sets both

  exchange  : the singlet pair mixes with the closed-shell ground state through
              <GS|H|S_RP> = √2 t, the triplet cannot. Second-order perturbation
              raises the singlet by 2t²/Δ, so with H_ex = J S1·S2 (singlet–triplet
              gap E_T − E_S = J)
                    J = −2 t² / Δ ,            |J| = 2 t² / Δ

  lifetime  : non-adiabatic Marcus rate for BET with driving force ΔG = −Δ
                    k = (2π/ħ) t² FC ,
                    FC = (4πλk_BT)^(−1/2) exp[−(ΔG + λ)² / (4λk_BT)]

Their ratio does not contain t:

        (|J|/ħ) τ_BET = 1 / (π Δ FC)

so the number of exchange precessions the pair completes before it recombines is
fixed by the driving force and the reorganisation energy alone. The coupling —
the least certain input — cancels.

Second-order (superexchange) J is valid while √2 t ≪ Δ; `t_over_delta` reports it.
"""

import numpy as np

HBAR_EVS = 6.582119569e-16       # eV·s
KB_EV = 8.617333262e-5           # eV/K
EV_TO_MHZ = 2.417989242e8        # 1 eV = h · 241.8 THz


def J_superexchange_MHz(t_meV, delta_eV):
    """|J| in MHz (ordinary frequency, J S1·S2 convention) from t and Δ."""
    t = np.asarray(t_meV) * 1e-3
    return 2.0 * t ** 2 / delta_eV * EV_TO_MHZ


def fc_weighted_dos(dG_eV, lam_eV, T=310.0):
    """Classical Franck–Condon weighted density of states, 1/eV."""
    kT = KB_EV * T
    return (4 * np.pi * lam_eV * kT) ** -0.5 * np.exp(-(dG_eV + lam_eV) ** 2 / (4 * lam_eV * kT))


def k_marcus(t_meV, dG_eV, lam_eV, T=310.0):
    """Non-adiabatic Marcus rate, 1/s."""
    t = np.asarray(t_meV) * 1e-3
    return 2 * np.pi / HBAR_EVS * t ** 2 * fc_weighted_dos(dG_eV, lam_eV, T)


NU_N = 1e13                      # effective nuclear frequency for the adiabatic limit, 1/s


def k_interpolated(t_meV, dG_eV, lam_eV, T=310.0, nu_n=NU_N):
    """Rate bridging the non-adiabatic (Marcus) and adiabatic (TST) limits.

    1/k = 1/k_NA + 1/k_ad, with k_ad = ν_n exp(−ΔG‡/k_BT). For t ≳ k_BT the
    golden-rule rate overshoots the nuclear frequency; this caps it there. The cap
    lengthens lifetimes, i.e. it errs towards magnetosensitivity.
    """
    kT = KB_EV * T
    k_ad = nu_n * np.exp(-(dG_eV + lam_eV) ** 2 / (4 * lam_eV * kT))
    return 1.0 / (1.0 / k_marcus(t_meV, dG_eV, lam_eV, T) + 1.0 / k_ad)


def J_tau(delta_eV, lam_eV, T=310.0):
    """(|J|/ħ)·τ_BET — independent of t. Angular exchange frequency × lifetime."""
    return 1.0 / (np.pi * delta_eV * fc_weighted_dos(-delta_eV, lam_eV, T))


def k_forward(t_meV, delta_eV, lam_eV, T=310.0):
    """Thermal forward SET rate (Fl + amine → Fl•− + amine•+), detailed balance."""
    return k_marcus(t_meV, +delta_eV, lam_eV, T)


def delta_max_for_turnover(t_meV, lam_eV, k_cat, T=310.0):
    """Largest Δ at which thermal SET still forms the pair as fast as turnover.

    A radical pair on the catalytic pathway must form at least as fast as the
    enzyme turns over. The forward rate is Marcus with ΔG = +Δ; solve
    k_forward(Δ) = k_cat for Δ by bisection (k_forward falls monotonically in Δ
    for Δ > −λ).
    """
    lo, hi = 0.0, 3.0
    if k_forward(t_meV, lo, lam_eV, T) < k_cat:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if k_forward(t_meV, mid, lam_eV, T) >= k_cat:
            lo = mid
        else:
            hi = mid
    return lo


def fit_decay(r, t):
    """Fit t(r) = t0 exp(−β r / 2); returns (t0, β). β is the rate decay constant."""
    r = np.asarray(r, float)
    lt = np.log(np.asarray(t, float))
    slope, icpt = np.polyfit(r, lt, 1)
    return float(np.exp(icpt)), float(-2 * slope)


if __name__ == "__main__":
    # self-check of the t-cancellation and of the J convention
    for t in (10.0, 100.0, 300.0):
        d, lam = 1.5, 0.9
        J_ang = J_superexchange_MHz(t, d) * 2 * np.pi * 1e6        # rad/s
        tau = 1 / k_marcus(t, -d, lam)
        assert np.isclose(J_ang * tau, J_tau(d, lam), rtol=1e-10), (t, J_ang * tau, J_tau(d, lam))
    # J = 2t²/Δ: t = 100 meV, Δ = 1 eV → 20 meV = 4.836e6 MHz
    assert np.isclose(J_superexchange_MHz(100.0, 1.0), 0.02 * EV_TO_MHZ)
    print("rp_calibration self-check passed")
