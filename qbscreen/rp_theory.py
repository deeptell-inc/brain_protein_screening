#!/usr/bin/env python3
"""Consistent rates and reactive exchange for a flavin–amine radical pair.

One electron-transfer process connects the singlet radical pair to the
closed-shell ground state, with many-electron matrix element V_S = √2 t
(t is the one-electron LUMO(Fl)–HOMO(amine) coupling). The triplet pair is not
coupled to the closed-shell state. From the same V_S and the same classical
Franck–Condon density

    p(E) = exp[−(E − (λ − Δ))² / 4λk_BT] / √(4πλk_BT),

second-order perturbation theory gives the singlet back-transfer rate and the
reactive (charge-transfer) exchange shift of the singlet level
(Fay, Lindoy & Manolopoulos, J. Chem. Phys. 149, 064107 (2018)):

    k_S   = (2π/ħ) V_S² p(0)
    J_gap = E_T − E_S = V_S² · PV∫ p(E)/E dE = V_S² Dawson(x) / √(λk_BT),
            x = (λ − Δ) / (2√(λk_BT))

so that J_gap/(ħ k_S) = ½ erfi(x): positive in the normal region, zero at the
activationless point Δ = λ, negative in the inverted region. Δ here is the
free energy of the singlet channel above the closed-shell state (Δ_S in
competence_sensitivity, which exceeds the spin-summed Δ by k_BT ln 4).

Correspondence with Fay et al.: their diabatic coupling "Δ" is V_S here, their
bias ε is Δ here, their eq. 81 is k_back, and their eq. 82 gives J^(2) with the
exchange written as −2J^(2) S1·S2 (eq. 66a), so J_gap = −2J^(2). The
fourth-order singlet–triplet dephasing k_d (their eq. 75a) is handled in
product_yield as a sensitivity parameter.

Forward (thermal formation) and backward rates use the same golden-rule
expression, so k_f/k_b = exp(−Δ/k_BT) exactly. The golden rule is only valid
while the electron transfer is non-adiabatic; `adiabaticity` returns the ratio
of the golden-rule rate to the transition-state rate ν_n exp(−ΔG‡/k_BT), and
points where it exceeds `ADIABATIC_LIMIT` are outside the model's validity and
must be excluded rather than patched.
"""

import numpy as np
from scipy.special import dawsn, erfi

HBAR_EVS = 6.582119569e-16
KB_EV = 8.617333262e-5
EV_TO_MHZ = 2.417989242e8
NU_N = 1e13                  # effective nuclear frequency, 1/s
ADIABATIC_LIMIT = 0.3        # golden rule trusted while k_NA ≤ 0.3 k_TST


def _kT(T):
    return KB_EV * T


def fc_density(E, delta, lam, T=310.0):
    kT = _kT(T)
    return np.exp(-(E - (lam - delta)) ** 2 / (4 * lam * kT)) / np.sqrt(4 * np.pi * lam * kT)


def V_S(t_meV):
    """Many-electron singlet–ground-state coupling, eV."""
    return np.sqrt(2.0) * np.asarray(t_meV) * 1e-3


def k_back(t_meV, delta, lam, T=310.0):
    """Singlet back electron transfer, radical pair → closed shell (ΔG = −Δ), 1/s."""
    return 2 * np.pi / HBAR_EVS * V_S(t_meV) ** 2 * fc_density(0.0, delta, lam, T)


def k_forward(t_meV, delta, lam, T=310.0):
    """Thermal formation, closed shell → singlet radical pair (ΔG = +Δ), 1/s.

    Same golden-rule expression with the driving force reversed. Formation
    produces the singlet pair only (the closed shell is a singlet)."""
    return 2 * np.pi / HBAR_EVS * V_S(t_meV) ** 2 * fc_density(0.0, -delta, lam, T)


def J_reactive_MHz(t_meV, delta, lam, T=310.0):
    """Reactive exchange J_gap = E_T − E_S (MHz, ordinary frequency)."""
    x = (lam - delta) / (2 * np.sqrt(lam * _kT(T)))
    return V_S(t_meV) ** 2 * dawsn(x) / np.sqrt(lam * _kT(T)) * EV_TO_MHZ


def J_over_k(delta, lam, T=310.0):
    """(J_gap/ħ)/k_S = ½ erfi(x), dimensionless (angular exchange frequency × lifetime)."""
    x = (lam - delta) / (2 * np.sqrt(lam * _kT(T)))
    return 0.5 * erfi(x)


def adiabaticity(t_meV, delta, lam, T=310.0, nu_n=NU_N):
    """k_NA / k_TST for the back transfer. ≲ 1 non-adiabatic; ≫ 1 adiabatic."""
    k_tst = nu_n * np.exp(-(lam - delta) ** 2 / (4 * lam * _kT(T)))
    return k_back(t_meV, delta, lam, T) / k_tst


def valid(t_meV, delta, lam, T=310.0):
    """Golden rule applicable and the perturbative exchange small against the gap."""
    return bool(adiabaticity(t_meV, delta, lam, T) <= ADIABATIC_LIMIT)


if __name__ == "__main__":
    from scipy.integrate import quad
    # 1. exchange by direct principal-value integration
    for delta, lam in ((0.5, 1.0), (1.78, 0.97), (1.0, 1.0)):
        kT = _kT(310.0)
        c = lam - delta
        w = 12 * np.sqrt(lam * kT)
        f = lambda E: fc_density(E, delta, lam)
        pv = quad(f, c - w, c + w, weight="cauchy", wvar=0.0)[0] if abs(c) < w else quad(lambda E: f(E) / E, c - w, c + w)[0]
        x = (lam - delta) / (2 * np.sqrt(lam * kT))
        assert np.isclose(pv, dawsn(x) / np.sqrt(lam * kT), rtol=1e-6, atol=1e-9), (delta, lam, pv)
    # 2. ratio identity and the photolyase value quoted by both panellists
    for t in (1.0, 22.0):
        r = J_reactive_MHz(t, 1.78, 0.97) * 2 * np.pi * 1e6 / k_back(t, 1.78, 0.97)
        assert np.isclose(r, J_over_k(1.78, 0.97), rtol=1e-8)
    assert np.isclose(abs(J_over_k(1.78, 0.97)), 70.05, rtol=1e-3)
    assert J_over_k(1.0, 1.0) == 0.0
    assert J_reactive_MHz(10.0, 0.5, 1.0) > 0 > J_reactive_MHz(10.0, 1.5, 1.0)
    # 2b. literal transcription of Fay, Lindoy & Manolopoulos (2018) eq. 82, J_gap = −2 J^(2)
    for t, d, lam in ((5.0, 0.4, 1.0), (22.0, 1.78, 0.97), (3.0, 1.2, 0.8)):
        kT = _kT(310.0)
        Vs = V_S(t)
        J2 = Vs ** 2 / 4 * np.sqrt(np.pi / (kT * lam)) * np.exp(-(lam - d) ** 2 / (4 * lam * kT)) \
            * erfi((d - lam) / (2 * np.sqrt(kT * lam)))
        assert np.isclose(-2 * J2 * EV_TO_MHZ, J_reactive_MHz(t, d, lam), rtol=1e-10), (t, d, lam)
        k81 = Vs ** 2 / HBAR_EVS * np.sqrt(np.pi / (kT * lam)) * np.exp(-(lam - d) ** 2 / (4 * lam * kT))
        assert np.isclose(k81, k_back(t, d, lam), rtol=1e-10)
    # 3. detailed balance
    for d in (0.1, 0.5, 0.9):
        assert np.isclose(k_forward(10.0, d, 1.0) / k_back(10.0, d, 1.0), np.exp(-d / _kT(310.0)), rtol=1e-10)
    print("rp_theory self-checks passed")
