# B3 — driving force Δ versus the competence bound (2026-09-30)

## Coupling-free competence bound
Any thermal formation step has an activation free energy ≥ Δ, so
k_f ≤ ν_n exp(−Δ/k_BT) whatever t and λ are. Requiring k_f ≥ k_cat:

    Δ_max = k_BT ln(ν_n / k_cat)
          = 0.62–0.80 eV  (ν_n = 1e13 s⁻¹, k_cat = 1e3–1 s⁻¹)
          = 0.68–0.86 eV  (ν_n = 1e14 s⁻¹)

The golden-rule version (rp_theory.k_forward) is not used: at the contact
couplings of the v5 scan (B1) the transfer is adiabatic.

## Experimental anchor (primary sources read)
- E°(R2NH•+/R2NH), dialkylamines, water: 1.30 ± 0.05 V vs NHE
  (dimethylamine 1.30 by pulse radiolysis, 1.27 by CV; diethylamine,
  pyrrolidine, piperidine 1.26–1.36 by CV). Jonsson, Wayner & Lusztyk,
  J. Phys. Chem. 100, 17539 (1996), Table 2 and text. Primary alkylamines
  were not measured; their gas-phase IPs are higher (Table 2 trend), so
  1.30 V is taken as a lower bound for methylamine/benzylamine.
- E(FMN ox/semiquinone), pH 7, 20 °C: −0.313 V vs NHE. Mayhew, Eur. J.
  Biochem. 265, 698 (1999), from Anderson's pulse-radiolysis data (1983).

Aqueous, separated ions: Δ ≥ 1.30 − (−0.313) = 1.61 eV (contact Coulomb in
water ≈ 0.05 eV). k_f ≤ 1e13·exp(−1.61/k_BT) ≈ 6e−14 s⁻¹.

## DFT versus the anchor (qbscreen/driving_force.py aqueous → aqueous_potentials.json)
Same protocol (B3LYP/def2-SVP/PCM, ε = 78.4):

| couple                 | DFT      | experiment | error   |
|------------------------|----------|------------|---------|
| dimethylamine•+/amine  | +0.73 V  | +1.30 V    | −0.57 V |
| methylamine•+/amine    | +1.26 V  | —          |         |
| benzylamine•+/amine    | +1.45 V  | —          |         |
| lumiflavin/Fl•−        | −1.16 V  | −0.313 V   | −0.84 V |

The half-reaction errors are large and of opposite sign in Δ; for the
anchored pair DFT overestimates Δ by 0.28 eV. Correcting each half by its
error gives aqueous Δ ≈ 2.1 eV (methylamine) and 2.3 eV (benzylamine).
The raw ε = 4–10 values (2.47–2.99 eV, driving_force.json) are therefore
upper estimates; they are not used as the bound.

## Conclusion for the manuscript
Robust statement: Δ ≥ 1.6 eV (experimental potentials, free cofactor, water)
against Δ_max ≤ 0.86 eV. Thermal formation at turnover would need the
protein to narrow the amine/flavin potential gap by ≥ 0.75 V relative to
water — e.g. a flavin ox/semiquinone potential ≥ +0.44 V vs NHE with the
amine unchanged. That is the falsification condition; a measured
one-electron potential of MAO-bound FAD is not yet in hand.

## Enzyme kinetics (abstracts, PubMed)
- Walker & Edmondson, Biochemistry 33, 7088 (1994): MAO B, no flavin
  radical intermediates detected in anaerobic reduction; ᴰk 6.5–14.1.
- Miller & Edmondson, Biochemistry 38, 13670 (1999): MAO A, α-C–H cleavage
  rate-limiting (KIE 6–13); ρ ≈ +2.0 supports proton abstraction.
