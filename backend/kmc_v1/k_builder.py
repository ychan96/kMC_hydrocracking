"""
k_builder.py
============
Maps a compact shape-parameter vector onto the full 66-key k_const dict that
init_v4.BaseKineticMC expects.

Why
---
Sensitivity analysis over 66 raw constants is meaningless (66 parameters, 39
degrees of freedom per condition).  The 62 ads/des values are not independent:
within each chain-length regime they lie on k(N) = k0 * exp(alpha * N).  So the
model has 16 effective parameters, and those are what should be perturbed.

    ads_A_k0, ads_A_alpha    C1-C3     |  des_A_k0, des_A_alpha
    ads_B_k0, ads_B_alpha    C4-C11    |  des_B_k0, des_B_alpha
    ads_C_k0, ads_C_alpha    C12-C30+  |  des_C_k0, des_C_alpha
    dmc_i, dmc_t, crk_i, crk_t

c31plus is generated as regime C evaluated at N = 31, matching init_v4.

Units
-----
alpha is stored per carbon (dimensionless in the exponent).  Multiply by
kB*T = 0.04508 eV at 250 C to report as an energy per CH2.

Perturbation convention
-----------------------
k0     : multiplicative, delta in log10 decades
alpha  : ADDITIVE, delta in eV per CH2 (alpha can be negative, so a
         multiplicative perturbation is not defined)
dmc/crk: multiplicative, log10 decades

Because k0 and alpha carry different units, their sensitivities are not
directly comparable to one another.  Comparison ACROSS REACTION FAMILIES
(ads / des / dmc / crk) is well posed and is what mode='family' provides.
"""

import numpy as np

REGIMES = {'A': (1, 3), 'B': (4, 11), 'C': (12, 30)}
C31_N = 31
KB_EV = 8.617333e-5

# Baseline shape parameters, refit from the init_v4 defaults.
BASE_SHAPE = {
    'ads_A_k0': 1.687976e-04, 'ads_A_alpha': +0.498165,
    'ads_B_k0': 2.308763e-03, 'ads_B_alpha': -0.729088,
    'ads_C_k0': 1.459201e-05, 'ads_C_alpha': +0.188240,
    'des_A_k0': 4.928056e-01, 'des_A_alpha': -3.144021,
    'des_B_k0': 4.401119e-04, 'des_B_alpha': +0.288718,
    'des_C_k0': 1.682774e-01, 'des_C_alpha': -0.090221,
	'dmc_i': 2.061422e-02,    'dmc_t': 2.18325e-03,
    'crk_i': 9.641641e-04,    'crk_t': 1.946220e-03,
}

SHAPE_KEYS = list(BASE_SHAPE.keys())
FAMILIES = {'ads': ['ads_A_k0', 'ads_B_k0', 'ads_C_k0'],
            'des': ['des_A_k0', 'des_B_k0', 'des_C_k0'],
            'dmc': ['dmc_i', 'dmc_t'],
            'crk': ['crk_i', 'crk_t']}


def build_k_const(shape: dict) -> dict:
    """Expand 16 shape parameters into the 66-key dict init_v4 consumes."""
    k = {}
    for rxn in ('ads', 'des'):
        for tag, (lo, hi) in REGIMES.items():
            k0 = shape[f'{rxn}_{tag}_k0']
            a = shape[f'{rxn}_{tag}_alpha']
            for n in range(lo, hi + 1):
                k[f'{rxn}_c{n}'] = float(k0 * np.exp(a * n))
        k0, a = shape[f'{rxn}_C_k0'], shape[f'{rxn}_C_alpha']
        k[f'{rxn}_c31plus'] = float(k0 * np.exp(a * C31_N))
    for key in ('dmc_i', 'dmc_t', 'crk_i', 'crk_t'):
        k[key] = float(shape[key])
    return k


def perturb(shape: dict, key: str, delta: float) -> dict:
    """One-at-a-time perturbation. delta is decades for k0/dmc/crk,
    eV per CH2 for alpha."""
    s = dict(shape)
    if key.endswith('_alpha'):
        s[key] = s[key] + delta / (KB_EV * 523.15)      # eV -> per-carbon
    else:
        s[key] = s[key] * 10.0 ** delta
    return s


def perturb_family(shape: dict, family: str, delta: float) -> dict:
    """Scale every k0 in a reaction family by 10**delta, alphas unchanged.
    This is the well-posed comparison across reaction types."""
    s = dict(shape)
    for key in FAMILIES[family]:
        s[key] = s[key] * 10.0 ** delta
    return s


def alpha_eV(shape: dict, key: str, temp_K: float = 523.15) -> float:
    """Report an alpha as energy per CH2."""
    return shape[key] * KB_EV * temp_K


if __name__ == '__main__':
    import re, sys
    src = open(sys.argv[1] if len(sys.argv) > 1 else 'init_v4.py').read()
    ref = {a: float(b) for a, b in
           re.findall(r"'(\w+)':\s*([\d.]+e[+-]\d+)", src)}
    built = build_k_const(BASE_SHAPE)
    bad = [(kk, ref[kk], built[kk]) for kk in built
           if kk in ref and abs(built[kk] / ref[kk] - 1) > 0.02]
    print(f'keys built : {len(built)}   reference keys : {len(ref)}')
    print(f'max relative error : '
          f'{max(abs(built[kk]/ref[kk]-1) for kk in built if kk in ref):.2e}')
    print(f'keys off by >2% : {[b[0] for b in bad] or "none"}')
    for kk, r, b in bad:
        print(f'   {kk}: ref={r:.4e} built={b:.4e} ({100*(b/r-1):+.1f}%)')