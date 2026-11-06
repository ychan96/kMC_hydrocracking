# kMC_hydrocracking

Kinetic Monte Carlo (kMC) framework for hydrogenolysis of long-chain alkanes (HDPE model chains) on a metal surface. Three model formulations, M1–M3, are included; they differ in how rate constants are resolved by chain length.

All section, table and equation numbers below refer to the Supporting Information (SI) of:

> Y. C. Park, C.-J. Yoo, K. Kim, _A Computationally Efficient Kinetic Monte Carlo Framework for Long-Chain Alkane Hydrogenolysis_, Supporting Information.

---

## 1. Models at a glance

|                            | **M1** Base                    | **M2** Chain-length-resolved | **M3** Regime-compressed           |
| -------------------------- | ------------------------------ | ---------------------------- | ---------------------------------- |
| Kinetic parameters         | 4                              | 66                           | 18 (+ θ_H)                         |
| Ads/des constants          | 1 each                         | 31 each (C1–C30, C31+)       | `k(L) = k_i·exp(α·L)` in 3 regimes |
| dMC / scission             | 1 each                         | internal + terminal          | internal + terminal                |
| Surface                    | M×M grid, `m_bond` / `m_chain` | M×M grid                     | Bridge/hollow lattice (`Lattice`)  |
| Optimizer (SI Table S2)    | Bayesian optimization          | CMA-ES                       | CMA-ES                             |
| RMSE, C1–C30 (SI Table S4) | 1.66                           | 0.41                         | 0.38                               |
| Φ = Y_C1 / (Y_C2 + Y_C3)   | 0.75                           | 2.17                         | 2.61 (exp. 2.88)                   |
| Package                    | `pykmc_M1`                     | `backend/pykMC_M2`           | `pykmc_M3`                         |
| Run file                   | `simulation_v0.py`             | `main_m2.py`                 | `main_m3.py`                       |

Reference condition for all three: **543 K (270 °C), 25 bar H₂, 10 h**, on a 5 × 5 patch (SI Section S2).

---

## 2. Repository layout

```
kMC_hydrocracking/
├── main_m2.py                 # M2 command-line run file
├── main_m3.py                 # M3 command-line run file
├── simulation.py              # M3 driver  (run_simulation, run_multiple_simulations, calculate_conversion)
├── simulation_v0.py           # M1 driver
├── pykmc_M1/                  # M1: init_v0, count_sites_v0, reactions_v0, coverage_v0, utils
├── pykmc_M3/                  # M3: base, constants, lattice, count_sites, reactions, utils
└── backend/pykMC_M2/          # M2: init_v4, count_sites_v4, reactions_v4, coverage, simulation_v4, utils_v4
```

M1 and M3 import their packages as top-level modules (`from pykmc_M3 import KMC`), so run them from the repository root. M2 imports `backend.pykMC_M2.*`. If your layout differs, edit the import lines at the top of the run files.

---

## 3. Installation

```bash
pip install numpy scipy pandas matplotlib openpyxl imageio
```

`openpyxl` is needed only to read experimental data from `.xlsx`, and `imageio` only for coverage GIFs.

---

## 4. Quick start

### M3 (regime-compressed, lattice)

```bash
python main_m3.py --temp 270 --time 36000 --sims 100 --seed 0 --output-dir results_m3
```

| Flag                      | Default        | Meaning                                                                       |
| ------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `--temp`                  | 250            | Temperature (°C); a label only, because rates are fixed at the reference      |
| `--time`                  | 7200           | Simulated reaction time (s); use 36000 (10 h)                                 |
| `--length`                | None           | Initial chain length; `None` draws from N(280, 10)                            |
| `--sims`                  | 10             | Number of independent trajectories                                            |
| `--msize`                 | 5              | Surface patch size (msize × msize metal atoms)                                |
| `--facet`                 | 111            | `111`, `0001` (hexagonal) or `100` (square)                                   |
| `--pressure`              | 25             | P_H₂ (bar)                                                                    |
| `--theta-H`               | 0.61           | H coverage; a negative value means θ_H is computed from `--pressure` (Eq. S5) |
| `--seed`                  | None           | Random seed (trajectory _i_ uses `seed + i`)                                  |
| `--exp-data`, `--exp-col` | `data.xlsx`, 2 | Experimental distribution to overlay                                          |
| `--coverage`              | off            | Save lattice snapshots of run 1 and a GIF                                     |
| `--verbose`               | off            | Print every event                                                             |
| `--output-dir`            | `results`      | Location of the JSON summary and PNG plot                                     |

From Python:

```python
from simulation import run_multiple_simulations, calculate_conversion
from pykmc_M3.utils import plot_distribution

res = run_multiple_simulations(100, temp_C=270, reaction_time=36000, seed=0)
calculate_conversion(res)
plot_distribution(res, exp_data_file='data.xlsx', exp_col=2)
```

### M2 (chain-length-resolved)

```bash
python main_m2.py --temp 270 --time 36000 --sims 100 --msize 5 --output-dir results_m2
```

Flags: `--temp --time --length --sims --msize --exp-data --verbose --output-dir --coverage` (same meaning as for M3).

### M1 (base)

```python
from simulation_v0 import run_multiple_simulations
from pykmc_M1.utils import plot_distribution

SI_M1 = {'ads': 7.04e-5, 'des': 1.69e-3, 'dmc': 1.97e-4, 'crk': 1.80e-3}   # SI Table S3
res = run_multiple_simulations(100, temp_C=270, reaction_time=36000, m_size=5, P_H2=25,
                               rate_constants=SI_M1, seed=0)
plot_distribution(res, exp_data_file='data.xlsx', exp_col=11)
```

---

## 5. Model description

### 5.1 Chain representation (all models)

| Array            | Length | Meaning                                                                    |
| ---------------- | ------ | -------------------------------------------------------------------------- |
| `carbon_array`   | N      | 0 = unbound, 1 = bound to the surface                                      |
| `chain_array`    | N + 1  | 1 = C–C bond, 0 = break or chain end; fragments are the runs between zeros |
| `carbon_to_site` | N      | Bridge id holding carbon _j_, −1 if unbound (SI array **p**; M3 only)      |

A C–C scission sets one entry of `chain_array` to 0 and invalidates the cached fragment list (`sim.chains`).

### 5.2 Surface representation

**M3** (SI Section S1, Table S1). Carbon binds only to bridge sites and H only to hollow sites. dMC partners are bridges that share exactly one metal atom at the nearest separation.

| SI symbol | Code (`sim.surface.…`)  | Encoding                                    |
| --------- | ----------------------- | ------------------------------------------- |
| **o**     | `occupancy[b]`          | −1 absent, 0 vacant, 1 sMC, 2 dMC           |
| **q**     | `carbon_at_site[b]`     | Carbon index, or −1                         |
| **m**     | `chain_at_site[b]`      | +N sMC on an N-mer, −N dMC member, 0 vacant |
| **p**     | `sim.carbon_to_site[j]` | Bridge id of carbon _j_, or −1              |
| **H**     | `hollow_H[u]`           | Drawn once at θ_H and held fixed            |

**M1 / M2**: an M × M grid with `m_bond` (0 vacant, 1 sMC, 2 dMC) and `m_chain` (±N). M1 also keeps an explicit `dmc_pairs` list so that each scission splits the correct pair of sites.

### 5.3 Surface events

| Event         | Eligible fragment                                                                    | Count _n_e_      | M2/M3 keys                        |
| ------------- | ------------------------------------------------------------------------------------ | ---------------- | --------------------------------- |
| Adsorption    | L ≥ 2, no bound carbon; M3 also needs a free bridge                                  | L per fragment   | `ads_c2 … ads_c30`, `ads_c31plus` |
| Desorption    | Exactly one sMC carbon                                                               | 1 per fragment   | `des_c1 … des_c30`, `des_c31plus` |
| dMC formation | Exactly one sMC carbon, unbound chain neighbour; M3 also needs a free partner bridge | 0–2 per fragment | `dmc_t`, `dmc_i`                  |
| C–C scission  | Adjacent dMC pair                                                                    | 1 per pair       | `crk_t`, `crk_i`                  |

- **Terminal vs internal.** The bond between carbons _i_ and _i_+1 is terminal if _i_ = 0 or _i_ = L − 2; otherwise it is internal. All dMC on C2–C3 is terminal.
- **C31+ lump.** Fragments with L > 30 share one constant: the heavy-regime function evaluated at L = 31 (`constants.py`: `MAX_LENGTH = 30`, `cn()`).
- **C1** can desorb but cannot adsorb, form dMC or crack.

### 5.4 Rates and time step

Each event propensity follows SI Eq. S12:

```
r_e = k_e · n_e · f_e(θ_H)
f_e = (1 − θ_H)²   for H-releasing events (adsorption, dMC)
f_e = θ_H²         for H-consuming events (desorption, scission)
```

Events are chosen by the Gillespie/BKL method: the event with cumulative share above u₁ is selected, and time advances by Δt = −ln(u₂)/R, with R = Σ r_e.

M3 rate constants are generated from 16 shape parameters plus the regime boundaries (`base.py → build_k_const`):

```
k(L) = k0_r · exp(alpha_r · L),   r = g (L ≤ x_g), l (x_g < L ≤ x_l), h (L > x_l)
x_g = 3, x_l = 11                 (SI Table S3, Fig. S1)
```

---

## 6. Parameter sets (SI Table S3, 543 K, 25 bar H₂, 10 h)

### M1

| k_ads       | k_des       | k_dMC       | k_crk       |
| ----------- | ----------- | ----------- | ----------- |
| 7.04 × 10⁻⁵ | 1.69 × 10⁻³ | 1.97 × 10⁻⁴ | 1.80 × 10⁻³ |

### M2

The 31 + 31 adsorption and desorption constants are listed in SI Table S3. The remaining four are:

| k_dMC,i     | k_dMC,t     | k_crk,i     | k_crk,t     |
| ----------- | ----------- | ----------- | ----------- |
| 2.96 × 10⁻² | 2.96 × 10⁻³ | 6.42 × 10⁻⁴ | 2.08 × 10⁻³ |

### M3

| Regime                 | k_i,ads (s⁻¹) | α_ads  | k_i,des (s⁻¹) | α_des  |
| ---------------------- | ------------- | ------ | ------------- | ------ |
| g: gas (L ≤ 3)         | 8.33 × 10⁻⁵   | +0.396 | 7.63 × 10⁻²   | −1.461 |
| l: light liquid (4–11) | 8.12 × 10⁻⁴   | −0.662 | 1.86 × 10⁻⁴   | +0.667 |
| h: heavy (≥ 12, C31+)  | 2.76 × 10⁻⁴   | +0.051 | 4.43 × 10⁻²   | −0.021 |

| k_dMC,i     | k_dMC,t    | k_crk,i     | k_crk,t     | θ_H,ref |
| ----------- | ---------- | ----------- | ----------- | ------- |
| 1.98 × 10⁻² | 6.0 × 10⁻³ | 9.13 × 10⁻⁴ | 3.51 × 10⁻³ | 0.5829  |

To run M3 with the SI parameters:

```python
from pykmc_M3 import BaseKineticMC
from simulation import run_multiple_simulations

SI_M3 = {'ads_k0_g': 8.33e-5, 'ads_alpha_g': 0.396,  'des_k0_g': 7.63e-2, 'des_alpha_g': -1.461,
         'ads_k0_l': 8.12e-4, 'ads_alpha_l': -0.662, 'des_k0_l': 1.86e-4, 'des_alpha_l': 0.667,
         'ads_k0_h': 2.76e-4, 'ads_alpha_h': 0.051,  'des_k0_h': 4.43e-2, 'des_alpha_h': -0.021,
         'dmc_i': 1.98e-2, 'dmc_t': 6.0e-3, 'crk_i': 9.13e-4, 'crk_t': 3.51e-3}

k = BaseKineticMC(init_len=1).build_k_const(SI_M3)        # expands to the 66-key dict
res = run_multiple_simulations(10000, temp_C=270, reaction_time=36000,
                               k_const=k, theta_H=0.5829, seed=0)
```

---

## 7. Temperature and H₂-pressure dependence (SI Sections S3.2–S3.5)

Reference T_ref = 543 K. Rate constants at the reference are held fixed.

| Quantity               | Expression                                                                             | SI           |
| ---------------------- | -------------------------------------------------------------------------------------- | ------------ |
| Rate constant          | `k_e(T) = k_e(T_ref) · exp[−E_a,e/R · (1/T − 1/T_ref)]`                                | Eq. S1       |
| Pre-exponential factor | `A_e = k_e(T_ref) · exp(E_a,e / R T_ref)`                                              | Eq. S2       |
| Ads/des E_a profile    | `E_a,e(L) = E_a∞ + (E_a1 − E_a∞) · exp[−(L − 1)/λ_e]`                                  | Table S5     |
| K_H₂ at reference      | `K_ref = (1/P) · [θ_ref/(1 − θ_ref)]² = 0.0781 bar⁻¹`                                  | Eq. S3       |
| K_H₂(T)                | van 't Hoff with ΔH_ads = −13.73 kJ mol⁻¹                                              | Eq. S4       |
| θ_H(T, P)              | `√(K·P) / (1 + √(K·P))`                                                                | Eq. S5       |
| Apparent E_a           | E_a + (1 − θ_H)·ΔH_ads for H-consuming events; E_a − θ_H·ΔH_ads for H-releasing events | Eqs. S14–S15 |

Optimized activation energies (SI Table S5):

|            | E_a1 (C1) | E_a∞   | λ    |
| ---------- | --------- | ------ | ---- |
| Adsorption | 125.40    | 20.25  | 1.45 |
| Desorption | 91.95     | 154.81 | 6.36 |

| dMC, int. | dMC, term. | Scission, int. | Scission, term. |
| --------- | ---------- | -------------- | --------------- |
| 43.42     | 54.96      | 70.32          | 30.34           |

All values are in kJ mol⁻¹. Per-L values of E_a and A are in SI Table S6; dMC and scission values in Table S7.

Pressure dependence (SI Section S2.1, stage 5): at 523 K, θ_H is computed at each pressure, and correction factors c_p,dMC and c_p,crk (equal to 1 at 25 bar) multiply f_e for dMC and scission. They were fitted at 5, 10, 40 and 65 bar.

> The temperature and pressure stages live in the optimization scripts, not in the M1–M3 core packages. The core packages run at a single fixed set of _k_ and θ_H.

---

## 8. Outputs

`run_simulation` returns a dictionary per trajectory:

| Key                           | Content                                                             |
| ----------------------------- | ------------------------------------------------------------------- |
| `products`                    | List of fragment lengths at the end of the run                      |
| `carbon_array`, `chain_array` | Final state                                                         |
| `time`, `steps`, `failed`     | Final simulated time, executed events, refused events (should be 0) |
| `theta_C`                     | Final carbon coverage (M1, M3)                                      |
| `computation_time`            | Wall-clock time (s)                                                 |

`main_m2.py` and `main_m3.py` write the following to `--output-dir`:

- `summary_<timestamp>.json`: parameters, rate constants, conversion and product counts.
- `comparison_<timestamp>.png`: simulated mass-% distribution (C1–C30, mass = 14n + 2) overlaid on the experimental data.
- With `--coverage`: `coverage_steps/*.png` and `coverage.gif`.

---

## 9. Code defaults vs SI: check before reproducing SI results

| Item                             | Code default                                     | SI value                                       | Action                                                           |
| -------------------------------- | ------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------------------------- |
| M1 rate constants (`init_v0.py`) | 3.52e-4, 8.46e-3, 9.84e-4, 8.98e-3               | Table S3                                       | Pass `SI_M1` (§4)                                                |
| M1 θ_H factor                    | θ_H computed but not applied to rates            | f_e enters all propensities (Table S3, note a) | Confirm the M1 rate expression                                   |
| M3 `BASE_SHAPE` (`base.py`)      | e.g. `ads_k0_g` = 1.69e-4, `ads_alpha_g` = 0.498 | 8.33e-5, 0.396                                 | Different fit, not a time rescale (α differs); pass `SI_M3` (§6) |
| M3 θ_H                           | 0.61                                             | 0.5829                                         | Pass `theta_H=0.5829`                                            |
| M3 `K_H2` (`base.py`)            | 0.0158                                           | 0.0781 bar⁻¹                                   | Used only when `theta_H=None`                                    |
| Reaction time                    | 7,200 s                                          | 36,000 s (10 h)                                | Set `--time 36000`                                               |
| Trajectories per condition       | 10 (CLI default)                                 | 10,000                                         | Set `--sims 10000`                                               |
| `--temp`                         | 250 °C                                           | 543 K = 270 °C                                 | A label only in M1–M3                                            |

---

## 10. Citation

If you use this code, please cite the main article and its Supporting Information (Y.C. Park, C-J. Yoo*, KS. Kim*).
