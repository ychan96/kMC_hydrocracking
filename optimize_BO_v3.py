#!/usr/bin/env python
"""
Bayesian Optimization for KMC Arrhenius parameters (10-parameter system).

Parameters being optimized
--------------------------
Adsorption / Desorption
    A_ads    : pre-exponential for adsorption        (log10 scale)
    E0_ads   : base adsorption barrier (eV)
    A_d      : pre-exponential for desorption        (log10 scale)
    E0_d     : base desorption barrier (eV)
    alpha_vdw: vdW chain-length scaling (eV/carbon)

Reaction (dMC & cracking)
    A_base   : shared pre-exponential               (log10 scale)
    E_dMC    : dMC formation barrier (eV)
    E_crk    : C-C scission barrier (eV)
    beta_int : internal-position penalty (eV)

Hydrogen equilibrium
    K_H2     : Langmuir equilibrium constant (bar⁻¹) (log10 scale)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import json
from datetime import datetime

from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence

from kmc_v3.simulation import run_multiple_simulations

# ══════════════════════════════════════════════════════════════════
#  Experimental data
# ══════════════════════════════════════════════════════════════════

def load_experimental_data(file_path: str, sheet_name: str = 'Sheet1',
                           col_index: int = 4, max_length: int = 30) -> dict:
    try:
        data = pd.read_excel(file_path, sheet_name=sheet_name)
        values = data.iloc[:max_length, col_index].values
        return {i + 1: float(v) for i, v in enumerate(values) if not np.isnan(v)}
    except Exception as e:
        print(f'Error loading experimental data: {e}')
        return None

# ══════════════════════════════════════════════════════════════════
#  Error metric
# ══════════════════════════════════════════════════════════════════

def calculate_rmse(sim_dist: dict, exp_dist: dict, max_length: int = 30) -> float:
    weights = np.array([
        1.0 if c <= 4 else 
        1.0 if c <= 12 else
        1.0
        for c in range(1, max_length + 1) 
    ])
    lengths    = range(1, max_length + 1)
    sim_values = np.array([sim_dist.get(c, 0.0) for c in lengths])
    exp_values = np.array([exp_dist.get(c, 0.0) for c in lengths])
    return float(np.sqrt(np.mean(weights*((sim_values - exp_values) ** 2))))


# ══════════════════════════════════════════════════════════════════
#  Objective
# ══════════════════════════════════════════════════════════════════

_call_count = 0

def objective_function(x, exp_dist: dict, sim_cfg: dict, space: list) -> float:
    global _call_count
    _call_count += 1
    print(f'[call {_call_count}] params: {dict(zip([d.name for d in space], x))}')

    keys = ['k_ads_i', 'k_ads_t', 'k_d_i', 'k_d_t', 'alpha_vdw_gas', 'alpha_vdw_light', 'alpha_vdw_heavy', 
            'k_dMC_i', 'k_dMC_t', 'k_crk_i', 'k_crk_t', 'K_H2']

    params = dict(zip(keys, x))
    try:
        results = run_multiple_simulations(
            num_sims      = sim_cfg['num_sims'],
            temp_C        = sim_cfg['temp_C'],
            reaction_time = sim_cfg['reaction_time'],
            chain_length  = sim_cfg['chain_length'],
            P_H2          = sim_cfg['P_H2'],
            params        = params,
            verbose       = False,
            max_steps     = sim_cfg.get('max_steps'),
            min_products  = sim_cfg.get('min_products', 10),  
            max_products  = sim_cfg.get('max_products', 100),
        )

    except KeyboardInterrupt:
        print(f'[call {_call_count}] SKIPPED manually — searching next params ...')
        return 1e6

    except Exception as e:
        print(f'[call {_call_count}] simulation error: {e}')
        return 1e6
    
    if results is None:
        return 1e6

    # ── Early stop gate ───────────────────────────────────────────
    all_products = []
    for r in results:
        all_products.extend(r['products'])

    total_products = len(all_products)

    if total_products < sim_cfg['num_sims']:
        print(f'[call {_call_count}] PRUNED — too few products ({total_products} < {sim_cfg["num_sims"]})')
        return 1e6

    if total_products > 300*sim_cfg['num_sims']:
        print(f'[call {_call_count}] PRUNED — too many products ({total_products} > {300*sim_cfg["num_sims"]})')
        return 1e6
    # ─────────────────────────────────────────────────────────────

    if not all_products:
        return 1e6

    counts     = Counter(all_products)
    max_length = sim_cfg['max_length']

    input_mass = sum((14 * r['carbon_array'].shape[0] + 2) for r in results)
    mass_raw   = {L: (14 * L + 2) * c for L, c in counts.items() if L <= max_length}
    sim_dist   = {L: m / input_mass * 100 for L, m in mass_raw.items()} if input_mass > 0 else {}

    error = calculate_rmse(sim_dist, exp_dist, max_length)
    print(f'[call {_call_count}] RMSE = {error:.4f}  (products: {total_products})')

    return error


# ══════════════════════════════════════════════════════════════════
#  Main optimisation routine
# ══════════════════════════════════════════════════════════════════

def optimize_parameters(
    exp_data_file: str,
    output_dir:    str  = 'optimization_results',
    n_calls:       int  = 60,
    n_initial:     int  = 15,
    sim_cfg:       dict = None,
) -> dict:
    """
    Run Bayesian optimisation over the 10 Arrhenius parameters.

    Returns the optimal parameter dict ready to pass to KMC / run_simulation.
    """
    if sim_cfg is None:
        sim_cfg = {
            'num_sims':      5,
            'temp_C':        250,
            'reaction_time': 7200,
            'chain_length':  300,
            'P_H2':          50,
            'max_length':    30,
            'max_steps':     None,
        }

    os.makedirs(output_dir, exist_ok=True)

    print('Loading experimental data ...')
    exp_dist = load_experimental_data(
        exp_data_file, max_length=sim_cfg['max_length'])
    print(f'  Loaded {len(exp_dist)} data points.')

    # ── Search space ──────────────────────────────────────────────
    # Physical bounds informed by typical heterogeneous catalysis values.
    space = [
        Real(1e-3, 1e-1,  name='k_ads_i'),
        Real(1e-4, 1e-2,  name='k_ads_t'),
        Real(1e-3, 1e-1,  name='k_d_i'),
        Real(1e-3, 1e-1,  name='k_d_t'),

        Real(0.001, 0.02, name='alpha_vdw_gas'),
        Real(0.001, 0.02, name='alpha_vdw_light'),
        Real(0.001, 0.02, name='alpha_vdw_heavy'),

        Real(1e-4, 1e-2,  name='k_dMC_i'),
        Real(1e-4, 1e-2,  name='k_dMC_t'),
        Real(1e-4, 1e-2,  name='k_crk_i'),
        Real(1e-4, 1e-2,  name='k_crk_t'),

        Real(0.1,  2.0,   name='K_H2'),
    ]

    def objective(x):
        return objective_function(x, exp_dist, sim_cfg, space)

    # ── Optimise ──────────────────────────────────────────────────
    print(f'\nStarting Bayesian optimisation  ({n_calls} calls, '
          f'{n_initial} random initial points) ...')
    global _call_count; _call_count = 0

    result = gp_minimize(
        objective,
        space,
        n_calls          = n_calls,
        n_initial_points = n_initial,
        acq_func         = 'LCB',          # Expected Improvement
        random_state     = 42,
        verbose          = False,
        n_jobs           = 1,             # set >1 only if objective is thread-safe
    )

    # ── Decode optimal params ─────────────────────────────────────
    optimal_params = dict(zip([d.name for d in space], result.x))

    # ── Save ──────────────────────────────────────────────────────
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'bo_results_{timestamp}.json')

    save_data = {
        'optimal_params':     optimal_params,
        'final_rmse':         float(result.fun),
        'n_calls':            n_calls,
        'simulation_config':  sim_cfg,
        'all_rmse_values':    result.func_vals.tolist(),
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    # ── Convergence plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_convergence(result, ax=ax)
    ax.set_title('Bayesian Optimization Convergence', fontweight='bold')
    plt.tight_layout()
    conv_file = os.path.join(output_dir, f'convergence_{timestamp}.png')
    plt.savefig(conv_file, dpi=300, bbox_inches='tight')
    plt.close()

    # ── Report ────────────────────────────────────────────────────
    print(f'\nOptimisation complete.')
    print(f'  Final RMSE : {result.fun:.4f}')
    print(f'  Results    : {results_file}')
    print(f'  Plot       : {conv_file}')
    print('\nOptimal parameters:')
    for k, v in optimal_params.items():
        print(f'  {k:<12} = {v:.6e}')

    return optimal_params


# ══════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Bayesian optimisation of KMC Arrhenius parameters')
    parser.add_argument('--exp-data',   type=str,   required=True,
                        help='Path to experimental Excel file')
    parser.add_argument('--temp',       type=float, default=250)
    parser.add_argument('--time',       type=float, default=7200)
    parser.add_argument('--length',     type=int,   default=300)
    parser.add_argument('--P-H2',       type=float, default=50)
    parser.add_argument('--sims',       type=int,   default=5,
                        help='Simulations per objective evaluation')
    parser.add_argument('--n-calls',    type=int,   default=60,
                        help='Total BO evaluations')
    parser.add_argument('--n-initial',  type=int,   default=15,
                        help='Random initial points before GP fits')
    parser.add_argument('--max-length', type=int,   default=30)
    parser.add_argument('--max-steps',  type=int,   default=None)
    parser.add_argument('--output-dir', type=str,   default='optimization_results')
    args = parser.parse_args()

    sim_cfg = {
        'num_sims':      args.sims,
        'temp_C':        args.temp,
        'reaction_time': args.time,
        'chain_length':  args.length,
        'P_H2':          args.P_H2,
        'max_length':    args.max_length,
        'max_steps':     args.max_steps,
        'min_products': 2,
        'max_products': 100,
    }

    optimize_parameters(
        exp_data_file = args.exp_data,
        output_dir    = args.output_dir,
        n_calls       = args.n_calls,
        n_initial     = args.n_initial,
        sim_cfg       = sim_cfg,
    )