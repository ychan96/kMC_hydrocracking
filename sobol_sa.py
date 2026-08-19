#!/usr/bin/env python
"""
Sobol Sensitivity Analysis for KMC Arrhenius parameters.

Workflow:
    1. Generate Sobol sample matrix X  (N*(k+2) runs)
    2. Evaluate KMC objective at each row of X
    3. Compute S1 and ST indices via SALib
    4. Plot and save results
    5. Print recommended reduced BO space
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
from collections import Counter

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from backend.kmc_v3.simulation import run_multiple_simulations

# ══════════════════════════════════════════════════════════════════
#  Experimental data loader  (same as optimize_BO_v3.py)
# ══════════════════════════════════════════════════════════════════

def load_experimental_data(file_path, sheet_name='Sheet1',
                           col_index=4, max_length=30):
    try:
        data   = pd.read_excel(file_path, sheet_name=sheet_name)
        values = data.iloc[:max_length, col_index].values
        return {i + 1: float(v) for i, v in enumerate(values) if not np.isnan(v)}
    except Exception as e:
        print(f'Error loading experimental data: {e}')
        return None


# ══════════════════════════════════════════════════════════════════
#  RMSE metric  (same as optimize_BO_v3.py)
# ══════════════════════════════════════════════════════════════════

def calculate_rmse(sim_dist, exp_dist, max_length=30):
    lengths    = range(1, max_length + 1)
    sim_values = np.array([sim_dist.get(c, 0.0) for c in lengths])
    exp_values = np.array([exp_dist.get(c, 0.0) for c in lengths])
    return float(np.sqrt(np.mean((sim_values - exp_values) ** 2)))


# ══════════════════════════════════════════════════════════════════
#  KMC objective — single parameter vector x -> scalar RMSE
# ══════════════════════════════════════════════════════════════════

def kmc_objective(x, exp_dist, sim_cfg):
    """
    Evaluate RMSE for one parameter combination x.
    Returns 1e6 (penalty) if simulation fails or produces no products.
    """
    keys   = ['k_ads_i', 'k_ads_t', 'k_d_i', 'k_d_t',
              'alpha_vdw_gas', 'alpha_vdw_light', 'alpha_vdw_heavy',
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
        )
    except Exception as e:
        print(f'  [sim error] {e}')
        return 1e6

    if not results:
        return 1e6

    all_products = []
    for r in results:
        all_products.extend(r['products'])

    if len(all_products) < sim_cfg['num_sims']:
        return 1e6

    counts     = Counter(all_products)
    max_length = sim_cfg['max_length']
    input_mass = sum((14 * r['carbon_array'].shape[0] + 2) for r in results)
    mass_raw   = {L: (14 * L + 2) * c for L, c in counts.items() if L <= max_length}
    sim_dist   = {L: m / input_mass * 100 for L, m in mass_raw.items()} if input_mass > 0 else {}

    return calculate_rmse(sim_dist, exp_dist, max_length)


# ══════════════════════════════════════════════════════════════════
#  Sobol SA
# ══════════════════════════════════════════════════════════════════

def run_sobol_sa(
    exp_data_file,
    output_dir   = 'sobol_results',
    N            = 64,        # power of 2 — total runs = N*(k+2)
    sim_cfg      = None,
    st_threshold = 0.05,      # parameters below this are considered insensitive
):
    """
    Run Sobol SA and return ranked sensitivity indices.

    Parameters
    ----------
    N            : Sobol base sample size. Total evals = N*(12+2).
                   N=64  ->  896 runs  (fast screen)
                   N=128 -> 1792 runs  (moderate)
                   N=512 -> 7168 runs  (high confidence)
    st_threshold : ST cutoff below which a parameter is flagged insensitive.
    """
    if sim_cfg is None:
        sim_cfg = {
            'num_sims':      3,      # keep low for SA — speed over accuracy
            'temp_C':        250,
            'reaction_time': 7200,
            'chain_length':  300,
            'P_H2':          50,
            'max_length':    30,
            'max_steps':     None,
        }

    os.makedirs(output_dir, exist_ok=True)

    print('Loading experimental data ...')
    exp_dist = load_experimental_data(exp_data_file, max_length=sim_cfg['max_length'])
    print(f'  Loaded {len(exp_dist)} data points.\n')

    # ── Problem definition ────────────────────────────────────────
    problem = {
        'num_vars': 12,
        'names': ['k_ads_i', 'k_ads_t', 'k_d_i', 'k_d_t',
                  'alpha_vdw_gas', 'alpha_vdw_light', 'alpha_vdw_heavy',
                  'k_dMC_i', 'k_dMC_t', 'k_crk_i', 'k_crk_t', 'K_H2'],
        'bounds': [
        [1e-4, 1e-3],   # k_ads_i   (nominal ~6e-4)
        [5e-5, 5e-4],   # k_ads_t   (nominal ~1.5e-4)
        [1e-2, 5e-1],   # k_d_i     (nominal ~8e-2)
        [1e-2, 3e-1],   # k_d_t     (nominal ~7e-2)
        [0.001, 0.1],   # alpha_vdw_gas
        [0.001, 0.05],  # alpha_vdw_light
        [0.001, 0.05],  # alpha_vdw_heavy
        [1e-3, 1e-2],   # k_dMC_i   (nominal ~6e-3)
        [5e-3, 5e-2],   # k_dMC_t   (nominal ~2e-2)
        [1e-4, 1e-3],   # k_crk_i   (nominal ~6e-4)
        [2e-3, 2e-2],   # k_crk_t   (nominal ~9e-3)
        [0.1,  2.0],    # K_H2
    ]
    }

    # ── Step 1: Generate sample matrix ───────────────────────────
    # X shape: (N*(k+2), k) = (896, 12) for N=64, k=12
    print(f'Generating Sobol sample matrix  (N={N}, total runs={N*(problem["num_vars"]+2)}) ...')
    X = sobol_sample.sample(problem, N=N, calc_second_order=False)
    print(f'  Sample matrix shape: {X.shape}\n')

    # ── Step 2: Evaluate KMC at each row ─────────────────────────
    n_runs = X.shape[0]
    Y      = np.zeros(n_runs)

    print(f'Running {n_runs} KMC evaluations ...')
    for i, x in enumerate(X):
        Y[i] = kmc_objective(x, exp_dist, sim_cfg)
        if (i + 1) % 10 == 0:
            valid = np.sum(Y[:i+1] < 1e5)
            print(f'  [{i+1}/{n_runs}]  valid={valid}  latest RMSE={Y[i]:.4f}')

    # ── Step 3: Analyze ───────────────────────────────────────────
    print('\nComputing Sobol indices ...')
    Si = sobol_analyze.analyze(problem, Y,
                                calc_second_order=False,
                                print_to_console=False)

    S1 = Si['S1']
    ST = Si['ST']
    S1_conf = Si['S1_conf']
    ST_conf = Si['ST_conf']

    # ── Step 4: Rank and print ────────────────────────────────────
    order = np.argsort(ST)[::-1]   # descending by ST

    print(f'\n{"Parameter":<20} {"S1":>6} {"±":>4} {"ST":>6} {"±":>4}  {"Status"}')
    print('-' * 60)
    active   = []
    inactive = []
    for idx in order:
        name   = problem['names'][idx]
        status = 'ACTIVE' if ST[idx] > st_threshold else 'fix'
        print(f'{name:<20} {S1[idx]:>6.3f} {S1_conf[idx]:>4.3f} '
              f'{ST[idx]:>6.3f} {ST_conf[idx]:>4.3f}  {status}')
        if ST[idx] > st_threshold:
            active.append(name)
        else:
            inactive.append(name)

    print(f'\nActive parameters  ({len(active)})  : {active}')
    print(f'Insensitive params ({len(inactive)}) : {inactive}')
    print(f'\nReducing BO from {problem["num_vars"]}D -> {len(active)}D')

    # ── Step 5: Save results ──────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_data = {
        'N':              N,
        'total_runs':     n_runs,
        'sim_cfg':        sim_cfg,
        'S1':             dict(zip(problem['names'], S1.tolist())),
        'ST':             dict(zip(problem['names'], ST.tolist())),
        'S1_conf':        dict(zip(problem['names'], S1_conf.tolist())),
        'ST_conf':        dict(zip(problem['names'], ST_conf.tolist())),
        'active_params':  active,
        'inactive_params':inactive,
        'Y_valid_fraction': float(np.mean(Y < 1e5)),
    }

    results_file = os.path.join(output_dir, f'sobol_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f'\nResults saved: {results_file}')

    # ── Step 6: Plot ──────────────────────────────────────────────
    names_sorted = [problem['names'][i] for i in order]
    S1_sorted    = S1[order]
    ST_sorted    = ST[order]
    S1c_sorted   = S1_conf[order]
    STc_sorted   = ST_conf[order]

    x_pos = np.arange(len(names_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x_pos - width/2, S1_sorted, width,
           label='S1 (first-order)', color='steelblue', alpha=0.85,
           yerr=S1c_sorted, capsize=3)
    ax.bar(x_pos + width/2, ST_sorted, width,
           label='ST (total-order)', color='coral', alpha=0.85,
           yerr=STc_sorted, capsize=3)

    ax.axhline(st_threshold, color='gray', linestyle='--', linewidth=1,
               label=f'ST threshold ({st_threshold})')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_sorted, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Sensitivity index', fontsize=12)
    ax.set_title(f'Sensitivity Analysis (Sample size={N})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, min(1.1, max(ST_sorted) + 0.1))
    plt.tight_layout()

    plot_file = os.path.join(output_dir, f'sobol_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Plot saved:    {plot_file}')

    return Si, active, inactive


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sobol SA for KMC parameters')
    parser.add_argument('--exp-data',    type=str,   required=True)
    parser.add_argument('--N',           type=int,   default=64,
                        help='Sobol base sample size (power of 2). Total runs = N*(k+2)')
    parser.add_argument('--num-sims',    type=int,   default=3,
                        help='KMC simulations per parameter combination')
    parser.add_argument('--temp',        type=float, default=250)
    parser.add_argument('--time',        type=float, default=7200)
    parser.add_argument('--length',      type=int,   default=300)
    parser.add_argument('--P-H2',        type=float, default=50)
    parser.add_argument('--threshold',   type=float, default=0.05,
                        help='ST threshold below which params are considered insensitive')
    parser.add_argument('--output-dir',  type=str,   default='sobol_results')
    args = parser.parse_args()

    sim_cfg = {
        'num_sims':      args.num_sims,
        'temp_C':        args.temp,
        'reaction_time': args.time,
        'chain_length':  args.length,
        'P_H2':          args.P_H2,
        'max_length':    30,
        'max_steps':     None,
    }

    run_sobol_sa(
        exp_data_file = args.exp_data,
        output_dir    = args.output_dir,
        N             = args.N,
        sim_cfg       = sim_cfg,
        st_threshold  = args.threshold,
    )