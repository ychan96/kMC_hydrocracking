#!/usr/bin/env python
"""
KMC Simulation for Hydrocarbon Chain Reactions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import argparse
import json
from datetime import datetime
from collections import Counter

from backend.kmc_v3.simulation import run_simulation, run_multiple_simulations
from backend.kmc_v3.utils import plot_distribution


def main():
    parser = argparse.ArgumentParser(
        description='Run KMC simulations for hydrocarbon chain reactions')
    parser.add_argument('--temp',       type=float, default=250,      help='Temperature (°C)')
    parser.add_argument('--time',       type=float, default=7200,     help='Reaction time (s)')
    parser.add_argument('--length',     type=int,   default=None,      help='Initial chain length')
    parser.add_argument('--sims',       type=int,   default=10,       help='Number of simulations')
    parser.add_argument('--P-H2',       type=float, default=50,       help='H2 pressure (bar)')
    parser.add_argument('--max-steps',  type=int,   default=None,     help='Max steps per simulation')
    parser.add_argument('--exp-data',   type=str,   default='data.xlsx', help='Experimental data file')
    parser.add_argument('--verbose',    action='store_true',          help='Print step-by-step output')
    parser.add_argument('--output-dir', type=str,   default='results', help='Output directory')
    parser.add_argument('--gui',        action='store_true',          help='Launch live GUI instead of batch run')
    args = parser.parse_args()

    # ── GUI mode ──────────────────────────────────────────────────
    if args.gui:
        from backend.kmc_v3.simulation import launch_gui
        launch_gui()
        return

    # ── Batch mode ────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting {args.sims} KMC simulations at {args.temp}°C ...")
    print(f"chain_length={args.length}  time={args.time}s  P_H2={args.P_H2} bar")

    results = run_multiple_simulations(
        num_sims      = args.sims,
        temp_C        = args.temp,
        reaction_time = args.time,
        chain_length  = args.length,
        P_H2          = args.P_H2,
        verbose       = args.verbose,
        max_steps     = args.max_steps,
    )

    # ── Aggregate products ────────────────────────────────────────
    all_products   = []
    for result in results:
        all_products.extend(result['products'])

    if not all_products:
        print("No products found — simulation may have stalled.")
        return

    product_counts = Counter(all_products)
    total_products = len(all_products)

    # ── Save JSON summary ─────────────────────────────────────────
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(args.output_dir, f'summary_{timestamp}.json')

    summary_data = {
        'parameters': {
            'temperature':    args.temp,
            'reaction_time':  args.time,
            'chain_length':   args.length,
            'P_H2':           args.P_H2,
            'num_simulations': args.sims,
        },
        'timestamp': timestamp,
        'results_summary': [
            {
                'products':         r['products'],
                'steps':            r['steps'],
                'time':             r['time'],
                'computation_time': r['computation_time'],
            }
            for r in results
        ],
        'product_distribution': {
            str(k): v for k, v in sorted(product_counts.items())
        },
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    # ── Print distribution table ──────────────────────────────────
    print("\nProduct Distribution:")
    print("Carbon # | Count | Mass%")
    print("-" * 32)
    mass_total = sum((14 * L + 2) * c for L, c in product_counts.items())
    for length in sorted(product_counts.keys()):
        count   = product_counts[length]
        mass_pc = (14 * length + 2) * count / mass_total * 100
        print(f"C{length:<7} | {count:<5} | {mass_pc:.2f}%")

    # ── Selectivity ───────────────────────────────────────────────
    max_c    = max(product_counts.keys())
    c1_c4    = sum(product_counts.get(i, 0) for i in range(1,  5))
    c5_c12   = sum(product_counts.get(i, 0) for i in range(5,  13))
    c13_plus = sum(product_counts.get(i, 0) for i in range(13, max_c + 1))

    print("\nSelectivity (count basis):")
    print(f"  C1–C4  : {c1_c4    / total_products * 100:.2f}%")
    print(f"  C5–C12 : {c5_c12   / total_products * 100:.2f}%")
    print(f"  C13+   : {c13_plus / total_products * 100:.2f}%")

    # ── Plot ──────────────────────────────────────────────────────
    print("\nGenerating distribution plot ...")
    try:
        plot_distribution(
            results,
            max_length    = 30,
            exp_data_file = args.exp_data,
            use_mass_basis= True,
            save_prefix   = os.path.join(args.output_dir, f'comparison_{timestamp}'),
        )
        print(f"Plot saved to {args.output_dir}/comparison_{timestamp}.png")
    except Exception as e:
        print(f"Warning: could not create plot — {e}")

    print("\nSimulation complete!")


if __name__ == '__main__':
    main()