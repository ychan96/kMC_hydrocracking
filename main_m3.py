#!/usr/bin/env python
import os
import sys
import json
import argparse
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.pykmc_M3 import BaseKineticMC
from backend.pykmc_M3.utils import plot_distribution
from backend.pykmc_M3.simulation import run_multiple_simulations, calculate_conversion


def main():
    parser = argparse.ArgumentParser(description='Run KMC simulations for hydrocarbon chain reactions')
    parser.add_argument('--temp', type=float, default=250, help='Temperature in Celsius')
    parser.add_argument('--time', type=float, default=7200, help='Reaction time in seconds')
    parser.add_argument('--length', type=int, default=None, help='Initial chain length (None=random)')
    parser.add_argument('--sims', type=int, default=10, help='Number of simulations')
    parser.add_argument('--msize', type=int, default=5, help='Metal surface size (msize x msize atoms)')
    parser.add_argument('--facet', type=str, default='111', help="Surface facet ('111', '0001', '100')")
    parser.add_argument('--pressure', type=float, default=25, help='H2 pressure in bar')
    parser.add_argument('--theta-H', type=float, default=0.61, help='H coverage (<0 -> Langmuir from --pressure)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None=random)')
    parser.add_argument('--exp-data', type=str, default='data.xlsx', help='Experimental data file')
    parser.add_argument('--exp-col', type=int, default=2, help='Column index of experimental data')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--coverage', action='store_true', help='Track coverage and create GIF')
    args = parser.parse_args()

    theta_H = None if args.theta_H < 0 else args.theta_H
    catalyst_config = {'facet': args.facet, 'nx': args.msize, 'ny': args.msize}

    base = BaseKineticMC(temp_C=args.temp, P_H2=args.pressure, theta_H=theta_H,
                         catalyst_config=catalyst_config, init_len=1)
    rate_constants = base.k_const

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting {args.sims} KMC simulations at {args.temp}°C...")
    print(f"Parameters: time={args.time}s, length={args.length}, surface={args.facet} "
          f"{args.msize}x{args.msize}, P_H2={args.pressure} bar, theta_H={base.theta_H:.4f}")

    # Run simulations
    results = run_multiple_simulations(
        num_sims=args.sims,
        temp_C=args.temp,
        reaction_time=args.time,
        init_len=args.length,
        k_const=None,  # Uses defaults from base.py
        P_H2=args.pressure,
        theta_H=theta_H,
        catalyst_config=catalyst_config,
        seed=args.seed,
        verbose=args.verbose,
        track_coverage=args.coverage
    )

    conversion_data = calculate_conversion(results)
    total_reactants = conversion_data['total_reactants']
    conversion      = conversion_data['conversion']

    # Collect products
    all_products = [p for r in results for p in r['products']]
    product_counts = Counter(all_products)
    total_products = len(all_products)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")

    summary_data = {
        'parameters': {
            'temperature': args.temp,
            'reaction_time': args.time,
            'chain_length': args.length,
            'num_simulations': args.sims,
            'catalyst_config': catalyst_config,
            'P_H2': args.pressure,
            'theta_H': base.theta_H,
            'seed': args.seed,
            'rate_constants': rate_constants or 'defaults',
            'conversion_data': conversion_data
        },
        'timestamp': timestamp,
        'results_summary': [
            {
                'steps': r['steps'],
                'failed': r['failed'],
                'time': r['time'],
                'theta_C': r['theta_C'],
                'computation_time': r['computation_time']
            } for r in results
        ],
        'product_distribution': {str(k): v for k, v in sorted(product_counts.items())}
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to {summary_file}")

    if total_products == 0:
        print("No products found.")
        return

    # Print statistics
    print("\nProduct Distribution:")
    print("Carbon # | Count | Percentage")
    print("-" * 30)

    for length in sorted(product_counts.keys()):
        count = product_counts[length]
        percentage = (count / total_products) * 100
        print(f"C{length:<7} | {count:<5} | {percentage:.2f}%")

    # Selectivity
    c1_c4    = sum(c for n, c in product_counts.items() if n <= 4)
    c5_c12   = sum(c for n, c in product_counts.items() if 5 <= n <= 12)
    c13_plus = sum(c for n, c in product_counts.items() if n >= 13)

    print("\nSelectivity:")
    print(f"C1-C4:  {c1_c4/total_products*100:.2f}%")
    print(f"C5-C12: {c5_c12/total_products*100:.2f}%")
    print(f"C13+:   {c13_plus/total_products*100:.2f}%")

    # Conversion
    print(f"\nConversion: {total_products} products from {total_reactants} chains "
          f"({conversion:.2f} fragments/chain)")

    # Generate plot
    print("\nGenerating comparison plot...")
    try:
        plot_distribution(
            results,
            max_length=30,
            exp_data_file=args.exp_data,
            exp_col=args.exp_col,
            mass_basis=True,
            save_prefix=os.path.join(args.output_dir, f"comparison_{timestamp}")
        )
        print(f"Plot saved to {args.output_dir}/comparison_{timestamp}.png")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()