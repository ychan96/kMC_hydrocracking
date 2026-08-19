#!/usr/bin/env python
"""
KMC Simulation for Hydrocarbon Chain Reactions
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from datetime import datetime
from collections import Counter
from backend.kmc_v1.simulation_v4 import run_simulation, run_multiple_simulations, calculate_conversion
from backend.kmc_v1.utils import plot_distribution

def main():
    parser = argparse.ArgumentParser(description='Run KMC simulations for hydrocarbon chain reactions')
    parser.add_argument('--temp', type=float, default=250, help='Temperature in Celsius')
    parser.add_argument('--time', type=float, default=7200, help='Reaction time in seconds')
    parser.add_argument('--length', type=int, default=None, help='Initial chain length (None=random)')
    parser.add_argument('--sims', type=int, default=10, help='Number of simulations')
    parser.add_argument('--msize', type=int, default=5, help='Metal surface grid size')
    parser.add_argument('--exp-data', type=str, default='data.xlsx', help='Experimental data file')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--coverage', action='store_true', help='Track coverage and create GIF')
    args = parser.parse_args()

    from backend.kmc_v1.init_v4 import BaseKineticMC
    rate_constants = BaseKineticMC().k_const  # load defaults

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting {args.sims} KMC simulations at {args.temp}°C...")
    print(f"Parameters: time={args.time}s, length={args.length}, m_size={args.msize}")

    # Run simulations
    results = run_multiple_simulations(
        num_sims=args.sims,
        temp_C=args.temp,
        reaction_time=args.time,
        m_size=args.msize,
        chain_length=args.length,
        rate_constants=None,  # Uses defaults from init.py
        verbose=args.verbose,
        track_coverage=args.coverage 
    )

    total_reactants = len(results)
    total_products  = sum(len(r['products']) for r in results)
    conversion      = total_products / total_reactants if total_reactants > 0 else 0.0

    conversion_data = {'total_reactants': total_reactants, 'total_products': total_products, 'conversion': conversion}

    # Collect products
    all_products = []
    for result in results:
        all_products.extend(result['products'])
    
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
            'm_size': args.msize,
            'rate_constants': rate_constants or 'defaults',
            'conversion_data': conversion_data
        },
        'timestamp': timestamp,
        #'results_summary': [
        #    {
        #        'steps': result['steps'],
        #        'time': result['time'],
        #        'computation_time': result['computation_time']
        #    } for result in results
        #],
        'product_distribution': {str(k): v for k, v in sorted(product_counts.items())}
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")
    
    # Print statistics
    print("\nProduct Distribution:")
    print("Carbon # | Count | Percentage")
    print("-" * 30)
    
    for length in sorted(product_counts.keys()):  # Show first 20
        count = product_counts[length]
        percentage = (count / total_products) * 100
        print(f"C{length:<7} | {count:<5} | {percentage:.2f}%")
    
    # Selectivity
    max_carbon = max(product_counts.keys()) if product_counts else 0
    c1_c4 = sum(product_counts.get(i, 0) for i in range(1, 5))
    c5_c12 = sum(product_counts.get(i, 0) for i in range(5, 13))
    c13_plus = sum(product_counts.get(i, 0) for i in range(13, max_carbon + 1))
    
    print("\nSelectivity:")
    print(f"C1-C4:  {c1_c4/total_products*100:.2f}%")
    print(f"C5-C12: {c5_c12/total_products*100:.2f}%")
    print(f"C13+:   {c13_plus/total_products*100:.2f}%")

    #Conversion
    print(f"\nConversion: {total_products} products from {total_reactants} chains "
          f"({conversion:.2f} fragments/chain)")
    
    # Generate plot
    print("\nGenerating comparison plot...")
    try:
        fig = plot_distribution(
            results, 
            max_length=30,
            exp_data_file=args.exp_data,
            use_mass_basis=True,
            save_prefix=os.path.join(args.output_dir, f"comparison_{timestamp}")
        )
        print(f"Plot saved to {args.output_dir}/comparison_{timestamp}.png")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()