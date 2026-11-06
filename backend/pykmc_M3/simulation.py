import os
import time

import numpy as np
import matplotlib.pyplot as plt

from pykmc_M3 import KMC
from pykmc_M3.utils import identify_final_products, plot_surface_coverage, create_coverage_animation


def run_simulation(temp_C=250, reaction_time=7200, init_len=None, k_const=None,
                   P_H2=25, theta_H=0.61, catalyst_config=None, seed=None,
                   max_steps=None, verbose=False, track_coverage=False):
    if seed is not None:
        np.random.seed(seed)                      # reactions/base use the global RNG

    sim = KMC(temp_C=temp_C, reaction_time=reaction_time, init_len=init_len,
              k_const=k_const, P_H2=P_H2, theta_H=theta_H,
              catalyst_config=catalyst_config, seed=seed)

    if track_coverage:
        os.makedirs('coverage_steps', exist_ok=True)

    steps = failed = 0
    t0 = time.time()

    while sim.current_time < sim.reaction_time and (max_steps is None or steps < max_steps):
        key, dt = sim.select_reaction(sim.update_configuration())
        if key is None:
            break

        sim.current_time += dt                    # time advances even on a refused event
        if not sim.perform_reaction(key):
            failed += 1
            continue
        steps += 1

        if verbose:
            print(f"step {steps:5d}  t={sim.current_time:9.2f}s  {key:12s}  "
                  f"products={identify_final_products(sim.chain_array)}")
        if track_coverage:
            fig = plot_surface_coverage(sim, f'coverage_steps/coverage_{steps:05d}.png')
            plt.close(fig)

    return {'products': identify_final_products(sim.chain_array),
            'carbon_array': sim.carbon_array.copy(),
            'chain_array': sim.chain_array.copy(),
            'time': sim.current_time,
            'steps': steps,
            'failed': failed,
            'theta_C': sim.surface.theta,
            'computation_time': time.time() - t0}


def run_multiple_simulations(num_sims, track_coverage=False, seed=None, **kw):
    print(f"Running {num_sims} simulations at {kw.get('temp_C', 250)} °C...")
    t0, results = time.time(), []
    for i in range(num_sims):
        r = run_simulation(track_coverage=track_coverage and i == 0,
                           seed=None if seed is None else seed + i, **kw)
        results.append(r)
        print(f"Sim {i + 1}: {len(r['carbon_array'])} C -> {len(r['products'])} products, "
              f"{r['steps']} steps ({r['failed']} refused), {r['computation_time']:.2f}s")

    if track_coverage:
        create_coverage_animation('coverage_steps')
    print(f"All {num_sims} simulations done in {time.time() - t0:.2f}s")
    return results


def calculate_conversion(results):
    n_in = len(results)
    n_out = sum(len(r['products']) for r in results)
    conv = n_out / n_in if n_in else 0.0
    print(f"Conversion: {n_out} products from {n_in} chains ({conv:.2f} fragments/chain)")
    return {'total_reactants': n_in, 'total_products': n_out, 'conversion': conv}


if __name__ == '__main__':
    from pykmc_M3.utils import plot_distribution
    res = run_multiple_simulations(5, temp_C=250, reaction_time=7200, seed=0)
    calculate_conversion(res)
    plot_distribution(res)