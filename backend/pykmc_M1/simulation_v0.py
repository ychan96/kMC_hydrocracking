import os
import time

import numpy as np
import matplotlib.pyplot as plt

from pykmc_M1 import KMC
from pykmc_M1.utils import identify_final_products, plot_surface_coverage, create_coverage_animation


def run_simulation(temp_C=250, reaction_time=7200, m_size=5, P_H2=40.0, chain_length=None,
                   rate_constants=None, seed=None, max_steps=None, track_coverage=False):
    if seed is not None:
        np.random.seed(seed)

    sim = KMC(temp_C=temp_C, reaction_time=reaction_time, chain_length=chain_length,
              rate_constants=rate_constants, P_H2=P_H2, m_size=m_size)

    if track_coverage:
        os.makedirs('coverage_steps', exist_ok=True)

    steps = failed = 0
    t0 = time.time()

    while sim.current_time < sim.reaction_time and (max_steps is None or steps < max_steps):
        key, dt = sim.select_reaction(sim.update_configuration())
        if key is None:
            break

        sim.current_time += dt
        ok, info = sim.perform_reaction(key)
        if not ok or not sim.metal_surface(key, info):
            failed += 1
            continue
        steps += 1

        if track_coverage:
            plt.close(plot_surface_coverage(sim, f'coverage_steps/coverage_{steps:05d}.png'))

    return {'products': identify_final_products(sim.chain_array),
            'carbon_array': sim.carbon_array.copy(),
            'chain_array': sim.chain_array.copy(),
            'time': sim.current_time,
            'steps': steps,
            'failed': failed,
            'theta_C': sim.theta,
            'computation_time': time.time() - t0}


def run_multiple_simulations(num_sims, track_coverage=False, seed=None, **kw):
    print(f"Running {num_sims} simulations at {kw.get('temp_C', 250)} °C, P_H2 = {kw.get('P_H2', 40.0)} bar...")
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