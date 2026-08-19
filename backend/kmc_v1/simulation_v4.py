import numpy as np
import time
import os
from typing import Optional, Union
import matplotlib.pyplot as plt
from kmc_v1.init_v4 import BaseKineticMC
from kmc_v1.count_sites_v4 import ConfigMixin
from kmc_v1.reactions_v4 import ReactionMixin
from kmc_v1.coverage import CoverageMixin
from kmc_v1.utils import create_coverage_animation, identify_final_products

class KMC(BaseKineticMC, ConfigMixin, ReactionMixin, CoverageMixin):
    pass

def run_simulation(
    temp_C: float = 250,
    reaction_time: float = 7200,
    m_size: int = 5,
    chain_length: Optional[Union[int, np.ndarray]] = None,
    rate_constants: dict = None,
    verbose: bool = False,
    track_coverage: bool = False,
    max_steps: Optional[int] = None
):
    # Initialize
    sim = KMC(
        temp_C=temp_C,
        reaction_time=reaction_time, 
        chain_length=chain_length, 
        m_size=m_size,
        rate_constants=rate_constants,
    )

    if track_coverage:
        coverage_dir = 'coverage_steps'
        os.makedirs(coverage_dir, exist_ok=True)

    history = [] if verbose else None
    steps_performed = 0
    start_time = time.time()

    # Reaction name mapping for verbose output
    reaction_names = {
        'dmc_i': 'Internal Double M-C Formation',
        'dmc_t': 'Terminal Double M-C Formation',
        'crk_i': 'Internal Cracking',
        'crk_t': 'Terminal Cracking',
    }
    for n in range(1, 31):
        reaction_names[f'ads_c{n}'] = f'C{n} Adsorption'
        reaction_names[f'des_c{n}'] = f'C{n} Desorption'
    reaction_names['ads_c31plus'] = 'C31+ Adsorption'
    reaction_names['des_c31plus'] = 'C31+ Desorption'

    # Main loop
    while sim.current_time < sim.reaction_time and (max_steps is None or steps_performed < max_steps):
        # Count available sites
        counts = sim.update_configuration()
        
        # Select reaction
        reaction_key, dt = sim.select_reaction(counts)
        
        if reaction_key is None:
            break
        
        # Perform reaction
        success, chain_info = sim.perform_reaction(reaction_key)
        
        if success:
            # Update surface
            sim.metal_surface(reaction_key, chain_info)
            # Update time
            sim.current_time += dt
            steps_performed += 1

            if track_coverage:
                from kmc_v1.utils import plot_surface_coverage
                fig = plot_surface_coverage(sim, save_path=None)
                plt.savefig(f'{coverage_dir}/coverage_step_{steps_performed:04d}.png', dpi=150)
                plt.close(fig)
            
            if verbose:
                products = identify_final_products(sim.chain_array)
                history.append({
                    'step': steps_performed,
                    'time': sim.current_time,
                    'reaction': reaction_names.get(reaction_key, reaction_key), #get(reaction_key, default)
                    'carbon_array': sim.carbon_array.copy(),
                    'chain_array': sim.chain_array.copy(),
                    'products': products
                })
                #if reaction_key.startswith("crk_"):
                print(f"Step {steps_performed}, Time: {sim.current_time:.2f}s")
                print(f"Reaction: {reaction_names.get(reaction_key, reaction_key)}")
                print(f"Carbon array: {sim.carbon_array}")
                print(f"Chain array: {sim.chain_array}")
                print(f"Products after reaction: {products}")
                print("-" * 50)

                # Plot surface coverage
                from kmc_v1.utils import plot_surface_coverage
                fig = plot_surface_coverage(sim, save_path=None)
                plt.show(block=False)
                plt.pause(0.1)  # Pause to update the plot

                #Pauses here
                input("Press Enter to continue...")  
                plt.close(fig)

    elapsed_time = time.time() - start_time
    products = identify_final_products(sim.chain_array)

    if verbose:
        print(f"Simulation completed in {elapsed_time:.2f} seconds")
        print(f"Final time: {sim.current_time:.2f}s, Steps: {steps_performed}")
        print(f"Final products: {products}")
        

    return {
        'carbon_array': sim.carbon_array.copy(),
        'chain_array': sim.chain_array.copy(),
        'time': sim.current_time,
        'history': history, 
        'products': products,
        'steps':steps_performed,
        'computation_time':elapsed_time
    }

def run_multiple_simulations(
        num_sims: int,
        temp_C: float,
        reaction_time: float,
        m_size: int,
        chain_length: Optional[Union[int, np.ndarray]] = None,
        rate_constants: np.ndarray = None,
        verbose: bool = False,
        track_coverage: bool = False,
        max_steps: Optional[int] = None
):  

    results = []
    total_start_time = time.time()

    print(f"Running {num_sims} simulations at {temp_C}°C...")

    for i in range(num_sims):
        if verbose:
            print(f"\nSimulation {i+1}/{num_sims}")
        
        #only first simulation
        track = (i == 0) and track_coverage

        result = run_simulation(
            temp_C=temp_C,
            reaction_time=reaction_time,
            m_size=m_size,
            chain_length=chain_length,
            rate_constants=rate_constants,
            verbose=verbose,
            track_coverage=track,
            max_steps=max_steps 
        )

        results.append(result)

        """# ── per-simulation product filter ─────────────────────────
        if any(p < 2 or p > 150 for p in result['products']):
            print(f"  [abort] Simulation {i+1} outlier products detected — aborting call")
            return None   # signal to optimizer to skip this candidate"""

        print(f"Simulation {i+1}: {len(result['carbon_array'])} carbon chains"
              f"→ {len(result['products'])} products in {result['steps']} steps"
              f"({result['computation_time']:.2f}s)")
    

    if track_coverage and os.path.exists('coverage_steps'):
        create_coverage_animation('coverage_steps')
    
    total_elapsed_time = time.time() - total_start_time
    print(f"\nAll {num_sims} simulations completed in {total_elapsed_time:.2f} seconds")

    return results

def calculate_conversion(results: list) -> dict:
    total_reactants = len(results)
    total_products  = sum(len(r['products']) for r in results)
    conversion      = total_products / total_reactants if total_reactants > 0 else 0.0

    print(f"\nConversion: {total_products} products from {total_reactants} chains "
          f"({conversion:.2f} fragments/chain)")

    return {'total_reactants': total_reactants,
            'total_products':  total_products,
            'conversion':      conversion}