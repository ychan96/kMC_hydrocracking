import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any


def identify_final_products(chain_array):
    """Extract chain lengths from final state"""
    products = []
    start = 0
    
    for i in range(1, len(chain_array)):
        if chain_array[i] == 0: # Chain break found
            length = i - start
            if length > 0:
                products.append(length)
            start = i

    return products

def plot_distribution(results: List[Dict[str, Any]], max_length=30, exp_data_file='data.xlsx', sheet = "Sheet1",
                     use_mass_basis=True, save_prefix='product_distribution'):
    """
    Parameters:
        results: List of simulation result dictionaries
        max_length: Maximum chain length to display
        use_mass_basis: If True, use mass %, otherwise use count %
        save_prefix: Prefix for saved files
    """
    # Collect all products
    all_products = []
    for result in results:
        all_products.extend(result['products']) #allow duplicates
    
    if len(all_products) == 0:
        print("Warning: No products found")
        return None
    
    # Count occurrences
    counts = Counter(all_products) #count all products
    lengths = list(range(1, max_length + 1))
    
    # Calculate distribution
    if use_mass_basis:
        # Mass-based distribution (CnH2n+2: mass = 14*n + 2)
        mass_by_length = {}
        for length, count in counts.items():
            product_mass = (14 * length + 2) * count
            mass_by_length[length] = product_mass
        
        sim_total = sum(mass_by_length.values())
        percentages = [(mass_by_length.get(i, 0) / sim_total * 100) for i in lengths]
        ylabel = 'Mass Percentage (%)'
    else:
        # Count-based distribution
        total = len(all_products)
        percentages = [(counts.get(i, 0) / total * 100) for i in lengths]
        ylabel = 'Count Percentage (%)'

    #Load experimental data for comparison
    exp_data = None
    try:
        if os.path.exists(exp_data_file):
            data = pd.read_excel(exp_data_file, sheet_name=sheet)
            exp_data = data.iloc[0:max_length, 2].values #pressure setting

    except Exception as e:
        print(f"Warning: Could not load experimental data: {e}")
    
    # Create publication-quality plot
    fig, ax = plt.subplots(1,1,figsize=(12, 6))
    
    bars = ax.bar(lengths, percentages, 
                   color='steelblue', alpha=0.7, 
                   edgecolor='black', linewidth=1.5,
                   label='Simulation (Mass%)') #legend label
    if exp_data is not None and len(exp_data) > 0:
        ax.plot(lengths, exp_data, 
                color='red', linestyle='-', linewidth=2.5, marker='o', 
                markersize=6, label='Experimental Data ' + r'$\bf{(10bar, 250°C)}$', #legend label and pressure setting
                markeredgewidth=1.5, markeredgecolor='darkred')
        
    # Styling
    ax.set_xlabel('Carbon Chain Length', fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.set_xlim(0.5, max_length + 0.5)
    #ax.set_ylim(0, 10)
    if exp_data is not None and len(exp_data) > 0:
        ax.set_ylim(0, max(max(percentages) + 2, max(exp_data) + 2, 5)) #110% max for better visualization
    else:
        ax.set_ylim(0, max(percentages) + 2 if percentages else 5) #set y-limit based on data or default to 5%

    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc='upper left', fontsize=17, framealpha=0.9, edgecolor='black', fancybox=True)
    
    #subplot labels 
    #ax.text(-0.05, 1.10, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    
    # Add value labels for significant bars
    for bar, percentage in zip(bars, percentages):
        if percentage > 10.0:
            ax.text(bar.get_x() + bar.get_width()/2, percentage + 0.5, 
                   f"{percentage:.1f}%", 
                   ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with publication-quality settings
    plt.savefig(f'{save_prefix}.png', dpi=600, bbox_inches='tight', facecolor='white')
    #plt.savefig(f'{save_prefix}.pdf', dpi=600, bbox_inches='tight', facecolor='white')
    #plt.savefig(f'{save_prefix}.svg', bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    return fig

def plot_surface_coverage(sim, figsize=(8, 8), save_path=None):
    """
    Visualize metal surface coverage as a grid with chain lengths.
    
    Parameters:
        sim: KMC simulation object with m_bond and m_chain matrices
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n = sim.m_size
    
    # Color scheme
    vacant_color = '#2C5F7C'  # Dark blue
    occupied_color = '#D97742'  # Orange
    dMC_color = '#D32F2F'  # Red for double M-C bonds
    
    for i in range(n):
        for j in range(n):
            # Determine color and text
            if sim.m_bond[i, j] == 0:
                color = vacant_color
                chain_text = ''
            else:
                chain_len = abs(sim.m_chain[i, j])
                chain_text = str(chain_len)

                # Red for double bonds (negative values)
                if sim.m_chain[i, j] < 0:
                    color = dMC_color
                else:
                    color = occupied_color 
            # Draw circle
            circle = patches.Circle((j, n-1-i), 0.4, 
                                   facecolor=color, 
                                   edgecolor='white', 
                                   linewidth=2)
            ax.add_patch(circle)
            
            # Add chain length text
            if chain_text:
                ax.text(j, n-1-i, chain_text, 
                       ha='center', va='center', 
                       fontsize=14, fontweight='bold', 
                       color='white')
    
    # Set axis properties
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, n-0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    coverage_pct = sim.theta * 100
    ax.set_title(f'Surface Coverage: {coverage_pct:.1f}%\n'
                f'Time: {sim.current_time:.2f}s', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_coverage_animation(image_folder='./', output_name=None):
    """
    Create GIF animation from saved coverage plots.
    
    Parameters:
        image_folder: Folder containing coverage_*.png files
        output_name: Name of output GIF file
    """
    import imageio
    import glob
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_name is None:
        output_name = f'coverage_animation_{timestamp}.gif'

    images = []
    filenames = sorted(glob.glob(f'{image_folder}/coverage_*.png'))
    
    if not filenames:
        print(f"No coverage images found in {image_folder}")
        return
    
    for filename in filenames:
        images.append(imageio.imread(filename))
    
    imageio.mimsave(output_name, images, fps=2.0, loop=1) #duration in seconds per frame
    print(f"Animation saved: {output_name} ({len(images)} frames)")