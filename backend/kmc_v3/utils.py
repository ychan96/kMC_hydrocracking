import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------------
# Product identification
# ------------------------------------------------------------------

def identify_final_products(chain_array: np.ndarray) -> list:
    """
    Parse chain_array (0=boundary, 1=bond) into a list of fragment lengths.

    chain_array has length N_total + 1; boundaries at index 0 and N_total are 0.
    Returns a flat list of fragment lengths (one entry per fragment, duplicates allowed).
    """
    products = []
    start    = 0

    for i in range(1, len(chain_array)):
        if chain_array[i] == 0:
            length = i - start
            if length > 0:
                products.append(length)
            start = i

    return products


# ------------------------------------------------------------------
# Product distribution plot
# ------------------------------------------------------------------

def plot_distribution(
    results:        List[Dict[str, Any]],
    max_length:     int             = 30,
    exp_data_file:  str             = 'data.xlsx',
    sheet:          str             = 'Sheet1',
    use_mass_basis: bool            = True,
    save_prefix:    str             = 'product_distribution',
) -> Optional[plt.Figure]:
    """
    Bar chart of simulated product distribution vs. optional experimental data.

    Parameters
    ----------
    results       : list of run_simulation() return dicts
    max_length    : maximum carbon chain length to display
    exp_data_file : Excel file with experimental reference data
    sheet         : sheet name in the Excel file
    use_mass_basis: True → mass %, False → count %
    save_prefix   : filename prefix for saved PNG
    """
    all_products = []
    for result in results:
        all_products.extend(result['products'])

    if not all_products:
        print("Warning: No products found.")
        return None

    counts  = Counter(all_products)
    lengths = list(range(1, max_length + 1))

    if use_mass_basis:
        mass_by_length = {L: (14 * L + 2) * c for L, c in counts.items()}
        total          = sum(mass_by_length.values())
        percentages    = [mass_by_length.get(i, 0) / total * 100 for i in lengths]
        ylabel         = 'Mass Percentage (%)'
    else:
        total       = len(all_products)
        percentages = [counts.get(i, 0) / total * 100 for i in lengths]
        ylabel      = 'Count Percentage (%)'

    # Optional experimental overlay
    exp_data = None
    try:
        if os.path.exists(exp_data_file):
            data     = pd.read_excel(exp_data_file, sheet_name=sheet)
            exp_data = data.iloc[0:max_length, 4].values
    except Exception as e:
        print(f"Warning: Could not load experimental data: {e}")

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(lengths, percentages,
                  color='steelblue', alpha=0.7,
                  edgecolor='black', linewidth=1.5,
                  label='Simulation (Mass%)' if use_mass_basis else 'Simulation (Count%)')

    if exp_data is not None and len(exp_data) > 0:
        ax.plot(lengths, exp_data,
                color='red', linestyle='-', linewidth=2.5,
                marker='o', markersize=6,
                label=r'Experimental Data $\bf{(40bar)}$',
                markeredgewidth=1.5, markeredgecolor='darkred')
    
    #labels
    ax.set_xlabel('Carbon Number', fontsize=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=20, fontweight='bold')
    ax.set_xlim(0.5, max_length + 0.5)

    #y-axis limit: max of sim and exp + 2% margin, with a minimum of 5%
    y_max = max(max(percentages, default=0),
                max(exp_data, default=0) if exp_data is not None else 0) + 2
    ax.set_ylim(0, max(y_max, 5))

    #x-ticks and styling
    ax.set_xticks(lengths[::2]) #show every 2nd label for readability
    ax.tick_params(axis='both', which='major', labelsize=18) #major ticks only 
    ax.legend(loc='upper left', fontsize=17, framealpha=0.9,
              edgecolor='black', fancybox=True)

    # Annotate bars with percentages if >10%
    for bar, pct in zip(bars, percentages):
        if pct > 10.0:
            ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    plt.tight_layout() #automatic adjustment to prevent clipping
    plt.savefig(f'{save_prefix}.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()

    return fig


# ------------------------------------------------------------------
# Surface coverage visualisation  (top-view + slab cross-section)
# ------------------------------------------------------------------
 
def plot_surface_coverage(sim, figsize=(14, 7), save_path=None) -> plt.Figure:
    """
    Two-panel figure:
        Left  — top-view of the Pt(111) surface (atop C sites + hollow H sites)
        Right — cross-section slab model showing Pt layers, adsorbed C and H
 
    Colour key
    ----------
    #2C5F7C  — vacant atop C site
    #D97742  — single M-C
    #D32F2F  — dMC pair
    #3CB371  — occupied hollow H site (small plain circle, no label)
    #C8E6C9  — vacant hollow H site (faint, small circle)
    #B8B8C8  — Pt metal atom (slab)

    Example;

    surface.get_coordinates_array() output:
    array([
    [0.00, 0.00, 0.0],  # Site 0 (Atop)
    [0.00, 3.92, 0.0],  # Site 1 (Atop)
    [3.92, 0.00, 0.0],  # Site 2 (Atop)
    [3.92, 3.92, 0.0],  # Site 3 (Atop)
    [1.96, 1.96, 0.0]   # Site 4 (Hollow - center of the square)
    ])
    """
    surface    = sim.surface
    coords     = surface.get_coordinates_array()   # matrix of (n_total_sites, 3); row: site index, col: x,y,z coordinates 
    c_indices  = surface.c_site_indices            # list[int]  atop
    h_indices  = surface.h_site_indices            # list[int]  hollow
 
    c_coords   = coords[c_indices]                 # (n_c, 3) = x,y,z coordinates of ATOP sites
    h_coords   = coords[h_indices]                 # (n_h, 3) = x,y,z coordinates of HOLLOW sites
    occ        = sim.occupancy                     # (n_c, 1) = state of C sites: 0=vacant, 1=single M-C, 2=dMC
    chain_info = sim.chain_at_site                 # (n_c, 1) = fragment length at C site (0 if vacant), negative if dMC (e.g., -3 for a C3 fragment in dMC state)
    h_occ      = sim.h_occupancy                   # (n_h, 1) = state of H sites: 0=vacant, 1=occupied
 
    C_COLOR  = {0: '#2C5F7C', 1: '#D97742', 2: '#D32F2F'}
    C_LABEL  = {0: 'Vacant C (atop)', 1: 'Single M-C', 2: 'dMC'}
    H_OCC    = '#3CB371'
    H_VAC    = '#C8E6C9'
    PT_COLOR = '#9E9EAF'
 
    fig, ax_top = plt.subplots(figsize=figsize)
 
    # ── Left: top-view ──────────────────────────────────────────────
 
    # H HOLLOW sites — small plain circles (no label)
    h_s = 40   # marker size for H (small)
    ax_top.scatter(h_coords[h_occ == 0, 0], h_coords[h_occ == 0, 1],        #mask: vacant , 0 -> x /1 -> y: pulls x and y coordinates of vacant H sites
                   s=h_s, c=H_VAC, edgecolors='#9DC89E', linewidths=0.5,  #size, color, edge color, line width
                   label='H hollow (vacant)', zorder=2)                     #label for legend, stacking order(higher zorder -> plotted on top)
    ax_top.scatter(h_coords[h_occ == 1, 0], h_coords[h_occ == 1, 1],
                   s=h_s, c=H_OCC, edgecolors='#256D3E', linewidths=0.5,
                   label='H hollow (occupied)', zorder=2)
 
    # C ATOP sites — larger circles with fragment-length labels
    #Shapes
    for state in (0, 1, 2):
        mask = (occ == state)
        if not np.any(mask): #skip if no sites in current state
            continue
        ax_top.scatter(c_coords[mask, 0], c_coords[mask, 1], #pulls x and y coordinates of sites in this state
                       s=280, c=C_COLOR[state], edgecolors='white',
                       linewidths=1.2, label=C_LABEL[state], zorder=3)
    #labels 
    for local_i in range(len(c_indices)):
        if occ[local_i] != 0:
            ax_top.text(c_coords[local_i, 0], c_coords[local_i, 1],
                        str(abs(chain_info[local_i])),
                        ha='center', va='center',
                        fontsize=6, fontweight='bold', color='white', zorder=4)
 
    n_c          = len(c_indices)          # total number of ATOP sites
    n_c_occ      = int(np.sum(occ > 0))    # number of occupied ATOP sites (single M-C or dMC)
    n_h_occ      = int(np.sum(h_occ == 1)) # number of occupied HOLLOW sites
    coverage_pct = n_c_occ / n_c * 100 if n_c else 0.0
    theta_H_pct  = n_h_occ / len(h_indices) * 100 if h_indices else 0.0
 
    ax_top.set_aspect('equal')
    ax_top.axis('off')
    ax_top.set_title(
        f'Top view  —  θ_C: {coverage_pct:.1f}%   θ_H: {theta_H_pct:.1f}%\n'
        f't = {sim.current_time:.4f} s',
        fontsize=12, fontweight='bold', pad=10,
    )
    ax_top.legend(loc='upper right', fontsize=8, framealpha=0.9,
                  edgecolor='gray', fancybox=True, markerscale=0.9)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
 
    return fig
 
 
# ------------------------------------------------------------------
# GIF animation from saved frames
# ------------------------------------------------------------------
 
def create_coverage_animation(
    image_folder: str          = './',
    output_name:  Optional[str] = None,
    fps:          float         = 2.0,
):
    """
    Stitch saved coverage_*.png frames into a GIF.
 
    Parameters
    ----------
    image_folder : directory containing coverage_*.png files
    output_name  : output filename; auto-timestamped if None
    fps          : frames per second
    """
    import imageio
    from datetime import datetime
 
    if output_name is None:
        ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'coverage_animation_{ts}.gif'
 
    filenames = sorted(glob.glob(os.path.join(image_folder, 'coverage_*.png')))
 
    if not filenames:
        print(f"No coverage images found in '{image_folder}'.")
        return
 
    images = [imageio.imread(f) for f in filenames]
    imageio.mimsave(output_name, images, fps=fps, loop=1)
    print(f"Animation saved: {output_name}  ({len(images)} frames)")
 