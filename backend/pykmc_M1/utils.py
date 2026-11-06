import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def identify_final_products(chain_array):
    breaks = np.flatnonzero(np.asarray(chain_array) == 0)
    return [int(n) for n in np.diff(breaks) if n > 0]


def product_distribution(results, max_length=30, mass_basis=True):
    counts = Counter(p for r in results for p in r['products'])
    lengths = np.arange(1, max_length + 1)
    w = {n: c * ((14 * n + 2) if mass_basis else 1) for n, c in counts.items()}
    total = sum(w.values()) or 1
    return lengths, np.array([w.get(n, 0) / total * 100 for n in lengths])


def plot_distribution(results, max_length=30, exp_data_file='data.xlsx', sheet='Sheet1',
                      exp_col=11, exp_label='Experiment', mass_basis=True, ylim=None,
                      save_prefix='product_distribution'):
    lengths, pct = product_distribution(results, max_length, mass_basis)

    exp = None
    if exp_data_file and os.path.exists(exp_data_file):
        try:
            exp = pd.read_excel(exp_data_file, sheet_name=sheet).iloc[:max_length, exp_col].values
        except Exception as e:
            print(f"Warning: could not load experimental data: {e}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(lengths, pct, color='steelblue', alpha=0.7, edgecolor='black', lw=1.5,
           label=f"Simulation ({'mass' if mass_basis else 'count'}%)")
    if exp is not None:
        ax.plot(lengths[:len(exp)], exp, 'o-', color='red', lw=2.5, ms=6, mec='darkred', label=exp_label)

    ymax = ylim or max(pct.max(), np.nanmax(exp) if exp is not None else 0, 3) + 2
    ax.set(xlim=(0.5, max_length + 0.5), ylim=(0, ymax), xticks=[1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel('Carbon Chain Length', fontsize=20, fontweight='bold')
    ax.set_ylabel(f"{'Mass' if mass_basis else 'Count'} Percentage (%)", fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=18)
    ax.legend(loc='upper left', fontsize=17, edgecolor='black')
    fig.tight_layout()
    fig.savefig(f'{save_prefix}.png', dpi=600, bbox_inches='tight', facecolor='white')
    return fig


def plot_surface_coverage(sim, save_path=None, figsize=(8, 8)):
    n = sim.m_size
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n):
        for j in range(n):
            v = sim.m_chain[i, j]
            color = '#2C5F7C' if sim.m_bond[i, j] == 0 else ('#D32F2F' if v < 0 else '#D97742')
            ax.add_patch(Circle((j, n - 1 - i), 0.4, facecolor=color, edgecolor='white', lw=2))
            if sim.m_bond[i, j]:
                ax.text(j, n - 1 - i, str(abs(v)), ha='center', va='center',
                        fontsize=14, fontweight='bold', color='white')
    ax.set(xlim=(-0.5, n - 0.5), ylim=(-0.5, n - 0.5), aspect='equal')
    ax.axis('off')
    ax.set_title(f'Surface Coverage: {sim.theta * 100:.1f}%\nTime: {sim.current_time:.2f}s',
                 fontsize=16, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_coverage_animation(image_folder='coverage_steps', output_name='coverage.gif', fps=2):
    import imageio.v2 as imageio
    files = sorted(glob.glob(os.path.join(image_folder, 'coverage_*.png')))
    if not files:
        print(f"No coverage images in {image_folder}")
        return
    imageio.mimsave(output_name, [imageio.imread(f) for f in files], fps=fps, loop=0)
    print(f"Animation saved: {output_name} ({len(files)} frames)")