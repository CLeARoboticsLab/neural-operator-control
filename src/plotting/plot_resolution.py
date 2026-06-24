"""
Figure 5: Task resolution invariance — 2x2 grid.

Plots median MSE with IQR shading for each environment.
Blue x-ticks = seen during training, red = unseen.

Usage:
    python src/plotting/plot_resolution.py \
        --results outputs/results/resolution \
        --output outputs/figures/figure5.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_SEED = 42

# Style
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
})

ENV_FILES = [
    'p2p_cost.npz',
    'p2p_dynamics.npz',
    'quadrotor.npz',
    'obstacle_avoidance.npz',
]


def plot_single_env(ax, data_path, title_override=None, show_legend=True):
    """Plot one environment's resolution results on a single axis."""
    d = np.load(str(data_path), allow_pickle=True)
    x_values = list(d['x_values'])
    train_values = set(int(v) for v in d['train_values'])
    env_name = str(d['env'])
    x_label = str(d['x_label'])

    medians, q25s, q75s = [], [], []
    for xv in x_values:
        errors = d[f'errors_{xv}']
        if len(errors) == 0:
            medians.append(np.nan)
            q25s.append(np.nan)
            q75s.append(np.nan)
        else:
            medians.append(np.median(errors))
            q25s.append(np.percentile(errors, 25))
            q75s.append(np.percentile(errors, 75))

    positions = np.arange(len(x_values))

    ax.plot(positions, medians, 's-', color='#2ca02c', linewidth=2, markersize=7, label='Median')
    ax.fill_between(positions, q25s, q75s, alpha=0.3, color='#2ca02c', label='IQR (25-75%)')

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha='right')

    # Color ticks: blue = seen during training, red = unseen
    for tick, val in zip(ax.xaxis.get_major_ticks(), x_values):
        color = 'blue' if val in train_values else 'red'
        tick.label1.set_color(color)
        tick.label1.set_fontweight('bold')
        tick.tick1line.set_color(color)
        tick.tick1line.set_markeredgewidth(2)

    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_title(title_override or env_name, fontweight='bold', fontsize=14)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    if show_legend:
        ax.legend(loc='upper right')


def main():
    parser = argparse.ArgumentParser(description="Plot task resolution invariance (Figure 5)")
    parser.add_argument("--results", default="outputs/results/resolution")
    parser.add_argument("--output", default="outputs/figures/figure5.pdf")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')

    for i, (ax, env_file) in enumerate(zip(axes.flat, ENV_FILES)):
        fpath = results_dir / env_file
        if fpath.exists():
            plot_single_env(ax, fpath, show_legend=(i == 0))
        else:
            ax.text(0.5, 0.5, f'{env_file}\nnot found', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_title(env_file.replace('.npz', ''))

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
