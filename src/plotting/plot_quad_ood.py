"""
Figure 8: OOD Quadrotor adaptation — trajectory panels + bar chart.

Left two panels: representative rollout trajectories for SetONet-FT and
SetONet-Meta-Full adapting to a shifted goal location.
Right panel: bar plot comparing relative L2 error across all methods.

Usage (matches Makefile):
    python src/plotting/plot_quad_ood.py \
        --results outputs/results/quadrotor_ood \
        --output outputs/figures/figure8.pdf
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ── Style ──

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

TRAJ_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

BAR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2']

METHOD_ORDER = ['SetONet', 'SetONet-FT', 'Full-Branch', 'Last-Branch',
                'Last-Both', 'SetONet-Meta', 'SetONet-Meta-Full']


def plot_trajectory_panel(ax, trajectories, training_goal, ood_goal, title,
                          max_trajs=5):
    """Plot rollout trajectories in (y, z) space."""
    n = min(max_trajs, len(trajectories))
    for i in range(n):
        traj = trajectories[i]
        ax.plot(traj[:, 0], traj[:, 1], color=TRAJ_COLORS[i % len(TRAJ_COLORS)],
                linewidth=1.8, alpha=0.9)
        # Start marker
        ax.scatter(traj[0, 0], traj[0, 1], color=TRAJ_COLORS[i % len(TRAJ_COLORS)],
                   s=50, zorder=5, edgecolors='white', linewidths=0.5)
        # End marker
        ax.scatter(traj[-1, 0], traj[-1, 1], color=TRAJ_COLORS[i % len(TRAJ_COLORS)],
                   s=50, zorder=5, marker='o', edgecolors='white', linewidths=0.5)

    # Training goal
    ax.scatter(training_goal[0], training_goal[1], color='black', s=200,
               marker='*', zorder=10, edgecolors='black', linewidths=0.5)
    ax.annotate('training goal', (training_goal[0], training_goal[1]),
                fontsize=10, fontweight='bold',
                textcoords='offset points', xytext=(-15, -18),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))

    # OOD goal
    ax.scatter(ood_goal[0], ood_goal[1], color='red', s=200,
               marker='*', zorder=10, edgecolors='darkred', linewidths=0.5)
    ax.annotate('new goal', (ood_goal[0], ood_goal[1]),
                fontsize=10, fontweight='bold', color='red',
                textcoords='offset points', xytext=(-15, -18),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))

    ax.set_xlabel('y (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_facecolor('white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def plot_bar_chart(ax, results, methods):
    """Bar chart of relative L2 error across methods."""
    means = [np.mean(results[m]) for m in methods]
    stds = [np.std(results[m]) for m in methods]
    x = np.arange(len(methods))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=BAR_COLORS[:len(methods)],
                  edgecolor='black', linewidth=1.0, alpha=0.85, width=0.6)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.008,
                f'{mean:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=20, ha='right')
    ax.set_ylabel(r'Relative $L^2$ Error', fontsize=12)
    ax.set_title('Quadrotor OOD Adaptation', fontsize=14, fontweight='bold')
    ax.set_facecolor('white')
    ax.tick_params(axis='x', length=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 8: Quadrotor OOD")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", required=True, help="Output figure path")
    args = parser.parse_args()

    data = np.load(str(Path(args.results) / "results.npz"), allow_pickle=True)

    ood_goal = data['ood_goal'][0]
    training_goal = data['training_goal'][0]

    # Load bar results
    results = {}
    for m in METHOD_ORDER:
        key = m.replace('-', '_')
        results[m] = data[key]

    # Load trajectories
    ft_trajs = data['SetONet_FT_trajectories']
    mf_trajs = data['SetONet_Meta_Full_trajectories']

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor('white')

    plot_trajectory_panel(axes[0], ft_trajs, training_goal, ood_goal,
                          'SetONet-FT')
    plot_trajectory_panel(axes[1], mf_trajs, training_goal, ood_goal,
                          'SetONet-Meta-Full')
    plot_bar_chart(axes[2], results, METHOD_ORDER)

    fig.tight_layout(w_pad=3.0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
