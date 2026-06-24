"""
Figure 7: Cost-based fine-tuning plots.

(a) P2P-Cost OOD: trajectory plots + distance-to-goal bar chart
(b) Obstacle: trajectory plots + collision bar chart

Usage (matches Makefile):
    python src/plotting/plot_cost_adapt.py \
        --p2p-results outputs/results/cost_adapt_p2p_ood \
        --obstacle-results outputs/results/cost_adapt_obstacle \
        --output outputs/figures/figure7.pdf
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'font.size': 11,
})

BAR_COLORS = {
    'SetONet': '#1f77b4',
    'FT': '#ff7f0e',
    'Last-Branch': '#2ca02c',
    'Last-Both': '#d62728',
    'Expert': '#2ca02c',
}


def assign_color(gx, gy):
    """Assign trajectory color based on goal angle."""
    angle = np.arctan2(gy, gx)
    if 0 <= angle < np.pi / 2:
        return 'red'
    elif angle >= np.pi / 2:
        return 'blue'
    elif -np.pi <= angle < -np.pi / 2:
        return 'green'
    else:
        return 'orange'


def plot_p2p_ood(results_dir, axes_traj, ax_bar):
    """Plot Figure 7a: P2P-Cost OOD trajectories + bar chart."""
    data = np.load(str(Path(results_dir) / "results.npz"), allow_pickle=True)

    goal_positions = data['goal_positions']
    start_center = data['start_center']
    half_width = float(data['half_width'])
    train_range = float(data['train_range'])

    setonet_trajs = data['setonet_trajectories']
    lqr_trajs = data.get('lqr_trajectories', None)

    # Find the trajectory key for each FT method
    methods_traj = {
        'SetONet': data['setonet_trajectories'],
        'Last-Branch': data['last_branch_trajectories'],
    }

    plot_lim = max(half_width, abs(start_center[0])) + 3
    num_starts_per_goal = setonet_trajs.shape[0] // len(goal_positions)

    titles = ['SetONet', 'Last-Branch']
    for ax_idx, (title, key) in enumerate(zip(titles, ['SetONet', 'Last-Branch'])):
        ax = axes_traj[ax_idx]
        trajs = methods_traj[key]

        # Training region
        rect = plt.Rectangle((-train_range, -train_range), 2 * train_range, 2 * train_range,
                              fill=True, facecolor='#e0e0e0', edgecolor='#666666',
                              linestyle='-', linewidth=1.5, alpha=0.35, zorder=0)
        ax.add_patch(rect)
        ax.text(0, 0, 'Training\nRegion', fontsize=10, fontweight='bold',
                color='#555555', va='center', ha='center', zorder=1)

        # Trajectories
        for t_idx in range(len(trajs)):
            g_idx = t_idx // num_starts_per_goal
            gx, gy = goal_positions[g_idx]
            color = assign_color(gx, gy)
            ax.plot(trajs[t_idx, :, 0], trajs[t_idx, :, 1],
                    color=color, alpha=0.5, linewidth=1.5)

        # Goal markers
        for gx, gy in goal_positions:
            ax.scatter(gx, gy, color=assign_color(gx, gy), s=80, marker='*',
                       edgecolors='black', linewidths=0.5, zorder=10, alpha=0.7)

        ax.set_xlim(-plot_lim, plot_lim)
        ax.set_ylim(-plot_lim, plot_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bar chart
    methods_bar = ['SetONet', 'FT', 'Last-Branch', 'Last-Both']
    means, stds = [], []
    for m in methods_bar:
        key = m.lower().replace('-', '_') + '_distances'
        d = data[key]
        means.append(np.mean(d))
        stds.append(np.std(d))

    colors = [BAR_COLORS[m] for m in methods_bar]
    x = np.arange(len(methods_bar))
    bars = ax_bar.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1.5})

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{mean:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax_bar.set_ylabel('Distance To Goal', fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods_bar, fontweight='bold', fontsize=9)
    ax_bar.set_ylim(bottom=0)


def _plot_obstacle_trajectories(ax, trajectories, obstacles, goal_pos, cmap_name, title):
    """Plot trajectories with obstacles on a single axis."""
    # Draw obstacles
    for obs in obstacles:
        x, y, radius = float(obs[0]), float(obs[1]), float(obs[2])
        circle = mpatches.Circle((x, y), radius, color='#ffaaaa', alpha=0.6, zorder=2)
        ax.add_patch(circle)
        border = mpatches.Circle((x, y), radius, fill=False,
                                 edgecolor='darkred', linewidth=1.5, zorder=3)
        ax.add_patch(border)

    # Goal marker
    ax.scatter(float(goal_pos[0]), float(goal_pos[1]), color='gold', s=150,
               marker='*', edgecolors='black', linewidths=1.5, zorder=10)

    # Trajectories
    colors = plt.colormaps[cmap_name](np.linspace(0.3, 0.9, len(trajectories)))
    for traj, color in zip(trajectories, colors):
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=40, marker='o',
                   edgecolors='black', linewidths=0.8, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=60, marker='^',
                   edgecolors='black', linewidths=0.8, zorder=5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_obstacle(results_dir, axes_grid, ax_bar):
    """Plot Figure 7b: Obstacle trajectory grid + collision bar chart.

    axes_grid: 2x3 array of axes for trajectory plots
    ax_bar: axis for collision bar chart
    """
    data = np.load(str(Path(results_dir) / "results.npz"), allow_pickle=True)
    plot_data = data['plot_data']
    goal_pos = data['goal_position']

    # Select tasks where FT improves over pretrained, prefer 2 and 4 obstacles
    # Compute per-task collisions to find good examples
    expert_c = data['expert_collisions']
    setonet_c = data['setonet_collisions']
    ft_c = data['ft_collisions']

    num_traj_per_task = [len(t['expert_trajectories']) for t in plot_data]
    offset = 0
    task_stats = []
    for i, nt in enumerate(num_traj_per_task):
        s = int(np.sum(setonet_c[offset:offset+nt]))
        f = int(np.sum(ft_c[offset:offset+nt]))
        task_stats.append({'setonet': s, 'ft': f, 'improved': f < s})
        offset += nt

    # Pick two 4-obstacle tasks where FT improved most
    candidates = []
    for i, t in enumerate(plot_data):
        if t['num_obstacles'] == 4 and task_stats[i]['improved']:
            improvement = task_stats[i]['setonet'] - task_stats[i]['ft']
            candidates.append((i, improvement))
    candidates.sort(key=lambda x: -x[1])  # Best improvement first
    task_indices = [c[0] for c in candidates[:2]]
    # Fallback
    while len(task_indices) < 2:
        for i in range(len(plot_data)):
            if i not in task_indices:
                task_indices.append(i)
                break
    print(f"  Selected tasks: {task_indices} (4 obstacles, FT improved)")

    # Plot columns: Expert, SetONet, Branch (FT)
    columns = [
        ('Expert', 'expert', 'Greens'),
        ('SetONet', 'setonet', 'Blues'),
        ('Branch', 'ft', 'Oranges'),
    ]

    for row_i, task_i in enumerate(task_indices):
        task = plot_data[task_i]
        obstacles = task['obstacles']
        num_total = len(task['expert_trajectories'])
        # Evenly subsample from both start lines (first half = left edge, second half = bottom edge)
        half = num_total // 2
        left_idx = np.linspace(0, half - 1, 4, dtype=int)
        bottom_idx = np.linspace(half, num_total - 1, 4, dtype=int)
        plot_idx = np.concatenate([left_idx, bottom_idx])

        for col_i, (title, key, cmap) in enumerate(columns):
            ax = axes_grid[row_i][col_i]
            traj_key = f'{key}_trajectories'
            trajs = task[traj_key][plot_idx]
            row_title = title if row_i == 0 else ''
            _plot_obstacle_trajectories(ax, trajs, obstacles, goal_pos, cmap, row_title)

    # Bar chart — use ALL tasks for proper statistics
    methods_bar = ['Expert', 'SetONet', 'FT', 'Last-Branch', 'Last-Both']
    # Compute per-task totals for mean ± std
    offset_map = np.cumsum([0] + num_traj_per_task)
    num_all_tasks = len(plot_data)
    means = []
    stds = []
    for m in methods_bar:
        key = m.lower().replace('-', '_') + '_collisions'
        per_task = []
        for ti in range(num_all_tasks):
            c = data[key][offset_map[ti]:offset_map[ti+1]]
            per_task.append(np.sum(c))
        means.append(np.mean(per_task))
        stds.append(np.std(per_task))

    colors = [BAR_COLORS.get(m, '#9467bd') for m in methods_bar]
    x = np.arange(len(methods_bar))
    bars = ax_bar.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1.5})

    for bar, mean in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.04,
                    f'{int(mean)}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax_bar.set_ylabel('Time in Collision', fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods_bar, fontweight='bold', fontsize=9)
    ax_bar.set_ylim(bottom=0)


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 7: Cost-based adaptation")
    parser.add_argument("--p2p-results", default="outputs/results/cost_adapt_p2p_ood")
    parser.add_argument("--obstacle-results", default="outputs/results/cost_adapt_obstacle")
    parser.add_argument("--output", default="outputs/figures/figure7")
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    output_base = Path(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    if args.part in ("a", "both"):
        fig_a, axes_a = plt.subplots(1, 3, figsize=(14, 4.5),
                                      gridspec_kw={'width_ratios': [1.2, 1.2, 1]})
        fig_a.patch.set_facecolor('white')
        p2p_path = Path(args.p2p_results) / "results.npz"
        if p2p_path.exists():
            plot_p2p_ood(args.p2p_results, [axes_a[0], axes_a[1]], axes_a[2])
        fig_a.suptitle('(a) P2P-Cost (out-of-distribution, cost-based adaptation)',
                        fontsize=12, fontweight='bold', y=1.02)
        fig_a.tight_layout()
        out_a = str(output_base) + '_a.pdf'
        fig_a.savefig(out_a, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure 7a saved to: {out_a}")
        plt.close(fig_a)

    if args.part in ("b", "both"):
        obs_path = Path(args.obstacle_results) / "results.npz"
        if not obs_path.exists():
            print("Obstacle results not found, skipping part b")
        else:
            data = np.load(str(obs_path), allow_pickle=True)
            plot_data = data['plot_data']
            goal_pos = data['goal_position']

            # Compute per-task collision stats for task selection
            expert_c = data['expert_collisions']
            setonet_c = data['setonet_collisions']
            ft_c = data['ft_collisions']
            num_traj_per_task = [len(t['expert_trajectories']) for t in plot_data]
            offset = 0
            task_stats = []
            for i, nt in enumerate(num_traj_per_task):
                s = int(np.sum(setonet_c[offset:offset+nt]))
                f = int(np.sum(ft_c[offset:offset+nt]))
                task_stats.append({'setonet': s, 'ft': f, 'improved': f < s})
                offset += nt

            # Pick two 4-obstacle tasks where FT improved most
            candidates = [(i, task_stats[i]['setonet'] - task_stats[i]['ft'])
                          for i in range(len(plot_data))
                          if plot_data[i]['num_obstacles'] == 4 and task_stats[i]['improved']]
            candidates.sort(key=lambda x: -x[1])
            task_indices = [c[0] for c in candidates[:2]]
            while len(task_indices) < 2:
                for i in range(len(plot_data)):
                    if i not in task_indices:
                        task_indices.append(i)
                        break
            print(f"  Selected tasks: {task_indices}")

            columns = [
                ('Expert', 'expert', 'Greens'),
                ('SetONet', 'setonet', 'Blues'),
                ('Branch', 'ft', 'Oranges'),
            ]

            # Generate individual trajectory plots
            for row_i, task_i in enumerate(task_indices):
                task = plot_data[task_i]
                obstacles = task['obstacles']
                num_total = len(task['expert_trajectories'])
                half = num_total // 2
                left_idx = np.linspace(0, half - 1, 4, dtype=int)
                bottom_idx = np.linspace(half, num_total - 1, 4, dtype=int)
                plot_idx = np.concatenate([left_idx, bottom_idx])

                for col_i, (title, key, cmap) in enumerate(columns):
                    fig_t, ax_t = plt.subplots(1, 1, figsize=(5, 5))
                    fig_t.patch.set_facecolor('white')
                    traj_key = f'{key}_trajectories'
                    trajs = task[traj_key][plot_idx]
                    _plot_obstacle_trajectories(ax_t, trajs, obstacles, goal_pos, cmap, title)
                    fig_t.tight_layout()
                    out_t = str(output_base) + f'_b_task{row_i}_{key}.pdf'
                    fig_t.savefig(out_t, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_t)
                    print(f"  Saved: {out_t}")

            # Generate bar chart separately
            fig_bar, ax_bar = plt.subplots(1, 1, figsize=(6, 5))
            fig_bar.patch.set_facecolor('white')

            methods_bar = ['Expert', 'SetONet', 'FT', 'Last-Branch', 'Last-Both']
            offset_map = np.cumsum([0] + num_traj_per_task)
            num_all_tasks = len(plot_data)
            means, stds = [], []
            for m in methods_bar:
                mkey = m.lower().replace('-', '_') + '_collisions'
                per_task = [np.sum(data[mkey][offset_map[ti]:offset_map[ti+1]])
                            for ti in range(num_all_tasks)]
                means.append(np.mean(per_task))
                stds.append(np.std(per_task))

            colors = [BAR_COLORS.get(m, '#9467bd') for m in methods_bar]
            x = np.arange(len(methods_bar))
            bars = ax_bar.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                              edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1.5})
            for bar, mean in zip(bars, means):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.04,
                            f'{int(mean)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax_bar.set_ylabel('Time in Collision', fontweight='bold')
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(methods_bar, fontweight='bold', fontsize=10, rotation=30, ha='right')
            ax_bar.set_ylim(bottom=0)
            fig_bar.tight_layout()
            out_bar = str(output_base) + '_b_barplot.pdf'
            fig_bar.savefig(out_bar, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig_bar)
            print(f"  Saved: {out_bar}")


if __name__ == "__main__":
    main()
