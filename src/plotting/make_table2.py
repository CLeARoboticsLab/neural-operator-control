"""Generate Table 2: Adaptation performance LaTeX table.

Reads grid_results.npz files from each environment and generates a LaTeX table
matching the format of the paper.

Usage:
    python src/plotting/make_table2.py \
        --results outputs/results \
        --output outputs/tables/table2.tex
"""

import argparse
import numpy as np
from pathlib import Path


METHOD_DISPLAY = {
    'setonet_ft': 'SetONet-FT',
    'last_branch': 'Last-Branch',
    'last_both': 'Last-Both',
    'maml': 'MAML',
    'setonet_meta': 'SetONet-Meta',
    'setonet_meta_full': 'SetONet-Meta-Full',
}

# For 0-step, only these methods make sense (no adaptation for FT methods)
ZERO_SHOT_METHODS = ['setonet_ft', 'maml', 'setonet_meta', 'setonet_meta_full']
ZERO_SHOT_DISPLAY = {
    'setonet_ft': 'Pretrained (zero-shot)',
    'maml': 'MAML',
    'setonet_meta': 'SetONet-Meta',
    'setonet_meta_full': 'SetONet-Meta-Full',
}

ADAPT_METHODS = ['setonet_ft', 'last_branch', 'last_both', 'maml', 'setonet_meta', 'setonet_meta_full']

ENV_NAMES = {
    'p2p_cost': 'P2P-Cost',
    'p2p_cost_small': 'P2P-Cost-Small',
    'p2p_dynamics': 'P2P-Dynamics',
    'quadrotor': 'Quadrotor',
    'obstacle': 'Obstacle',
}


def load_env_results(results_dir, env_name):
    """Load grid results for an environment."""
    path = Path(results_dir) / env_name / "grid_results.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data


def format_cell(mean, std, is_best=False):
    """Format a table cell as mean ± std, optionally bold."""
    if np.isnan(mean):
        return "nan $\\pm$ nan"
    s = f"{mean:.3f} $\\pm$ {std:.3f}"
    if is_best:
        s = f"\\textbf{{{mean:.3f}}} $\\pm$ {std:.3f}"
    return s


def generate_table(results_dir, output_path, num_demos=1):
    """Generate the adaptation table LaTeX."""
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    envs = ['p2p_cost', 'p2p_cost_small', 'p2p_dynamics', 'quadrotor', 'obstacle']
    env_data = {}
    for env in envs:
        data = load_env_results(results_dir, env)
        if data is not None:
            env_data[env] = data

    if not env_data:
        print("No results found!")
        return

    available_envs = [e for e in envs if e in env_data]
    env_headers = " & ".join(ENV_NAMES[e] for e in available_envs)

    # Get grid params from first available env
    first = env_data[available_envs[0]]
    grad_steps_list = list(first['grad_steps_list'])
    num_demos_list = list(first['num_demos_list'])

    # Find demo index
    if num_demos not in num_demos_list:
        print(f"num_demos={num_demos} not in results. Available: {num_demos_list}")
        return
    demo_idx = num_demos_list.index(num_demos)

    lines = []
    lines.append(f"% Adaptation table: {num_demos} demos")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Adaptation performance (relative $L^2$ error, mean $\\pm$ std over "
                 f"{int(first['num_seeds'])} seeds) with {num_demos} expert demonstrations.}}")
    lines.append(f"\\label{{tab:adaptation_{num_demos}demos}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    cols = "l" + "c" * len(available_envs)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append(f"Method & {env_headers} \\\\")
    lines.append("\\midrule")

    for gs_idx, gs in enumerate(grad_steps_list):
        if gs == 0:
            lines.append(f"\\multicolumn{{{1 + len(available_envs)}}}{{l}}{{\\textit{{0 steps (zero-shot)}}}} \\\\")
            methods = ZERO_SHOT_METHODS
            display = ZERO_SHOT_DISPLAY
        else:
            lines.append(f"\\multicolumn{{{1 + len(available_envs)}}}{{l}}{{\\textit{{{gs} step{'s' if gs > 1 else ''}}}}}} \\\\")
            methods = ADAPT_METHODS
            display = METHOD_DISPLAY

        # Find best per env for this step count
        best_per_env = {}
        for env in available_envs:
            best_val = float('inf')
            for method in methods:
                if method in env_data[env].files:
                    arr = env_data[env][method]
                    mean = np.mean(arr[demo_idx, gs_idx, :])
                    if mean < best_val:
                        best_val = mean
            best_per_env[env] = best_val

        for method in methods:
            name = display.get(method, METHOD_DISPLAY.get(method, method))
            cells = []
            for env in available_envs:
                if method in env_data[env].files:
                    arr = env_data[env][method]
                    mean = np.mean(arr[demo_idx, gs_idx, :])
                    std = np.std(arr[demo_idx, gs_idx, :])
                    is_best = abs(mean - best_per_env[env]) < 1e-6
                    cells.append(format_cell(mean, std, is_best))
                else:
                    cells.append("---")

            row = f"  {name}-{gs} & " + " & ".join(cells) + " \\\\"
            lines.append(row)

        if gs_idx < len(grad_steps_list) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(tex)
    print(f"Table written to {output_path}")
    print(tex)


def main():
    parser = argparse.ArgumentParser(description="Generate adaptation table")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--output", required=True, help="Output .tex file")
    parser.add_argument("--num-demos", type=int, default=1, help="Number of demos for this table")
    args = parser.parse_args()

    generate_table(args.results, args.output, args.num_demos)


if __name__ == "__main__":
    main()
