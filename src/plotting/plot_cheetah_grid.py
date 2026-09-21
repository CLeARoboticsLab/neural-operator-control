"""Figure 10: HalfCheetah-v3 adaptation error vs gradient steps, one panel per demo count.

Reads either the ``grid_results.npz`` / ``grid_checkpoint.json`` written by
``src/evaluation/evaluate_halfcheetah.py`` or the paper's shipped results in
``paper_results/halfcheetah_grid.json``. Lines are the mean relative L2 error
over seeds; shading is +/- one standard deviation.

Usage:
    python src/plotting/plot_cheetah_grid.py \
        --results outputs/results/halfcheetah_grid/grid_results.npz \
        --output outputs/figures/figure10.pdf
    python src/plotting/plot_cheetah_grid.py \
        --results paper_results/halfcheetah_grid.json --output outputs/figures/figure10.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})

# Same four methods as the paper figure (last_layer is evaluated but not plotted).
METHODS = [
    ("setonet_ft", "SetONet-FT", "#1f77b4", "-", "o"),
    ("maml", "MAML", "#ff7f0e", "-.", "^"),
    ("setonet_meta", "SetONet-Meta", "#2ca02c", "-", "s"),
    ("setonet_meta_full", "SetONet-Meta-Full", "#d62728", "--", "v"),
]


def load_results(path):
    """Return ``{method: {num_demos: {grad_steps: [values over seeds]}}}``."""
    path = Path(path)
    res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if path.suffix == ".json":
        for e in json.load(open(path))["results"]:
            res[e["method"]][int(e["num_demos"])][int(e["grad_steps"])].append(float(e["value"]))
    else:
        d = np.load(path, allow_pickle=True)
        demos, steps = [int(x) for x in d["num_demos_list"]], [int(x) for x in d["grad_steps_list"]]
        for m in d["methods"]:
            m = str(m)
            for i, nd in enumerate(demos):
                for j, gs in enumerate(steps):
                    res[m][nd][gs] = [float(v) for v in d[m][i, j] if not np.isnan(v)]
    return res


def plot(res, output, max_steps=100, log_x=False):
    demos = sorted({nd for m in res for nd in res[m]})
    fig, axes = plt.subplots(1, len(demos), figsize=(3.0 * len(demos), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    handles = {}
    for ax, nd in zip(axes, demos):
        for key, label, color, ls, marker in METHODS:
            if key not in res or nd not in res[key]:
                continue
            steps = sorted(gs for gs in res[key][nd] if gs <= max_steps and res[key][nd][gs])
            if not steps:
                continue
            mean = np.array([np.mean(res[key][nd][gs]) for gs in steps])
            std = np.array([np.std(res[key][nd][gs]) for gs in steps])
            (h,) = ax.plot(steps, mean, color=color, ls=ls, marker=marker, ms=4, lw=1.6, label=label)
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
            handles[label] = h
        ax.set_title(f"{nd} demo" + ("s" if nd != 1 else ""))
        ax.set_xlabel("Gradient Steps")
        if log_x:
            ax.set_xscale("log")
        else:
            ax.set_xticks([s for s in (1, 25, 50, 100, 200) if s <= max_steps])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Relative $L^2$ Error")
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(Path(output).with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="grid_results.npz, grid_checkpoint.json or paper_results json")
    p.add_argument("--output", required=True)
    p.add_argument("--max-steps", type=int, default=100, help="largest gradient-step count to show")
    p.add_argument("--log-x", action="store_true")
    a = p.parse_args()
    plot(load_results(a.results), a.output, a.max_steps, a.log_x)


if __name__ == "__main__":
    main()
