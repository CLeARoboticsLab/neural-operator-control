"""Generate the zero-shot baseline comparison table (LaTeX, booktabs).

Reads the per-env `baseline_zeroshot.json` files produced by
`src/evaluation/evaluate_baseline.py` and emits a Table-2-style LaTeX table with
three rows — pretrained SetONet, Baseline 1 (task-parameter MLP), Baseline 2
(context-conditioned MLP) — bolding the best (lowest) relative-L2 per column.

Usage:
    python src/plotting/make_baseline_table.py \
        --results outputs/results --output outputs/tables/baseline_table.tex
"""

import argparse
import json
from pathlib import Path

ENVS = ["p2p_cost", "p2p_cost_small", "p2p_dynamics", "quadrotor", "obstacle"]
ENV_HEADERS = {
    "p2p_cost": "P2P-Cost", "p2p_cost_small": "P2P-Small", "p2p_dynamics": "P2P-Dyn.",
    "quadrotor": "Quadrotor", "obstacle": "Obstacle",
}
# (json key, display label)
ROWS = [
    ("pretrained", "SetONet (pretrained)"),
    ("baseline", "MLP, task params (B1)"),
    ("baseline2", "MLP, context (B2)"),
]
# Entries flagged as non-converged: (env, method) -> note marker
DIVERGED = {("p2p_cost_small", "baseline2")}


def load(results_dir):
    data = {}
    for env in ENVS:
        p = Path(results_dir) / env / "baseline_zeroshot.json"
        if p.exists():
            data[env] = json.load(open(p))
    return data


def cell(mean, std, best, diverged):
    s = f"{mean:.3f} $\\pm$ {std:.3f}"
    if diverged:
        s += "$^{\\dagger}$"
    elif best:
        s = f"\\textbf{{{mean:.3f}}} $\\pm$ {std:.3f}"
    return s


def generate(results_dir, output_path):
    data = load(results_dir)
    envs = [e for e in ENVS if e in data]

    # Best (lowest) per env, ignoring diverged entries.
    best = {}
    for e in envs:
        vals = [(m, data[e][m]["mean"]) for m, _ in ROWS
                if m in data[e] and (e, m) not in DIVERGED]
        best[e] = min(vals, key=lambda kv: kv[1])[0] if vals else None

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Zero-shot relative $L^2$ error (mean $\\pm$ std over 5 seeds) of the "
        "pretrained SetONet operator versus two MLP baselines: \\textbf{B1} is handed the "
        "ground-truth task parameters; \\textbf{B2} must infer the task from the same context "
        "the operator's branch receives, but flattened into a vector. Lower is better; bold "
        "marks the best per column. $^{\\dagger}$ did not converge within the environment's "
        "training budget.}",
        "\\label{tab:baselines}",
        f"\\begin{{tabular}}{{l{'c' * len(envs)}}}",
        "\\toprule",
        "Method & " + " & ".join(ENV_HEADERS[e] for e in envs) + " \\\\",
        "\\midrule",
    ]
    for m, label in ROWS:
        cells = []
        for e in envs:
            if m in data[e]:
                v = data[e][m]
                cells.append(cell(v["mean"], v["std"], best[e] == m, (e, m) in DIVERGED))
            else:
                cells.append("---")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex + "\n")
    print(tex)
    print(f"\nWritten to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--output", default="outputs/tables/baseline_table.tex")
    args = ap.parse_args()
    generate(args.results, args.output)


if __name__ == "__main__":
    main()
