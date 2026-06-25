"""Fixed-K (no-padding) fair comparison across the 3 control envs.

Trains BOTH the SetONet operator and B2 at a single fixed context size K used for
training *and* eval, with B2's slot bank == K so there is **zero padding** (every
slot filled). This removes the variable-size / padding mechanism entirely, making
the operator-vs-B2 comparison fully apples-to-apples.

  operator: trained with K=[K]   (config override)
  B2:       trained with baseline2_fixed_K=K  -> slots == n_ctx == K, no -1 padding
  eval:     both at K (the existing evaluate_baseline K=64 protocol)
  B1:       context-free, reused from the existing checkpoints as a reference row

Usage:
  uv run python run_fixedk_table.py                       # K=64, operators best-of-(config num_runs)
  uv run python run_fixedk_table.py --operator-runs 1     # quick single-run operators
  uv run python run_fixedk_table.py --skip-train          # just re-eval + re-table existing fixed-K ckpts
"""
import argparse, copy, json, importlib
import yaml
from pathlib import Path

ENVS = ["p2p_cost", "p2p_dynamics", "quadrotor"]
LABELS = {"p2p_cost": "P2P-Cost", "p2p_dynamics": "P2P-Dyn.", "quadrotor": "Quadrotor"}
OP_TRAIN = {
    "p2p_cost": "src.training.train_p2p_cost",
    "p2p_dynamics": "src.training.train_p2p_dynamics",
    "quadrotor": "src.training.train_quadrotor",
}
B2_TRAIN = {
    "p2p_cost": "src.training.train_baseline2_p2p_cost",
    "p2p_dynamics": "src.training.train_baseline2_p2p_dynamics",
    "quadrotor": "src.training.train_baseline2_quadrotor",
}


def eval_env(env, cfg, data_dir, b1_dir, op_dir, b2_dir, out, seed):
    from src.evaluation.evaluate_baseline import run_p2p_cost, run_p2p_dynamics, run_quadrotor
    runner = {"p2p_cost": run_p2p_cost, "p2p_dynamics": run_p2p_dynamics,
              "quadrotor": run_quadrotor}[env]
    kw = dict(pretrained_dir=op_dir, baseline2_dir=b2_dir)
    if env == "p2p_cost":
        kw["context"] = "onpolicy"
    return runner(cfg, data_dir, b1_dir, out, seed, **kw)


def emit_table(results, K, out_tex):
    rows = [("pretrained", "SetONet (pretrained)"), ("baseline2", "MLP, context (B2)"),
            ("baseline", "MLP, task params (B1)")]
    envs = [e for e in ENVS if e in results]
    def cell(e, key):
        d = results[e].get(key)
        return f"{d['mean']:.3f} $\\pm$ {d['std']:.3f}" if d else "--"
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Zero-shot relative $L^2$ error (mean $\\pm$ std over 5 seeds) at a "
        f"\\emph{{fixed}} context size $K={K}$ for both training and evaluation. B2's context "
        "bank is exactly $K$ slots (every slot filled, \\textbf{no padding}), and the operator "
        "is trained with $K=" + str(K) + "$, so the comparison is fully matched. Lower is better.}",
        "\\label{tab:fixedk}",
        f"\\begin{{tabular}}{{l{'c' * len(envs)}}}", "\\toprule",
        "Method & " + " & ".join(LABELS[e] for e in envs) + " \\\\", "\\midrule",
    ]
    for key, lab in rows:
        lines.append(f"{lab} & " + " & ".join(cell(e, key) for e in envs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex = "\n".join(lines)
    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tex).write_text(tex + "\n")
    print("\n" + tex)
    print(f"\nWritten to {out_tex}")
    # console summary
    pm = " $\\pm$ "
    print(f"\n{'Method':<24}" + "".join(f"{LABELS[e]:>16}" for e in envs))
    for key, lab in rows:
        cells = [cell(e, key).replace(pm, "±") for e in envs]
        print(f"{lab:<24}" + "".join(f"{c:>16}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--operator-runs", type=int, default=None,
                    help="override operator num_runs (default: use each config's value)")
    ap.add_argument("--iters", type=int, default=None, help="override num_iterations (smoke testing)")
    ap.add_argument("--ckpt-root", default="checkpoints_fixedK")
    ap.add_argument("--results-root", default="outputs/results_fixedK")
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    envs = [e.strip() for e in a.envs.split(",") if e.strip()]
    results = {}
    for env in envs:
        cfg = yaml.safe_load(open(f"configs/{env}.yaml"))
        cfg["training"]["K"] = [a.K]
        cfg["training"]["baseline2_fixed_K"] = a.K
        if a.operator_runs is not None:
            cfg["training"]["num_runs"] = a.operator_runs
        if a.iters is not None:
            cfg["training"]["num_iterations"] = a.iters
        data_dir = f"data/{env}"
        op_dir = f"{a.ckpt_root}/{env}/pretrained"
        b2_dir = f"{a.ckpt_root}/{env}/baseline2"
        b1_dir = f"checkpoints/{env}/baseline"          # context-free B1, reused
        out = f"{a.results_root}/{env}/baseline_zeroshot.json"

        if not a.skip_train:
            runs = cfg["training"].get("num_runs", 1)
            print(f"\n{'='*70}\n{env}: train operator  (K=[{a.K}], num_runs={runs})\n{'='*70}")
            importlib.import_module(OP_TRAIN[env]).run_training(copy.deepcopy(cfg), data_dir, op_dir, seed=a.seed)
            print(f"\n{'='*70}\n{env}: train B2  (fixed K={a.K}, slots={a.K}, no padding)\n{'='*70}")
            importlib.import_module(B2_TRAIN[env]).run_training(copy.deepcopy(cfg), data_dir, b2_dir, seed=a.seed)

        print(f"\n{'='*70}\n{env}: eval at K={a.K}\n{'='*70}")
        results[env] = eval_env(env, cfg, data_dir, b1_dir, op_dir, b2_dir, out, a.seed)

    emit_table(results, a.K, f"outputs/tables/fixedk_table.tex")


if __name__ == "__main__":
    main()
