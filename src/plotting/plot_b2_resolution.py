"""P2P-Cost context-size (K) generalization: SetONet operator vs B2 (context MLP).

Companion to Figure 5. Sweeps the number of real context elements K and plots
median rel-L2 with IQR for both models, using the on-policy context both were
trained/evaluated on. Blue x-ticks = context sizes B2 saw in training {32,64,128};
red = unseen. The operator's set encoder degrades gracefully at unseen K, while
B2's fixed-slot flattening blows up off its trained sizes — the generalization gap.

Usage:
    python src/plotting/plot_b2_resolution.py --output outputs/figures/p2p_cost_b2_resolution.pdf
"""
import argparse, yaml, numpy as np
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from src.evaluation.evaluate import TransferTaskDataLoader
from src.evaluation.evaluate_baseline import build_setonet, relative_l2, predict_setonet, N_EVAL, NUM_EVAL_BATCHES
from src.training.baseline_mlp import BaselineMLP
from src.training.baseline2_common import pad_flatten_context, build_input
from src.envs.create_p2p_cost import generate_dataset

K_LIST = [1, 4, 8, 16, 32, 64, 128, 256]
B2_TRAIN_K = {32, 64, 128}
NUM_TASKS = 20

plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold', 'axes.labelsize': 14,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 11})


def load_op(d, cfg, key):
    m = build_setonet(cfg, 6, 1, 5, 2, key)
    m = eqx.tree_deserialise_leaves(d + "/setonet.eqx", m)
    h = np.load(d + "/training_history.npz")
    n = {"state_mean": np.array(h["state_mean"]), "state_std": np.array(h["state_std"]),
         "cost_mean": float(h["cost_mean"]), "cost_std": float(h["cost_std"])}
    return m, n, float(h["max_action"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs"); ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--output", default="outputs/figures/p2p_cost_b2_resolution.pdf")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(a.config) / "p2p_cost.yaml")); key = jr.PRNGKey(a.seed)

    op, opn, opmax = load_op(str(Path(a.checkpoints) / "p2p_cost" / "pretrained"), cfg, jr.split(key)[0])
    bh = np.load(str(Path(a.checkpoints) / "p2p_cost" / "baseline2" / "training_history.npz"))
    slots, elem, sd, ad = int(bh["slots"]), int(bh["elem_dim"]), int(bh["state_dim"]), int(bh["action_dim"])
    b2 = BaselineMLP(sd + 1 + slots * elem, ad, int(bh["width"]), int(bh["depth"]), jr.split(key)[1])
    b2 = eqx.tree_deserialise_leaves(str(Path(a.checkpoints) / "p2p_cost" / "baseline2" / "baseline2.eqx"), b2)
    b2n = {"state_mean": np.array(bh["state_mean"]), "state_std": np.array(bh["state_std"]),
           "cost_mean": float(bh["cost_mean"]), "cost_std": float(bh["cost_std"])}
    b2max = float(bh["max_action"]); ctx_scale = np.array(bh["ctx_scale"])
    st_mean, st_std = np.array(bh["st_mean"]), np.array(bh["st_std"])
    y_mean, y_std = np.array(bh["y_mean"]), np.array(bh["y_std"])

    tds = generate_dataset(num_goals=NUM_TASKS, trajectories_per_goal=50, horizon=50, dt=0.1,
                           Q_weight=1.0, R_weight=0.1, Qf_weight=10.0, goal_range=(-5., 5.),
                           state_range=(-10., 10.), vel_range=(-5., 5.), zero_velocity_goal=True, seed=a.seed + 1000)

    def per_task_errs(which, K):
        d = dict(tds); d["norm_stats"] = opn if which == "op" else b2n
        np.random.seed(7); dl = TransferTaskDataLoader(d, 0.25, True)
        dl.max_action = opmax if which == "op" else b2max
        out = []
        for tid in dl.get_task_ids():
            ee = []
            for _ in range(NUM_EVAL_BATCHES):
                sb = dl.sample_train(tid, K, N=1); eb = dl.sample_holdout(tid, K, N=N_EVAL)
                if which == "op":
                    si = jnp.asarray(np.concatenate([sb[0], sb[1]], -1)); sv = jnp.asarray(sb[2])
                    pred = np.array(predict_setonet(op, si, sv, eb[4])) * opmax
                    ee.append(relative_l2(pred, np.array(eb[5]) * opmax))
                else:
                    ctx = np.concatenate([sb[0], sb[1], sb[2]], -1); flat = pad_flatten_context(ctx, slots, ctx_scale)
                    X = build_input(eb[4], flat, st_mean, st_std)
                    pred = np.array(jax.vmap(b2)(jnp.asarray(X))) * y_std + y_mean
                    ee.append(relative_l2(pred, eb[5].reshape(-1, ad)))
            out.append(float(np.mean(ee)))
        return np.array(out)

    data = {"op": {}, "b2": {}}
    for which in ("op", "b2"):
        for K in K_LIST:
            data[which][K] = per_task_errs(which, K)
            print(f"{which:3} K={K:3d}: median rel-L2 = {np.median(data[which][K]):.4f}")

    pos = np.arange(len(K_LIST))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for which, color, label in [("op", "#2ca02c", "SetONet (operator)"), ("b2", "#d62728", "B2 (context MLP)")]:
        med = [np.median(data[which][K]) for K in K_LIST]
        q25 = [np.percentile(data[which][K], 25) for K in K_LIST]
        q75 = [np.percentile(data[which][K], 75) for K in K_LIST]
        ax.plot(pos, med, "-s", color=color, lw=2, ms=5, label=label)
        ax.fill_between(pos, q25, q75, color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_title("P2P-Cost"); ax.set_xlabel("K Samples"); ax.set_ylabel("Relative $L^2$ (log)")
    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in K_LIST])
    for t, k in zip(ax.get_xticklabels(), K_LIST):
        t.set_color("blue" if k in B2_TRAIN_K else "red")
    ax.legend(); ax.grid(True, which="both", alpha=0.2)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.output, dpi=150, bbox_inches="tight")
    fig.savefig(str(Path(a.output).with_suffix(".png")), dpi=150, bbox_inches="tight")
    print("saved", a.output)


if __name__ == "__main__":
    main()
