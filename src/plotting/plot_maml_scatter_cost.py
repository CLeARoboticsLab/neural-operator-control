"""Figure 6 companion — per-task comparison using ACHIEVED TASK COST (normalized
to the expert) instead of relative L2.

For each held-out task and each method (Pre-trained, SetONet-FT, Last-Branch,
Last-Both, MAML) we:
  1. adapt the model exactly as Figure 6 does (test-time only — NO change to training),
  2. roll the adapted policy out CLOSED-LOOP through the task's true dynamics,
  3. accumulate the achieved task cost (stage + terminal),
  4. divide by the EXPERT's stored cost on that task.
=> normalized cost: 1.0 == expert-level, < 1.0 == better than expert.

Outputs:
  - <output>_cost.npz : per-task normalized cost for every method/env
  - <output>_scatter.pdf : 3-panel per-task scatter (MAML vs each method), mirroring Fig 6
  - <output>_summary.pdf : mean ± std normalized cost per method/env (bar)

Usage:
    python src/plotting/plot_maml_scatter_cost.py \
        --config configs --data data --checkpoints checkpoints \
        --output outputs/figures/figure6_cost

NOTE: this is purely an evaluation/plotting script. It loads existing checkpoints
and does the same test-time adaptation as Figure 6.
"""

import argparse
import copy
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.setonet import SetONet
from src.training.maml import MAMLMLP
from src.envs.create_p2p_cost import generate_dataset
# Reuse Figure-6 helpers (prediction, dataloader, random-context sampling).
from src.plotting.plot_maml_scatter import (
    predict_setonet, predict_maml, TransferTaskDataLoader, _get_random_context,
    GRAD_STEPS, NUM_DEMOS, NUM_TASKS, K, FT_LR, DEFAULT_SEED,
)

METHODS = ["pretrained", "setonet_ft", "last_branch", "last_both", "maml"]
METHOD_LABELS = {"pretrained": "Pre-trained", "setonet_ft": "SetONet-FT",
                 "last_branch": "Last-Branch", "last_both": "Last-Both"}
METHOD_STYLE = {"pretrained": ("#4477AA", "s"), "setonet_ft": ("#EE7733", "o"),
                "last_branch": ("#66CCEE", "^"), "last_both": ("#228833", "D")}


# ── SetONet adaptation (returns the ADAPTED model; mirrors plot_maml_scatter) ──

def _adapt_setonet(model, dl, tid, dataset, mode, grad_steps, num_demos):
    """Return an adapted copy of `model`. mode: 'none'|'ft'|'last_branch'|'last_both'."""
    if mode == "none" or grad_steps == 0:
        return model
    m = copy.deepcopy(model)
    filt = jax.tree_util.tree_map(lambda _: False, m)
    if mode == "ft":  # branch only (freeze trunk)
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), m)
        filt = eqx.tree_at(lambda mm: mm.trunk, filt,
                           replace=jax.tree_util.tree_map(lambda _: False, m.trunk))
    elif mode == "last_branch":
        filt = eqx.tree_at(lambda mm: mm.rho.layers[-1], filt,
                           replace=jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]))
    elif mode == "last_both":
        filt = eqx.tree_at(lambda mm: (mm.rho.layers[-1], mm.trunk.layers[-1]), filt,
                           replace=(jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]),
                                    jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.trunk.layers[-1])))
    trainable, frozen = eqx.partition(m, filt)
    opt = optax.adam(FT_LR)
    opt_state = opt.init(trainable)
    for _ in range(grad_steps):
        si, sv = _get_random_context(dl, tid, dataset)
        b = dl.sample_train(tid, K, N=num_demos)
        ts, tc = jnp.array(b[4]), jnp.array(b[5])

        def loss_fn(tr):
            mm = eqx.combine(tr, frozen)
            p = jax.vmap(jax.vmap(mm, in_axes=(None, None, 0)), in_axes=(None, None, 0))(si, sv, ts)
            return jnp.mean(jnp.square(p - tc))

        _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = opt.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
    return eqx.combine(trainable, frozen)


def _adapt_maml(maml_model, dl, tid, inner_lr, grad_steps, num_demos):
    b = dl.sample_train(tid, K, N=num_demos)
    sup_s = jnp.array(b[4].reshape(-1, b[4].shape[-1]))
    sup_c = jnp.array(b[5].reshape(-1, b[5].shape[-1]))
    adapted = maml_model
    for _ in range(grad_steps):
        def loss_fn(mm):
            return jnp.mean(jnp.square(jax.vmap(mm)(sup_s) - sup_c))
        _, grads = eqx.filter_value_and_grad(loss_fn)(adapted)
        adapted = eqx.apply_updates(adapted, jax.tree_util.tree_map(lambda g: -inner_lr * g, grads))
    return adapted


# ── Generic closed-loop rollout + cost ──────────────────────────────────────

def rollout_cost(predict_norm_ctrl, start_state, step_fn, stage_cost, terminal_cost,
                 horizon, max_action, state_mean, state_std):
    """Closed-loop rollout; returns achieved cost (sum stage + terminal).

    predict_norm_ctrl(state_norm, t_norm) -> normalized control (the model output).
    step_fn(state_phys, ctrl_phys) -> next_state_phys.
    """
    state = np.asarray(start_state, dtype=np.float64)
    total = 0.0
    for t in range(horizon):
        s_norm = (state - state_mean) / state_std
        u = np.asarray(predict_norm_ctrl(s_norm, t / horizon)) * max_action
        total += float(stage_cost(state, u))
        state = np.asarray(step_fn(state, u), dtype=np.float64)
    total += float(terminal_cost(state))
    return total


# ── P2P-Cost runner ─────────────────────────────────────────────────────────

def run_p2p_cost(config_dir, data_dir, checkpoint_dir, seed):
    print("\n=== P2P-Cost (achieved cost) ===")
    cfg = yaml.safe_load(open(Path(config_dir) / "p2p_cost.yaml"))
    mc = cfg["model"]
    dt, horizon = 0.1, 50
    Qw, Rw, Qfw = 1.0, 0.1, 10.0
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    B = np.array([[0, 0], [0, 0], [dt, 0], [0, dt]], float)

    keys = jr.split(jr.PRNGKey(seed), 3)
    pretrained = SetONet(input_size_src=6, output_size_src=1, input_size_tgt=5, output_size_tgt=2,
                         **{k: mc[k] for k in ['p', 'phi_hidden_size', 'phi_output_size', 'rho_hidden_size',
                            'trunk_hidden_size', 'n_phi_layers', 'n_rho_layers', 'n_trunk_layers',
                            'aggregation_type', 'attention_n_heads', 'attention_n_tokens', 'use_bias']}, key=keys[0])
    pretrained = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "setonet.eqx"), pretrained)

    maml_path = Path(checkpoint_dir) / "p2p_cost" / "maml" / "maml.eqx"
    maml_model = None
    if maml_path.exists():
        mcfg = cfg.get("maml_model", {})
        maml_model = MAMLMLP(input_dim=5, output_dim=2, hidden_size=mcfg.get("hidden_size", 128),
                             num_layers=mcfg.get("num_layers", 16), key=keys[1])
        maml_model = eqx.tree_deserialise_leaves(str(maml_path), maml_model)
    inner_lr = cfg.get("maml", {}).get("inner_lr", 0.01)

    hist = np.load(str(Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "training_history.npz"))
    sm, ss = np.array(hist["state_mean"]), np.array(hist["state_std"])
    norm = {"state_mean": sm, "state_std": ss, "cost_mean": float(hist["cost_mean"]),
            "cost_std": float(hist["cost_std"])}
    max_action = float(hist["max_action"])

    np.random.seed(seed)
    ds = generate_dataset(num_goals=NUM_TASKS, trajectories_per_goal=50, horizon=horizon, dt=dt,
                          Q_weight=Qw, R_weight=Rw, Qf_weight=Qfw, goal_range=(-5., 5.),
                          state_range=(-10., 10.), vel_range=(-5., 5.), zero_velocity_goal=True, seed=seed + 1000)
    ds["norm_stats"] = norm
    np.random.seed(seed)
    dl = TransferTaskDataLoader(ds, train_holdout_split=0.25, normalize=True)
    dl.max_action = max_action
    task_ids = dl.get_task_ids()

    def make_cost(goal):
        def stage(s, u):
            e = s - goal
            return Qw * (e @ e) + Rw * (u @ u)
        def terminal(s):
            e = s - goal
            return Qfw * (e @ e)
        return stage, terminal

    def step_fn(s, u):
        return A @ s + B @ np.clip(u, -50, 50)

    return _eval_tasks(
        env="P2P-Cost", dl=dl, task_ids=task_ids, dataset=ds,
        pretrained=pretrained, maml_model=maml_model, inner_lr=inner_lr,
        make_cost=make_cost, step_fn=step_fn, horizon=horizon, max_action=max_action,
        sm=sm, ss=ss, goal_of=lambda tid: np.asarray(ds["goal_states"][tid], float),
        is_setonet_ctx=True)


# ── Per-task evaluation loop (shared) ───────────────────────────────────────

def _eval_tasks(env, dl, task_ids, dataset, pretrained, maml_model, inner_lr,
                make_cost, step_fn, horizon, max_action, sm, ss, goal_of, is_setonet_ctx):
    """Returns {method: np.array(normalized_cost per task)}."""
    out = {m: [] for m in METHODS}
    for tid in task_ids:
        goal = goal_of(tid)
        stage, terminal = make_cost(goal)
        # held-out expert trajectories (physical) for start states + expert cost
        eb = dl.sample_holdout(tid, K, N=8)
        tgt_states = np.asarray(eb[4])                       # (N,H,state+1) normalized
        tgt_ctrls = np.asarray(eb[5]) * max_action           # (N,H,ctrl) physical
        starts = tgt_states[:, 0, :-1] * ss + sm             # (N, state) physical start states

        # expert achieved cost per trajectory (stored states/actions, same cost fn)
        exp_states = tgt_states[..., :-1] * ss + sm          # (N,H,state)
        expert_costs = []
        for n in range(exp_states.shape[0]):
            c = sum(float(stage(exp_states[n, t], tgt_ctrls[n, t])) for t in range(exp_states.shape[1]))
            c += float(terminal(exp_states[n, -1]))
            expert_costs.append(c)
        expert_costs = np.array(expert_costs)

        # branch context for SetONet methods (fixed per task)
        if is_setonet_ctx:
            si, sv = _get_random_context(dl, tid, dataset)

        for method in METHODS:
            if method == "maml":
                if maml_model is None:
                    out[method].append(np.nan); continue
                am = _adapt_maml(maml_model, dl, tid, inner_lr, GRAD_STEPS, NUM_DEMOS)
                predict = lambda sn, tn, _am=am: np.asarray(_am(jnp.array([*sn, tn])))
            else:
                mode = {"pretrained": "none", "setonet_ft": "ft",
                        "last_branch": "last_branch", "last_both": "last_both"}[method]
                am = _adapt_setonet(pretrained, dl, tid, dataset, mode, GRAD_STEPS, NUM_DEMOS)
                predict = lambda sn, tn, _am=am: np.asarray(
                    _am(si, sv, jnp.array([*sn, tn])))
            ratios = []
            for n in range(starts.shape[0]):
                pc = rollout_cost(predict, starts[n], step_fn, stage, terminal,
                                  horizon, max_action, sm, ss)
                ratios.append(pc / (expert_costs[n] + 1e-9))
            out[method].append(float(np.mean(ratios)))
    return {m: np.array(v) for m, v in out.items()}


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_scatter(results, output):
    envs = list(results.keys())
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4.2))
    if len(envs) == 1:
        axes = [axes]
    for ax, env in zip(axes, envs):
        r = results[env]
        maml = r.get("maml")
        lim = 0.0
        for m in METHOD_LABELS:
            if m not in r:
                continue
            c, mk = METHOD_STYLE[m]
            ax.scatter(r[m], maml, s=30, c=c, marker=mk, alpha=0.8, label=METHOD_LABELS[m])
            lim = max(lim, np.nanmax(r[m]), np.nanmax(maml))
        lim *= 1.05
        ax.plot([0, lim], [0, lim], "--", color="#BBBBBB", label="y = x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(env); ax.set_xlabel("Model normalized cost")
        if ax is axes[0]:
            ax.set_ylabel("MAML normalized cost")
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{output}_scatter.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"saved {output}_scatter.pdf")


def plot_summary(results, output):
    envs = list(results.keys())
    methods = [m for m in METHODS]
    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(envs), 4))
    x = np.arange(len(envs)); w = 0.16
    for i, m in enumerate(methods):
        means = [np.nanmean(results[e][m]) for e in envs]
        stds = [np.nanstd(results[e][m]) for e in envs]
        color = METHOD_STYLE.get(m, ("#AA3377", "x"))[0]
        ax.bar(x + (i - len(methods) / 2) * w, means, w, yerr=stds, capsize=2,
               label=METHOD_LABELS.get(m, "MAML"), color=color)
    ax.axhline(1.0, ls="--", color="k", lw=1, label="expert (1.0)")
    ax.set_xticks(x); ax.set_xticklabels(envs); ax.set_ylabel("normalized achieved cost")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(f"{output}_summary.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"saved {output}_summary.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--output", default="outputs/figures/figure6_cost")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--envs", default="p2p_cost",
                    help="comma-separated: p2p_cost,p2p_dynamics,quadrotor")
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    RUNNERS = {"p2p_cost": run_p2p_cost}  # p2p_dynamics / quadrotor added next
    results = {}
    for env in args.envs.split(","):
        env = env.strip()
        if env not in RUNNERS:
            print(f"(skipping {env}: runner not implemented yet)"); continue
        r = RUNNERS[env](args.config, args.data, args.checkpoints, args.seed)
        results[{"p2p_cost": "P2P-Cost"}.get(env, env)] = r
        for m in METHODS:
            print(f"  {env:14} {m:12} normalized cost = {np.nanmean(r[m]):.3f} ± {np.nanstd(r[m]):.3f}")

    np.savez(f"{args.output}_cost.npz", **{f"{e}__{m}": v for e, rr in results.items() for m, v in rr.items()})
    plot_scatter(results, args.output)
    plot_summary(results, args.output)
    print(f"\nSaved per-task costs to {args.output}_cost.npz")


if __name__ == "__main__":
    main()
