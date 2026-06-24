"""Zero-shot comparison: task-conditioned MLP baseline vs pretrained SetONet.

Neither model adapts here, so this reports the relative L2 error (Eq. 19) of each
model's zero-shot predictions on the **same held-out test tasks**, looping over
the standard seed convention (``42 + 100*i``).

Fairness / no leakage:
- Stored-dataset envs (P2P-Dynamics, Quadrotor, Obstacle) use the canonical
  train/test split in ``baseline_data``. The pretrained SetONet trainers are
  restricted to the canonical *train* split, so the test tasks evaluated here were
  never seen by either model. For these envs both models are scored on the
  *identical* sampled trajectories per seed (a true head-to-head).
- P2P-Cost / Small generate fresh transfer tasks online; both models are scored on
  the same tasks and seeds (each with its own training normalization).

Usage:
    python src/evaluation/evaluate_baseline.py \
        --config configs/p2p_dynamics.yaml \
        --data data/p2p_dynamics \
        --checkpoint checkpoints/p2p_dynamics/baseline \
        --pretrained checkpoints/p2p_dynamics/pretrained \
        --output outputs/results/p2p_dynamics/baseline_zeroshot.json
"""

import argparse
import json
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import yaml
from pathlib import Path

from src.training.baseline_mlp import BaselineMLP, relative_l2

# Match src/evaluation/evaluate.py
NUM_SEEDS = 5
NUM_TASKS = 20
K = 64
N_EVAL = 32
NUM_EVAL_BATCHES = 5


# ── SetONet helpers ──────────────────────────────────────────────────────────

def build_setonet(cfg, in_src, out_src, in_tgt, out_tgt, key):
    """Construct a SetONet with the given I/O dims from the config's model block."""
    from src.setonet import SetONet
    m = cfg.get("model", {})
    return SetONet(
        input_size_src=in_src, output_size_src=out_src,
        input_size_tgt=in_tgt, output_size_tgt=out_tgt,
        p=m.get("p", 128),
        phi_hidden_size=m.get("phi_hidden_size", 128),
        phi_output_size=m.get("phi_output_size", 128),
        rho_hidden_size=m.get("rho_hidden_size", 128),
        trunk_hidden_size=m.get("trunk_hidden_size", 128),
        n_phi_layers=m.get("n_phi_layers", 2),
        n_rho_layers=m.get("n_rho_layers", 2),
        n_trunk_layers=m.get("n_trunk_layers", 2),
        aggregation_type=m.get("aggregation_type", "attention"),
        attention_n_heads=m.get("attention_n_heads", 4),
        attention_n_tokens=m.get("attention_n_tokens", 4),
        use_bias=m.get("use_bias", True),
        key=key,
    )


def predict_setonet(model, src_input, src_values, tgt_states):
    """Branch encodes (src_input, src_values); trunk predicts at each tgt state."""
    pred = jax.vmap(
        jax.vmap(model, in_axes=(None, None, 0)),
        in_axes=(None, None, 0),
    )(src_input, src_values, jnp.asarray(tgt_states))
    return np.array(pred)


# ── Pretrained predictors (stored envs) ──────────────────────────────────────

def make_pretrained_dynamics(time_mode, action_norm, normalized=False):
    """Build a pretrained-SetONet predictor for VaryingDynamics-style envs.

    time_mode: 'int' (P2P-Dynamics, raw integer time) or 'norm' (Quadrotor,
    normalized time over H+1 points with the last dropped).
    action_norm: if True, un-normalize predictions with the checkpoint's
    action_mean/action_std (Quadrotor).
    normalized: if True, the checkpoint is a NormalizedSetONet (P2P-Dynamics) —
    reconstruct the wrapper so deserialization matches; predictions are already
    in physical action space (no action_norm needed).
    """
    def maker(cfg, data_dir, pretrained_dir, info, key):
        sd, ad = info["state_dim"], info["action_dim"]
        model = build_setonet(cfg, sd + ad, sd, sd + 1, ad, key)
        if normalized:
            from src.setonet import NormalizedSetONet
            from src.normalization import GridEnvironmentNormalizer
            dc = cfg.get("data", {})
            xr = dc.get("workspace", {}).get("x_range", [-5.0, 5.0])
            mv = dc.get("dynamics_ranges", {}).get("max_velocity_range", [10.0, 15.0])[1]
            # max_acceleration is a placeholder — its leaves are overwritten on deserialize.
            normr = GridEnvironmentNormalizer(position_range=(xr[0], xr[1]), max_velocity=mv,
                                              max_acceleration=1.0)
            model = NormalizedSetONet(model, normr)
        model = eqx.tree_deserialise_leaves(Path(pretrained_dir) / "setonet.eqx", model)
        amean = astd = None
        if action_norm:
            hist = np.load(Path(pretrained_dir) / "training_history.npz")
            amean = np.array(hist["action_mean"]); astd = np.array(hist["action_std"])

        def predict(task, ti, ctx_idx):
            # Branch context drawn from ctx_idx trajectories (the adapt half).
            cs, ca, cns = task["states"][ctx_idx], task["actions"][ctx_idx], task["next_states"][ctx_idx]
            fs, fa, fns = cs.reshape(-1, sd), ca.reshape(-1, ad), cns.reshape(-1, sd)
            ci = np.random.choice(fs.shape[0], size=K, replace=K > fs.shape[0])
            src_input = jnp.asarray(np.concatenate([fs[ci], fa[ci]], axis=-1))
            src_values = jnp.asarray(fns[ci])

            ss = task["states"][ti]
            h = ss.shape[1]
            if time_mode == "int":
                t = np.arange(h, dtype=np.float32)[None, :, None]
            else:
                t = np.linspace(0.0, 1.0, h + 1, dtype=np.float32)[:-1][None, :, None]
            t = np.broadcast_to(t, (ss.shape[0], h, 1))
            tgt = np.concatenate([ss, t], axis=-1)

            pred = predict_setonet(model, src_input, src_values, tgt)
            if amean is not None:
                pred = pred * astd + amean
            return pred.reshape(-1, ad), task["actions"][ti].reshape(-1, ad)

        return predict
    return maker


def make_pretrained_obstacle():
    """Build a pretrained-SetONet predictor for the Obstacle env.

    Normalization constants (max_pos/max_vel/max_action) are taken from an
    ObstacleAvoidanceImitation built on the canonical *train* subset — exactly
    what the (restricted) pretrained trainer used.
    """
    def maker(cfg, data_dir, pretrained_dir, info, key):
        from src.envs.dataloader import ObstacleAvoidanceImitation
        from src.training.baseline_data import obstacle_split, restrict_obstacle_dataset

        model = build_setonet(cfg, 2, 1, 4 + 1, 2, key)
        model = eqx.tree_deserialise_leaves(Path(pretrained_dir) / "setonet.eqx", model)

        split_seed = cfg.get("training", {}).get("split_seed", 42)
        data = np.load(Path(data_dir) / "trajectories.npy", allow_pickle=True).item()
        train_keys, _ = obstacle_split(data_dir, split_seed=split_seed)
        dtrain = restrict_obstacle_dataset(data, train_keys)
        dl = ObstacleAvoidanceImitation(
            dtrain, train_perc=cfg.get("training", {}).get("train_perc", 0.8),
            normalize=True, seed=split_seed)
        max_pos, max_vel, max_action = float(dl.max_pos), float(dl.max_vel), float(dl.max_action)

        def predict(task, ti, ctx_idx=None):   # branch context = obstacle set (no trajectories)
            s = task["states"][ti].copy()      # (M, H, 4) raw
            a = task["actions"][ti]
            obs = task["obstacles"]            # (n_obs, 3) raw
            op = jnp.asarray(obs[:, :2] / max_pos)
            ov = jnp.asarray(obs[:, 2:3] / max_pos)
            s[..., :2] = s[..., :2] / max_pos
            s[..., 2:] = s[..., 2:] / max_vel
            h = s.shape[1]
            t = (np.arange(h + 1, dtype=np.float32) / h)[:-1][None, :, None]
            t = np.broadcast_to(t, (s.shape[0], h, 1))
            tgt = np.concatenate([s, t], axis=-1)
            pred = predict_setonet(model, op, ov, tgt) * max_action
            return pred.reshape(-1, 2), a.reshape(-1, 2)

        return predict
    return maker


# ── Baseline 2 predictor (context-conditioned MLP, stored envs) ──────────────

def make_b2_dynamics(time_mode):
    """Context-conditioned MLP predictor for VaryingDynamics-style envs.

    Reconstructs the same flattened-context input used in training: sample n_ctx
    (state, action, next_state) transitions from the task, scale + pad to `slots`,
    concat with the standardized [state, time] queries.
    """
    def maker(cfg, data_dir, b2_dir, info, key):
        from src.training.baseline2_common import pad_flatten_context, build_input
        h = np.load(Path(b2_dir) / "training_history.npz")
        sd, ad = int(h["state_dim"]), int(h["action_dim"])
        slots, elem_dim, n_ctx = int(h["slots"]), int(h["elem_dim"]), int(h["n_ctx"])
        ctx_scale = np.array(h["ctx_scale"])
        st_mean, st_std = np.array(h["st_mean"]), np.array(h["st_std"])
        y_mean, y_std = np.array(h["y_mean"]), np.array(h["y_std"])
        in_dim = sd + 1 + slots * elem_dim
        model = BaselineMLP(in_dim, ad, int(h["width"]), int(h["depth"]), key)
        model = eqx.tree_deserialise_leaves(Path(b2_dir) / "baseline2.eqx", model)

        def predict(task, ti, ctx_idx):
            cs, ca, cns = task["states"][ctx_idx], task["actions"][ctx_idx], task["next_states"][ctx_idx]
            fs, fa, fns = cs.reshape(-1, sd), ca.reshape(-1, ad), cns.reshape(-1, sd)
            ci = np.random.choice(fs.shape[0], size=n_ctx, replace=n_ctx > fs.shape[0])
            ctx = np.concatenate([fs[ci], fa[ci], fns[ci]], axis=-1)       # (n_ctx, elem)
            flat = pad_flatten_context(ctx, slots, ctx_scale)

            ss = task["states"][ti]
            hh = ss.shape[1]
            if time_mode == "int":
                t = np.arange(hh, dtype=np.float32)[None, :, None]
            else:
                t = np.linspace(0.0, 1.0, hh + 1, dtype=np.float32)[:-1][None, :, None]
            t = np.broadcast_to(t, (ss.shape[0], hh, 1))
            queries = np.concatenate([ss, t], axis=-1)
            X = build_input(queries, flat, st_mean, st_std)
            pred = np.array(jax.vmap(model)(jnp.asarray(X))) * y_std + y_mean
            return pred.reshape(-1, ad), task["actions"][ti].reshape(-1, ad)

        return predict
    return maker


def make_b2_obstacle():
    """Context-conditioned MLP predictor for Obstacle (context = obstacle set)."""
    def maker(cfg, data_dir, b2_dir, info, key):
        from src.training.baseline2_common import pad_flatten_context, build_input
        h = np.load(Path(b2_dir) / "training_history.npz")
        sd, ad = int(h["state_dim"]), int(h["action_dim"])
        slots, elem_dim = int(h["slots"]), int(h["elem_dim"])
        ctx_scale = np.array(h["ctx_scale"])
        st_mean, st_std = np.array(h["st_mean"]), np.array(h["st_std"])
        y_mean, y_std = np.array(h["y_mean"]), np.array(h["y_std"])
        model = BaselineMLP(sd + 1 + slots * elem_dim, ad, int(h["width"]), int(h["depth"]), key)
        model = eqx.tree_deserialise_leaves(Path(b2_dir) / "baseline2.eqx", model)

        def predict(task, ti, ctx_idx=None):   # context = obstacle set
            flat = pad_flatten_context(np.asarray(task["obstacles"]), slots, ctx_scale)
            ss = task["states"][ti]
            hh = ss.shape[1]
            t = np.broadcast_to(np.linspace(0.0, 1.0, hh, endpoint=False)[None, :, None],
                                (ss.shape[0], hh, 1))
            queries = np.concatenate([ss, t], axis=-1)
            X = build_input(queries, flat, st_mean, st_std)
            pred = np.array(jax.vmap(model)(jnp.asarray(X))) * y_std + y_mean
            return pred.reshape(-1, ad), task["actions"][ti].reshape(-1, ad)

        return predict
    return maker


# ── Baseline loaders ─────────────────────────────────────────────────────────

def _load_baseline(checkpoint_dir, input_dim, output_dim, key):
    """Load the P2P-Cost BaselineMLP and its saved normalization stats."""
    ckpt = Path(checkpoint_dir)
    hist = np.load(ckpt / "training_history.npz")
    model = BaselineMLP(input_dim, output_dim, int(hist["hidden_size"]), int(hist["num_layers"]), key)
    model = eqx.tree_deserialise_leaves(ckpt / "baseline.eqx", model)
    norm_stats = {
        "state_mean": np.array(hist["state_mean"]), "state_std": np.array(hist["state_std"]),
        "cost_mean": float(hist["cost_mean"]), "cost_std": float(hist["cost_std"]),
    }
    return model, norm_stats, float(hist["max_action"])


def _load_stored_baseline(checkpoint_dir, key):
    """Load a stored-env BaselineMLP plus its saved standardization stats."""
    ckpt = Path(checkpoint_dir)
    hist = np.load(ckpt / "training_history.npz")
    state_dim, action_dim, param_dim = int(hist["state_dim"]), int(hist["action_dim"]), int(hist["param_dim"])
    model = BaselineMLP(state_dim + 1 + param_dim, action_dim,
                        int(hist["hidden_size"]), int(hist["num_layers"]), key)
    model = eqx.tree_deserialise_leaves(ckpt / "baseline.eqx", model)
    stats = {
        "st_mean": np.array(hist["st_mean"]), "st_std": np.array(hist["st_std"]),
        "y_mean": np.array(hist["y_mean"]), "y_std": np.array(hist["y_std"]),
        "split_seed": int(hist["split_seed"]),
        "state_dim": state_dim, "action_dim": action_dim, "param_dim": param_dim,
    }
    return model, stats


# ── Output helper ────────────────────────────────────────────────────────────

def _summary(seed_means):
    return {"seed_means": seed_means, "mean": float(np.mean(seed_means)), "std": float(np.std(seed_means))}


def _save(result, output_dir):
    out = Path(output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")


# ── P2P-Cost (online transfer tasks) ─────────────────────────────────────────

def _random_context(goal, K, ranges, Qw, Rw):
    """Random (state, control) context with immediate cost — matches how the
    P2P-Cost operator/B2 branch context is sampled during *training* (uniform
    state/control), as opposed to on-policy transitions from optimal trajectories."""
    (s_lo, s_hi), (v_lo, v_hi), (c_lo, c_hi) = ranges
    sp = np.random.uniform(s_lo, s_hi, (K, 2))
    sv = np.random.uniform(v_lo, v_hi, (K, 2))
    states = np.concatenate([sp, sv], axis=-1)
    ctrl = np.random.uniform(c_lo, c_hi, (K, 2))
    err = states - goal
    cost = (Qw * np.sum(err ** 2, axis=1) + Rw * np.sum(ctrl ** 2, axis=1)).reshape(-1, 1)
    return states, ctrl, cost


def run_p2p_cost(cfg, data_dir, checkpoint_dir, output_dir, seed=42,
                 pretrained_dir=None, baseline2_dir=None, context="onpolicy"):
    from src.envs.create_p2p_cost import generate_dataset
    from src.evaluation.evaluate import TransferTaskDataLoader

    state_dim, control_dim = 4, 2
    input_dim = state_dim + 1 + state_dim  # [state, time, goal]

    # Branch-context distribution for the context-using models (SetONet, B2).
    # "random" matches training; "onpolicy" uses transitions from optimal trajectories.
    tcfg = cfg.get("training", {})
    ranges = (tuple(tcfg.get("state_range", [-10.0, 10.0])),
              tuple(tcfg.get("src_vel_range", [-10.0, 10.0])),
              tuple(tcfg.get("src_control_range", [-5.0, 5.0])))
    Qw_eval, Rw_eval = 1.0, 0.1  # transfer-task cost weights (see generate_dataset below)
    num_tasks = int(tcfg.get("num_eval_tasks", NUM_TASKS))  # paper: 20 (cost) / 10 (small)
    print(f"P2P-Cost branch context: {context} | transfer tasks: {num_tasks}")

    key = jr.PRNGKey(seed)
    model, norm_stats, max_action = _load_baseline(checkpoint_dir, input_dim, control_dim, key)
    state_mean, state_std = norm_stats["state_mean"], norm_stats["state_std"]

    # Optional pretrained SetONet (its own normalization).
    pre_model = pre_norm = pre_max_action = None
    pdir = Path(pretrained_dir) if pretrained_dir else None
    if pdir and (pdir / "setonet.eqx").exists():
        pre_model = build_setonet(cfg, state_dim + control_dim, 1, state_dim + 1, control_dim, jr.split(key)[0])
        pre_model = eqx.tree_deserialise_leaves(pdir / "setonet.eqx", pre_model)
        ph = np.load(pdir / "training_history.npz")
        pre_norm = {"state_mean": np.array(ph["state_mean"]), "state_std": np.array(ph["state_std"]),
                    "cost_mean": float(ph["cost_mean"]), "cost_std": float(ph["cost_std"])}
        pre_max_action = float(ph["max_action"])
        print(f"Evaluating pretrained SetONet from {pdir}")
    elif pretrained_dir:
        print(f"(pretrained checkpoint not found at {pretrained_dir}; baseline only)")

    # Optional Baseline 2 (context-conditioned MLP, its own normalization).
    b2 = None
    b2dir = Path(baseline2_dir) if baseline2_dir else None
    if b2dir and (b2dir / "baseline2.eqx").exists():
        from src.training.baseline2_common import pad_flatten_context, build_input
        bh = np.load(b2dir / "training_history.npz")
        slots, elem, n_ctx = int(bh["slots"]), int(bh["elem_dim"]), int(bh["n_ctx"])
        sd, ad = int(bh["state_dim"]), int(bh["action_dim"])
        b2model = BaselineMLP(sd + 1 + slots * elem, ad, int(bh["width"]), int(bh["depth"]), jr.split(key)[1])
        b2model = eqx.tree_deserialise_leaves(b2dir / "baseline2.eqx", b2model)
        b2 = {"model": b2model, "slots": slots, "elem": elem, "n_ctx": n_ctx, "ad": ad,
              "ctx_scale": np.array(bh["ctx_scale"]),
              "st_mean": np.array(bh["st_mean"]), "st_std": np.array(bh["st_std"]),
              "y_mean": np.array(bh["y_mean"]), "y_std": np.array(bh["y_std"]),
              "norm": {"state_mean": np.array(bh["state_mean"]), "state_std": np.array(bh["state_std"]),
                       "cost_mean": float(bh["cost_mean"]), "cost_std": float(bh["cost_std"])},
              "max_action": float(bh["max_action"]),
              "pad": pad_flatten_context, "binput": build_input}
        print(f"Evaluating Baseline 2 (context MLP) from {b2dir}")

    base_means, pre_means, b2_means = [], [], []
    for seed_i in range(NUM_SEEDS):
        s = 42 + seed_i * 100
        transfer_ds = generate_dataset(
            num_goals=num_tasks, trajectories_per_goal=50, horizon=50, dt=0.1,
            Q_weight=1.0, R_weight=0.1, Qf_weight=10.0,
            goal_range=(-5.0, 5.0), state_range=(-10.0, 10.0), vel_range=(-5.0, 5.0),
            zero_velocity_goal=True, seed=s + 1000)

        # Baseline dataloader (baseline normalization).
        ds_b = dict(transfer_ds); ds_b["norm_stats"] = norm_stats
        np.random.seed(s)
        dl = TransferTaskDataLoader(ds_b, train_holdout_split=0.25, normalize=True)
        dl.max_action = max_action
        base_errs = []
        for tid in dl.get_task_ids():
            goal_n = (dl.task_data[tid]["goal_state"] - state_mean) / state_std
            errs = []
            for _ in range(NUM_EVAL_BATCHES):
                eb = dl.sample_holdout(tid, K, N=N_EVAL)
                ts, tc = eb[4], eb[5]
                n, h = ts.shape[0], ts.shape[1]
                goal_bc = np.broadcast_to(goal_n[None, None, :], (n, h, state_dim))
                x = np.concatenate([ts, goal_bc], axis=-1).reshape(-1, input_dim)
                pred = np.array(jax.vmap(model)(jnp.asarray(x)))
                errs.append(relative_l2(pred * max_action, tc.reshape(-1, control_dim) * max_action))
            base_errs.append(float(np.mean(errs)))
        base_means.append(float(np.mean(base_errs)))

        msg = f"  Seed {s}: baseline={base_means[-1]:.6f}"

        if pre_model is not None:
            ds_p = dict(transfer_ds); ds_p["norm_stats"] = pre_norm
            np.random.seed(s)
            dlp = TransferTaskDataLoader(ds_p, train_holdout_split=0.25, normalize=True)
            dlp.max_action = pre_max_action
            pn = pre_norm
            pre_errs = []
            for tid in dlp.get_task_ids():
                errs = []
                for _ in range(NUM_EVAL_BATCHES):
                    if context == "random":
                        rs, rc, rco = _random_context(dlp.task_data[tid]["goal_state"], K, ranges, Qw_eval, Rw_eval)
                        si = jnp.asarray(np.concatenate([(rs - pn["state_mean"]) / pn["state_std"],
                                                         rc / pre_max_action], axis=-1))
                        sv = jnp.asarray((rco - pn["cost_mean"]) / pn["cost_std"])
                    else:
                        sb = dlp.sample_train(tid, K, N=1)
                        si = jnp.asarray(np.concatenate([sb[0], sb[1]], axis=-1))
                        sv = jnp.asarray(sb[2])
                    eb = dlp.sample_holdout(tid, K, N=N_EVAL)
                    pred = predict_setonet(pre_model, si, sv, eb[4])
                    errs.append(relative_l2(pred * pre_max_action, eb[5] * pre_max_action))
                pre_errs.append(float(np.mean(errs)))
            pre_means.append(float(np.mean(pre_errs)))
            msg += f" | pretrained={pre_means[-1]:.6f}"

        if b2 is not None:
            ds_2 = dict(transfer_ds); ds_2["norm_stats"] = b2["norm"]
            np.random.seed(s)
            dl2 = TransferTaskDataLoader(ds_2, train_holdout_split=0.25, normalize=True)
            dl2.max_action = b2["max_action"]
            b2n = b2["norm"]
            b2_errs = []
            for tid in dl2.get_task_ids():
                errs = []
                for _ in range(NUM_EVAL_BATCHES):
                    if context == "random":
                        rs, rc, rco = _random_context(dl2.task_data[tid]["goal_state"], b2["n_ctx"], ranges, Qw_eval, Rw_eval)
                        ctx = np.concatenate([(rs - b2n["state_mean"]) / b2n["state_std"],
                                              rc / b2["max_action"],
                                              (rco - b2n["cost_mean"]) / b2n["cost_std"]], axis=-1)
                    else:
                        sb = dl2.sample_train(tid, b2["n_ctx"], N=1)
                        ctx = np.concatenate([sb[0], sb[1], sb[2]], axis=-1)   # (n_ctx, 7)
                    flat = b2["pad"](ctx, b2["slots"], b2["ctx_scale"])
                    eb = dl2.sample_holdout(tid, K, N=N_EVAL)
                    X = b2["binput"](eb[4], flat, b2["st_mean"], b2["st_std"])
                    pred = np.array(jax.vmap(b2["model"])(jnp.asarray(X))) * b2["y_std"] + b2["y_mean"]
                    errs.append(relative_l2(pred, eb[5].reshape(-1, b2["ad"])))
                b2_errs.append(float(np.mean(errs)))
            b2_means.append(float(np.mean(b2_errs)))
            msg += f" | baseline2={b2_means[-1]:.6f}"
        print(msg)

    result = {"env": "p2p_cost", "metric": "relative_l2", "grad_steps": 0,
              "num_seeds": NUM_SEEDS, "baseline": _summary(base_means)}
    if b2_means:
        result["baseline2"] = _summary(b2_means)
    if pre_means:
        result["pretrained"] = _summary(pre_means)
    parts = [f"baseline {result['baseline']['mean']:.4f}±{result['baseline']['std']:.4f}"]
    if b2_means:
        parts.append(f"baseline2 {result['baseline2']['mean']:.4f}±{result['baseline2']['std']:.4f}")
    if pre_means:
        parts.append(f"pretrained {result['pretrained']['mean']:.4f}±{result['pretrained']['std']:.4f}")
    print("\nP2P-Cost zero-shot: " + " | ".join(parts))
    _save(result, output_dir)
    return result


# ── Stored-dataset envs ──────────────────────────────────────────────────────

def run_stored(env_name, extractor, cfg, data_dir, checkpoint_dir, output_dir, seed=42,
               pretrained_dir=None, make_pretrained=None, baseline2_dir=None, make_b2=None,
               n_eval=N_EVAL, holdout_split=None):
    """Head-to-head zero-shot eval on a stored env's canonical held-out test tasks.

    Scores up to three models on the *same* sampled trajectories per seed:
    baseline (oracle task params), baseline2 (context-conditioned MLP), and the
    pretrained SetONet operator. Each predictor receives ``(task, ti, ctx_idx)``:
    ``ti`` = eval trajectory indices, ``ctx_idx`` = trajectories the branch context
    is drawn from. With ``holdout_split`` (paper protocol) each task's trajectories
    are split so context and eval come from disjoint halves.
    """
    key = jr.PRNGKey(seed)
    model, stats = _load_stored_baseline(checkpoint_dir, key)
    state_dim, action_dim = stats["state_dim"], stats["action_dim"]
    st = state_dim + 1
    st_mean, st_std, y_mean, y_std = stats["st_mean"], stats["st_std"], stats["y_mean"], stats["y_std"]

    _train, test_tasks, info = extractor(data_dir, split_seed=stats["split_seed"])
    input_dim = state_dim + 1 + stats["param_dim"]

    def baseline_predict(task, ti, ctx_idx):  # oracle task params — no branch context
        s, a = task["states"][ti], task["actions"][ti]
        n, h = s.shape[0], s.shape[1]
        time = np.broadcast_to(np.linspace(0.0, 1.0, h, endpoint=False)[None, :, None], (n, h, 1))
        param = np.broadcast_to(task["param"][None, None, :], (n, h, task["param"].shape[0]))
        x = np.concatenate([s, time, param], axis=-1).reshape(-1, input_dim)
        x[:, :st] = (x[:, :st] - st_mean) / st_std
        pred = np.array(jax.vmap(model)(jnp.asarray(x))) * y_std + y_mean
        return pred, a.reshape(-1, action_dim)

    # Optional models. Each `predict(task, ti) -> (pred_phys, target_phys)`.
    predictors = {"baseline": baseline_predict}
    if baseline2_dir and make_b2 and (Path(baseline2_dir) / "baseline2.eqx").exists():
        predictors["baseline2"] = make_b2(cfg, data_dir, Path(baseline2_dir), info, jr.split(key)[1])
        print(f"Evaluating Baseline 2 (context MLP) from {baseline2_dir}")
    pdir = Path(pretrained_dir) if pretrained_dir else None
    if pdir and make_pretrained and (pdir / "setonet.eqx").exists():
        predictors["pretrained"] = make_pretrained(cfg, data_dir, pdir, info, jr.split(key)[0])
        print(f"Evaluating pretrained SetONet from {pdir}")
    elif pretrained_dir:
        print(f"(pretrained checkpoint not found at {pretrained_dir})")

    means = {name: [] for name in predictors}
    for seed_i in range(NUM_SEEDS):
        s = 42 + seed_i * 100
        np.random.seed(s)
        per_task = {name: [] for name in predictors}
        for task in test_tasks:
            n_traj = task["states"].shape[0]
            errs = {name: [] for name in predictors}
            for _ in range(NUM_EVAL_BATCHES):
                if holdout_split is not None:
                    perm = np.random.permutation(n_traj)
                    cut = max(1, int(n_traj * holdout_split))
                    ctx_idx, eval_pool = perm[:cut], perm[cut:]
                    if len(eval_pool) == 0:
                        eval_pool = perm[cut - 1:]
                else:
                    ctx_idx = eval_pool = np.arange(n_traj)
                ti = eval_pool[np.random.choice(len(eval_pool), size=n_eval, replace=n_eval > len(eval_pool))]
                for name, fn in predictors.items():
                    p, t = fn(task, ti, ctx_idx)
                    errs[name].append(relative_l2(p, t))
            for name in predictors:
                per_task[name].append(float(np.mean(errs[name])))
        for name in predictors:
            means[name].append(float(np.mean(per_task[name])))
        print("  Seed %d: " % s + " | ".join(f"{n}={means[n][-1]:.6f}" for n in predictors))

    result = {"env": env_name, "metric": "relative_l2", "grad_steps": 0,
              "num_seeds": NUM_SEEDS, "num_test_tasks": len(test_tasks)}
    for name in predictors:
        result[name] = _summary(means[name])
    print(f"\n{env_name} zero-shot: " + " | ".join(
        f"{n} {result[n]['mean']:.4f}±{result[n]['std']:.4f}" for n in predictors))
    _save(result, output_dir)
    return result


def run_p2p_dynamics(cfg, data_dir, checkpoint_dir, output_dir, seed=42,
                     pretrained_dir=None, baseline2_dir=None):
    from src.training.baseline_data import extract_dynamics
    return run_stored("p2p_dynamics", extract_dynamics, cfg, data_dir, checkpoint_dir,
                      output_dir, seed, pretrained_dir,
                      make_pretrained_dynamics(time_mode="int", action_norm=False, normalized=True),
                      baseline2_dir, make_b2_dynamics(time_mode="int"),
                      n_eval=8, holdout_split=0.5)   # paper protocol


def run_quadrotor(cfg, data_dir, checkpoint_dir, output_dir, seed=42,
                  pretrained_dir=None, baseline2_dir=None):
    from src.training.baseline_data import extract_dynamics
    return run_stored("quadrotor", extract_dynamics, cfg, data_dir, checkpoint_dir,
                      output_dir, seed, pretrained_dir,
                      make_pretrained_dynamics(time_mode="norm", action_norm=True),
                      baseline2_dir, make_b2_dynamics(time_mode="norm"),
                      n_eval=4, holdout_split=0.5)   # paper protocol


def run_obstacle(cfg, data_dir, checkpoint_dir, output_dir, seed=42,
                 pretrained_dir=None, baseline2_dir=None):
    from src.training.baseline_data import extract_obstacle
    return run_stored("obstacle", extract_obstacle, cfg, data_dir, checkpoint_dir,
                      output_dir, seed, pretrained_dir, make_pretrained_obstacle(),
                      baseline2_dir, make_b2_obstacle())


RUNNERS = {
    "p2p_cost": run_p2p_cost,
    "p2p_dynamics": run_p2p_dynamics,
    "quadrotor": run_quadrotor,
    "obstacle": run_obstacle,
}


def main():
    parser = argparse.ArgumentParser(description="Zero-shot eval: MLP baseline vs pretrained SetONet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="baseline checkpoint dir")
    parser.add_argument("--pretrained", default=None, help="pretrained SetONet checkpoint dir (optional)")
    parser.add_argument("--baseline2", default=None, help="context-MLP baseline2 checkpoint dir (optional)")
    parser.add_argument("--context", default="onpolicy", choices=["random", "onpolicy"],
                        help="P2P-Cost branch context for SetONet/B2 (random matches training)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    generator = cfg.get("generator")

    runner = RUNNERS.get(generator)
    if runner is None:
        print(f"Baseline eval for '{generator}' not implemented (available: {list(RUNNERS)})",
              file=sys.stderr)
        sys.exit(1)

    extra = {"pretrained_dir": args.pretrained, "baseline2_dir": args.baseline2}
    if generator == "p2p_cost":
        extra["context"] = args.context  # only the P2P-Cost path uses a context distribution
    runner(cfg, args.data, args.checkpoint, args.output, args.seed, **extra)


if __name__ == "__main__":
    main()
