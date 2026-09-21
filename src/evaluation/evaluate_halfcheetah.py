"""HalfCheetah-v3 adaptation grid (paper Figure 10).

For every held-out configuration, ``num_demos`` episodes are used for adaptation
and the remaining episodes for evaluation (random ``H``-transition window per
episode). The adaptation transitions also form the operator's context set.
Metric: relative L2 error of the predicted actions in the original action scale,
averaged over held-out configurations; one value per (method, demos, steps, seed).

Methods:
  setonet_ft         full-network fine-tuning of the pretrained operator (Adam)
  last_layer         last trunk + rho layer fine-tuning (Adam)
  maml               MAML MLP baseline, SGD inner steps
  setonet_meta       SetONet-Meta, branch-only SGD steps (trunk frozen)
  setonet_meta_full  SetONet-Meta-Full, full-network SGD steps

Usage (matches Makefile):
    python src/evaluation/evaluate_halfcheetah.py \
        --config configs/halfcheetah.yaml --data data/halfcheetah \
        --checkpoints checkpoints/halfcheetah --output outputs/results/halfcheetah_grid

Progress is checkpointed to ``<output>/grid_checkpoint.json`` and resumed on rerun.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import yaml

from src.envs.imujoco_dataloader import IMuJoCoImitation
from src.training.halfcheetah_common import (
    branch_only_filter_spec, last_layer_filter_spec, load_maml, load_setonet,
    predict_setonet, relative_l2,
)

METHODS = ["setonet_ft", "last_layer", "maml", "setonet_meta", "setonet_meta_full"]
CHECKPOINT_FILES = {
    "pretrained": ("pretrained", "setonet.eqx"),
    "maml": ("maml", "maml.eqx"),
    "meta_branch": ("meta_branch", "setonet.eqx"),
    "meta_full": ("meta_full", "setonet.eqx"),
}


# ── Adaptation primitives ──

def _adam_finetune(model, filter_spec, context_sa, context_ns, adapt_s, adapt_a, steps, lr):
    """Adam fine-tuning of the parameters selected by ``filter_spec``."""
    trainable, frozen = eqx.partition(model, filter_spec)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(trainable)

    @eqx.filter_jit
    def step(trainable, opt_state):
        def loss_fn(t):
            m = eqx.combine(t, frozen)
            pred = predict_setonet(m, context_sa, context_ns, adapt_s)
            return jnp.mean(jnp.square(pred - adapt_a))
        _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(trainable, updates), opt_state

    for _ in range(steps):
        trainable, opt_state = step(trainable, opt_state)
    return eqx.combine(trainable, frozen)


def _sgd_adapt(model, filter_spec, context_sa, context_ns, adapt_s, adapt_a, steps, lr):
    """Plain SGD inner-loop steps (MAML-style) on the parameters selected by ``filter_spec``."""
    diff, static = eqx.partition(model, filter_spec)

    @eqx.filter_jit
    def step(diff):
        def loss_fn(d):
            m = eqx.combine(d, static)
            pred = predict_setonet(m, context_sa, context_ns, adapt_s)
            return jnp.mean(jnp.square(pred - adapt_a))
        _, grads = eqx.filter_value_and_grad(loss_fn)(diff)
        return eqx.apply_updates(diff, jax.tree_util.tree_map(lambda g: -lr * g, grads))

    for _ in range(steps):
        diff = step(diff)
    return eqx.combine(diff, static)


def adapt_maml(model, adapt_s, adapt_a, steps, lr):
    @eqx.filter_jit
    def step(m):
        def loss_fn(mm):
            return jnp.mean(jnp.square(jax.vmap(mm)(adapt_s) - adapt_a))
        _, grads = eqx.filter_value_and_grad(loss_fn)(m)
        return eqx.apply_updates(m, jax.tree_util.tree_map(lambda g: -lr * g, grads))

    for _ in range(steps):
        model = step(model)
    return model


def adapt(method, models, lrs, context_sa, context_ns, adapt_s, adapt_a, steps):
    """Return the adapted model for ``method``; ``steps == 0`` returns it unchanged."""
    if method == "setonet_ft":
        spec = jax.tree_util.tree_map(eqx.is_array, models["pretrained"])
        return _adam_finetune(models["pretrained"], spec, context_sa, context_ns,
                              adapt_s, adapt_a, steps, lrs["setonet_ft"])
    if method == "last_layer":
        return _adam_finetune(models["pretrained"], last_layer_filter_spec(models["pretrained"]),
                              context_sa, context_ns, adapt_s, adapt_a, steps, lrs["last_layer"])
    if method == "maml":
        return adapt_maml(models["maml"], adapt_s, adapt_a, steps, lrs["maml"])
    if method == "setonet_meta":
        return _sgd_adapt(models["meta_branch"], branch_only_filter_spec(models["meta_branch"]),
                          context_sa, context_ns, adapt_s, adapt_a, steps, lrs["setonet_meta"])
    if method == "setonet_meta_full":
        spec = jax.tree_util.tree_map(eqx.is_array, models["meta_full"])
        return _sgd_adapt(models["meta_full"], spec, context_sa, context_ns,
                          adapt_s, adapt_a, steps, lrs["setonet_meta_full"])
    raise ValueError(method)


def predict(method, model, context_sa, context_ns, eval_s):
    if method == "maml":
        return np.array(jax.vmap(model)(eval_s))
    return np.array(predict_setonet(model, context_sa, context_ns, eval_s))


# ── Data helpers ──

def split_task_episodes(data_loader, episodes, num_demos, H, rng):
    """Adaptation/evaluation windows for one held-out task, or ``None`` if too few episodes."""
    n = len(episodes)
    if num_demos >= n:
        return None
    perm = rng.permutation(n)
    adapt_idx, eval_idx = perm[:num_demos], perm[num_demos:]

    def windows(idx):
        s_l, a_l, ns_l = [], [], []
        for i in idx:
            s, a, ns = data_loader.sample_window(episodes[i], H, rng)
            s_l.append(s)
            a_l.append(a)
            ns_l.append(ns)
        return (np.concatenate(s_l), np.concatenate(a_l), np.concatenate(ns_l))

    adapt_s, adapt_a, adapt_ns = windows(adapt_idx)
    eval_s, eval_a, _ = windows(eval_idx)
    return dict(
        context_sa=jnp.array(np.concatenate([adapt_s, adapt_a], axis=-1)),
        context_ns=jnp.array(adapt_ns),
        adapt_s=jnp.array(adapt_s), adapt_a=jnp.array(adapt_a),
        eval_s=jnp.array(eval_s),
        eval_a_denorm=np.array(data_loader.denormalize_actions(eval_a)),
    )


def load_models(checkpoint_dir, cfg, obs_size, act_size, seed=0):
    """Load whichever checkpoints exist; methods whose checkpoint is missing are skipped."""
    ckpt = Path(checkpoint_dir)
    keys = jr.split(jr.PRNGKey(seed), 4)
    models = {}
    for i, (name, (sub, fname)) in enumerate(CHECKPOINT_FILES.items()):
        path = ckpt / sub / fname
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        if name == "maml":
            models[name] = load_maml(path, cfg, obs_size, act_size, keys[i])
        else:
            models[name] = load_setonet(path, cfg.get("model", {}), obs_size, act_size, keys[i])
        print(f"  loaded {name:12s} <- {path}")
    return models


def available_methods(models, requested):
    need = {"setonet_ft": "pretrained", "last_layer": "pretrained", "maml": "maml",
            "setonet_meta": "meta_branch", "setonet_meta_full": "meta_full"}
    return [m for m in requested if need[m] in models]


# ── Grid ──

def run_grid(cfg, data_dir, checkpoint_dir, output_dir, seed=42,
             methods=None, num_demos_list=None, grad_steps_list=None, num_seeds=None):
    data_cfg, ev = cfg["data"], cfg.get("evaluation", {})
    num_demos_list = list(num_demos_list or ev.get("num_demos_list", [1, 5, 10, 25]))
    grad_steps_list = list(grad_steps_list or ev.get("grad_steps_list", [1, 5, 10, 25, 50, 100, 200]))
    num_seeds = int(num_seeds or ev.get("num_seeds", 5))
    H = int(ev.get("H", 100))
    lrs = {
        "setonet_ft": float(ev.get("setonet_ft_lr", 1e-4)),
        "last_layer": float(ev.get("last_layer_lr", 1e-4)),
        "maml": float(ev.get("maml_inner_lr", cfg.get("maml", {}).get("inner_lr", 0.01))),
        "setonet_meta": float(ev.get("setonet_meta_inner_lr",
                                     cfg.get("setonet_meta", {}).get("inner_lr", 0.01))),
        "setonet_meta_full": float(ev.get("setonet_meta_full_inner_lr",
                                          cfg.get("setonet_meta", {}).get("inner_lr", 0.01))),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_loader = IMuJoCoImitation(
        data_dir=data_dir, env_name=data_cfg["env_name"],
        train_perc=data_cfg.get("train_perc", 0.8), seed=data_cfg.get("split_seed", 42),
        normalize=data_cfg.get("normalize", True),
    )
    obs_size, act_size = data_loader.obs_size, data_loader.act_size
    num_test_tasks = len(data_loader.test_data)

    print("Loading models...")
    models = load_models(checkpoint_dir, cfg, obs_size, act_size)
    methods = available_methods(models, methods or METHODS)
    if not methods:
        raise SystemExit("No checkpoints found; train with `make train ENV=halfcheetah` etc.")

    total = len(methods) * len(num_demos_list) * len(grad_steps_list) * num_seeds
    print(f"\nGrid: {methods} x demos {num_demos_list} x steps {grad_steps_list} "
          f"x {num_seeds} seeds = {total} combos on {num_test_tasks} held-out configs")

    # Resume support
    ckpt_path = out / "grid_checkpoint.json"
    results = {m: {nd: {gs: [None] * num_seeds for gs in grad_steps_list}
                   for nd in num_demos_list} for m in methods}
    if ckpt_path.exists():
        for e in json.load(open(ckpt_path))["results"]:
            m, nd, gs, si = e["method"], e["num_demos"], e["grad_steps"], e["seed_i"]
            if m in results and nd in results[m] and gs in results[m][nd] and si < num_seeds:
                results[m][nd][gs][si] = e["value"]
        n_done = sum(v is not None for m in results.values() for d in m.values()
                     for g in d.values() for v in g)
        print(f"Resumed {n_done} completed evaluations from {ckpt_path}")

    def save_checkpoint():
        entries = [{"method": m, "num_demos": nd, "grad_steps": gs, "seed_i": si, "value": v}
                   for m in results for nd in results[m] for gs in results[m][nd]
                   for si, v in enumerate(results[m][nd][gs]) if v is not None]
        json.dump({"results": entries, "env": data_cfg["env_name"], "H": H,
                   "num_test_tasks": num_test_tasks, "lrs": lrs}, open(ckpt_path, "w"))

    done = 0
    for seed_i in range(num_seeds):
        s = seed + seed_i * 100
        print(f"\n{'=' * 60}\nSeed {s} ({seed_i + 1}/{num_seeds})\n{'=' * 60}")
        for nd, gs in product(num_demos_list, grad_steps_list):
            todo = [m for m in methods if results[m][nd][gs][seed_i] is None]
            if not todo:
                continue
            per_task = {m: [] for m in todo}
            for task_idx in range(num_test_tasks):
                rng = np.random.RandomState(s * 10000 + task_idx)
                split = split_task_episodes(data_loader, data_loader.test_data[task_idx], nd, H, rng)
                if split is None:
                    continue
                for m in todo:
                    model = adapt(m, models, lrs, split["context_sa"], split["context_ns"],
                                  split["adapt_s"], split["adapt_a"], gs)
                    pred = predict(m, model, split["context_sa"], split["context_ns"], split["eval_s"])
                    pred = data_loader.denormalize_actions(pred)
                    per_task[m].append(relative_l2(pred, split["eval_a_denorm"]))
            for m in todo:
                val = float(np.mean(per_task[m])) if per_task[m] else float("nan")
                results[m][nd][gs][seed_i] = val
                done += 1
                print(f"  [{done:3d}/{total}] {m:18s} demos={nd:2d} steps={gs:3d} => {val:.6f}")
            save_checkpoint()

    # Save final arrays
    save_dict = {
        "methods": np.array(methods), "num_demos_list": np.array(num_demos_list),
        "grad_steps_list": np.array(grad_steps_list), "num_seeds": num_seeds,
        "num_tasks": num_test_tasks, "env_name": data_cfg["env_name"], "H": H,
        "metric": "relative_l2",
    }
    for m in methods:
        arr = np.full((len(num_demos_list), len(grad_steps_list), num_seeds), np.nan)
        for i, nd in enumerate(num_demos_list):
            for j, gs in enumerate(grad_steps_list):
                arr[i, j, :] = results[m][nd][gs]
        save_dict[m] = arr
    np.savez(out / "grid_results.npz", **save_dict)
    save_checkpoint()
    print(f"\nResults saved to {out / 'grid_results.npz'} (and grid_checkpoint.json)")

    print("\nMean relative L2 over seeds (rows: demos, cols: gradient steps)")
    for m in methods:
        print(f"\n  {m}:  steps " + " ".join(f"{gs:>7d}" for gs in grad_steps_list))
        for i, nd in enumerate(num_demos_list):
            print(f"    {nd:3d} demos " + " ".join(f"{np.nanmean(save_dict[m][i, j]):7.4f}"
                                              for j in range(len(grad_steps_list))))


def main():
    p = argparse.ArgumentParser(description="HalfCheetah-v3 adaptation grid (Figure 10)")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoints", required=True, help="dir with pretrained/, maml/, meta_branch/, meta_full/")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="*", default=None, choices=METHODS)
    p.add_argument("--demos", nargs="*", type=int, default=None)
    p.add_argument("--steps", nargs="*", type=int, default=None)
    p.add_argument("--num-seeds", type=int, default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_grid(cfg, args.data, args.checkpoints, args.output, args.seed,
             args.methods, args.demos, args.steps, args.num_seeds)


if __name__ == "__main__":
    main()
