"""Figure 9: control predictions on a held-out HalfCheetah-v3 configuration.

Rows are the first control dimensions (u1, u2, u3); columns are the zero-shot
pretrained operator (SetONet), SetONet-FT and SetONet-Meta-Full after
``grad_steps`` adaptation steps on ``num_demos`` expert episodes (``H``-step
windows). The expert action is drawn in black.

Usage:
    python src/plotting/plot_cheetah_ctrl.py \
        --config configs/halfcheetah.yaml --data data/halfcheetah \
        --checkpoints checkpoints/halfcheetah --output outputs/figures/figure9.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.envs.imujoco_dataloader import IMuJoCoImitation
from src.evaluation.evaluate_halfcheetah import adapt, load_models, predict, split_task_episodes

plt.rcParams.update({
    "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.labelsize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

COLUMNS = [  # (method key, checkpoint needed, title, colour)
    ("pretrained", "pretrained", "SetONet", "#1f77b4"),
    ("setonet_ft", "pretrained", "SetONet-FT", "#ff7f0e"),
    ("setonet_meta_full", "meta_full", "SetONet-Meta-Full", "#c51b8a"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ev = cfg.get("evaluation", {})
    f9 = ev.get("figure9", {})
    task_idx = a.task_idx if a.task_idx is not None else int(f9.get("task_idx", 0))
    num_demos = int(f9.get("num_demos", 10))
    grad_steps = int(f9.get("grad_steps", 25))
    T = int(f9.get("plot_timesteps", 50))
    n_dims = int(f9.get("control_dims", 3))
    H = int(ev.get("H", 100))
    lrs = {
        "setonet_ft": float(ev.get("setonet_ft_lr", 1e-4)),
        "last_layer": float(ev.get("last_layer_lr", 1e-4)),
        "maml": float(ev.get("maml_inner_lr", 0.01)),
        "setonet_meta": float(ev.get("setonet_meta_inner_lr", 0.01)),
        "setonet_meta_full": float(ev.get("setonet_meta_full_inner_lr", 0.01)),
    }

    dc = cfg["data"]
    dl = IMuJoCoImitation(a.data, dc["env_name"], dc.get("train_perc", 0.8),
                          dc.get("split_seed", 42), dc.get("normalize", True))
    models = load_models(a.checkpoints, cfg, dl.obs_size, dl.act_size)

    episodes = dl.test_data[task_idx]
    rng = np.random.RandomState(a.seed * 10000 + task_idx)
    split = split_task_episodes(dl, episodes, num_demos, H, rng)
    if split is None:
        raise SystemExit(f"Held-out task {task_idx} has too few episodes for {num_demos} demos")

    # Plot the first T steps of one evaluation episode (the first one after the demos).
    perm = np.random.RandomState(a.seed * 10000 + task_idx).permutation(len(episodes))
    plot_ep = episodes[perm[num_demos]]
    s, act, _ = dl.get_transitions(plot_ep)
    s, act = s[:T], act[:T]
    expert = dl.denormalize_actions(act)
    t = np.arange(len(s))

    preds = {}
    for key, needed, title, color in COLUMNS:
        if needed not in models:
            print(f"  [skip] {title}: checkpoint '{needed}' missing")
            continue
        if key == "pretrained":
            model, m = models["pretrained"], "setonet_ft"
        else:
            m = key
            model = adapt(m, models, lrs, split["context_sa"], split["context_ns"],
                          split["adapt_s"], split["adapt_a"], grad_steps)
        pred = predict(m, model, split["context_sa"], split["context_ns"], s)
        preds[key] = dl.denormalize_actions(pred)

    cols = [c for c in COLUMNS if c[0] in preds]
    fig, axes = plt.subplots(n_dims, len(cols), figsize=(3.2 * len(cols), 1.8 * n_dims),
                             sharex=True, squeeze=False)
    for j, (key, _needed, title, color) in enumerate(cols):
        for i in range(n_dims):
            ax = axes[i, j]
            ax.plot(t, expert[:, i], color="black", lw=1.4)
            ax.plot(t, preds[key][:, i], color=color, lw=1.2)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"$u_{i + 1}$")
            if i == n_dims - 1:
                ax.set_xlabel("time")
    fig.tight_layout()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, bbox_inches="tight")
    fig.savefig(Path(a.output).with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {a.output}  (task {task_idx}, {num_demos} demos, {grad_steps} steps)")


if __name__ == "__main__":
    main()
