"""SetONet meta-training for P2P-Cost (Double Integrator LQR).

Both meta_branch and meta_full variants use the same data sampling and loss.
The difference is only in which parameters the inner loop adapts.

Branch input: (state, control) -> cost (encodes task from cost observations)
Trunk input: (state + time) -> predicted control
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.setonet import SetONet
from src.envs.dataloader import DoubleIntegratorLQRData
from src.training.setonet_meta import (
    get_inner_update, meta_task_loss, meta_train_step,
)


def meta_sample_batch(data_loader, data_type, num_tasks, K_support, K_query,
                      N_support, N_query):
    """Sample meta-batch with support/query sets for SetONet.

    Both sets contain src data (branch) and tgt data (trunk/targets).
    """
    support_lists = [[] for _ in range(6)]
    query_lists = [[] for _ in range(6)]

    for _ in range(num_tasks):
        meta_train, meta_test = data_loader.get_meta_task(
            data_type, K_train=K_support, N_train=N_support,
            K_test=K_query, N_test=N_query,
        )

        src_s, src_c, src_cost, src_t, tgt_s, tgt_c, goal = meta_train
        for lst, val in zip(support_lists, [src_s, src_c, src_cost, tgt_s, tgt_c, goal]):
            lst.append(val)

        src_s_q, src_c_q, src_cost_q, src_t_q, tgt_s_q, tgt_c_q, goal_q = meta_test
        for lst, val in zip(query_lists, [src_s_q, src_c_q, src_cost_q, tgt_s_q, tgt_c_q, goal_q]):
            lst.append(val)

    support_batch = tuple(jnp.stack(lst) for lst in support_lists)
    query_batch = tuple(jnp.stack(lst) for lst in query_lists)
    return support_batch, query_batch


def compute_setonet_loss(model, batch):
    """SetONet loss: branch encodes cost observations, trunk predicts controls."""
    src_states, src_controls, src_costs, tgt_states, tgt_controls, _ = batch

    sensor_locations = jnp.concatenate([src_states, src_controls], axis=-1)
    sensor_values = src_costs

    if tgt_states.ndim == 3:
        tgt_states_flat = tgt_states.reshape(-1, tgt_states.shape[-1])
        tgt_controls_flat = tgt_controls.reshape(-1, tgt_controls.shape[-1])
    else:
        tgt_states_flat = tgt_states
        tgt_controls_flat = tgt_controls

    pred = jax.vmap(lambda q: model(sensor_locations, sensor_values, q))(tgt_states_flat)
    return jnp.mean(jnp.square(pred - tgt_controls_flat))


def batch_meta_loss(model, batch, alpha, inner_update_fn):
    """Meta-loss over batch of tasks."""
    support_batch, query_batch = batch
    num_tasks = support_batch[0].shape[0]

    losses = []
    for i in range(num_tasks):
        sup_task = tuple(b[i] for b in support_batch)
        qry_task = tuple(b[i] for b in query_batch)
        loss = meta_task_loss(model, sup_task, qry_task, alpha,
                              compute_setonet_loss, inner_update_fn)
        losses.append(loss)

    return jnp.mean(jnp.array(losses))


def evaluate(model, data_loader, num_tasks, K_support, K_query, N_support, N_query,
             alpha, inner_update_fn, num_batches=10):
    total_loss = 0.0
    for _ in range(num_batches):
        batch = meta_sample_batch(data_loader, "test", num_tasks,
                                  K_support, K_query, N_support, N_query)
        loss = batch_meta_loss(model, batch, alpha, inner_update_fn)
        total_loss += float(loss)
    return total_loss / num_batches


def run_training(cfg, data_dir, output_dir, variant, seed=42, device="cpu"):
    """Meta-train SetONet on P2P-Cost."""
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    meta_cfg = cfg.get("setonet_meta", {})
    model_cfg = cfg.get("model", {})

    inner_update_fn = get_inner_update(variant)

    # Load data
    data_path = str(Path(data_dir) / "trajectories.npz")
    print(f"Loading data from {data_path}...")
    raw = np.load(data_path, allow_pickle=True)
    dataset = {k: raw[k] for k in raw.files}
    if "norm_stats" in dataset:
        dataset["norm_stats"] = dataset["norm_stats"].item()

    data_loader = DoubleIntegratorLQRData(dataset, train_perc=0.8, normalize=True)

    state_dim = dataset["states"].shape[-1]
    control_dim = dataset["actions"].shape[-1]

    # Meta params
    inner_lr = meta_cfg.get("inner_lr", 0.01)
    outer_lr = meta_cfg.get("outer_lr", 0.001)
    K_support = meta_cfg.get("K_support", 64)
    K_query = meta_cfg.get("K_query", 64)
    N_support = meta_cfg.get("N_support", 4)
    N_query = meta_cfg.get("N_query", 4)
    num_tasks = meta_cfg.get("num_tasks", 16)
    num_iterations = meta_cfg.get("num_iterations", 5000)
    eval_interval = meta_cfg.get("eval_every", 100)
    num_eval_batches = meta_cfg.get("num_eval_batches", 10)

    print("=" * 60)
    print(f"SetONet Meta-Training — P2P-Cost ({variant})")
    print("=" * 60)
    print(f"State dim: {state_dim}, Control dim: {control_dim}")
    print(f"Variant: {variant}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"K_support: {K_support}, K_query: {K_query}")
    print(f"N_support: {N_support}, N_query: {N_query}")
    print(f"Tasks per batch: {num_tasks}")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    # Init model (same architecture as pretrained SetONet)
    key, model_key = jr.split(key)
    model = SetONet(
        input_size_src=state_dim + control_dim,
        output_size_src=1,
        input_size_tgt=state_dim + 1,
        output_size_tgt=control_dim,
        p=model_cfg.get("p", 128),
        phi_hidden_size=model_cfg.get("phi_hidden_size", 128),
        phi_output_size=model_cfg.get("phi_output_size", 128),
        rho_hidden_size=model_cfg.get("rho_hidden_size", 128),
        trunk_hidden_size=model_cfg.get("trunk_hidden_size", 128),
        n_phi_layers=model_cfg.get("n_phi_layers", 2),
        n_rho_layers=model_cfg.get("n_rho_layers", 2),
        n_trunk_layers=model_cfg.get("n_trunk_layers", 2),
        aggregation_type=model_cfg.get("aggregation_type", "attention"),
        attention_n_heads=model_cfg.get("attention_n_heads", 4),
        attention_n_tokens=model_cfg.get("attention_n_tokens", 4),
        use_bias=model_cfg.get("use_bias", True),
        key=model_key,
    )

    # Load pretrained checkpoint if available
    pretrained_path = Path(output_dir).parent / "pretrained" / "setonet.eqx"
    if pretrained_path.exists():
        model = eqx.tree_deserialise_leaves(pretrained_path, model)
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print("No pretrained checkpoint found, training from scratch")

    optim = optax.adam(outer_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Wrap batch_meta_loss with the inner_update_fn
    def batch_loss_fn(model, batch, alpha):
        return batch_meta_loss(model, batch, alpha, inner_update_fn)

    train_losses = []
    eval_losses = []

    print("\nStarting meta-training...")

    for iteration in range(num_iterations):
        batch = meta_sample_batch(data_loader, "train", num_tasks,
                                  K_support, K_query, N_support, N_query)
        loss, model, opt_state = meta_train_step(
            model, optim, opt_state, batch, inner_lr, batch_loss_fn
        )
        train_losses.append(float(loss))

        if iteration % eval_interval == 0:
            eval_loss = evaluate(
                model, data_loader, num_tasks, K_support, K_query,
                N_support, N_query, inner_lr, inner_update_fn, num_eval_batches
            )
            eval_losses.append(eval_loss)
            print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(out / "setonet.eqx", model)
    print(f"\nModel saved to {out / 'setonet.eqx'}")

    np.savez(
        out / "training_history.npz",
        train_losses=np.array(train_losses),
        eval_losses=np.array(eval_losses),
        eval_iterations=np.arange(0, num_iterations, eval_interval)[:len(eval_losses)],
        variant=variant,
    )
    print(f"Training history saved to {out / 'training_history.npz'}")
