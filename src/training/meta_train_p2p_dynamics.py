"""SetONet meta-training for P2P-Dynamics (varying dynamics, fixed goal).

Branch input: (state, action) -> next_state (dynamics-based encoding)
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
from src.envs.dataloader import VaryingDynamicsData
from src.training.setonet_meta import (
    get_inner_update, meta_task_loss, meta_train_step,
)


def meta_sample_batch(data_set, data_type, num_tasks, K, N_support, N_query):
    """Sample meta-batch: support/query are (transitions + trajectories) from same dynamics config."""
    support_lists = [[] for _ in range(8)]
    query_lists = [[] for _ in range(8)]

    for _ in range(num_tasks):
        meta_train, meta_test = data_set.get_meta_task_dynamics(
            data_type,
            K_train=K, N_train=N_support,
            K_test=K, N_test=N_query,
        )

        for lst, val in zip(support_lists, meta_train):
            lst.append(val)
        for lst, val in zip(query_lists, meta_test):
            lst.append(val)

    support_batch = tuple(jnp.array(np.stack(lst)) for lst in support_lists)
    query_batch = tuple(jnp.array(np.stack(lst)) for lst in query_lists)
    return support_batch, query_batch


def compute_setonet_loss(model, batch):
    """SetONet loss: branch encodes dynamics, trunk predicts controls.

    batch = (random_states, random_actions, random_next_states, random_costs,
             expert_trajectories, expert_actions, expert_goal_states, dynamics_params)
    """
    random_states, random_actions, random_next_states, _costs, \
        expert_trajectories, expert_actions, _goal_states, *_ = batch

    # Branch: (state, action) -> next_state
    src_input = jnp.concatenate([random_states, random_actions], axis=-1)
    src_output = random_next_states

    # Trunk: expert trajectory states (drop last timestep, already has time appended)
    tgt_input = expert_trajectories[:, :-1, :]  # (N, horizon, state_dim+1)

    # Predict actions
    pred = jax.vmap(
        jax.vmap(lambda q: model(src_input, src_output, q))
    )(tgt_input)

    return jnp.mean(jnp.square(pred - expert_actions))


def batch_meta_loss(model, batch, alpha, inner_update_fn):
    support_batch, query_batch = batch
    num_tasks = support_batch[0].shape[0]

    losses = []
    for i in range(num_tasks):
        sup = tuple(b[i] for b in support_batch)
        qry = tuple(b[i] for b in query_batch)
        loss = meta_task_loss(model, sup, qry, alpha,
                              compute_setonet_loss, inner_update_fn)
        losses.append(loss)

    return jnp.mean(jnp.array(losses))


def evaluate(model, data_set, num_tasks, K, N_support, N_query, alpha,
             inner_update_fn, num_batches=10):
    total = 0.0
    for _ in range(num_batches):
        batch = meta_sample_batch(data_set, "test", num_tasks, K, N_support, N_query)
        loss = batch_meta_loss(model, batch, alpha, inner_update_fn)
        total += float(loss)
    return total / num_batches


def run_training(cfg, data_dir, output_dir, variant, seed=42, device="cpu"):
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
    if "dynamics_params" in dataset:
        dataset["dynamics_params"] = dataset["dynamics_params"].tolist() \
            if hasattr(dataset["dynamics_params"], "tolist") else dataset["dynamics_params"]

    goal_idx = meta_cfg.get("goal_idx", 0)
    data_set = VaryingDynamicsData.load_for_goal(goal_idx, dataset, train_perc=0.8)

    state_dim = dataset["states"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    # Meta params
    inner_lr = meta_cfg.get("inner_lr", 0.01)
    outer_lr = meta_cfg.get("outer_lr", 0.001)
    K = meta_cfg.get("K", 32)
    N_support = meta_cfg.get("N_support", 8)
    N_query = meta_cfg.get("N_query", 8)
    num_tasks = meta_cfg.get("num_tasks", 16)
    num_iterations = meta_cfg.get("num_iterations", 1500)
    eval_interval = meta_cfg.get("eval_every", 100)
    num_eval_batches = meta_cfg.get("num_eval_batches", 10)

    print("=" * 60)
    print(f"SetONet Meta-Training — P2P-Dynamics ({variant})")
    print("=" * 60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Variant: {variant}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"K (branch context): {K}")
    print(f"N_support: {N_support}, N_query: {N_query}")
    print(f"Tasks per batch: {num_tasks}")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    key, model_key = jr.split(key)
    model = SetONet(
        input_size_src=state_dim + action_dim,
        output_size_src=state_dim,
        input_size_tgt=state_dim + 1,
        output_size_tgt=action_dim,
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

    pretrained_path = Path(output_dir).parent / "pretrained" / "setonet.eqx"
    if pretrained_path.exists():
        model = eqx.tree_deserialise_leaves(pretrained_path, model)
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print("No pretrained checkpoint found, training from scratch")

    optim = optax.adam(outer_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def batch_loss_fn(model, batch, alpha):
        return batch_meta_loss(model, batch, alpha, inner_update_fn)

    train_losses = []
    eval_losses = []

    print("\nStarting meta-training...")

    for iteration in range(num_iterations):
        batch = meta_sample_batch(data_set, "train", num_tasks, K, N_support, N_query)
        loss, model, opt_state = meta_train_step(
            model, optim, opt_state, batch, inner_lr, batch_loss_fn
        )
        train_losses.append(float(loss))

        if iteration % eval_interval == 0:
            eval_loss = evaluate(
                model, data_set, num_tasks, K, N_support, N_query,
                inner_lr, inner_update_fn, num_eval_batches
            )
            eval_losses.append(eval_loss)
            print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

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
