"""Shared MAML core: model, inner update, outer loop.

All environment-specific MAML trainers import from here.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


class MAMLMLP(eqx.Module):
    """MLP policy network for MAML.

    Takes state (with time) as input and outputs control action.
    Task information is implicitly learned through adaptation.
    """
    mlp: eqx.nn.MLP

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 num_layers: int, key: jr.PRNGKey):
        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=output_dim,
            width_size=hidden_size,
            depth=num_layers,
            key=key,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x)


def compute_loss_flat(model, states, actions):
    """MSE loss on flattened (state, action) pairs.

    Args:
        model: MAMLMLP (state -> action)
        states: (K, input_dim)
        actions: (K, action_dim)
    """
    pred = jax.vmap(model)(states)
    return jnp.mean(jnp.square(pred - actions))


def compute_loss_trajectories(model, traj_states, traj_actions):
    """MSE loss on trajectory data — flattens before computing.

    Args:
        model: MAMLMLP
        traj_states: (N, horizon, input_dim)
        traj_actions: (N, horizon, action_dim)
    """
    flat_states = traj_states.reshape(-1, traj_states.shape[-1])
    flat_actions = traj_actions.reshape(-1, traj_actions.shape[-1])
    pred = jax.vmap(model)(flat_states)
    return jnp.mean(jnp.square(pred - flat_actions))


def inner_update(model, support_states, support_actions, alpha, support_loss_fn):
    """Inner loop: one gradient step on support set.

    Args:
        model: MAMLMLP
        support_states: support inputs
        support_actions: support targets
        alpha: inner learning rate
        support_loss_fn: callable(model, states, actions) -> scalar loss
    """
    _, grads = eqx.filter_value_and_grad(support_loss_fn)(
        model, support_states, support_actions
    )
    updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
    return eqx.apply_updates(model, updates)


def maml_task_loss(model, support_states, support_actions,
                   query_states, query_actions, alpha,
                   support_loss_fn, query_loss_fn):
    """MAML loss for a single task: adapt on support, evaluate on query."""
    adapted = inner_update(model, support_states, support_actions, alpha, support_loss_fn)
    return query_loss_fn(adapted, query_states, query_actions)


@eqx.filter_jit
def train_step(model, optim, opt_state, batch, alpha, batch_loss_fn):
    """Outer loop training step."""
    loss, grad = eqx.filter_value_and_grad(batch_loss_fn)(model, batch, alpha)
    updates, new_state = optim.update(grad, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return loss, new_model, new_state
