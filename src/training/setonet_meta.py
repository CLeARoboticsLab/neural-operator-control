"""Shared SetONet meta-training core.

Provides inner_update with branch-only or full adaptation,
and the common meta-loss / outer loop step.
"""

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Inner update variants
# ---------------------------------------------------------------------------

def inner_update_branch(model, loss_fn, batch, alpha):
    """Inner loop: adapt branch only (phi, aggregator, rho). Trunk frozen.

    Args:
        model: SetONet
        loss_fn: callable(model, batch) -> scalar
        batch: support batch
        alpha: inner learning rate
    """
    # Build filter: True for arrays we want to differentiate
    filter_spec = jax.tree_util.tree_map(lambda x: eqx.is_array(x), model)

    # Freeze trunk
    filter_spec = eqx.tree_at(
        lambda m: m.trunk,
        filter_spec,
        replace=jax.tree_util.tree_map(lambda _: False, model.trunk),
    )

    diff_model, static_model = eqx.partition(model, filter_spec)

    def wrapped_loss(diff_part):
        combined = eqx.combine(diff_part, static_model)
        return loss_fn(combined, batch)

    _, grads = eqx.filter_value_and_grad(wrapped_loss)(diff_model)
    updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
    updated_diff = eqx.apply_updates(diff_model, updates)
    return eqx.combine(updated_diff, static_model)


def inner_update_full(model, loss_fn, batch, alpha):
    """Inner loop: adapt all parameters (branch + trunk).

    Args:
        model: SetONet
        loss_fn: callable(model, batch) -> scalar
        batch: support batch
        alpha: inner learning rate
    """
    _, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
    return eqx.apply_updates(model, updates)


def get_inner_update(variant: str):
    """Return the appropriate inner_update function for the variant."""
    if variant == "meta_branch":
        return inner_update_branch
    elif variant == "meta_full":
        return inner_update_full
    else:
        raise ValueError(f"Unknown variant '{variant}'. Must be 'meta_branch' or 'meta_full'.")


# ---------------------------------------------------------------------------
# Meta-loss and outer loop
# ---------------------------------------------------------------------------

def meta_task_loss(model, support_batch, query_batch, alpha, loss_fn, inner_update_fn):
    """Single-task meta-loss: adapt on support, evaluate on query."""
    adapted = inner_update_fn(model, loss_fn, support_batch, alpha)
    return loss_fn(adapted, query_batch)


@eqx.filter_jit
def meta_train_step(model, optim, opt_state, batch, alpha, batch_meta_loss_fn):
    """Outer loop training step."""
    loss, grad = eqx.filter_value_and_grad(batch_meta_loss_fn)(model, batch, alpha)
    updates, new_state = optim.update(grad, opt_state)
    new_model = eqx.apply_updates(model, updates)
    return loss, new_model, new_state
