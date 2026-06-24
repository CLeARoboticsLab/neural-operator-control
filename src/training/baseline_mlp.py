"""Supervised multi-task MLP baseline (task-parameter conditioned).

A plain MLP policy that receives the task parameters *explicitly*:

    [state, time, task_params] -> action

It is trained with ordinary behavioral cloning jointly across all training
tasks (no meta-learning), and evaluated **zero-shot** (no gradient steps at
test time). This is the natural point of comparison for the pretrained SetONet
operator's zero-shot row.

Contrast with ``MAMLMLP`` (``src/training/maml.py``): that model gets *no* task
information and must infer the task through inner-loop adaptation. Here the task
is handed to the network directly, so depth/width are reused from each config's
``maml_model`` block to keep the two MLP baselines parameter-comparable.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx


class BaselineMLP(eqx.Module):
    """MLP policy: [state, time, task_params] -> action."""

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


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """||pred - target|| / ||target|| (Eq. 19), in whatever space is passed in.

    Matches ``src/evaluation/evaluate.py``: callers pass *physical* (un-normalized)
    actions so the metric is comparable across models with different normalizers.
    """
    diff = pred - target
    return float(np.sqrt(np.sum(diff ** 2)) / (np.sqrt(np.sum(target ** 2)) + 1e-12))


@eqx.filter_jit
def train_step(model, optim, opt_state, x, y):
    """Single Adam BC step on a flat batch of (input, target) pairs."""

    def loss_fn(m):
        pred = jax.vmap(m)(x)
        return jnp.mean(jnp.square(pred - y))

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def count_params(model) -> int:
    """Total number of array parameters (for reporting param-comparability)."""
    return sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )


def width_for_target(in_dim: int, out_dim: int, depth: int, target: int) -> int:
    """Hidden width that makes an eqx.nn.MLP(in,out,W,depth) ≈ `target` params.

    eqx.nn.MLP params = (in*W + W) + (depth-1)*(W*W + W) + (W*out + out).
    Solve the quadratic (depth-1)*W^2 + (in+out+depth)*W + (out - target) = 0.
    Used to keep Baseline-2 parameter-comparable to SetONet despite its much
    larger (flattened-context) input.
    """
    import math
    a = depth - 1
    b = in_dim + out_dim + depth
    c = out_dim - target
    if a <= 0:
        return max(1, (target - out_dim) // max(1, in_dim + out_dim))
    w = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return max(8, int(round(w)))
