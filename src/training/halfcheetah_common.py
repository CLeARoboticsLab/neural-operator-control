"""Shared building blocks for the HalfCheetah-v3 (iMuJoCo) experiments.

Used by the pretraining, meta-training and MAML trainers, the adaptation-grid
evaluation (Figure 10) and the control-prediction figure (Figure 9).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.setonet import SetONet


def k_options(value):
    """Config ``K`` may be an int or a list; always return a list of ints."""
    if isinstance(value, (list, tuple)):
        return [int(k) for k in value]
    return [int(value)]


def build_setonet(model_cfg: dict, obs_size: int, act_size: int, key) -> SetONet:
    """SetONet with the paper's HalfCheetah branch/trunk interface.

    Branch location (state, action) -> value next_state; trunk state -> action.
    """
    return SetONet(
        input_size_src=obs_size + act_size,
        output_size_src=obs_size,
        input_size_tgt=obs_size,
        output_size_tgt=act_size,
        p=model_cfg.get("p", 128),
        phi_hidden_size=model_cfg.get("phi_hidden_size", 128),
        phi_output_size=model_cfg.get("phi_output_size", 128),
        rho_hidden_size=model_cfg.get("rho_hidden_size", 128),
        trunk_hidden_size=model_cfg.get("trunk_hidden_size", 128),
        n_phi_layers=model_cfg.get("n_phi_layers", 4),
        n_rho_layers=model_cfg.get("n_rho_layers", 4),
        n_trunk_layers=model_cfg.get("n_trunk_layers", 4),
        aggregation_type=model_cfg.get("aggregation_type", "attention"),
        attention_n_heads=model_cfg.get("attention_n_heads", 4),
        attention_n_tokens=model_cfg.get("attention_n_tokens", 4),
        use_bias=model_cfg.get("use_bias", True),
        key=key,
    )


class MAMLMLP(eqx.Module):
    """Monolithic MLP policy ``state -> action`` for the MAML baseline."""

    mlp: eqx.nn.MLP

    def __init__(self, obs_size, act_size, hidden_size, num_layers, key):
        self.mlp = eqx.nn.MLP(
            in_size=obs_size, out_size=act_size,
            width_size=hidden_size, depth=num_layers, key=key,
        )

    def __call__(self, state):
        return self.mlp(state)


def build_maml(cfg: dict, obs_size: int, act_size: int, key) -> MAMLMLP:
    mm = cfg.get("maml_model", {})
    return MAMLMLP(obs_size, act_size, mm.get("hidden_size", 256), mm.get("num_layers", 4), key)


def load_setonet(path, model_cfg, obs_size, act_size, key) -> SetONet:
    model = build_setonet(model_cfg, obs_size, act_size, key)
    return eqx.tree_deserialise_leaves(str(path), model)


def load_maml(path, cfg, obs_size, act_size, key) -> MAMLMLP:
    model = build_maml(cfg, obs_size, act_size, key)
    return eqx.tree_deserialise_leaves(str(path), model)


def predict_setonet(model, context_sa, context_ns, query_states):
    """Vectorised operator evaluation at many query states for one task."""
    return jax.vmap(lambda q: model(context_sa, context_ns, q))(query_states)


def task_mse(model, context_sa, context_ns, query_states, query_actions):
    pred = predict_setonet(model, context_sa, context_ns, query_states)
    return jnp.mean(jnp.square(pred - query_actions))


def setonet_task_loss(model, batch):
    """Loss on one task, ``batch = (context_sa, context_ns, query_states, query_actions)``."""
    return task_mse(model, *batch)


def relative_l2(pred, target):
    diff = np.asarray(pred) - np.asarray(target)
    return float(np.sqrt(np.sum(diff ** 2)) / (np.sqrt(np.sum(np.asarray(target) ** 2)) + 1e-12))


def last_layer_filter_spec(model):
    """Trainable mask for last-layer fine-tuning: final trunk and rho layers only."""
    spec = jax.tree_util.tree_map(lambda _: False, model)
    spec = eqx.tree_at(lambda m: m.trunk.layers[-1], spec,
                       replace=jax.tree_util.tree_map(lambda _: True, model.trunk.layers[-1]))
    spec = eqx.tree_at(lambda m: m.rho.layers[-1], spec,
                       replace=jax.tree_util.tree_map(lambda _: True, model.rho.layers[-1]))
    return spec


def branch_only_filter_spec(model):
    """Trainable mask for SetONet-Meta adaptation: everything except the trunk."""
    spec = jax.tree_util.tree_map(lambda x: eqx.is_array(x), model)
    return eqx.tree_at(lambda m: m.trunk, spec,
                       replace=jax.tree_util.tree_map(lambda _: False, model.trunk))


@eqx.filter_jit
def train_step(batch_loss_fn, model, optim, opt_state, batch):
    loss, grads = batch_loss_fn(model, batch)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
