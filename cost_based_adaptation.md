# Cost-Based Adaptation (Figure 7)

This document explains **cost-based adaptation** (a.k.a. cost-based fine-tuning), the
test-time procedure that produces **Figure 7** in the paper. It is distinct from the
supervised pretraining of SetONet. All code lives in
[`src/adaptation/cost_adapt.py`](src/adaptation/cost_adapt.py).

## The core idea

After SetONet is pretrained (via imitation of expert demonstrations), you get a new
task — possibly **out-of-distribution (OOD)** — where you have **no expert
demonstrations**. But you *do* know two things: the **dynamics** and the **cost
function**. Cost-based adaptation fine-tunes the network's weights by
**backpropagating the task cost through a differentiable rollout of the known
dynamics** — no demos needed. It is essentially trajectory optimization, but the
decision variables are the operator's parameters instead of an open-loop control
sequence.

The whole loop is:

```
loss(θ) = Σ_starts  cost_of_rollout( policy_θ , known_dynamics )
θ ← θ − lr · ∇θ loss(θ)      (Adam, gradient-clipped)
```

## The two key ingredients

### 1. A differentiable rollout that accumulates cost

For **P2P-Cost**, `differentiable_rollout_p2p`
([cost_adapt.py:163](src/adaptation/cost_adapt.py:163)) runs a `jax.lax.scan`: at each
step the SetONet predicts the action from the current state, the **known linear
dynamics** `A·x + B·u` advance the state, and the running cost accumulates
`(x − goal)ᵀ Q (x − goal) + uᵀ R u`, plus a terminal `Qf` penalty. Because every
operation is differentiable, `eqx.filter_value_and_grad` gets `∂cost/∂θ` straight
through the entire trajectory.

For **Obstacle** ([cost_adapt.py:391](src/adaptation/cost_adapt.py:391)) the structure
is identical but with RK4 dynamics and a different cost: a **soft collision penalty**
(`collision_cost`, a smooth exponential barrier so it stays differentiable,
[cost_adapt.py:348](src/adaptation/cost_adapt.py:348)) + goal + control + terminal
terms.

### 2. Selective fine-tuning (`make_filter_spec`)

`make_filter_spec` ([cost_adapt.py:74](src/adaptation/cost_adapt.py:74)) controls
*which subset of weights* is adapted — this is the main experimental variable:

- **`FT`** — train everything *except* the trunk (trunk frozen).
- **`last_branch`** — only the last layer of the branch (`rho`) network.
- **`last_both`** — last layer of both trunk and branch.
- **`SetONet`** — the pretrained model with **no** adaptation, used as the baseline.

It uses `eqx.partition` / `eqx.combine` to split trainable vs. frozen leaves, then
runs Adam for a fixed number of steps (`finetune_p2p`
[cost_adapt.py:181](src/adaptation/cost_adapt.py:181), `finetune_obstacle`
[cost_adapt.py:424](src/adaptation/cost_adapt.py:424)).

## The two experiments

### Figure 7a — P2P-Cost OOD

`run_p2p_ood` ([cost_adapt.py:210](src/adaptation/cost_adapt.py:210)). Goals are placed
on the border of a ±15 square — *outside* the training region — with starts far out at
(−20, −20). For each goal it:

1. samples the cost context for the branch (`sample_cost_context`,
   [cost_adapt.py:130](src/adaptation/cost_adapt.py:130)),
2. fine-tunes 3 variants on a couple of *train* start states (25 steps, lr 1e-4),
3. evaluates each on *held-out* start states, measuring **final distance to goal**,
   and compares against the true tvLQR expert.

### Figure 7b — Obstacle

`run_obstacle` ([cost_adapt.py:456](src/adaptation/cost_adapt.py:456)). Held-out
obstacle configurations; fine-tunes for 200 steps (lr 1e-5) on expert start states
using the collision-aware cost, then measures **collision counts** vs. the expert and
the un-adapted SetONet.

Both experiments save a `results.npz` consumed by
[`src/plotting/plot_cost_adapt.py`](src/plotting/plot_cost_adapt.py) to render
Figure 7.

## How it differs from pretraining

| | Pretraining ([train_p2p_cost.py](src/training/train_p2p_cost.py)) | Cost-based adaptation |
|---|---|---|
| Loss | MSE vs. expert controls | Task cost via differentiable rollout |
| Supervision | Expert demos required | **No demos** — needs dynamics + cost |
| What's trained | Whole network from scratch | A frozen-pretrained model, partially |
| When | Offline | Test time, per new task |

So "cost-based" means the adaptation signal is the **cost itself**, differentiated
through the dynamics, rather than imitation of demonstrations.

## How to run (from the Makefile / module docstring)

```bash
# P2P-Cost OOD (Figure 7a)
python src/adaptation/cost_adapt.py \
    --config configs/p2p_cost.yaml \
    --data data/p2p_cost \
    --checkpoint checkpoints/p2p_cost/pretrained \
    --ood \
    --output outputs/results/cost_adapt_p2p_ood

# Obstacle (Figure 7b)
python src/adaptation/cost_adapt.py \
    --config configs/obstacle.yaml \
    --data data/obstacle \
    --checkpoint checkpoints/obstacle/pretrained \
    --output outputs/results/cost_adapt_obstacle
```
