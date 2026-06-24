# Context Generation per Environment

This document describes **how the context set is generated for each environment**, what its
values physically represent, and what a practitioner would need to construct it on a real
system. It is intended to make concrete the practical cost of applying the approach, which is
a leading motivation of the paper.

---

## 1. What the "context set" is

The branch network of SetONet consumes a **context set** `C = {(location_j, value_j)}_{j=1}^{m}`:
an unordered, variable-sized collection of pointwise samples that *identifies the task* without
ever exposing the task parameters `(φ, ψ)` to the model. The branch maps this set to the
task-dependent coefficients `{c_k}`; the trunk maps a query `y = (state, time)` to basis
functions `{b_k(y)}`; the predicted control is `π̂(y) = Σ_k c_k b_k(y)`.

Because the branch is a **permutation-invariant set encoder**, the context can have any size and
any ordering at train or test time (this is the *task-resolution invariance* of §5.2). A
practitioner therefore does not need a fixed sensor grid — only the ability to produce *some*
collection of `(location, value)` samples that characterize the task.

The crucial practical question — and the reviewer's question — is **what the `(location, value)`
pairs are and how hard they are to obtain**. This differs by environment, and falls into three
qualitatively different regimes:

| Context type | Location | Value | What you need to produce it |
|---|---|---|---|
| **Cost evaluations** (P2P-Cost) | `(state, control)` | cost `ℓ(state, control)` | Ability to **evaluate the cost function** at chosen points |
| **Dynamics transitions** (P2P-Dynamics, Quadrotor, HalfCheetah) | `(state, action)` | next state | Ability to **observe / roll out transitions** of the system |
| **Task geometry** (Obstacle) | obstacle `(x, y)` | radius `r` | Ability to **observe the scene** (e.g., perception) |

The key takeaway: **none of these require solving the optimal control problem at deployment.**
The context is built from quantities a practitioner already has access to (a cost model, a
dynamics model or logged rollouts, or sensor readings of the environment), and the operator then
predicts the policy in a single forward pass.

---

## 2. Per-environment details

### P2P-Cost / P2P-Cost-Small  — context = cost evaluations
- **Context point:** `location = (state, control) ∈ ℝ^6`, `value = ℓ(state, control; x_g) ∈ ℝ`.
- **How it is generated (training):** sampled **analytically on the fly**
  ([`src/training/train_p2p_cost.py`](src/training/train_p2p_cost.py), `sample_task_data`):
  - states drawn uniformly over the workspace (`state_range`, `vel_range`),
  - controls drawn uniformly (`src_control_range`),
  - the value is the **known quadratic stage cost** `ℓ = (x−x_g)ᵀQ(x−x_g) + uᵀRu`
    evaluated at those points (`compute_immediate_cost`).
  The branch input is `concat(state, control)`; the value is the scalar cost.
- **Equivalent path from logged data:** `DoubleIntegratorLQRData` in
  [`dataloader.py`](src/envs/dataloader.py) builds the *same* `(state, control, cost)` context by
  sampling `K` transitions from stored expert trajectories. Either route yields the identical
  context format.
- **Number of context points:** `K` sampled per iteration from `{32, 64, 128}` (resolution-invariant).
- **Practitioner cost:** *Low.* You only need to be able to **evaluate the cost** at arbitrary
  `(state, control)` points — no solver, no rollouts, no expert. This is the cheapest regime.

### P2P-Dynamics — context = dynamics transitions
- **Context point:** `location = (state, action) ∈ ℝ^6`, `value = next_state ∈ ℝ^4`.
- **How it is generated:** `VaryingDynamicsData.get_task` in
  [`dataloader.py`](src/envs/dataloader.py) enumerates all one-step transitions
  `(state_t, action_t, state_{t+1})` from the task's trajectories and samples `K` of them at
  random (the cost is also available but the dynamics-varying branch uses next-state as the value).
  Branch input is `concat(state, action)`, value is `next_state`
  ([`src/training/train_p2p_dynamics.py`](src/training/train_p2p_dynamics.py)).
- **Number of context points:** `K` from `{32, 64, 128}`.
- **Practitioner cost:** *Moderate.* You need to **observe transitions** of the system under the
  new dynamics — e.g., a short interaction log or a simulator rollout. Crucially these are
  *transitions*, not expert demonstrations: any control sequence that excites the dynamics works.

### Quadrotor — context = dynamics transitions (6-D)
- **Context point:** `location = (state, action) ∈ ℝ^8`, `value = next_state ∈ ℝ^6`.
- **How it is generated:** same transition-sampling scheme as P2P-Dynamics
  ([`src/training/train_quadrotor.py`](src/training/train_quadrotor.py):
  `src_input = concat(states, actions)`, `src_output = next_states`).
- **Number of context points:** `K = 64`.
- **Practitioner cost:** *Moderate*, identical in kind to P2P-Dynamics — observed transitions of
  the platform under its (unknown-to-the-model) physical parameters.

### Obstacle — context = task geometry
- **Context point:** `location = obstacle center (x, y) ∈ ℝ^2`, `value = radius r ∈ ℝ`.
- **How it is generated:** the context is **the obstacle field itself**, read directly from the
  task definition ([`ObstacleAvoidanceImitation`](src/envs/dataloader.py),
  [`train_obstacle.py`](src/training/train_obstacle.py): `obs_positions = obstacles[..., :2]`,
  `obs_values = obstacles[..., 2:3]`). No trajectory sampling is involved for the context — every
  obstacle is one context element, and the set size equals the number of obstacles.
- **Number of context points:** the actual obstacle count (`n_obs ∈ {2, 4, 6}` in training;
  generalizes to 3 and 5 at test time — §5.2). Here `K = {32, 64, 128}` controls the number of
  *expert trajectories in the loss*, **not** the branch context size.
- **Practitioner cost:** *Low.* The context is a **direct observation of the scene geometry**
  (positions and sizes of obstacles), exactly what an onboard perception system would provide.
  This illustrates that the context need not be a smooth cost: it can be a finite-dimensional
  proxy for the constraint structure.

### HalfCheetah-v3 (iMuJoCo) — context = dynamics transitions (logged RL rollouts)
- **Context point:** `location = (state, action) ∈ ℝ^{23}`, `value = next_state ∈ ℝ^{17}`.
- **How it is generated:** state–action transitions are taken from **logged rollouts of an
  SAC expert** for each task configuration (varying mass, limb length, joint range, friction).
  The data is not bundled in this archive (see `README.md`); it uses the same transition-context
  pipeline as the other dynamics-varying environments.
- **Practitioner cost:** *Moderate.* Same as P2P-Dynamics/Quadrotor — a log of state–action
  transitions for the new agent configuration.

---

## 3. Context at training vs. deployment

- **Training:** for each task, `K` context points are sampled (per the rules above) at every
  iteration; `K` is itself drawn from a set so the branch never overfits to a fixed cardinality.
- **Deployment (new, held-out task):** the operator receives a *single* context set for the new
  task — sampled cost evaluations, observed transitions, or the observed obstacle field — and
  predicts the policy at arbitrary query states in one forward pass, **without re-solving the
  optimal control problem** (paper §4.1). If accuracy is insufficient, the same context (plus
  optionally a few demonstrations or the known cost) drives the lightweight adaptation strategies
  of §4.2.

---

## 4. Practical considerations (real-world applicability)

The three regimes map onto increasing real-world effort:

1. **Cost-evaluation context (P2P-Cost):** needs only a queryable cost model — the easiest to
   satisfy, since the cost is often the very thing the practitioner is designing.
2. **Transition context (P2P-Dynamics, Quadrotor, HalfCheetah):** needs the ability to observe
   the system's transitions. These are *exploratory* transitions, not expert trajectories, so a
   simulator, a system-identification dataset, or routine operational logs all suffice.
3. **Geometry/observation context (Obstacle):** needs a perception of the task configuration —
   typically already available from sensors.

What the method **avoids** at deployment is the expensive step: it does **not** require solving the
pOCP, nor (except where explicitly studied in §4.2) expert demonstrations for the new task.
The honest open challenge — and a good direction for practitioners — is that the *informativeness*
of the context matters: the context must sufficiently characterize the task (e.g., transitions
must excite the relevant dynamics modes). The resolution-invariance study (§5.2, Figure 5) gives
empirical guidance: roughly 16–32 samples suffice in the control environments before error
plateaus, and the dynamics-varying tasks need only a handful of transitions to identify the
parameters.

---

## 5. Where this lives in the code

| Environment | Context-building code |
|---|---|
| P2P-Cost / Small | `sample_task_data` in [`src/training/train_p2p_cost.py`](src/training/train_p2p_cost.py); `DoubleIntegratorLQRData.get_task` in [`src/envs/dataloader.py`](src/envs/dataloader.py) |
| P2P-Dynamics | `VaryingDynamicsData.get_task` in [`dataloader.py`](src/envs/dataloader.py); [`train_p2p_dynamics.py`](src/training/train_p2p_dynamics.py) |
| Quadrotor | `VaryingDynamicsData.get_task`; [`train_quadrotor.py`](src/training/train_quadrotor.py) |
| Obstacle | `ObstacleAvoidanceImitation` in [`dataloader.py`](src/envs/dataloader.py); [`train_obstacle.py`](src/training/train_obstacle.py) |
| HalfCheetah | Same transition pipeline (data external, see `README.md`) |

Branch/trunk input dimensions for each environment are tabulated in
[`configuration_overview.md`](configuration_overview.md) §2.
