# Configuration Overview: Architectures and Hyperparameters

This document summarizes the model architectures and the main hyperparameters used
throughout the experiments in *Neural Operators for Multi-Task Control and Adaptation*.
All values are taken directly from the per-environment YAML files in `configs/` and the
model construction code in `src/`. Each environment has its own config file; the model,
training, meta-training (`setonet_meta`), and MAML baseline (`maml`) hyperparameters all
live in that single file.

---

## 1. Common Setup (shared across all experiments)

| Item | Value |
|---|---|
| Operator architecture | **SetONet** (branch–trunk, permutation-invariant set-encoder branch) — `src/setonet.py` |
| Framework | JAX + [Equinox](https://github.com/patrick-kidger/equinox) |
| Optimizer | **Adam** (`optax`) |
| Base learning rate | `1e-3` |
| Activation | ReLU (`jax.nn.relu`) |
| Training loss | MSE / behavioral-cloning loss (Eq. 9), evaluated as relative L² error (Eq. 19) |
| Branch aggregation | Multi-head attention pooling (Set-Transformer style), `n_heads = 4`, `n_tokens = 4` |
| Output bias | Enabled (`use_bias: true`), initialized `N(0, 0.01)` |
| Input normalization | Zero-mean / unit-variance, statistics computed on the training set (`normalize: true`) |
| Train / test split | 80% / 20% over tasks (`train_perc: 0.8`, `split_seed: 42`) |
| Default seed | `42` |
| Context-size sampling | Number of context points `K` sampled uniformly from a predefined set each iteration (resolution invariance) |
| Hardware | Single NVIDIA GPU (RTX 5070 in the paper) |

### SetONet architecture (branch–trunk)

- **Branch (DeepOSet set encoder):** per-element MLP `φ` → attention-pool aggregation →
  post-aggregation MLP `ρ` → `p` task-dependent coefficients `{c_k}`.
- **Trunk:** MLP mapping the query location `y` → `p × output_dim` basis values `{b_k(y)}`.
- **Output:** `π̂(y) = Σ_{k=1}^{p} c_k · b_k(y) + bias`.
- `ρ` output size = `p`; trunk output size = `p × output_size_tgt`.

The branch `φ` input is the concatenation of a sensor *location* and its *value*; what these
encode changes per environment (see §2), but the architecture is identical.

---

## 2. Per-Environment Branch / Trunk I/O Dimensions

These are set in each `src/training/train_<env>.py`. `d_x` / `d_u` are state / control dims.

| Environment | Branch input (location) | Branch value | Trunk input | Output |
|---|---|---|---|---|
| P2P-Cost / P2P-Cost-Small | `(state, control)` = `4+2` | cost `ℓ` (1) | `state + time` = `4+1` | control (2) |
| P2P-Dynamics | `(state, action)` = `4+2` | next state (4) | `state + time` = `4+1` | action (2) |
| Quadrotor | `(state, action)` = `6+2` | next state (6) | `state + time` = `6+1` | control (2) |
| Obstacle | obstacle `(x, y)` (2) | radius `r` (1) | `state + time` = `4+1` | control (2) |
| HalfCheetah-v3 (iMuJoCo) | `(state, action)` = `17+6` | next state (17) | `state + time` = `17+1` | action (6) |

> iMuJoCo note: HalfCheetah-v3 (`d_x = 17`, `d_u = 6`, 53 configurations, 100 SAC expert
> episodes each) is the only iMuJoCo environment reported in the paper. Its data is not
> generated but downloaded (`make data-halfcheetah`, Patacchiola et al. 2023); the loader is
> `src/envs/imujoco_dataloader.py` and the settings are in `configs/halfcheetah.yaml`:
> p=128, 4 hidden layers, attention pooling (4 heads, 4 tokens); pretraining 10k iterations,
> M=16 tasks, K in {1,3,5} context episodes of H=100 transitions, lr 1e-3 with cosine decay,
> best-of-5; meta-training and MAML 10k iterations, inner lr 0.01 (SGD), outer lr 5e-4 (Adam),
> 16 tasks per meta-batch; the MAML MLP is 256 wide and 4 deep.

---

## 3. SetONet Model Architecture per Environment

All models use `p = 128`, attention aggregation (`n_heads = 4`, `n_tokens = 4`), and
`use_bias = true`. Hidden widths are 128 everywhere; only the **depth** changes per environment.

| Environment | `p` | φ hidden / out | ρ hidden | trunk hidden | `n_phi` | `n_rho` | `n_trunk` |
|---|---|---|---|---|---|---|---|
| P2P-Cost | 128 | 128 / 128 | 128 | 128 | 2 | 2 | 2 |
| P2P-Cost-Small | 128 | 128 / 128 | 128 | 128 | 2 | 2 | 2 |
| P2P-Dynamics | 128 | 128 / 128 | 128 | 128 | 3 | 3 | 3 |
| Quadrotor | 128 | 128 / 128 | 128 | 128 | 3 | 3 | 3 |
| Obstacle | 128 | 128 / 128 | 128 | 128 | 4 | 4 | 4 |

(`depth` is the number of hidden layers in each Equinox MLP.)

---

## 4. Pretraining (Behavioral Cloning) Hyperparameters

| Environment | Iterations | LR | `M` (tasks/batch) | `K` (branch context) | `N` (traj/loss) | eval_every |
|---|---|---|---|---|---|---|
| P2P-Cost | 7500 | 1e-3 | 32 | {32, 64, 128} | — | 100 |
| P2P-Cost-Small | 500 | 1e-3 | 16 | {32, 64, 128} | — | 50 |
| P2P-Dynamics | 7500 | 1e-3 | 32 | {32, 64, 128} | 16 | 100 |
| Quadrotor | 7500 | 1e-3 | 16 | 64 | 5 | 100 |
| Obstacle | 20000 | 1e-3 | 16 | {32, 64, 128} | — | 100 |

Notes:
- `K` (when a list) is the branch context size, **randomly sampled per iteration** from the
  listed values to encourage task-resolution invariance.
- For Obstacle, the branch context is always the full set of `num_obs` obstacles; the listed
  `K = {32, 64, 128}` is the number of expert trajectories used in the loss.
- `N` is the number of expert trajectories contributing to the loss (where fixed).
  A **"—"** in the `N` column means no fixed `N` is set; the loss uses the sampled `K` context
  (for Obstacle, `K` is itself the trajectory count, see above).
- All runs use `num_runs: 1`, `split_seed: 42`.

---

## 5. Data Generation Hyperparameters

| Environment | Expert | #Tasks | Traj/task | Horizon | dt | Notes |
|---|---|---|---|---|---|---|
| P2P-Cost | LQR | 500 goals | 100 | 50 | 0.1 | Q=1.0, R=0.1, Qf=10.0; goal/state ∈ [−10,10]², vel ∈ [−5,5] |
| P2P-Cost-Small | LQR | 50 goals | 10 | 50 | 0.1 | Same cost weights as P2P-Cost |
| P2P-Dynamics | iLQR | 100 dyn cfgs | 100 | 30 | 0.1 | friction ∈ [0.3,0.9], v_max ∈ [10,15], a_max ∈ [3,5]; max_iter 100 |
| Quadrotor | iLQR | 100 dyn cfgs | 20 | 100 | 0.02 (50 Hz) | mass ∈ [0.1,0.5], inertia_scale ∈ [0.8,1.2], arm ∈ [0.05,0.15], max_torque 0.1; max_iter 150 |
| Obstacle | IPOPT (CasADi) | 500 | 30 (×2 starts) | N=50 | T=1.0 | num_obs=12 slots, qf=1000, qp=1000, 4 solver workers |
| HalfCheetah-v3 | SAC | 53 cfgs | 100 | 100 | — | varies mass, limb, joints, friction (external dataset) |

**P2P-Dynamics cost weights:** position 1.0, velocity 0.5, control 0.01, terminal position 20.0, terminal velocity 10.0.
**Quadrotor cost weights:** position 10.0, velocity 1.0, angle 50.0, ang-vel 5.0, control 0.1, terminal position 100.0, terminal velocity 50.0, terminal angle 100.0.

---

## 6. Task-Specific Adaptation (fine-tuning the pretrained operator)

Strategies (`src/adaptation/`, `ADAPT_METHODS` in `Makefile`). All use the **pretrained SetONet
as a warm start**; they differ only in *which parameters* are unfrozen:

| Method | Trainable parameters |
|---|---|
| **SetONet-FT** | Full network (branch + trunk) |
| **Full-Branch** | Entire branch (`φ`, aggregator, `ρ`); trunk frozen |
| **Last-Branch** | Final layer of the branch only |
| **Last-Trunk** | Final layer of the trunk only |
| **Last-Both** | Final layer of both branch and trunk |

- Reported at **0, 1, and 25 gradient steps** (Table 2).
- Adaptation with expert demonstrations minimizes the imitation loss (Eq. 10) on a small
  support set (Table 2 uses a single expert demonstration; figures use up to 25 demos).
- **Cost-based adaptation** (`src/adaptation/cost_adapt.py`, Eq. 11): no expert data; the
  operator is fine-tuned by differentiating the control objective through a differentiable
  rollout. Obstacle surrogate cost uses `w_coll = 10`, `w_ctrl = Δt`, collision sharpness
  `α = 15`, safety margin `m = 0.2`.

---

## 7. Meta-Training Variants (`setonet_meta` block)

Two MAML-style variants over the SetONet operator (`src/training/setonet_meta.py`):
- **SetONet-Meta** (`meta_branch`): inner loop updates **branch only** (trunk frozen).
- **SetONet-Meta-Full** (`meta_full`): inner loop updates **all** parameters (second-order).

Both use a single inner gradient step; the outer loop updates all parameters.
The underlying SetONet architecture is identical to §3 for each environment.

| Environment | inner_lr (α) | outer_lr (β) | Support | Query | tasks/batch | Iterations | eval_every |
|---|---|---|---|---|---|---|---|
| P2P-Cost | 0.01 | 0.001 | K=64, N=4 | K=64, N=4 | 16 | 5000 | 100 |
| P2P-Cost-Small | 0.01 | 0.001 | K=64, N=4 | K=64, N=4 | 8 | 500 | 50 |
| P2P-Dynamics | 0.01 | 0.001 | K=32, N=8 | K=32, N=8 | 16 | 1500 | 100 |
| Quadrotor | 0.01 | 0.001 | K=64, N=5 | K=64, N=5 | 16 | 1000 | 100 |
| Obstacle | 0.01 | 0.001 | K=10 | (10 eval batches) | 16 | 2000 | 100 |

Here `K` = branch context samples (transitions / source samples) and `N` = number of target
expert trajectories used in the support/query inner and outer losses.

---

## 8. MAML Baseline (`maml` block)

A monolithic MLP policy `state(+time) → action` (`MAMLMLP` in `src/training/maml.py`).
Width 128 everywhere; depth (`num_layers` = MLP hidden depth) varies per environment.

| Environment | hidden_size | num_layers | inner_lr (α) | outer_lr (β) | Support | Query | tasks/batch | Iterations |
|---|---|---|---|---|---|---|---|---|
| P2P-Cost | 128 | 16 | 0.01 | 0.001 | K=64 | K=8 | 16 | 5000 |
| P2P-Cost-Small | 128 | 16 | 0.01 | 0.001 | K=64 | K=8 | 16 | 500 |
| P2P-Dynamics | 128 | 17 | 0.01 | 0.001 | N=8 | N=8 | 16 | 1500 |
| Quadrotor | 128 | 17 | 0.01 | 0.001 | K=64 | K=32 | 16 | 1000 |
| Obstacle | 128 | 22 | 0.01 | 0.001 | K=8 | K=8 | 16 | 2000 |

All MAML runs use one inner gradient step, Adam outer optimizer, `split_seed: 42`,
`num_eval_batches: 10`.

---

## 9. Where to Find Each Value in the Code

| Hyperparameter group | Location |
|---|---|
| Model architecture | `model:` block in `configs/<env>.yaml`; constructed in `src/training/train_<env>.py` |
| Branch/trunk I/O dims | `SetONet(...)` call in each `src/training/train_<env>.py` |
| Pretraining | `training:` block in `configs/<env>.yaml`; loop in `src/training/train.py` → `train_<env>.py` |
| Data generation | `data:` block in `configs/<env>.yaml`; `src/envs/create_<env>.py`, `generate_data.py` |
| Adaptation | `src/adaptation/cost_adapt.py`, `ADAPT_METHODS` in `Makefile` |
| Meta-training | `setonet_meta:` block; `src/training/setonet_meta.py`, `meta_train_<env>.py` |
| MAML | `maml:` / `maml_model:` blocks; `src/training/maml.py`, `train_maml_<env>.py` |
| Normalization | `src/normalization.py` (`GridEnvironmentNormalizer`) |
