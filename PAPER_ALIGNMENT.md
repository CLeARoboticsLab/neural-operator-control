# Paper-Codebase Alignment Reference

Authoritative settings extracted from the paper codebase
`multi_op_control` (branch `paper_table_2`), to align this reproduction package
for submission. Per-env diff of **paper → this repo**, plus an execution plan.

## Key architecture fact
`SetONet-1` is **not** a separate (smaller) class — it's a config string. There is
one `SetONet` class. The dynamics model is small only because its config passes no
`model_params`, so it uses SetONet **defaults**: `p=32`, `n_{phi,rho,trunk}=4`,
**`attention_n_tokens=1`**. The dynamics/quad/obstacle operators are wrapped in
`NormalizedSetONet` (a `GridEnvironmentNormalizer`, min-max state/action scaling,
**no learnable params**). P2P-Cost uses a bare `SetONet` with data-level norm.

## Paper SetONet architecture, per env
| Env | wrapper | p | layers (φ/ρ/trunk) | attn_tokens |
|---|---|---|---|---|
| P2P-Cost / Small | SetONet (bare) | 128 | 2/2/2 | 4 |
| P2P-Dynamics | NormalizedSetONet | **32** | **4/4/4** | **1** |
| Quadrotor | NormalizedSetONet | **64** | 3/3/3 | 4 |
| Obstacle | NormalizedSetONet | 128 | 4/4/4 | 4 |

This repo currently uses `p=128` and 2/3/3/4 layers, no `NormalizedSetONet`, and
matches only on **P2P-Cost** (and Obstacle on p/layers). Dynamics (p=128/3L) and
Quadrotor (p=128) are oversized.

## Paper SetONet training, per env
| Env | episodes | LR | num_runs | M | K | N | extras |
|---|---|---|---|---|---|---|---|
| P2P-Cost | 7500 | 1e-3 | 10 | 32 | [128] | 32 | **stored dataset** |
| P2P-Small | 7500 | 1e-3 | 10 | 32 | [128] | 10 | stored (50 goals×10) |
| P2P-Dynamics | 15000 | **5e-4** | 10 | 32 | [16,32,64,128] | 16 | cosine decay, grad-clip 1.0 |
| Quadrotor | **2000** | 1e-3 | 10 | 16 | 64 | 5 | — |
| Obstacle | 20000 | 1e-3 | 10 | 16 | [32,64,128] | — | L1 1e-5, cosine, seed **132**, train_perc **0.9** |

This repo: `num_runs=1` everywhere; cost is **online** (not stored) with N=1; dynamics
7500/1e-3 (no decay/clip); quadrotor 7500 (vs 2000); obstacle seed 42 / train_perc 0.8.

## Paper eval protocol (zero-shot, grad_steps=0), per env
| Env | K (ctx) | N_EVAL | holdout split | NUM_TASKS | context source |
|---|---|---|---|---|---|
| P2P-Cost | 64 | 32 | 0.25 | 20 | random (state,control,cost) from train split |
| P2P-Small | 64 | 32 | 0.25 | **10** | same |
| P2P-Dynamics | 64 | **8** | **0.5** | 20 | random (state,action,next_state) transitions |
| Quadrotor | 64 | **4** | **0.5** | 20 | random transitions |
| Obstacle | obstacle set | per-task | train_perc 0.9 | 50 | obstacle (x,y)+radius |

This repo's eval: P2P-Cost matches (K=64,N=32,0.25,20). Dynamics/Quad use N_EVAL=32 and
no holdout split (context+eval from same pool) — needs N_EVAL=8/4 + 0.5 split. Obstacle
uses our canonical split, not the paper's train_perc=0.9 / 50-task protocol.

## Paper MLP baseline (= our Baseline 1)
Plain MLP `[state, task_params, time] → control`, stored-dataset, **100 epochs**,
batch 2048, LR 1e-3, **num_runs 10**, eval_every 5.
| Env | in_size | width | depth | ~params |
|---|---|---|---|---|
| P2P-Cost | 9 (s4+goal4+t1) | 256 | 5 | ~266K |
| P2P-Dynamics | 10 (s4+dyn5+t1) | 128 | 5 | ~68K |
| Quadrotor | 10 (s6+dyn3+t1) | 128 | 5 | ~68K |
| Obstacle | 17 (s4+xy12+t1) | 256 | 6 | ~333K |

This repo's B1: depth 4, auto-sized width, online/iteration training, num_runs 1.

## Reported paper zero-shot SetONet numbers (`paper_exp/grid_*.npz`, setonet_ft, grad=0)
P2P-Cost 0.048 · P2P-Small 0.101 · P2P-Dyn 0.179 · Quadrotor 0.063 · Obstacle 0.238.

## Baseline 2 (context-conditioned MLP)
**Not in the paper.** No transition-consuming MLP baseline exists in the paper
codebase. B2 is an extension authored in this repo.

## Source-repo inconsistencies to be aware of
- Dynamics data path: gen outputs `..._H=30.npz`, training reads `...varying_dynamics.npz`.
- Obstacle episodes: `models/` config says 50000, `train_obstacle.yaml` says 20000 (operative).
- Obstacle train_perc: 0.9 (train/compare) vs 0.8 (mlp-baseline eval helper).

## Alignment status (single-run validations; server does best-of-10)
Each env's SetONet was retrained once (num_runs=1) with the aligned config to validate;
the configs are set to num_runs=10 for the paper's best-of-10 on a server.

| Env | aligned (1 run) | paper | what changed |
|---|---|---|---|
| P2P-Cost | 0.064 | 0.048 | stored dataset + on-policy K=128/N=32 (was online/random/N=1) |
| P2P-Dynamics | 0.090 | 0.179 | p=32/4L/1-tok + NormalizedSetONet + LR/episodes/decay; residual = data |
| Quadrotor | 0.071 | 0.063 | p=128→64; 2000 episodes |
| Obstacle | 0.252 (pre-change) | 0.238 | seed 132 / train_perc 0.9 (already matched) |
| P2P-Small | pending | 0.101 | stored small dataset (50×10), N=10, 10-task eval |

Eval protocol baked in: per-env N_EVAL (dyn 8, quad 4), 0.5 holdout split, p2p_cost
on-policy context, p2p_cost/small transfer-task count from config (20/10). Best-of-10
will land operators slightly below these single-run numbers.

Run everything with `bash run_paper_sweep.sh` (best-of-10 + 5 eval seeds, all 5 envs).
Removed dead code: `src/training/p2p_cost_pool.py` (buffer no longer used).

## Missing data (not checked into the paper repo)
`double_integrator_lqr.npz` (full), `double_integrator_lqr_100traj.npz`,
`trajectories_single_goal_varying_dynamics.npz`, `obstacle_data_hard_2-6.npy`.
Only `_small`, `planar_quadrotor_varying_dynamics.npz`, and eval-result grids are present.
Matching the stored-dataset training requires regenerating these via the data-gen configs.
