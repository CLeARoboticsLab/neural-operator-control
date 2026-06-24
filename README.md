# Multi-Task Operator Learning for Control

Supplementary code for reproducing all experiments in the paper.

## Quick Start

### Option 1: Docker

```bash
# Build the container
make docker-build

# Reproduce all results from pretrained checkpoints
make docker-run CMD="make all"

# Reproduce a specific figure
make docker-run CMD="make figure7"
```

### Option 2: Native Installation

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (CPU)
pip install -e .

# Or with GPU support
pip install -e ".[cuda]"

# Reproduce all results from pretrained checkpoints
make all
```

## Reproducing Results

### From Pretrained Checkpoints (Fast)

Pretrained model checkpoints are included in `checkpoints/`. To reproduce all tables and figures:

```bash
make all
```

Or reproduce individual results:

```bash
make table2       # Table 2: Adaptation results
make figure4      # Figure 4: Operator fitting
make figure5      # Figure 5: Task resolution invariance
make figure6      # Figure 6: MAML vs SetONet comparison
make figure7      # Figure 7: Cost-based adaptation
make figure8      # Figure 8: Quadrotor OOD adaptation
```

Outputs are saved to `outputs/figures/` and `outputs/tables/`.

### From Scratch (Slow)

To reproduce the OCP results from data generation through training and evaluation:

```bash
make all-from-scratch
```

## Training Individual Models

```bash
# Train SetONet on a specific environment
make train ENV=p2p_cost

# Meta-train SetONet-Meta or SetONet-Meta-Full
make train-meta ENV=p2p_dynamics VARIANT=meta
make train-meta ENV=p2p_dynamics VARIANT=meta_full

# Train MAML baseline
make train-maml ENV=p2p_dynamics
```

## Task-Conditioned MLP Baseline

A simple baseline that receives the task parameters **explicitly** (`[state, time,
task_params] -> action`), trained with ordinary multi-task behavioral cloning (no
meta-learning) and evaluated **zero-shot**. It is parameter-comparable to the
pretrained SetONet (reuses each env's `maml_model` width/depth) and is the natural
point of comparison for the SetONet zero-shot row.

Task parameters per environment: P2P-Cost/Small → goal state; P2P-Dynamics /
Quadrotor → dynamics-config vector; Obstacle → obstacle `(x,y)` positions padded
with `-1` up to the dataset's maximum obstacle count.

```bash
# Train on the same data as the pretrained operator, then evaluate zero-shot.
# Use 5 seeds to match the adaptation grid (seeds 42, 142, ..., 442).
for env in p2p_cost p2p_cost_small p2p_dynamics quadrotor obstacle; do
    make train ENV=$env             # pretrained SetONet (needed for side-by-side)
    make train-baseline ENV=$env    # task-conditioned MLP baseline
    make eval-baseline  ENV=$env    # reports BOTH, zero-shot, on the same test tasks
done
```

`eval-baseline` writes `outputs/results/<env>/baseline_zeroshot.json` with relative-L²
mean ± std over 5 seeds for the MLP baseline, the pretrained SetONet, and (if present)
Baseline 2. For a leak-free comparison the split is canonical
(`src/training/baseline_data.py`): the stored-dataset SetONet trainers are
restricted to the canonical *train* tasks, and all models are scored on the same
held-out *test* tasks (identical sampled trajectories per seed for the stored envs).

### Baseline 2 — context-conditioned MLP (operator ablation)

A second baseline that must **infer** the task instead of being told it: it gets the
same data SetONet puts in its branch — the (location, value) context set — but
consumes it as a flat, padded vector (`[state, time, flatten(context)]`) rather than
through a permutation-invariant set encoder. The context is sampled the way SetONet
samples its branch (variable size `N`, written into a fixed 128-slot bank, the rest
padded with -1). Width is auto-sized so it stays parameter-comparable to SetONet.

The gap **Baseline 2 → SetONet** isolates the value of the operator architecture;
the gap **Baseline 1 → Baseline 2** isolates the cost of inferring the task vs being
handed it.

```bash
for env in p2p_cost p2p_cost_small p2p_dynamics quadrotor obstacle; do
    make train-baseline2 ENV=$env
    make eval-baseline   ENV=$env   # now reports baseline | baseline2 | pretrained
done
```

## Project Structure

```
├── Dockerfile
├── Makefile
├── README.md
├── pyproject.toml
├── configs/               # YAML configuration files per environment
├── src/
│   ├── models/            # SetONet, DeepONet, MAML architectures
│   ├── envs/              # Environment definitions and expert solvers
│   ├── training/          # Training scripts (BC, meta-training, MAML)
│   ├── adaptation/        # Fine-tuning and cost-based adaptation
│   ├── evaluation/        # Metrics, rollouts, resolution sweeps
│   └── plotting/          # One script per figure/table
├── checkpoints/           # Pretrained model weights
├── data/                  # Datasets (generated or downloaded)
└── outputs/               # Reproduced figures and tables
```

## Environments

| Environment | Expert | Tasks | Description |
|---|---|---|---|
| P2P-Cost | LQR | 500 | 2D point mass, varying goal locations |
| P2P-Cost-Small | LQR | 50 | Reduced data variant |
| P2P-Dynamics | iLQR | 100 | Nonlinear dynamics, varying parameters |
| Quadrotor | iLQR | 100 | Planar quadrotor, varying physical params |
| Obstacle | IPOPT | 500 | Obstacle avoidance, varying configurations |

Pretrained checkpoints and data for all OCP environments above are included.

## Dependencies

- Python 3.10+
- JAX with CUDA support
- Equinox (neural network library for JAX)
- CasADi with IPOPT (for obstacle avoidance)

See `pyproject.toml` for pinned versions.

## iMuJoCo Environments

The iMuJoCo environments (Hopper, HalfCheetah, Walker2d) are not included in this archive. The iMuJoCo benchmark data is openly available at https://github.com/mpatacchiola/imujoco and can be used with the existing training and evaluation infrastructure in this codebase with minimal effort. The Makefile includes stub targets (`make data-imujoco`, `make figure9`, `make figure10`) for these environments.

## Hardware

All experiments were run on a single NVIDIA GPU.
