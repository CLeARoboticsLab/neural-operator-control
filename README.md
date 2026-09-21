# Neural Operators for Multi-Task Control and Adaptation

Code for the paper **Neural Operators for Multi-Task Control and Adaptation**
(David SeWell, Xingjian Li, Stepan Tretiakov, Krishna Kumar, David Fridovich-Keil),
*Transactions on Machine Learning Research*, 2026.
[OpenReview](https://openreview.net/forum?id=jciOb0z5Wm)

A single SetONet operator maps a task description (cost evaluations, dynamics
transitions, or obstacle geometry) to a feedback policy, generalises zero-shot to
held-out tasks, and adapts with a few gradient steps. This repository reproduces
every table and figure in the paper on four optimal-control environments
(P2P-Cost, P2P-Dynamics, Quadrotor, Obstacle) and the HalfCheetah-v3 locomotion task
from the iMuJoCo benchmark.

## Installation

```bash
git clone <this repository>
cd <repository>
python -m venv .venv && source .venv/bin/activate
pip install -e .            # CPU
pip install -e ".[cuda]"    # GPU (JAX with CUDA 12)
```

Or build the Docker image and run any `make` target inside it:

```bash
make docker-build
make docker-run CMD="make all"
```

Dependencies are pinned in `pyproject.toml` (JAX, Equinox, Optax, CasADi/IPOPT
for the obstacle solver). Python 3.10+.

## Data

Four of the optimal-control datasets (P2P-Cost, P2P-Cost-Small, P2P-Dynamics,
Quadrotor) are included in `data/` (35 MB). The Obstacle dataset is too large to
bundle (70 MB) and is generated with the IPOPT expert; any dataset can be
regenerated from its YAML config with the expert solvers (LQR, iLQR, IPOPT):

```bash
make data-obstacle            # required before any Obstacle target
make data-ocp                 # regenerate all five
```

HalfCheetah-v3 uses the offline SAC expert rollouts of the iMuJoCo benchmark
([Patacchiola et al., 2023](https://github.com/mpatacchiola/imujoco), Apache-2.0).
The target downloads `dataset.zip` (about 2 GB) from Zenodo and extracts the 53
HalfCheetah configurations into `data/halfcheetah/`:

```bash
make data-halfcheetah
```

## Pretrained checkpoints

`checkpoints/<env>/` ships, for each of the five optimal-control environments:

| Checkpoint | Model |
|---|---|
| `pretrained/setonet.eqx` | SetONet operator trained by behavioral cloning |
| `baseline/baseline.eqx` | B1, task-conditioned MLP |
| `baseline2/baseline2.eqx` | B2, context-conditioned MLP |

Each is a single training run with the config saved next to it. The operators in
the paper were selected best-of-10 by validation loss (`num_runs: 10` in
`configs/`), so retrain with `make train ENV=<env>` for paper-exact numbers; the
shipped checkpoints reproduce the zero-shot comparison and are a working starting
point for the fine-tuning and cost-based adaptation experiments. The MAML and
meta-trained operators (`maml/`, `meta_branch/`, `meta_full/`) and all HalfCheetah
models are not included and are produced by the training targets below.

## Reproducing the paper

Results are regenerated from the configs in `configs/`. Trained models are
written to `checkpoints/<env>/…` and results to `outputs/`.

| Paper item | Command | Needs |
|---|---|---|
| Table 2 (adaptation, OCP envs) | `make table2` | MAML + meta checkpoints (`make train-maml`, `make train-meta`) |
| Baseline table (B1/B2 vs SetONet) | `bash run_paper_sweep.sh` | trains + evaluates itself |
| Fixed-K fair comparison | `bash run_fixedk_table.sh` | B1 checkpoints |
| Figure 4 operator fitting | `make figure4` | shipped checkpoints |
| Figure 5 task-resolution invariance | `make figure5` | shipped checkpoints |
| Figure 6 MAML vs SetONet | `make figure6` | MAML + meta checkpoints |
| Figure 7 cost-based adaptation | `make figure7` | shipped checkpoints |
| Figure 8 quadrotor OOD adaptation | `make figure8` | Quadrotor meta checkpoints |
| Figure 9 HalfCheetah control predictions | `make figure9` | HalfCheetah data + checkpoints |
| Figure 10 HalfCheetah adaptation grid | `make figure10` | HalfCheetah data + checkpoints |
| Figure 10 from the paper's grid results | `make figure10-paper` | nothing |

`make all` regenerates Table 2 and Figures 4 to 8 from whatever is in
`checkpoints/`; `make all-from-scratch` first regenerates the data and retrains
everything. Individual models:

```bash
make train      ENV=p2p_cost                       # pretrained SetONet
make train-meta ENV=p2p_dynamics VARIANT=meta_branch   # SetONet-Meta
make train-meta ENV=p2p_dynamics VARIANT=meta_full     # SetONet-Meta-Full
make train-maml ENV=p2p_dynamics                   # MAML baseline
```

`ENV` is one of `p2p_cost p2p_cost_small p2p_dynamics quadrotor obstacle halfcheetah`.

### HalfCheetah-v3 (iMuJoCo)

Each of the 53 configurations varies body mass, limb length, joint range or
friction, with 100 expert episodes per configuration. Configurations are split
80/20 (seed 42) into training and held-out tasks. The operator's context set is
built from logged `(state, action) -> next_state` transitions; the trunk maps a
state to the expert action (`context_generation.md`, Appendix B of the paper).

```bash
make data-halfcheetah
make train-halfcheetah        # pretrained, MAML, SetONet-Meta, SetONet-Meta-Full
make figure9                  # zero-shot / FT / Meta-Full vs expert on a held-out task
make figure10                 # adaptation grid: 5 methods x {1,5,10,25} demos x {1..200} steps x 5 seeds
```

The grid (`src/evaluation/evaluate_halfcheetah.py`) checkpoints its progress to
`outputs/results/halfcheetah_grid/grid_checkpoint.json` and resumes on rerun;
`--methods`, `--demos`, `--steps` and `--num-seeds` restrict it. The exact grid
values behind Figure 10 in the paper are included in
`paper_results/halfcheetah_grid.json`, so the figure can be redrawn without any
training (`make figure10-paper`). Hyperparameters are in `configs/halfcheetah.yaml`.

## MLP baselines

Two non-operator baselines accompany the zero-shot SetONet row.

**B1, task-conditioned MLP.** Receives the task parameters explicitly
(`[state, time, task_params] -> action`), trained by ordinary multi-task
behavioral cloning and evaluated zero-shot. Parameter-comparable to SetONet.
Task parameters: P2P-Cost/-Small goal state; P2P-Dynamics and Quadrotor the
dynamics-config vector; Obstacle the obstacle `(x, y)` positions padded with -1.

**B2, context-conditioned MLP.** Must infer the task from the same
`(location, value)` context the operator's branch receives, but consumes it as a
flat, padded vector (variable size `N` written into a fixed 128-slot bank).
The gap B2 to SetONet isolates the operator architecture; the gap B1 to B2
isolates the cost of inferring the task instead of being handed it.

```bash
for env in p2p_cost p2p_dynamics quadrotor; do
    make train ENV=$env             # pretrained SetONet
    make train-baseline ENV=$env    # B1
    make train-baseline2 ENV=$env   # B2
    make eval-baseline  ENV=$env    # zero-shot: baseline | baseline2 | pretrained, 5 seeds
done
python src/plotting/make_baseline_table.py
```

`eval-baseline` writes `outputs/results/<env>/baseline_zeroshot.json` with
relative-L2 mean and std over 5 seeds. The split is canonical
(`src/training/baseline_data.py`): SetONet is pretrained on the canonical train
tasks only and all models are scored on the same held-out test tasks.
`bash run_paper_sweep.sh` runs the whole pipeline (best-of-10 operators) and
`bash run_fixedk_table.sh` produces the fixed-context-size comparison.

## Project structure

```
├── Makefile                   # every reproduction target (`make help`)
├── configs/                   # one YAML per environment: data, model, training, evaluation
├── data/                      # bundled OCP datasets (Obstacle generated, HalfCheetah downloaded)
├── checkpoints/               # bundled pretrained SetONet + B1/B2 per OCP environment
├── paper_results/             # grid results shipped with the paper (Figure 10)
├── src/
│   ├── setonet.py             # SetONet / DeepONet architecture
│   ├── normalization.py       # state/action normalisation wrappers
│   ├── envs/                  # data generation, expert solvers, data loaders
│   │   └── imujoco_dataloader.py   # iMuJoCo (HalfCheetah-v3) loader
│   ├── training/              # pretraining, meta-training, MAML and MLP baselines
│   ├── adaptation/            # cost-based adaptation
│   ├── evaluation/            # adaptation grids, resolution sweeps, OOD evaluation
│   └── plotting/              # one script per figure / table
├── configuration_overview.md  # per-environment architecture and hyperparameters
├── context_generation.md      # how each environment's context set is built
├── cost_based_adaptation.md   # cost-based fine-tuning details
├── run_paper_sweep.sh         # baseline table (B1 / B2 / SetONet)
└── run_fixedk_table.sh        # fixed-K comparison
```

## Environments

| Environment | Expert | Tasks | State / control | Task variation |
|---|---|---|---|---|
| P2P-Cost | LQR | 500 goals | 4 / 2 | goal location (cost) |
| P2P-Cost-Small | LQR | 50 goals | 4 / 2 | reduced-data variant |
| P2P-Dynamics | iLQR | 100 configs | 4 / 2 | friction, velocity and acceleration limits |
| Quadrotor | iLQR | 100 configs | 6 / 2 | mass, inertia, arm length |
| Obstacle | IPOPT | 500 | 4 / 2 | number and placement of obstacles |
| HalfCheetah-v3 | SAC (iMuJoCo) | 53 configs | 17 / 6 | mass, limb length, joint range, friction |

## Hardware

All experiments in the paper were run on a single NVIDIA GPU (RTX 5070). The
CPU install works for every target but the training runs are slow.

## Citation

```bibtex
@article{sewell2026neuraloperators,
  title   = {Neural Operators for Multi-Task Control and Adaptation},
  author  = {SeWell, David and Li, Xingjian and Tretiakov, Stepan and Kumar, Krishna and Fridovich-Keil, David},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=jciOb0z5Wm}
}
```

## License

MIT, see `LICENSE`. The iMuJoCo dataset is distributed separately under its own
Apache-2.0 license.
