#!/usr/bin/env bash
# ==============================================================================
# Full paper-aligned sweep: best-of-10 training + 5-seed evaluation, all 5 envs.
#
#   - Operators (SetONet) train with num_runs=10 (best-of-10 by val loss) via the
#     per-env configs.
#   - MLP baselines B1/B2 train once each (their fit is very stable; the eval's
#     5 seeds provide the error bars). Set BASELINE_RUNS>1 below if you want
#     best-of-N baselines too (requires the num_runs loop in the baseline trainers).
#   - eval-baseline scores baseline | baseline2 | pretrained over 5 eval seeds.
#
# Usage:  bash run_paper_sweep.sh
# Hand to a server with GPU; expect many hours (obstacle SetONet is the slow one).
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN="uv run"   # change to "python" if not using uv

echo "=================== 1/3  DATA GENERATION ==================="

# P2P-Cost: stored LQR dataset (500 goals x 25 trajectories) — paper protocol.
$RUN python - <<'PY'
from src.envs.create_p2p_cost import generate_dataset
import numpy as np, os
os.makedirs("data/p2p_cost", exist_ok=True)
ds = generate_dataset(num_goals=500, trajectories_per_goal=25, horizon=50, dt=0.1,
    Q_weight=1.0, R_weight=0.1, Qf_weight=10.0, goal_range=(-10.,10.), state_range=(-10.,10.),
    vel_range=(-5.,5.), zero_velocity_goal=True, seed=42)
np.savez("data/p2p_cost/trajectories.npz", **{k:v for k,v in ds.items() if isinstance(v,np.ndarray)})
print("p2p_cost stored dataset:", ds["states"].shape)
PY

# P2P-Cost-Small: reduced stored dataset (50 goals x 10 trajectories).
$RUN python - <<'PY'
from src.envs.create_p2p_cost import generate_dataset
import numpy as np, os
os.makedirs("data/p2p_cost_small", exist_ok=True)
ds = generate_dataset(num_goals=50, trajectories_per_goal=10, horizon=50, dt=0.1,
    Q_weight=1.0, R_weight=0.1, Qf_weight=10.0, goal_range=(-10.,10.), state_range=(-10.,10.),
    vel_range=(-5.,5.), zero_velocity_goal=True, seed=42)
np.savez("data/p2p_cost_small/trajectories.npz", **{k:v for k,v in ds.items() if isinstance(v,np.ndarray)})
print("p2p_cost_small stored dataset:", ds["states"].shape)
PY

# P2P-Dynamics (iLQR), Quadrotor (iLQR), Obstacle (IPOPT/CasADi) — regenerated
# from their configs. Obstacle is slow (IPOPT). Skip any whose data/<env> you've
# already copied over.
$RUN make data-p2p_dynamics
$RUN make data-quadrotor
$RUN make data-obstacle

echo "=================== 2/3  TRAIN + EVAL ==================="
for env in p2p_cost p2p_cost_small p2p_dynamics quadrotor obstacle; do
    echo "----- $env : SetONet operator (best-of-10) -----"
    $RUN make train ENV=$env
    echo "----- $env : MLP baseline B1 -----"
    $RUN make train-baseline ENV=$env
    echo "----- $env : MLP baseline B2 (context) -----"
    $RUN make train-baseline2 ENV=$env
    echo "----- $env : zero-shot eval (5 seeds) -----"
    $RUN make eval-baseline ENV=$env
done

echo "=================== 3/3  TABLE ==================="
$RUN python src/plotting/make_baseline_table.py
echo "Done. Results in outputs/results/<env>/baseline_zeroshot.json; table in outputs/tables/baseline_table.tex"
