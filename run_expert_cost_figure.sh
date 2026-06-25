#!/usr/bin/env bash
# ==============================================================================
# Figure 6 companion: per-task comparison using ACHIEVED TASK COST normalized to
# the expert (1.0 = expert-level, < 1.0 = better than expert), instead of rel-L2.
#
# Purely an evaluation/plotting step — trains nothing except the MAML baselines
# that Figure 6 itself already requires (and only if their checkpoints are absent).
#
# Usage:
#   bash run_expert_cost_figure.sh                 # P2P-Cost (validated)
#   bash run_expert_cost_figure.sh p2p_cost,p2p_dynamics,quadrotor
#
# Requires the pretrained SetONet checkpoints (from run_paper_sweep.sh) under
# checkpoints/<env>/pretrained.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN="uv run"                       # change to "python" if not using uv
ENVS="${1:-p2p_cost,p2p_dynamics,quadrotor}"   # all three Figure-6 envs

# Figure 6 compares against MAML — train it where missing (same as `make figure6`).
for env in $(echo "$ENVS" | tr ',' ' '); do
    if [ ! -f "checkpoints/$env/maml/maml.eqx" ]; then
        echo ">>> training MAML baseline for $env (needed for the scatter)"
        $RUN make train-maml ENV=$env
    fi
done

$RUN python src/plotting/plot_maml_scatter_cost.py \
    --config configs --data data --checkpoints checkpoints \
    --envs "$ENVS" --output outputs/figures/figure6_cost

echo "Done -> outputs/figures/figure6_cost_{scatter,summary}.pdf  (+ _cost.npz)"
