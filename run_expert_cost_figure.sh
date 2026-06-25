#!/usr/bin/env bash
# ==============================================================================
# Regenerate Figure 6 and its achieved-cost companion for the 3 control envs
# (P2P-Cost, P2P-Dynamics, Quadrotor), with the corrected eval conventions:
#   - P2P-Cost branch context is sampled ON-POLICY (not uniform-random).
#   - P2P-Dynamics trunk uses RAW integer time (P2P-Cost / Quadrotor use t/H).
# Both figures share the fixed code (src/plotting/plot_maml_scatter*.py), so this
# script just drives them end-to-end on the server.
#
# Produces:
#   outputs/figures/figure6.pdf                      rel-L2 scatter (MAML vs SetONet)
#   outputs/figures/figure6_cost_{scatter,summary}.pdf   achieved cost normalized to expert
#
# This is an evaluation/plotting step only — it changes no training. It trains the
# MAML baselines (which both figures compare against) only if their checkpoints are
# absent. Requires the pretrained SetONet checkpoints under checkpoints/<env>/pretrained
# (from run_paper_sweep.sh).
#
# Usage:
#   bash run_expert_cost_figure.sh                                  # all 3 envs
#   bash run_expert_cost_figure.sh p2p_cost,p2p_dynamics,quadrotor  # explicit
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
RUN="uv run"                       # change to "python" if not using uv
ENVS="${1:-p2p_cost,p2p_dynamics,quadrotor}"   # all three Figure-6 envs

# Both figures compare against MAML — train it where missing (same as `make figure6`).
for env in $(echo "$ENVS" | tr ',' ' '); do
    if [ ! -f "checkpoints/$env/maml/maml.eqx" ]; then
        echo ">>> training MAML baseline for $env (needed for both figures)"
        $RUN make train-maml ENV=$env
    fi
done

echo ">>> [1/2] rel-L2 Figure 6 (MAML vs SetONet scatter)"
$RUN python src/plotting/plot_maml_scatter.py \
    --config configs --data data --checkpoints checkpoints \
    --output outputs/figures/figure6.pdf

echo ">>> [2/2] achieved-cost companion (normalized to expert)"
$RUN python src/plotting/plot_maml_scatter_cost.py \
    --config configs --data data --checkpoints checkpoints \
    --envs "$ENVS" --output outputs/figures/figure6_cost

echo "Done ->"
echo "  outputs/figures/figure6.pdf"
echo "  outputs/figures/figure6_cost_{scatter,summary}.pdf  (+ _cost.npz)"
