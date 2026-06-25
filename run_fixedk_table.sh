#!/usr/bin/env bash
# ==============================================================================
# Fixed-K (no-padding) fair table: train BOTH B2 and the SetONet operator at a
# single fixed context size K (train == eval), with B2's slot bank == K so there
# is ZERO padding. Removes the variable-size / padding mechanism entirely.
#
#   operator: trained with K=[K]                 B2: baseline2_fixed_K=K (no pad)
#   eval:     both at K (existing K=64 protocol) B1: reused (context-free)
#
# Operators use each config's num_runs (best-of-10); pass --operator-runs 1 for a
# quick single-run pass. Requires data/<env> and the existing B1 checkpoints
# (checkpoints/<env>/baseline) — run run_paper_sweep.sh first if absent.
#
# Usage:
#   bash run_fixedk_table.sh                 # K=64, all 3 envs
#   bash run_fixedk_table.sh 64 --operator-runs 1
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
K="${1:-64}"; shift || true
uv run python run_fixedk_table.py --K "$K" "$@"
echo "Done -> outputs/tables/fixedk_table.tex  (+ outputs/results_fixedK/<env>/baseline_zeroshot.json)"
