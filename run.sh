#!/usr/bin/env bash
# Convenience wrapper for the common pipeline steps.
# Each subcommand maps to a script in scripts/. Pass --help to any underlying
# script for its own flags; this wrapper just gives short names.
#
# Usage:
#   ./run.sh smoke             # Smoke test the reward pipeline on a few synthesized samples
#   ./run.sh vllm-check        # Verify OuteTTS + vLLM compatibility on this box
#   ./run.sh roundtrip-check   # Verify token -> audio decode round-trip via the codec
#   ./run.sh grpo-rollout      # Reproduce a GRPO rollout outside TRL (diagnostic)
#   ./run.sh dataset           # Generate the scored preference dataset
#   ./run.sh dpo               # Train DPO
#   ./run.sh kto               # Train KTO
#   ./run.sh grpo              # Train GRPO
#   ./run.sh eval              # Held-out eval across base / DPO / KTO / GRPO

set -euo pipefail

cmd="${1:-}"
shift || true

case "$cmd" in
    smoke)            uv run python scripts/diagnostics/smoke_test_rewards.py "$@" ;;
    vllm-check)       uv run python scripts/diagnostics/check_vllm_compat.py "$@" ;;
    roundtrip-check)  uv run python scripts/diagnostics/check_audio_roundtrip.py "$@" ;;
    grpo-rollout)     uv run python scripts/diagnostics/check_grpo_rollout.py "$@" ;;
    dataset)          uv run python scripts/01_generate_dataset.py "$@" ;;
    dpo)              uv run python scripts/02_train_dpo.py "$@" ;;
    kto)              uv run python scripts/03_train_kto.py "$@" ;;
    grpo)             uv run python scripts/04_train_grpo.py "$@" ;;
    eval)             uv run python scripts/05_evaluate_all.py "$@" ;;
    "")               echo "Usage: ./run.sh {smoke|vllm-check|roundtrip-check|grpo-rollout|dataset|dpo|kto|grpo|eval}" && exit 1 ;;
    *)                echo "Unknown subcommand: $cmd" && exit 1 ;;
esac
