#!/usr/bin/env bash
# Convenience wrapper for the common pipeline steps.
# Each subcommand maps to a script in scripts/. Pass --help to any underlying
# script for its own flags; this wrapper just gives short names.
#
# Usage:
#   ./run.sh smoke            # Phase 1 smoke test of the reward pipeline
#   ./run.sh vllm-check       # Verify OuteTTS + vLLM compatibility on this box
#   ./run.sh dataset          # [Phase 2] Generate scored preference dataset
#   ./run.sh dpo              # [Phase 2] Train DPO
#   ./run.sh kto              # [Phase 3] Train KTO
#   ./run.sh grpo             # [Phase 4] Train GRPO
#   ./run.sh eval             # [Phase 5] Run held-out eval across methods
#   ./run.sh all              # [Phase 5] dataset -> dpo -> kto -> grpo -> eval

set -euo pipefail

cmd="${1:-}"
shift || true

case "$cmd" in
    smoke)        uv run python scripts/00_smoke_test_rewards.py "$@" ;;
    vllm-check)   uv run python scripts/00b_check_vllm_compat.py "$@" ;;
    dataset)      echo "[Phase 2] not yet implemented" && exit 2 ;;
    dpo)          echo "[Phase 2] not yet implemented" && exit 2 ;;
    kto)          echo "[Phase 3] not yet implemented" && exit 2 ;;
    grpo)         echo "[Phase 4] not yet implemented" && exit 2 ;;
    eval)         echo "[Phase 5] not yet implemented" && exit 2 ;;
    all)          echo "[Phase 5] not yet implemented" && exit 2 ;;
    "")           echo "Usage: ./run.sh {smoke|vllm-check|dataset|dpo|kto|grpo|eval|all}" && exit 1 ;;
    *)            echo "Unknown subcommand: $cmd" && exit 1 ;;
esac
