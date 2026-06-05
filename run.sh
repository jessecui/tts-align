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
    smoke)            uv run python scripts/00_smoke_test_rewards.py "$@" ;;
    vllm-check)       uv run python scripts/00b_check_vllm_compat.py "$@" ;;
    roundtrip-check)  uv run python scripts/00c_check_audio_roundtrip.py "$@" ;;
    bare-generate)    uv run python scripts/00d_check_bare_generate.py "$@" ;;
    grpo-rollout)     uv run python scripts/00e_check_grpo_rollout.py "$@" ;;
    dataset)          uv run python scripts/01_generate_dataset.py "$@" ;;
    dpo)              uv run python scripts/02_train_dpo.py "$@" ;;
    kto)              uv run python scripts/03_train_kto.py "$@" ;;
    grpo)             uv run python scripts/04_train_grpo.py "$@" ;;
    eval)         uv run python scripts/05_evaluate_all.py "$@" ;;
    all)          echo "[Phase 5] not yet implemented" && exit 2 ;;
    "")           echo "Usage: ./run.sh {smoke|vllm-check|dataset|dpo|kto|grpo|eval|all}" && exit 1 ;;
    *)            echo "Unknown subcommand: $cmd" && exit 1 ;;
esac
