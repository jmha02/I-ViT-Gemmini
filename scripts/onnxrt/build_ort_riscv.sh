#!/usr/bin/env bash
#
# Build helper for onnxruntime-riscv + ivit custom ops.
# Delegates to the canonical build script under /root/flexi/bench/ivit.
#
# Usage:
#   scripts/onnxrt/build_ort_riscv.sh [--debug]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IVIT_GEMMINI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
FLEXI_DIR="$(cd "$IVIT_GEMMINI_DIR/../.." && pwd)"
DELEGATE="${FLEXI_DIR}/bench/ivit/build_ort_riscv.sh"

if [ ! -x "$DELEGATE" ]; then
    echo "ERROR: delegate script not found: $DELEGATE"
    echo "  Expected workspace layout rooted at: $FLEXI_DIR"
    exit 1
fi

"$DELEGATE" "$@"
