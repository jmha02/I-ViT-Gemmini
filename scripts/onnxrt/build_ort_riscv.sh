#!/usr/bin/env bash
#
# Build onnxruntime-riscv + imagenet_runner (ort_test) + I-ViT custom ops.
#
# Usage:
#   scripts/onnxrt/build_ort_riscv.sh [--debug]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FLEXI_DIR="$(cd "$REPO_ROOT/../.." && pwd)"
ORT_RISCV_DIR="${REPO_ROOT}/tvm-gemmini/3rdparty/gemmini/software/onnxruntime-riscv"
IMAGENET_RUNNER_DIR="${ORT_RISCV_DIR}/systolic_runner/imagenet_runner"
IVIT_OPS_DIR="${SCRIPT_DIR}/ort_ivit_ops"

if [[ ! -d "$ORT_RISCV_DIR" ]]; then
    echo "ERROR: onnxruntime-riscv path not found: $ORT_RISCV_DIR"
    exit 1
fi

if [[ ! -d "$IVIT_OPS_DIR" ]]; then
    echo "ERROR: ivit custom ops dir not found: $IVIT_OPS_DIR"
    exit 1
fi

if ! command -v riscv64-unknown-linux-gnu-gcc >/dev/null 2>&1; then
    FALLBACK_TC_BIN="${FLEXI_DIR}/chipyard/.conda-env/riscv-tools/bin"
    if [[ -x "${FALLBACK_TC_BIN}/riscv64-unknown-linux-gnu-gcc" ]]; then
        export PATH="${FALLBACK_TC_BIN}:${PATH}"
        echo "[INFO] Added toolchain to PATH: ${FALLBACK_TC_BIN}"
    fi
fi

if ! command -v riscv64-unknown-linux-gnu-gcc >/dev/null 2>&1; then
    echo "ERROR: riscv64-unknown-linux-gnu-gcc not found in PATH"
    exit 1
fi

BUILD_FLAG="--config=Release"
if [[ "${1:-}" == "--debug" ]]; then
    BUILD_FLAG=""
fi

echo "=== Step 1: Build onnxruntime-riscv ==="
echo "Source: $ORT_RISCV_DIR"
(
    cd "$ORT_RISCV_DIR"
    ./build.sh --parallel $BUILD_FLAG \
        --cmake_extra_defines \
        onnxruntime_USE_SYSTOLIC=ON \
        onnxruntime_SYSTOLIC_INT8=ON \
        onnxruntime_SYSTOLIC_FP32=OFF
)

echo
echo "=== Step 2: Build ivit custom ops ==="
echo "Source: $IVIT_OPS_DIR"
make -C "$IVIT_OPS_DIR" ORT_RISCV_DIR="$ORT_RISCV_DIR"

if [[ ! -f "${IVIT_OPS_DIR}/libivit_ops.a" ]]; then
    echo "ERROR: failed to build ${IVIT_OPS_DIR}/libivit_ops.a"
    exit 1
fi

echo
echo "=== Step 3: Build imagenet_runner (ort_test) ==="
echo "Source: $IMAGENET_RUNNER_DIR"

(
    cd "$IMAGENET_RUNNER_DIR"

    if ! grep -q "USE_CUSTOM_OP_LIBRARY" Makefile; then
        sed -i \
            "s|CDEFS = -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS|CDEFS = -Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS -DUSE_CUSTOM_OP_LIBRARY|" \
            Makefile
    fi

    if grep -q "libivit_ops.a" Makefile; then
        sed -i -E "s|[^[:space:]]*libivit_ops\\.a|${IVIT_OPS_DIR}/libivit_ops.a|g" Makefile
    else
        sed -i \
            "s| -ldl -static| ${IVIT_OPS_DIR}/libivit_ops.a -ldl -static|" \
            Makefile
    fi

    ./build.sh $BUILD_FLAG
)

if [[ ! -x "${IMAGENET_RUNNER_DIR}/ort_test" ]]; then
    echo "ERROR: ort_test not found after build"
    exit 1
fi

echo
echo "Build complete:"
echo "  ort_test: ${IMAGENET_RUNNER_DIR}/ort_test"
echo "  custom ops: ${IVIT_OPS_DIR}/libivit_ops.a"
