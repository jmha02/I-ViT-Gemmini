# ONNX Runtime (Spike)

This directory provides an ONNX Runtime path for Gemmini Spike runs.

## 1) Build ORT runner (`ort_test`) + custom ops

```bash
cd /root/flexi/third-party/I-ViT-Gemmini
scripts/onnxrt/build_ort_riscv.sh
```

## 2) Export ONNX model

### DeiT-Tiny (I-ViT INT8 graph)

```bash
scripts/onnxrt/export_onnx.sh \
  --model-name deit_tiny_patch16_224 \
  --checkpoint /root/checkpoint_last.pth.tar \
  --output /root/flexi/bench/build/ort/ivit_tiny_int8.onnx
```

Backward-compatible wrapper:

```bash
scripts/onnxrt/export_ivit_onnx.sh \
  --checkpoint /root/checkpoint_last.pth.tar
```

### Swin-Tiny

```bash
scripts/onnxrt/export_onnx.sh \
  --model-name swin_tiny_patch4_window7_224 \
  --checkpoint /path/to/swin_tiny_checkpoint.pth.tar \
  --fp32-output /root/flexi/bench/build/ort/swin_tiny_fp32.onnx \
  --output /root/flexi/bench/build/ort/swin_tiny_int8.onnx \
  --swin-progress
```

`--swin-progress` enables a progress-focused approximation flow:

- uses reduced Swin depths (`1,1,1,1`) for faster Spike completion
- lifts constant RHS for `MatMul`
- applies per-tensor static quantization (`QLinearMatMul`)
- forces `QLinearMatMul` zero-points (`a/b/y`) to 0 for Gemmini systolic constraints

This prioritizes runnability/progress over accuracy.

If checkpoint is unavailable, test-only export with random weights:

```bash
scripts/onnxrt/export_onnx.sh \
  --model-name swin_tiny_patch4_window7_224 \
  --allow-random-init \
  --swin-progress
```

## 3) Run on Spike

### DeiT-Tiny

```bash
scripts/onnxrt/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  /root/flexi/bench/build/ort/ivit_tiny_int8.onnx \
  1 \
  --model-name deit_tiny_patch16_224
```

### Swin-Tiny

```bash
scripts/onnxrt/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  /root/flexi/bench/build/ort/swin_tiny_int8.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224
```

Recommended verification run:

```bash
scripts/onnxrt/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  /root/flexi/bench/build/ort/swin_tiny_int8.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224 \
  --log-file scripts/onnxrt/logs/swin_tiny_progress_x1.log
```

`mode` argument:

- `0`: CPU fallback
- `1`: Gemmini output-stationary (recommended)
- `2`: Gemmini weight-stationary

## 4) Verify Gemmini usage from logs

```bash
python3 scripts/onnxrt/verify_gemmini_usage.py \
  --x1-log /path/to/ort_spike_x1.log \
  --x0-log /path/to/ort_spike_x0.log
```
