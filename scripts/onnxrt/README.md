# ONNX Runtime (Spike)

This directory provides an ONNX Runtime path for Gemmini Spike runs.

## 1) Build ORT runner (`ort_test`) + custom ops

```bash
cd /root/flexi/third-party/I-ViT-Gemmini
scripts/onnxrt/build_ort_riscv.sh
```

Custom ops source is local to this repo:

- `scripts/onnxrt/ort_ivit_ops/ivit_ops.c`

## 2) Export ONNX model

### DeiT-Tiny (I-ViT INT8 graph)

```bash
scripts/onnxrt/export_onnx.sh \
  --model-name deit_tiny_patch16_224 \
  --checkpoint /root/checkpoint_last.pth.tar \
  --output /root/flexi/third-party/I-ViT-Gemmini/build/ort/ivit_tiny_int8.onnx
```

Backward-compatible wrapper:

```bash
scripts/onnxrt/export_ivit_onnx.sh \
  --checkpoint /root/checkpoint_last.pth.tar
```

Direct DeiT exporter script:

```bash
python3 scripts/onnxrt/export_deit_ivit_onnx.py \
  --checkpoint /root/checkpoint_last.pth.tar
```

### Swin-Tiny

```bash
scripts/onnxrt/export_onnx.sh \
  --model-name swin_tiny_patch4_window7_224 \
  --checkpoint /path/to/swin_tiny_checkpoint.pth.tar \
  --output /root/flexi/third-party/I-ViT-Gemmini/build/ort/swin_tiny_int8.onnx
```

Swin export now defaults to the same DeiT-style custom-op flow:

- `QLinearConv` + `QLinearMatMul` + `MatMulInteger` for Gemmini-friendly INT8 execution
- `ivit.QLayernorm`, `ivit.Shiftmax`, `ivit.ShiftGELU` custom ops
- no FP32->QOperator post-quantization pass on this default path
- fixed Swin depth configuration: `(2,2,6,2)`

If checkpoint is unavailable, test-only export with random weights:

```bash
scripts/onnxrt/export_onnx.sh \
  --model-name swin_tiny_patch4_window7_224 \
  --allow-random-init \
  --output /root/flexi/third-party/I-ViT-Gemmini/build/ort/swin_tiny_int8.onnx
```

Direct Swin custom exporter:

```bash
python3 scripts/onnxrt/export_swin_ivit_onnx.py \
  --checkpoint /path/to/swin_tiny_checkpoint.pth.tar \
  --output /root/flexi/third-party/I-ViT-Gemmini/build/ort/swin_tiny_int8.onnx
```

Wrapper (DeiT wrapper style):

```bash
scripts/onnxrt/export_swin_ivit_onnx.sh \
  --checkpoint /path/to/swin_tiny_checkpoint.pth.tar
```

Legacy static-quantized QOperator flow is still available only for compatibility:
`--swin-export-style legacy-qop`.

## 3) Run on Spike

### DeiT-Tiny

```bash
scripts/onnxrt/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  /root/flexi/third-party/I-ViT-Gemmini/build/ort/ivit_tiny_int8.onnx \
  1 \
  --model-name deit_tiny_patch16_224
```

### Swin-Tiny

```bash
scripts/onnxrt/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  /root/flexi/third-party/I-ViT-Gemmini/build/ort/swin_tiny_int8.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224
```

Recommended verification run:

```bash
scripts/onnxrt/run_ort_spike.sh \
  scripts/gemmini/test_cat.jpg \
  1 \
  /root/flexi/third-party/I-ViT-Gemmini/build/ort/swin_tiny_int8.onnx \
  1 \
  --model-name swin_tiny_patch4_window7_224 \
  --log-file scripts/onnxrt/logs/swin_custom_x1.log
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
