# I-ViT ORT Custom Ops

This directory contains ONNX Runtime custom ops used by the integer-only I-ViT
graphs:

- `ivit.QLayernorm`
- `ivit.Shiftmax`
- `ivit.ShiftGELU`

Build:

```bash
cd scripts/onnxrt/ort_ivit_ops
make ORT_RISCV_DIR=/path/to/onnxruntime-riscv
```

Outputs:

- `libivit_ops.a`

`scripts/onnxrt/build_ort_riscv.sh` builds this archive automatically and links
it into `systolic_runner/imagenet_runner/ort_test`.
