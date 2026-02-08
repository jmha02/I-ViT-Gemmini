# Gemmini Inference Scripts

This directory contains scripts for running I-ViT inference on Gemmini via Spike simulator.

## Quick Start

```bash
# Set up environment
export TVM_HOME=/path/to/tvm-gemmini
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export RISCV=/path/to/riscv-tools

# Run inference
python run_inference_spike.py \
    --image ./test_cat.jpg \
    --checkpoint /path/to/checkpoint.pth.tar
```

## Files

| File | Description |
|------|-------------|
| `run_inference_spike.py` | Main inference script - builds TVM model, compiles to C, runs on Spike |
| `test_cat.jpg` | Test image (cat) |
| `test_dog.jpg` | Test image (dog) |
| `imagenet_classes.txt` | ImageNet class labels |

## Verified Results

| Image | Top-1 Prediction | Status |
|-------|------------------|--------|
| test_cat.jpg | tiger cat | ✅ |
| test_dog.jpg | Labrador retriever | ✅ |

See the main [README](../../README.md) for full documentation.
