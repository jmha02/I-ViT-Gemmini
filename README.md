# I-ViT-Gemmini

Integer-only Vision Transformer (I-ViT) inference on Gemmini RISC-V accelerator via TVM.

## Overview

This repository provides a complete pipeline for running quantized DeiT models on the Gemmini systolic array accelerator:

- **I-ViT**: Integer-only quantized Vision Transformer (QAT trained)
- **TVM-Gemmini**: TVM backend for Gemmini accelerator code generation
- **Verification**: Spike ISS and Verilator RTL simulation support

## Verification Results

| Test | Image | Prediction | Status |
|------|-------|------------|--------|
| Cat | test_cat.jpg | tiger cat | ✅ |
| Dog | test_dog.jpg | Labrador retriever | ✅ |

**Model Agreement (PyTorch vs TVM)**: 78.6% Top-1, 93.4% Top-5 overlap

## Prerequisites

### Required Environment

- **Chipyard** with Gemmini configured (for Spike/Verilator)
- **RISC-V Toolchain** (riscv64-unknown-elf-gcc)
- **Python 3.8+** with conda

### Dependencies

```bash
pip install torch torchvision timm pillow numpy tqdm datasets scipy
```

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/jmha02/I-ViT-Gemmini.git
cd I-ViT-Gemmini
```

### 2. Build TVM-Gemmini

```bash
cd tvm-gemmini
mkdir build && cd build

# Configure
cp ../cmake/config.cmake .
echo "set(USE_LLVM ON)" >> config.cmake

# Build (takes ~30-60 minutes)
cmake ..
make -j$(nproc)

# Return to repo root
cd ../..

# Set environment
export TVM_HOME=$(pwd)/tvm-gemmini
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### 3. Set Up RISC-V Toolchain

You need a RISC-V toolchain with Gemmini support. If using Chipyard:

```bash
# Assuming Chipyard is already built
export RISCV=/path/to/chipyard/.conda-env/riscv-tools
```

### 4. Run Inference on Gemmini (Spike)

```bash
cd scripts/gemmini

# Run inference on test image
python run_real_image.py \
    --image ./test_cat.jpg \
    --checkpoint /path/to/checkpoint.pth.tar \
    --output-dir ./output \
    --timeout 600
```

Expected output:
```
Top-5 Predictions:
  1. Class 282: tiger cat
  2. Class 285: Egyptian cat
  3. Class 281: tabby
  ...
```

**Note**: Checkpoint file (`checkpoint_last.pth.tar`) is not included in this repo. You can:
- Train your own using I-ViT's `quant_train.py`
- Use a pretrained checkpoint (see I-ViT repo)

## Repository Structure

```
I-ViT-Gemmini/
├── README.md                       # This file
├── I-ViT/                          # I-ViT submodule (original repo)
│   ├── models/                     # PyTorch quantized model definitions
│   └── quant_train.py              # QAT training script
├── tvm-gemmini/                    # TVM-Gemmini submodule
│   └── python/tvm/contrib/gemmini/ # Gemmini backend
├── scripts/                        # Verification scripts
│   ├── convert_model.py            # PyTorch → TVM params converter
│   ├── compare_pytorch_tvm.py      # Model agreement test
│   ├── validate_tvm_vs_pytorch.py  # Detailed validation
│   ├── evaluate_accuracy_mapped.py # Batch accuracy evaluation
│   └── gemmini/
│       ├── run_real_image.py       # Main Gemmini inference script
│       ├── test_cat.jpg            # Test image
│       ├── test_dog.jpg            # Test image
│       └── README.md               # Gemmini-specific docs
└── tvm_models/                     # TVM model definitions
    └── build_model.py              # I-ViT model builder for TVM
```

## Scripts

### `scripts/gemmini/run_real_image.py`
End-to-end inference on Gemmini via Spike simulator.

```bash
python run_real_image.py --image <path> --checkpoint <path> [options]

Options:
  --image PATH         Input image (JPEG/PNG)
  --checkpoint PATH    PyTorch QAT checkpoint
  --output-dir PATH    Output directory (default: ./output)
  --timeout SECS       Spike timeout (default: 300)
```

### `scripts/convert_model.py`
Convert PyTorch checkpoint to TVM parameters.

```bash
python convert_model.py --model-path <ckpt> --params-path <out> --depth 12
```

### `scripts/compare_pytorch_tvm.py`
Compare PyTorch and TVM model outputs.

```bash
python compare_pytorch_tvm.py \
    --checkpoint <path> \
    --params-path ./params_dir/params.npy \
    --num-samples 1000
```

## Verilator Simulation (RTL)

For RTL-level verification (very slow, ~hours per image):

```bash
# Build Verilator simulator (in Chipyard)
cd chipyard/sims/verilator
make CONFIG=BigRocketSaturnGemminiConfig -j$(nproc) LOADMEM=1

# Run (example path)
./simulator-chipyard.harness-BigRocketSaturnGemminiConfig \
    +permissive +dramsim \
    +dramsim_ini_dir=$CHIPYARD_DIR/generators/testchipip/src/main/resources/dramsim2_ini \
    +max-cycles=1000000000 \
    +loadmem=<elf_path> \
    +permissive-off <elf_path>
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | DeiT-tiny (I-ViT quantized) |
| Embed dim | 192 |
| Heads | 3 |
| Depth | 12 blocks |
| Input | 224×224×3, INT8 |
| Output | 1000 classes (ImageNet) |
| Gemmini config | DIM=16 (16×16 systolic array) |

## Performance

| Backend | Time per Image | Cycles |
|---------|----------------|--------|
| Spike (ISS) | ~5 minutes | ~11B |
| Verilator (RTL) | ~hours | ~11B |

## References

- [I-ViT Paper](https://arxiv.org/abs/2207.01405): Integer-only Quantization for Efficient Vision Transformer Inference
- [Gemmini](https://github.com/ucb-bar/gemmini): Berkeley's systolic array generator
- [TVM](https://tvm.apache.org/): Deep learning compiler

## License

This project combines multiple components with their respective licenses:
- I-ViT: [Original License](https://github.com/zkkli/I-ViT)
- TVM-Gemmini: Apache 2.0
