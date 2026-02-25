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
python run_inference_spike.py \
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
├── README.md                           # This file
├── I-ViT/                              # I-ViT submodule (original repo)
│   ├── models/                         # PyTorch quantized model definitions
│   └── quant_train.py                  # QAT training script
├── tvm-gemmini/                        # TVM-Gemmini submodule
│   └── python/tvm/contrib/gemmini/     # Gemmini backend
├── scripts/                            # Verification scripts
│   ├── pytorch_to_tvm_params.py        # PyTorch checkpoint → TVM params
│   ├── compare_accuracy.py             # Model output comparison
│   ├── validate_model_outputs.py       # Detailed output validation
│   ├── evaluate_imagenet_accuracy.py   # ImageNet accuracy evaluation
│   └── gemmini/
│       ├── run_inference_spike.py      # Main Gemmini inference script
│       ├── test_cat.jpg                # Test image
│       ├── test_dog.jpg                # Test image
│       └── README.md                   # Gemmini-specific docs
└── models/                             # TVM model definitions
    ├── build_model.py                  # I-ViT model builder for TVM
    ├── quantized_vit.py                # Quantized ViT implementation
    ├── quantized_layers.py             # Quantized layer operations
    └── utils.py                        # Utility functions
```

## TVM Pipeline Flow

The inference pipeline works as follows:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PyTorch QAT     │───▶│ TVM Relay IR    │───▶│ Gemmini C Code  │
│ Checkpoint      │    │ Model           │    │ Generation      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Classification  │◀───│ Spike Simulator │◀───│ RISC-V ELF      │
│ Results         │    │ (Gemmini ISS)   │    │ Binary          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key TVM APIs Used

1. **`gemmini.Environment.init_overwrite()`** - Configure Gemmini hardware parameters
2. **`get_workload()`** → Creates TVM Relay model from quantized ViT definition
3. **`gemmini.preprocess_pass()`** - Apply Gemmini-specific optimization passes
4. **`relay.build()`** - Compile to C code with AOT executor
5. **`tvm.micro.export_model_library_format()`** - Export compiled model

## Scripts

### `scripts/gemmini/run_inference_spike.py`
End-to-end inference on Gemmini via Spike or Verilator simulator.

```bash
python run_inference_spike.py --image <path> --checkpoint <path> [options]

Options:
  --image PATH         Input image (JPEG/PNG)
  --checkpoint PATH    PyTorch QAT checkpoint
  --simulator MODE     spike (default) or verilator
  --output-dir PATH    Output directory (default: ivit_real_image_project)
  --timeout SECS       Simulation timeout (default: 600)
```

### `scripts/pytorch_to_tvm_params.py`
Convert PyTorch checkpoint to TVM parameters.

```bash
python pytorch_to_tvm_params.py --model-path <ckpt> --params-path <out> --depth 12
```

### `scripts/compare_accuracy.py`
Compare PyTorch and TVM model outputs.

```bash
python compare_accuracy.py \
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

# Run the same inference script on Verilator
cd /root/flexi/third-party/I-ViT-Gemmini/scripts/gemmini
python run_inference_spike.py \
    --image ./test_cat.jpg \
    --checkpoint /path/to/checkpoint.pth.tar \
    --simulator verilator \
    --chipyard-dir /root/flexi/chipyard \
    --verilator-config BigRocketSaturnGemminiConfig \
    --max-cycles 20000000000 \
    --timeout 14400
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
