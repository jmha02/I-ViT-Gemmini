#!/usr/bin/env python3
"""
Validate TVM int8 I-ViT model against PyTorch float32 reference.

This script compares the outputs of the TVM quantized model with the
PyTorch floating-point model to verify correctness of the TVM implementation.
"""

import sys
import os

# Add TVM_benchmark path FIRST (before ivit_dir) to get correct models module
tvm_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
ivit_dir = os.path.dirname(tvm_benchmark_dir)
# Put tvm_benchmark_dir first so its 'models' takes precedence
sys.path.insert(0, ivit_dir)  # For models_quant
sys.path.insert(0, tvm_benchmark_dir)  # For TVM models module (must be first!)

import argparse
import torch
import torch.nn as nn
import tvm
from tvm import relay
import numpy as np

import models.build_model as build_model
from models.quantized_layers import QuantizeContext
import convert_model

# Import I-ViT PyTorch models
from models.vit_quant import deit_tiny_patch16_224 as quant_deit_tiny
from models.vit_quant import deit_small_patch16_224 as quant_deit_small
from models.vit_quant import deit_base_patch16_224 as quant_deit_base
from models.model_utils import freeze_model


def get_pytorch_model(model_name):
    """Get the PyTorch model class by name."""
    models = {
        "deit_tiny_patch16_224": quant_deit_tiny,
        "deit_small_patch16_224": quant_deit_small,
        "deit_base_patch16_224": quant_deit_base,
    }
    return models[model_name]


def load_pytorch_model(checkpoint_path, model_name):
    """Load PyTorch model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Get model class
    model_fn = get_pytorch_model(model_name)
    model = model_fn(pretrained=False)

    # Load state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    freeze_model(model)
    return model


def build_tvm_model(model_name, checkpoint_path, params_path):
    """Build TVM model from parameters."""
    # Load checkpoint to extract qconfig
    model = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    pretrained_params = np.load(params_path, allow_pickle=True)[()]

    depth = 12
    convert_model.load_qconfig(model, depth)

    # Build TVM model
    batch_size = 1
    image_shape = (3, 224, 224)
    data_layout = "NCHW"
    kernel_layout = "OIHW"

    func, params = build_model.get_workload(
        name=model_name,
        batch_size=batch_size,
        image_shape=image_shape,
        dtype="int8",
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )

    # Build with LLVM target
    target = "llvm"
    pretrained_params = {**pretrained_params}

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=pretrained_params)

    runtime = tvm.contrib.graph_executor.GraphModule(
        lib["default"](tvm.device(target, 0))
    )

    return runtime


def generate_random_image(seed=None):
    """Generate a random normalized image tensor."""
    if seed is not None:
        np.random.seed(seed)

    # Generate random RGB image in [0, 255]
    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

    # Convert to float and normalize like ImageNet preprocessing
    img = img.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Convert to NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32)

    return img


def generate_structured_image(pattern="checkerboard", seed=None):
    """Generate a structured test image."""
    if seed is not None:
        np.random.seed(seed)

    img = np.zeros((224, 224, 3), dtype=np.float32)

    if pattern == "checkerboard":
        # 16x16 aligned checkerboard (matches ViT patches)
        for i in range(14):
            for j in range(14):
                if (i + j) % 2 == 0:
                    img[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, :] = 1.0
    elif pattern == "gradient":
        # Horizontal gradient
        for i in range(224):
            img[:, i, :] = i / 223.0
    elif pattern == "stripes":
        # Vertical stripes
        for i in range(0, 224, 32):
            img[:, i : i + 16, :] = 1.0
    elif pattern == "random_patches":
        # Random values per patch
        for i in range(14):
            for j in range(14):
                val = np.random.random(3)
                img[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16, :] = val

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Convert to NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32)

    return img


def quantize_input(float_input, input_scale):
    """Quantize float input to int8 for TVM."""
    quantized = float_input / input_scale
    quantized = np.clip(quantized, -128, 127)
    quantized = np.round(quantized)
    return quantized.astype("int8")


def run_pytorch(model, float_input):
    """Run PyTorch model on float input."""
    with torch.no_grad():
        input_tensor = torch.from_numpy(float_input)
        output = model(input_tensor)
        return output.numpy()


def run_tvm(runtime, int8_input):
    """Run TVM model on int8 input."""
    runtime.set_input("data", int8_input)
    runtime.run()
    return runtime.get_output(0).numpy()


def compare_outputs(pytorch_out, tvm_out, name=""):
    """Compare PyTorch and TVM outputs."""
    # Get top-5 predictions
    pt_top5 = np.argsort(pytorch_out[0])[::-1][:5]
    tvm_top5 = np.argsort(tvm_out[0])[::-1][:5]

    # Check if top-1 matches
    top1_match = pt_top5[0] == tvm_top5[0]

    # Check top-5 overlap
    top5_overlap = len(set(pt_top5) & set(tvm_top5))

    # Compute output differences
    diff = np.abs(pytorch_out - tvm_out)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Compute correlation
    corr = np.corrcoef(pytorch_out.flatten(), tvm_out.flatten())[0, 1]

    result = {
        "name": name,
        "pt_top1": pt_top5[0],
        "tvm_top1": tvm_top5[0],
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "correlation": corr,
        "pt_top5": pt_top5,
        "tvm_top5": tvm_top5,
    }

    return result


def print_result(result):
    """Print comparison result."""
    status = "✓" if result["top1_match"] else "✗"
    print(f"\n{status} {result['name']}:")
    print(f"  PyTorch top-1: {result['pt_top1']}, TVM top-1: {result['tvm_top1']}")
    print(f"  Top-5 overlap: {result['top5_overlap']}/5")
    print(f"  PyTorch top-5: {result['pt_top5']}")
    print(f"  TVM top-5:     {result['tvm_top5']}")
    print(f"  Max diff: {result['max_diff']:.6f}, Mean diff: {result['mean_diff']:.6f}")
    print(f"  Correlation: {result['correlation']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Validate TVM vs PyTorch I-ViT")
    parser.add_argument(
        "--model-name",
        default="deit_tiny_patch16_224",
        choices=[
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
        ],
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument("--params-path", required=True, help="Path to TVM params.npy")
    parser.add_argument(
        "--num-random", type=int, default=10, help="Number of random images to test"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("TVM vs PyTorch Validation for I-ViT")
    print("=" * 60)

    # Load models
    print("\nLoading PyTorch model...")
    pytorch_model = load_pytorch_model(args.model_path, args.model_name)

    print("Building TVM model...")
    tvm_runtime = build_tvm_model(args.model_name, args.model_path, args.params_path)

    # Get input scale for quantization
    input_scale = QuantizeContext.qconfig_dict["qconfig_embed_conv"].input_scale
    print(f"Input scale: {input_scale}")

    results = []

    # Test structured patterns
    patterns = ["checkerboard", "gradient", "stripes", "random_patches"]
    print("\n" + "-" * 40)
    print("Testing structured patterns:")
    print("-" * 40)

    for pattern in patterns:
        float_input = generate_structured_image(pattern, seed=args.seed)
        int8_input = quantize_input(float_input, input_scale)

        pt_out = run_pytorch(pytorch_model, float_input)
        tvm_out = run_tvm(tvm_runtime, int8_input)

        result = compare_outputs(pt_out, tvm_out, name=f"Pattern: {pattern}")
        results.append(result)
        print_result(result)

    # Test random images
    print("\n" + "-" * 40)
    print(f"Testing {args.num_random} random images:")
    print("-" * 40)

    np.random.seed(args.seed)
    for i in range(args.num_random):
        float_input = generate_random_image(seed=args.seed + i)
        int8_input = quantize_input(float_input, input_scale)

        pt_out = run_pytorch(pytorch_model, float_input)
        tvm_out = run_tvm(tvm_runtime, int8_input)

        result = compare_outputs(pt_out, tvm_out, name=f"Random image {i + 1}")
        results.append(result)
        print_result(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    top1_matches = sum(1 for r in results if r["top1_match"])
    total = len(results)
    avg_correlation = np.mean([r["correlation"] for r in results])
    avg_top5_overlap = np.mean([r["top5_overlap"] for r in results])

    print(
        f"Top-1 agreement: {top1_matches}/{total} ({100 * top1_matches / total:.1f}%)"
    )
    print(f"Average Top-5 overlap: {avg_top5_overlap:.2f}/5")
    print(f"Average correlation: {avg_correlation:.6f}")

    if top1_matches / total >= 0.8 and avg_correlation > 0.9:
        print("\n✓ VALIDATION PASSED: TVM model outputs match PyTorch closely")
        return 0
    else:
        print("\n✗ VALIDATION FAILED: Significant discrepancy between TVM and PyTorch")
        return 1


if __name__ == "__main__":
    sys.exit(main())
