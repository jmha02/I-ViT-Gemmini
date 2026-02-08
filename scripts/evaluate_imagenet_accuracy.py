#!/usr/bin/env python3
"""
Evaluate I-ViT TVM int8 model accuracy on Tiny-ImageNet with proper class mapping.

This script:
1. Loads the TVM quantized DeiT model
2. Maps Tiny-ImageNet classes to ImageNet classes using WordNet IDs
3. Evaluates on Tiny-ImageNet validation set using TVM LLVM backend
4. Reports Top-1 and Top-5 accuracy

NOTE: This uses TVM LLVM backend (CPU), NOT Gemmini/Spike.
      Gemmini/Spike verification is done separately via run_real_image.py

Usage:
    python evaluate_accuracy_mapped.py --checkpoint /root/checkpoint_last.pth.tar \
        --num-samples 500
"""

import sys
import os

tvm_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, tvm_benchmark_dir)

import argparse
import json
import urllib.request
import torch
import tvm
from tvm import relay
import numpy as np
from PIL import Image
from tqdm import tqdm

import models.build_model as build_model
from models.quantized_layers import QuantizeContext
import convert_model


# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def create_wnid_to_imagenet_mapping():
    """Create mapping from WordNet ID to ImageNet class index."""
    synset_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"
    with urllib.request.urlopen(synset_url) as response:
        imagenet_synsets = response.read().decode("utf-8").strip().split("\n")
    return {wnid: idx for idx, wnid in enumerate(imagenet_synsets)}


def get_tiny_to_imagenet_mapping(tiny_wnids, wnid_to_imagenet):
    """Create Tiny-ImageNet index -> ImageNet index mapping."""
    mapping = {}
    for tiny_idx, wnid in enumerate(tiny_wnids):
        if wnid in wnid_to_imagenet:
            mapping[tiny_idx] = wnid_to_imagenet[wnid]
    return mapping


def load_imagenet_class_names():
    """Load ImageNet class names."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8").strip().split("\n")


def build_tvm_model(model_name, checkpoint_path, params_path, target="llvm"):
    """Build TVM model for inference."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Load pre-converted params
    pretrained_params = np.load(params_path, allow_pickle=True)[()]

    # Determine model depth
    depth = 12  # DeiT-tiny default

    # Load qconfig
    convert_model.load_qconfig(checkpoint, depth)

    # Get input scale from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    input_scale = None
    for key in state_dict:
        if "qact_input" in key and "act_scaling_factor" in key:
            input_scale = state_dict[key].item()
            break

    if input_scale is None:
        input_scale = 0.035607241094112396  # Default from training
        print(f"Warning: Using default input scale {input_scale}")
    else:
        print(f"Input quantization scale: {input_scale}")

    # Build TVM model
    func, params_from_build = build_model.get_workload(
        name=model_name,
        batch_size=1,
        image_shape=(3, 224, 224),
        dtype="int8",
    )

    # Compile model
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=pretrained_params)

    # Create runtime module
    dev = tvm.device(target, 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    return module, dev, input_scale


def preprocess_image(image, input_scale):
    """Preprocess image for I-ViT model."""
    # Resize to 224x224
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)

    # Convert to numpy and normalize
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # Quantize to INT8
    img_int8 = np.clip(np.round(img_np / input_scale), -128, 127).astype(np.int8)

    # NHWC -> NCHW
    img_int8 = np.transpose(img_int8, (2, 0, 1))
    img_int8 = np.expand_dims(img_int8, axis=0)

    return img_int8


def evaluate_accuracy(
    module, dev, input_scale, dataset, mapping, num_samples=None, imagenet_classes=None
):
    """Evaluate model accuracy on dataset."""
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    top1_correct = 0
    top5_correct = 0
    total = 0
    skipped = 0

    # For detailed analysis
    class_correct = {}
    class_total = {}

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[i]
        image = sample["image"]
        tiny_label = sample["label"]

        # Skip if class not mapped
        if tiny_label not in mapping:
            skipped += 1
            continue

        imagenet_label = mapping[tiny_label]

        # Preprocess
        input_data = preprocess_image(image, input_scale)

        # Run inference
        module.set_input("data", tvm.nd.array(input_data, dev))
        module.run()
        output = module.get_output(0).numpy()[0]

        # Get predictions
        top5_preds = np.argsort(output)[-5:][::-1]
        top1_pred = top5_preds[0]

        # Check accuracy
        if top1_pred == imagenet_label:
            top1_correct += 1
        if imagenet_label in top5_preds:
            top5_correct += 1

        total += 1

        # Track per-class accuracy
        if imagenet_label not in class_total:
            class_total[imagenet_label] = 0
            class_correct[imagenet_label] = 0
        class_total[imagenet_label] += 1
        if top1_pred == imagenet_label:
            class_correct[imagenet_label] += 1

    top1_acc = top1_correct / total * 100 if total > 0 else 0
    top5_acc = top5_correct / total * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"Accuracy Results (TVM LLVM Backend)")
    print(f"{'=' * 60}")
    print(f"Samples evaluated: {total}")
    print(f"Samples skipped (unmapped classes): {skipped}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% ({top1_correct}/{total})")
    print(f"Top-5 Accuracy: {top5_acc:.2f}% ({top5_correct}/{total})")

    # Show worst performing classes
    if imagenet_classes and class_total:
        print(f"\n{'=' * 60}")
        print("Per-class accuracy (lowest 10):")
        print(f"{'=' * 60}")
        class_accs = [
            (cls, class_correct.get(cls, 0) / class_total[cls] * 100)
            for cls in class_total
            if class_total[cls] >= 3
        ]
        class_accs.sort(key=lambda x: x[1])
        for cls, acc in class_accs[:10]:
            print(
                f"  {imagenet_classes[cls][:30]:30s}: {acc:5.1f}% ({class_correct.get(cls, 0)}/{class_total[cls]})"
            )

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_samples": total,
        "skipped_samples": skipped,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate I-ViT accuracy on Tiny-ImageNet"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PyTorch QAT checkpoint"
    )
    parser.add_argument(
        "--params-path",
        type=str,
        default=None,
        help="Path to TVM params.npy (default: ./params_dir_test/params.npy)",
    )
    parser.add_argument(
        "--model-name", type=str, default="deit_tiny_patch16_224", help="Model name"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--target", type=str, default="llvm", help="TVM target (default: llvm)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("I-ViT Accuracy Evaluation on Tiny-ImageNet")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name}")
    print(f"Target: {args.target}")
    print(f"Samples: {args.num_samples or 'all'}")
    print()

    # Load dataset
    print("[1/4] Loading Tiny-ImageNet dataset...")
    from datasets import load_dataset

    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")
    print(f"       Dataset size: {len(dataset)} samples")

    # Get class mapping
    print("[2/4] Creating class mapping...")
    tiny_wnids = dataset.features["label"].names
    wnid_to_imagenet = create_wnid_to_imagenet_mapping()
    mapping = get_tiny_to_imagenet_mapping(tiny_wnids, wnid_to_imagenet)
    print(f"       Mapped {len(mapping)}/200 classes")

    # Load ImageNet class names
    imagenet_classes = load_imagenet_class_names()

    # Build model
    print("[3/4] Building TVM model...")
    params_path = args.params_path or "./params_dir_test/params.npy"
    module, dev, input_scale = build_tvm_model(
        args.model_name, args.checkpoint, params_path, args.target
    )
    print("       Model compiled successfully")

    # Evaluate
    print("[4/4] Running evaluation...")
    results = evaluate_accuracy(
        module, dev, input_scale, dataset, mapping, args.num_samples, imagenet_classes
    )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
