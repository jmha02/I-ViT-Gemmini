#!/usr/bin/env python3
"""
Compare PyTorch I-ViT vs TVM I-ViT accuracy on Tiny-ImageNet.

Both models output 1000 ImageNet classes. We compare:
1. PyTorch model accuracy on Tiny-ImageNet (200 classes, labels 0-199)
2. TVM model accuracy on same images
3. Agreement between the two models

No class mapping needed - just compare raw predictions.
"""

import sys
import os

tvm_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
ivit_dir = os.path.dirname(tvm_benchmark_dir)
sys.path.insert(0, ivit_dir)
sys.path.insert(0, tvm_benchmark_dir)

import argparse
import json
import torch
import tvm
from tvm import relay
import numpy as np
from PIL import Image
from tqdm import tqdm

import models.build_model as build_model
from models.quantized_layers import QuantizeContext
import convert_model

from models.vit_quant import deit_tiny_patch16_224 as quant_deit_tiny
from models.model_utils import freeze_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pytorch_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model = quant_deit_tiny(pretrained=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

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


def build_tvm_model(checkpoint_path, params_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    pretrained_params = np.load(params_path, allow_pickle=True)[()]

    depth = 12
    convert_model.load_qconfig(checkpoint, depth)

    func, _ = build_model.get_workload(
        name="deit_tiny_patch16_224",
        batch_size=1,
        image_shape=(3, 224, 224),
        dtype="int8",
    )

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, target=target, params=pretrained_params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    input_scale = QuantizeContext.qconfig_dict["qconfig_embed_conv"].input_scale
    return runtime, dev, input_scale


def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)

    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)

    return img_np.astype(np.float32)


def quantize_input(float_input, input_scale):
    quantized = float_input / input_scale
    quantized = np.clip(np.round(quantized), -128, 127)
    return quantized.astype("int8")


def run_pytorch(model, float_input):
    with torch.no_grad():
        input_tensor = torch.from_numpy(float_input)
        output = model(input_tensor)
        return output.numpy()[0]


def run_tvm(runtime, dev, int8_input):
    runtime.set_input("data", tvm.nd.array(int8_input, dev))
    runtime.run()
    return runtime.get_output(0).numpy()[0]


def evaluate(
    pytorch_model, tvm_runtime, tvm_dev, input_scale, dataset, num_samples=None
):
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    pt_top1_correct = 0
    pt_top5_correct = 0
    tvm_top1_correct = 0
    tvm_top5_correct = 0
    agreement_top1 = 0
    agreement_top5 = 0
    total = 0

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[i]
        image = sample["image"]
        label = sample["label"]

        float_input = preprocess_image(image)
        int8_input = quantize_input(float_input, input_scale)

        pt_out = run_pytorch(pytorch_model, float_input)
        tvm_out = run_tvm(tvm_runtime, tvm_dev, int8_input)

        pt_top5 = np.argsort(pt_out)[-5:][::-1]
        tvm_top5 = np.argsort(tvm_out)[-5:][::-1]

        pt_top1 = pt_top5[0]
        tvm_top1 = tvm_top5[0]

        if pt_top1 == label:
            pt_top1_correct += 1
        if label in pt_top5:
            pt_top5_correct += 1
        if tvm_top1 == label:
            tvm_top1_correct += 1
        if label in tvm_top5:
            tvm_top5_correct += 1

        if pt_top1 == tvm_top1:
            agreement_top1 += 1
        if len(set(pt_top5) & set(tvm_top5)) >= 3:
            agreement_top5 += 1

        total += 1

    results = {
        "total_samples": total,
        "pytorch": {
            "top1_accuracy": pt_top1_correct / total * 100,
            "top5_accuracy": pt_top5_correct / total * 100,
            "top1_correct": pt_top1_correct,
            "top5_correct": pt_top5_correct,
        },
        "tvm": {
            "top1_accuracy": tvm_top1_correct / total * 100,
            "top5_accuracy": tvm_top5_correct / total * 100,
            "top1_correct": tvm_top1_correct,
            "top5_correct": tvm_top5_correct,
        },
        "agreement": {
            "top1_match_rate": agreement_top1 / total * 100,
            "top5_overlap_rate": agreement_top5 / total * 100,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params-path", default="./params_dir_test/params.npy")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PyTorch vs TVM Accuracy on Tiny-ImageNet")
    print("=" * 60)

    print("\n[1/4] Loading Tiny-ImageNet...")
    from datasets import load_dataset

    dataset = load_dataset("zh-plus/tiny-imagenet", split="valid")
    print(f"       {len(dataset)} samples, 200 classes (labels 0-199)")

    print("\n[2/4] Loading PyTorch model...")
    pytorch_model = load_pytorch_model(args.checkpoint)

    print("\n[3/4] Building TVM model...")
    tvm_runtime, tvm_dev, input_scale = build_tvm_model(
        args.checkpoint, args.params_path
    )
    print(f"       Input scale: {input_scale}")

    print("\n[4/4] Running evaluation...")
    results = evaluate(
        pytorch_model, tvm_runtime, tvm_dev, input_scale, dataset, args.num_samples
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSamples: {results['total_samples']}")
    print(f"\nPyTorch I-ViT (FP32 inference):")
    print(f"  Top-1: {results['pytorch']['top1_accuracy']:.2f}%")
    print(f"  Top-5: {results['pytorch']['top5_accuracy']:.2f}%")
    print(f"\nTVM I-ViT (INT8 inference):")
    print(f"  Top-1: {results['tvm']['top1_accuracy']:.2f}%")
    print(f"  Top-5: {results['tvm']['top5_accuracy']:.2f}%")
    print(f"\nModel Agreement:")
    print(f"  Top-1 match: {results['agreement']['top1_match_rate']:.2f}%")
    print(f"  Top-5 overlap (>=3): {results['agreement']['top5_overlap_rate']:.2f}%")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
