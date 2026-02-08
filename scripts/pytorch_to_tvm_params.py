import torch
import numpy as np
import argparse

import os

from models.quantized_layers import QConfig, QuantizeContext


def clip_to_int32(arr):
    """Clip values to int32 range to prevent overflow."""
    INT32_MIN = -(2**31)
    INT32_MAX = 2**31 - 1
    return np.clip(arr, INT32_MIN, INT32_MAX).astype("int32")


# Threshold for detecting dead channels (LayerNorm weights near zero)
DEAD_CHANNEL_THRESHOLD = 0.01


def fix_dead_channels_in_norm(model, block_idx, norm_name, verbose=True):
    """
    Fix dead channels in LayerNorm where weight ≈ 0 causes overflow.

    When LayerNorm weight is near zero:
    - norm_scaling_factor becomes extremely small (e.g., 3e-13)
    - bias_integer = floor(bias / weight / norm_scaling_factor) overflows int32

    Fix:
    - Set bias_integer to 0 for dead channels (they contribute nothing anyway)
    - Replace norm_scaling_factor with mean of valid channels

    Returns: (fixed_bias_int, fixed_norm_sf, dead_mask)
    """
    weight_key = f"blocks.{block_idx}.{norm_name}.weight"
    sf_key = f"blocks.{block_idx}.{norm_name}.norm_scaling_factor"
    bias_int_key = f"blocks.{block_idx}.{norm_name}.bias_integer"

    weight = model[weight_key].cpu().numpy()
    norm_sf = model[sf_key].cpu().numpy().reshape(-1)
    bias_int = model[bias_int_key].cpu().numpy().reshape(-1)

    # Detect dead channels
    dead_mask = np.abs(weight) < DEAD_CHANNEL_THRESHOLD
    num_dead = np.sum(dead_mask)

    if num_dead > 0:
        valid_mask = ~dead_mask
        mean_sf = np.mean(
            np.abs(norm_sf[valid_mask])
        )  # Use abs since sf can be negative

        if verbose:
            dead_indices = np.where(dead_mask)[0]
            print(
                f"  [FIX] blocks.{block_idx}.{norm_name}: {num_dead} dead channel(s) at {dead_indices.tolist()}"
            )
            for idx in dead_indices:
                print(
                    f"         Channel {idx}: weight={weight[idx]:.2e}, sf={norm_sf[idx]:.2e} -> {mean_sf:.2e}, bias_int={bias_int[idx]:.2e} -> 0"
                )

        # Fix: zero out bias_int and replace sf with mean
        bias_int[dead_mask] = 0
        norm_sf[dead_mask] = mean_sf * np.sign(norm_sf[dead_mask])  # Preserve sign
        # Handle case where original sf was 0
        norm_sf[dead_mask & (norm_sf == 0)] = mean_sf

    return bias_int.astype("int32"), norm_sf, dead_mask


def fix_final_norm_dead_channels(model, verbose=True):
    """Fix dead channels in the final LayerNorm (norm)."""
    weight_key = "norm.weight"
    sf_key = "norm.norm_scaling_factor"
    bias_int_key = "norm.bias_integer"

    weight = model[weight_key].cpu().numpy()
    norm_sf = model[sf_key].cpu().numpy().reshape(-1)
    bias_int = model[bias_int_key].cpu().numpy().reshape(-1)

    dead_mask = np.abs(weight) < DEAD_CHANNEL_THRESHOLD
    num_dead = np.sum(dead_mask)

    if num_dead > 0:
        valid_mask = ~dead_mask
        mean_sf = np.mean(np.abs(norm_sf[valid_mask]))

        if verbose:
            dead_indices = np.where(dead_mask)[0]
            print(
                f"  [FIX] norm (final): {num_dead} dead channel(s) at {dead_indices.tolist()}"
            )

        bias_int[dead_mask] = 0
        norm_sf[dead_mask] = mean_sf * np.sign(norm_sf[dead_mask])
        norm_sf[dead_mask & (norm_sf == 0)] = mean_sf

    return bias_int.astype("int32"), norm_sf, dead_mask


def save_params(model, depth, save_path):
    ## weight and bias (conv and dense)
    params = {}
    for key, tensor in model.items():
        if "weight_integer" in key:
            print(key)
            params[key] = tensor.cpu().numpy().astype("int8")
        if "bias_integer" in key:
            print(key)
            # Clip to int32 range to prevent overflow
            params[key] = clip_to_int32(tensor.cpu().numpy())

    renamed_params = {}
    renamed_params["embed_conv_weight"] = params["patch_embed.proj.weight_integer"]
    renamed_params["embed_conv_bias"] = params["patch_embed.proj.bias_integer"].reshape(
        1, -1, 1, 1
    )

    for i in range(depth):
        for key in ["weight_integer", "bias_integer"]:
            old_name = "blocks.%d.attn.qkv." % (i) + key
            new_name = "block_%d_attn_qkv_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = "blocks.%d.attn.proj." % (i) + key
            new_name = "block_%d_attn_proj_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = "blocks.%d.mlp.fc1." % (i) + key
            new_name = "block_%d_mlp_fc1_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

            old_name = "blocks.%d.mlp.fc2." % (i) + key
            new_name = "block_%d_mlp_fc2_" % (i) + key[:-8]
            renamed_params[new_name] = params[old_name]

    renamed_params["head_weight"] = params["head.weight_integer"]
    renamed_params["head_bias"] = params["head.bias_integer"]

    ## norm - fix dead channels where weight ≈ 0 causes bias_integer overflow
    print("\n=== Fixing dead channels in LayerNorm ===")
    for i in range(depth):
        fixed_bias1, _, _ = fix_dead_channels_in_norm(model, i, "norm1")
        renamed_params["block_%d_norm1_bias" % i] = fixed_bias1

        fixed_bias2, _, _ = fix_dead_channels_in_norm(model, i, "norm2")
        renamed_params["block_%d_norm2_bias" % i] = fixed_bias2

    fixed_norm_bias, _, _ = fix_final_norm_dead_channels(model)
    renamed_params["norm_bias"] = fixed_norm_bias

    ## other params
    renamed_params["cls_token_weight"] = model["cls_token"].cpu().numpy()
    renamed_params["pos_embed_weight"] = model["pos_embed"].cpu().numpy()

    np.save(os.path.join(save_path, "params.npy"), renamed_params)


def load_qconfig(model, depth):
    params = {}
    for key, tensor in model.items():
        if "scaling_factor" in key:
            tensor_np = tensor.cpu().numpy().reshape((-1))
            params[key] = tensor_np
            if "act_scaling_factor" in key and np.ndim(tensor_np) == 1:
                params[key] = tensor_np[0]

    QuantizeContext.qconfig_dict["qconfig_pos"] = QConfig(
        output_scale=params["qact_pos.act_scaling_factor"]
    )
    QuantizeContext.qconfig_dict["qconfig_addpos"] = QConfig(
        input_scale=params["patch_embed.qact.act_scaling_factor"],
        input_dtype="int16",
        output_scale=params["qact1.act_scaling_factor"],
    )
    ## Embed
    conv_input_scale = params["qact_input.act_scaling_factor"]
    conv_kernel_scale = params["patch_embed.proj.conv_scaling_factor"]
    conv_output_scale = conv_input_scale * conv_kernel_scale
    QuantizeContext.qconfig_dict["qconfig_embed_conv"] = QConfig(
        input_scale=conv_input_scale,
        kernel_scale=conv_kernel_scale,
        output_scale=conv_output_scale,
    )

    for i in range(depth):
        input_scale = (
            params["qact1.act_scaling_factor"]
            if i == 0
            else params["blocks.%d.qact4.act_scaling_factor" % (i - 1)]
        )
        _, fixed_norm1_sf, _ = fix_dead_channels_in_norm(
            model, i, "norm1", verbose=False
        )
        QuantizeContext.qconfig_dict["block_%d_qconfig_norm1" % (i)] = QConfig(
            input_scale=input_scale, output_scale=fixed_norm1_sf
        )

        input_scale = params["blocks.%d.qact1.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.attn.qkv.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_qkv" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.attn.qact1.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.attn.matmul_1.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_matmul_1" % (i)] = QConfig(
            input_scale=input_scale, output_scale=output_scale
        )

        input_scale = params["blocks.%d.attn.qact_attn1.act_scaling_factor" % (i)]
        softmax_scale = params["blocks.%d.attn.int_softmax.act_scaling_factor" % (i)]
        # TVM softmax outputs int8 scaled to 1/128, while checkpoint uses 1/32768.
        # Rescale to match int8 output range and keep matmul_2 scaling consistent.
        softmax_scale_int8 = softmax_scale * 256.0
        QuantizeContext.qconfig_dict["block_%d_qconfig_softmax" % (i)] = QConfig(
            input_scale=input_scale, output_scale=softmax_scale_int8
        )

        input_scale = softmax_scale_int8
        qact1_scale = params["blocks.%d.attn.qact1.act_scaling_factor" % (i)]
        output_scale = input_scale * qact1_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_matmul_2" % (i)] = QConfig(
            input_scale=input_scale, output_scale=output_scale
        )

        input_scale = params["blocks.%d.attn.qact2.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.attn.proj.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_proj" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.attn.qact3.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.qact2.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_add1" % (i)] = QConfig(
            input_scale=input_scale, input_dtype="int16", output_scale=output_scale
        )

        input_scale = params["blocks.%d.qact2.act_scaling_factor" % (i)]
        _, fixed_norm2_sf, _ = fix_dead_channels_in_norm(
            model, i, "norm2", verbose=False
        )
        QuantizeContext.qconfig_dict["block_%d_qconfig_norm2" % (i)] = QConfig(
            input_scale=input_scale, output_scale=fixed_norm2_sf
        )

        input_scale = params["blocks.%d.qact3.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.mlp.fc1.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_fc1" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.mlp.qact_gelu.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.mlp.act.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_gelu" % (i)] = QConfig(
            input_scale=input_scale, output_scale=output_scale, input_dtype="int8"
        )

        input_scale = params["blocks.%d.mlp.qact1.act_scaling_factor" % (i)]
        kernel_scale = params["blocks.%d.mlp.fc2.fc_scaling_factor" % (i)]
        output_scale = input_scale * kernel_scale
        QuantizeContext.qconfig_dict["block_%d_qconfig_fc2" % (i)] = QConfig(
            input_scale=input_scale,
            kernel_scale=kernel_scale,
            output_scale=output_scale,
        )

        input_scale = params["blocks.%d.mlp.qact2.act_scaling_factor" % (i)]
        output_scale = params["blocks.%d.qact4.act_scaling_factor" % (i)]
        QuantizeContext.qconfig_dict["block_%d_qconfig_add2" % (i)] = QConfig(
            input_scale=input_scale, input_dtype="int16", output_scale=output_scale
        )

    _, fixed_final_norm_sf, _ = fix_final_norm_dead_channels(model, verbose=False)
    last_block_output_scale = params["blocks.%d.qact4.act_scaling_factor" % (depth - 1)]
    QuantizeContext.qconfig_dict["qconfig_norm"] = QConfig(
        input_scale=last_block_output_scale, output_scale=fixed_final_norm_sf
    )

    input_scale = params["qact2.act_scaling_factor"]
    kernel_scale = params["head.fc_scaling_factor"]
    output_scale = input_scale * kernel_scale
    QuantizeContext.qconfig_dict["qconfig_head"] = QConfig(
        input_scale=input_scale, kernel_scale=kernel_scale, output_scale=output_scale
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="I-ViT convert model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="saved checkpoint path in QAT (checkpoint.pth.tar)",
    )
    parser.add_argument("--params-path", default="", help="Saved parameters directory")
    parser.add_argument("--depth", default=12, type=int, help="Depth of ViT")

    args = parser.parse_args()
    model = torch.load(args.model_path, map_location=torch.device("cpu"))
    # print(model.keys())

    save_params(model, args.depth, args.params_path)
