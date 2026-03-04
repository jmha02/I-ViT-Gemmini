import numpy as np
from tvm import relay

from . import quantized_layers as layers


SWIN_TINY_DEPTHS = (2, 2, 6, 2)
SWIN_TINY_HEADS = (3, 6, 12, 24)


def _slice_nhwc(x, batch, h0, h1, w0, w1, c):
    return relay.strided_slice(
        x,
        begin=[0, h0, w0, 0],
        end=[batch, h1, w1, c],
        strides=[1, 1, 1, 1],
    )


def _cyclic_shift_neg(x, batch, h, w, c, shift_size):
    """Roll by (-shift_size, -shift_size) on H/W."""
    if shift_size == 0:
        return x

    x1 = _slice_nhwc(x, batch, shift_size, h, shift_size, w, c)
    x2 = _slice_nhwc(x, batch, shift_size, h, 0, shift_size, c)
    x3 = _slice_nhwc(x, batch, 0, shift_size, shift_size, w, c)
    x4 = _slice_nhwc(x, batch, 0, shift_size, 0, shift_size, c)

    top = relay.concatenate([x1, x2], axis=2)
    bottom = relay.concatenate([x3, x4], axis=2)
    return relay.concatenate([top, bottom], axis=1)


def _cyclic_shift_pos(x, batch, h, w, c, shift_size):
    """Roll by (+shift_size, +shift_size) on H/W."""
    if shift_size == 0:
        return x

    x1 = _slice_nhwc(x, batch, h - shift_size, h, w - shift_size, w, c)
    x2 = _slice_nhwc(x, batch, h - shift_size, h, 0, w - shift_size, c)
    x3 = _slice_nhwc(x, batch, 0, h - shift_size, w - shift_size, w, c)
    x4 = _slice_nhwc(x, batch, 0, h - shift_size, 0, w - shift_size, c)

    top = relay.concatenate([x1, x2], axis=2)
    bottom = relay.concatenate([x3, x4], axis=2)
    return relay.concatenate([top, bottom], axis=1)


def _window_partition(x, batch, h, w, c, window_size):
    x = relay.reshape(
        x,
        [
            batch,
            h // window_size,
            window_size,
            w // window_size,
            window_size,
            c,
        ],
    )
    x = relay.transpose(x, [0, 1, 3, 2, 4, 5])
    x = relay.reshape(
        x,
        [batch * (h // window_size) * (w // window_size), window_size * window_size, c],
    )
    return x


def _window_reverse(x, batch, h, w, c, window_size):
    x = relay.reshape(
        x,
        [batch, h // window_size, w // window_size, window_size, window_size, c],
    )
    x = relay.transpose(x, [0, 1, 3, 2, 4, 5])
    x = relay.reshape(x, [batch, h, w, c])
    return x


def _relative_position_index(window_size):
    coords_h = np.arange(window_size)
    coords_w = np.arange(window_size)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = np.transpose(relative_coords, (1, 2, 0))
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1).astype("int32").reshape(-1)


def _attention_mask(input_resolution, window_size, shift_size):
    if shift_size == 0:
        return None

    h, w = input_resolution
    img_mask = np.zeros((1, h, w, 1), dtype=np.float32)

    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1

    nwh = h // window_size
    nww = w // window_size
    mask_windows = img_mask.reshape(1, nwh, window_size, nww, window_size, 1)
    mask_windows = np.transpose(mask_windows, (0, 1, 3, 2, 4, 5)).reshape(
        -1, window_size * window_size
    )
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    attn_mask = np.where(attn_mask != 0, -100.0, 0.0).astype(np.float32)
    return attn_mask


def _as_scalar(scale):
    arr = np.array(scale)
    return float(arr.reshape(-1)[0])


def _softmax_scale_for_int8(raw_scale):
    s = _as_scalar(raw_scale)
    return s * 256.0 if s < (1.0 / 1024.0) else s


def _qblock_swin(
    data,
    name,
    dim,
    num_heads,
    input_resolution,
    window_size,
    shift_size,
    block_input_scale,
    batch_size,
):
    h, w = input_resolution
    num_windows = batch_size * (h // window_size) * (w // window_size)
    tokens_per_window = window_size * window_size
    head_dim = dim // num_heads

    shortcut = data

    qconfig_norm1 = layers.get_qconfig(name + "_qconfig_norm1")
    norm1_bias = relay.var(name + "_norm1_bias", shape=[dim], dtype="int32")
    norm1 = layers.quantized_layernorm(data, norm1_bias)

    qconfig_qkv = layers.get_qconfig(name + "_qconfig_qkv")
    req1 = layers.requantize_via_float(
        norm1,
        input_scale=qconfig_norm1.output_scale,
        output_scale=qconfig_qkv.input_scale,
        out_dtype=qconfig_qkv.input_dtype,
    )

    x = relay.reshape(req1, [batch_size, h, w, dim])
    x = _cyclic_shift_neg(x, batch_size, h, w, dim, shift_size)
    x_windows = _window_partition(x, batch_size, h, w, dim, window_size)

    x_windows = relay.reshape(x_windows, [-3, 0])
    qkv = layers.quantized_dense(
        data=x_windows,
        name=name + "_attn_qkv",
        input_scale=qconfig_qkv.input_scale,
        kernel_scale=qconfig_qkv.kernel_scale,
        units=dim * 3,
        kernel_shape=(dim * 3, dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    qconfig_matmul_1 = layers.get_qconfig(name + "_qconfig_matmul_1")
    qkv = layers.requantize(
        qkv,
        input_scale=qconfig_qkv.output_scale,
        output_scale=qconfig_matmul_1.input_scale,
        out_dtype=qconfig_matmul_1.input_dtype,
    )
    qkv = relay.reshape(qkv, [num_windows, tokens_per_window, 3, num_heads, head_dim])
    qkv = relay.transpose(qkv, [2, 0, 3, 1, 4])
    qkv = relay.split(qkv, 3, axis=0)
    q = relay.reshape(relay.squeeze(qkv[0], axis=[0]), [-3, -2])
    k = relay.reshape(relay.squeeze(qkv[1], axis=[0]), [-3, -2])
    v = relay.reshape(relay.squeeze(qkv[2], axis=[0]), [-3, -2])

    qconfig_softmax = layers.get_qconfig(name + "_qconfig_softmax")
    qk_scale = (dim // num_heads) ** -0.5
    attn = layers.quantized_matmul_via_dense(
        q,
        k,
        input_scale1=qconfig_matmul_1.input_scale,
        input_scale2=qconfig_matmul_1.input_scale,
        requant_input_scale=qconfig_matmul_1.output_scale * qk_scale,
        requant_output_scale=qconfig_softmax.input_scale,
        requant_out_dtype="int8",
    )
    attn = relay.reshape(
        attn, [num_windows, num_heads, tokens_per_window, tokens_per_window]
    )

    qconfig_rel = layers.get_qconfig(name + "_qconfig_rel_pos")
    rel_pos_table = relay.var(
        name + "_attn_rel_pos_bias_table_weight",
        shape=((2 * window_size - 1) * (2 * window_size - 1), num_heads),
        dtype="float32",
    )
    rel_pos = layers.quantize(
        rel_pos_table,
        output_scale=qconfig_rel.output_scale,
        out_dtype="int8",
    )
    rel_index = relay.const(_relative_position_index(window_size), dtype="int32")
    rel_pos = relay.take(rel_pos, rel_index, axis=0)
    rel_pos = relay.reshape(rel_pos, [tokens_per_window, tokens_per_window, num_heads])
    rel_pos = relay.transpose(rel_pos, [2, 0, 1])
    rel_pos = relay.expand_dims(rel_pos, axis=0)
    if num_windows > 1:
        rel_pos = relay.repeat(rel_pos, num_windows, axis=0)

    attn = layers.add_float(
        lhs=attn,
        rhs=rel_pos,
        lhs_scale=qconfig_softmax.input_scale,
        rhs_scale=qconfig_rel.output_scale,
        output_scale=qconfig_softmax.input_scale,
        out_dtype="int32",
    )

    if shift_size > 0:
        mask = _attention_mask(input_resolution, window_size, shift_size)
        softmax_scale = _as_scalar(qconfig_softmax.input_scale)
        mask_int = np.round(mask / softmax_scale).astype("int32")
        mask_const = relay.const(mask_int, dtype="int32")
        mask_const = relay.expand_dims(mask_const, axis=1)
        attn = attn + mask_const

    attn = layers.quantized_softmax(attn, qconfig_softmax.input_scale)

    qconfig_matmul_2 = layers.get_qconfig(name + "_qconfig_matmul_2")
    qconfig_proj = layers.get_qconfig(name + "_qconfig_proj")

    attn = relay.reshape(attn, [num_windows * num_heads, tokens_per_window, tokens_per_window])
    v = relay.transpose(v, [0, 2, 1])

    x = layers.quantized_matmul_via_dense(
        attn,
        v,
        input_scale1=qconfig_matmul_2.input_scale,
        input_scale2=qconfig_matmul_1.input_scale,
        requant_input_scale=qconfig_matmul_2.output_scale,
        requant_output_scale=qconfig_proj.input_scale,
        requant_out_dtype=qconfig_proj.input_dtype,
    )

    x = relay.reshape(x, [num_windows, num_heads, tokens_per_window, head_dim])
    x = relay.transpose(x, [0, 2, 1, 3])
    x = relay.reshape(x, [num_windows, tokens_per_window, dim])

    x = _window_reverse(x, batch_size, h, w, dim, window_size)
    x = _cyclic_shift_pos(x, batch_size, h, w, dim, shift_size)
    x = relay.reshape(x, [batch_size, h * w, dim])

    x = relay.reshape(x, [-3, 0])
    x = layers.quantized_dense(
        data=x,
        name=name + "_attn_proj",
        input_scale=qconfig_proj.input_scale,
        kernel_scale=qconfig_proj.kernel_scale,
        units=dim,
        kernel_shape=(dim, dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    qconfig_add1 = layers.get_qconfig(name + "_qconfig_add1")
    x = layers.requantize(
        x,
        input_scale=qconfig_proj.output_scale,
        output_scale=qconfig_add1.input_scale,
        out_dtype=qconfig_add1.input_dtype,
    )
    x = relay.reshape(x, [batch_size, h * w, dim])

    x = layers.add_float(
        lhs=x,
        rhs=shortcut,
        lhs_scale=qconfig_add1.input_scale,
        rhs_scale=block_input_scale,
        output_scale=qconfig_add1.output_scale,
    )

    shortcut = x

    qconfig_norm2 = layers.get_qconfig(name + "_qconfig_norm2")
    norm2_bias = relay.var(name + "_norm2_bias", shape=[dim], dtype="int32")
    x = layers.quantized_layernorm(x, norm2_bias)

    qconfig_fc1 = layers.get_qconfig(name + "_qconfig_fc1")
    x = layers.requantize_via_float(
        x,
        input_scale=qconfig_norm2.output_scale,
        output_scale=qconfig_fc1.input_scale,
        out_dtype=qconfig_fc1.input_dtype,
    )

    x = relay.reshape(x, [-3, 0])
    x = layers.quantized_dense(
        data=x,
        name=name + "_mlp_fc1",
        input_scale=qconfig_fc1.input_scale,
        kernel_scale=qconfig_fc1.kernel_scale,
        units=4 * dim,
        kernel_shape=(4 * dim, dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    qconfig_gelu = layers.get_qconfig(name + "_qconfig_gelu")
    x = layers.requantize(
        x,
        input_scale=qconfig_fc1.output_scale,
        output_scale=qconfig_gelu.input_scale,
        out_dtype=qconfig_gelu.input_dtype,
    )
    x = relay.reshape(x, [batch_size, h * w, 4 * dim])
    x = layers.quantized_gelu(x, qconfig_gelu.input_scale)

    qconfig_fc2 = layers.get_qconfig(name + "_qconfig_fc2")
    x = layers.requantize(
        x,
        input_scale=qconfig_gelu.output_scale,
        output_scale=qconfig_fc2.input_scale,
        out_dtype=qconfig_fc2.input_dtype,
    )

    x = relay.reshape(x, [-3, 0])
    x = layers.quantized_dense(
        data=x,
        name=name + "_mlp_fc2",
        input_scale=qconfig_fc2.input_scale,
        kernel_scale=qconfig_fc2.kernel_scale,
        units=dim,
        kernel_shape=(dim, 4 * dim),
        kernel_dtype="int8",
        add_bias=True,
    )

    qconfig_add2 = layers.get_qconfig(name + "_qconfig_add2")
    x = layers.requantize(
        x,
        input_scale=qconfig_fc2.output_scale,
        output_scale=qconfig_add2.input_scale,
        out_dtype=qconfig_add2.input_dtype,
    )
    x = relay.reshape(x, [batch_size, h * w, dim])

    x = layers.add_float(
        lhs=x,
        rhs=shortcut,
        lhs_scale=qconfig_add2.input_scale,
        rhs_scale=qconfig_add1.output_scale,
        output_scale=qconfig_add2.output_scale,
    )

    x = relay.annotation.stop_fusion(x)
    return x


def _patch_merging(data, name, input_resolution, dim, batch_size):
    h, w = input_resolution

    x = relay.reshape(data, [batch_size, h, w, dim])

    x0 = relay.strided_slice(
        x,
        begin=[0, 0, 0, 0],
        end=[batch_size, h, w, dim],
        strides=[1, 2, 2, 1],
    )
    x1 = relay.strided_slice(
        x,
        begin=[0, 1, 0, 0],
        end=[batch_size, h, w, dim],
        strides=[1, 2, 2, 1],
    )
    x2 = relay.strided_slice(
        x,
        begin=[0, 0, 1, 0],
        end=[batch_size, h, w, dim],
        strides=[1, 2, 2, 1],
    )
    x3 = relay.strided_slice(
        x,
        begin=[0, 1, 1, 0],
        end=[batch_size, h, w, dim],
        strides=[1, 2, 2, 1],
    )

    x = relay.concatenate([x0, x1, x2, x3], axis=3)
    x = relay.reshape(x, [batch_size, (h // 2) * (w // 2), 4 * dim])

    qconfig_norm = layers.get_qconfig(name + "_qconfig_norm")
    norm_bias = relay.var(name + "_norm_bias", shape=[4 * dim], dtype="int32")
    x = layers.quantized_layernorm(x, norm_bias)

    qconfig_reduction = layers.get_qconfig(name + "_qconfig_reduction")
    x = layers.requantize_via_float(
        x,
        input_scale=qconfig_norm.output_scale,
        output_scale=qconfig_reduction.input_scale,
        out_dtype=qconfig_reduction.input_dtype,
    )

    x = relay.reshape(x, [-3, 0])
    x = layers.quantized_dense(
        data=x,
        name=name + "_reduction",
        input_scale=qconfig_reduction.input_scale,
        kernel_scale=qconfig_reduction.kernel_scale,
        units=2 * dim,
        kernel_shape=(2 * dim, 4 * dim),
        kernel_dtype="int8",
        add_bias=False,
    )

    qconfig_out = layers.get_qconfig(name + "_qconfig_out")
    x = layers.requantize(
        x,
        input_scale=qconfig_reduction.output_scale,
        output_scale=qconfig_out.output_scale,
        out_dtype=qconfig_out.input_dtype,
    )

    x = relay.reshape(x, [batch_size, (h // 2) * (w // 2), 2 * dim])
    return x


def Q_SwinTransformerTiny(
    data_shape,
    dtype="int8",
    num_classes=1000,
    embed_dim=96,
    depths=SWIN_TINY_DEPTHS,
    num_heads=SWIN_TINY_HEADS,
    window_size=7,
    mlp_ratio=4,
    debug_unit=None,
):
    batch_size = data_shape[0]
    if batch_size != 1:
        raise RuntimeError("Swin path currently supports batch_size=1 only")

    data = relay.var("data", shape=data_shape, dtype=dtype)

    qconfig_embed_conv = layers.get_qconfig("qconfig_embed_conv")
    x = layers.quantized_conv2d(
        data=data,
        name="patch_embed_proj",
        add_bias=True,
        input_channels=3,
        output_channels=embed_dim,
        kernel_dtype=qconfig_embed_conv.kernel_dtype,
        input_scale=qconfig_embed_conv.input_scale,
        kernel_scale=qconfig_embed_conv.kernel_scale,
        kernel_size=(4, 4),
        strides=(4, 4),
        padding=(0, 0),
        data_layout="NCHW",
        kernel_layout="OIHW",
    )

    x = relay.reshape(x, [0, 0, -1])
    x = relay.transpose(x, [0, 2, 1])

    qconfig_patch_norm = layers.get_qconfig("qconfig_patch_norm")
    x = layers.requantize_via_float(
        x,
        input_scale=qconfig_embed_conv.output_scale,
        output_scale=qconfig_patch_norm.input_scale,
        out_dtype=qconfig_patch_norm.input_dtype,
    )

    patch_norm_bias = relay.var("patch_embed_norm_bias", shape=[embed_dim], dtype="int32")
    x = layers.quantized_layernorm(x, patch_norm_bias)

    qconfig_patch_out = layers.get_qconfig("qconfig_patch_out")
    x = layers.requantize_via_float(
        x,
        input_scale=qconfig_patch_norm.output_scale,
        output_scale=qconfig_patch_out.output_scale,
        out_dtype=qconfig_patch_out.input_dtype,
    )

    qconfig_stem = layers.get_qconfig("qconfig_stem")
    x = layers.requantize_via_float(
        x,
        input_scale=qconfig_stem.input_scale,
        output_scale=qconfig_stem.output_scale,
        out_dtype=qconfig_stem.input_dtype,
    )
    x = relay.annotation.stop_fusion(x)
    if debug_unit == "post_stem":
        return relay.Function(relay.analysis.free_vars(x), x)

    h = 56
    w = 56
    dim = embed_dim
    current_scale = qconfig_stem.output_scale
    global_block_idx = 0

    for stage_idx, stage_depth in enumerate(depths):
        for block_idx in range(stage_depth):
            block_name = f"stage{stage_idx}_block{block_idx}"
            shift_size = 0 if (block_idx % 2 == 0 or min(h, w) <= window_size) else window_size // 2
            x = _qblock_swin(
                x,
                name=block_name,
                dim=dim,
                num_heads=num_heads[stage_idx],
                input_resolution=(h, w),
                window_size=window_size,
                shift_size=shift_size,
                block_input_scale=current_scale,
                batch_size=batch_size,
            )
            current_scale = layers.get_qconfig(block_name + "_qconfig_add2").output_scale
            if debug_unit == f"post_stage{stage_idx}_block{block_idx}":
                return relay.Function(relay.analysis.free_vars(x), x)
            if debug_unit == f"post_block{global_block_idx}":
                return relay.Function(relay.analysis.free_vars(x), x)
            global_block_idx += 1

        if stage_idx < len(depths) - 1:
            downsample_name = f"stage{stage_idx}_downsample"
            x = _patch_merging(
                x,
                name=downsample_name,
                input_resolution=(h, w),
                dim=dim,
                batch_size=batch_size,
            )
            current_scale = layers.get_qconfig(downsample_name + "_qconfig_out").output_scale
            if debug_unit == f"post_stage{stage_idx}_downsample":
                return relay.Function(relay.analysis.free_vars(x), x)
            h //= 2
            w //= 2
            dim *= 2

    qconfig_norm = layers.get_qconfig("qconfig_norm")
    norm_bias = relay.var("norm_bias", shape=[dim], dtype="int32")
    x = layers.quantized_layernorm(x, norm_bias)

    qconfig_post_norm = layers.get_qconfig("qconfig_post_norm")
    x = layers.requantize_via_float(
        x,
        input_scale=qconfig_norm.output_scale,
        output_scale=qconfig_post_norm.output_scale,
        out_dtype=qconfig_post_norm.input_dtype,
    )

    x_float = layers.dequantize(x, input_scale=qconfig_post_norm.output_scale)
    x_float = relay.mean(x_float, axis=1)
    if debug_unit == "pre_head_float":
        return relay.Function(relay.analysis.free_vars(x_float), x_float)

    qconfig_pre_head = layers.get_qconfig("qconfig_pre_head")
    x = layers.quantize(
        x_float,
        output_scale=qconfig_pre_head.output_scale,
        out_dtype=qconfig_pre_head.input_dtype,
    )
    if debug_unit == "pre_head":
        return relay.Function(relay.analysis.free_vars(x), x)

    qconfig_head = layers.get_qconfig("qconfig_head")
    x = layers.requantize(
        x,
        input_scale=qconfig_pre_head.output_scale,
        output_scale=qconfig_head.input_scale,
        out_dtype=qconfig_head.input_dtype,
    )

    head = layers.quantized_dense(
        data=x,
        name="head",
        input_scale=qconfig_head.input_scale,
        kernel_scale=qconfig_head.kernel_scale,
        units=num_classes,
        kernel_shape=(num_classes, dim),
        kernel_dtype="int8",
        add_bias=True,
    )
    if debug_unit == "head_int":
        return relay.Function(relay.analysis.free_vars(head), head)

    net = layers.dequantize(head, input_scale=qconfig_head.output_scale)
    if debug_unit == "head_float":
        return relay.Function(relay.analysis.free_vars(net), net)

    net = relay.nn.softmax(data=net)
    return relay.Function(relay.analysis.free_vars(net), net)


def get_swin_tiny_model(data_shape, dtype="int8", debug_unit=None):
    return Q_SwinTransformerTiny(
        data_shape=data_shape,
        dtype=dtype,
        embed_dim=96,
        depths=SWIN_TINY_DEPTHS,
        num_heads=SWIN_TINY_HEADS,
        window_size=7,
        mlp_ratio=4,
        debug_unit=debug_unit,
    )
