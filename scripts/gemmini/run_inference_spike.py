#!/usr/bin/env python3
"""
Run I-ViT inference on a REAL IMAGE using Gemmini.

This script:
1. Loads a real image (JPEG/PNG)
2. Applies ImageNet preprocessing + quantization
3. Embeds the preprocessed image into the test harness
4. Runs on Spike or Verilator and reports the predicted class

Usage:
    python run_real_image.py --image /path/to/image.jpg
"""

import os
import sys
import argparse
import subprocess
import pathlib
import shutil
import tarfile
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
SCRIPTS_DIR = SCRIPT_DIR.parent  # scripts/
REPO_ROOT = SCRIPTS_DIR.parent  # I-ViT-Gemmini/
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
import tvm
from tvm import relay
import tvm.contrib.gemmini as gemmini

from models.build_model import get_workload
import pytorch_to_tvm_params as convert_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_CLASSES = None


def load_imagenet_classes():
    global IMAGENET_CLASSES
    if IMAGENET_CLASSES is not None:
        return IMAGENET_CLASSES

    classes_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    classes_file = SCRIPT_DIR / "imagenet_classes.txt"

    if not classes_file.exists():
        import urllib.request

        print(f"Downloading ImageNet classes...")
        urllib.request.urlretrieve(classes_url, classes_file)

    with open(classes_file, "r") as f:
        IMAGENET_CLASSES = [line.strip() for line in f.readlines()]

    return IMAGENET_CLASSES


def preprocess_image(image_path, input_scale):
    """
    Preprocess image for I-ViT inference.

    Steps:
    1. Resize to 256, center crop to 224x224
    2. Normalize with ImageNet mean/std
    3. Quantize to int8 using input_scale from checkpoint
    """
    img = Image.open(image_path).convert("RGB")

    width, height = img.size
    scale = 256 / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = img.resize((new_width, new_height), Image.BILINEAR)

    left = (new_width - 224) // 2
    top = (new_height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    img_np = np.array(img, dtype=np.float32) / 255.0

    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)

    img_int8 = np.clip(np.round(img_np / input_scale), -128, 127).astype(np.int8)

    return img_int8


def create_real_image_harness(output_dir, model_name, embed_dim, input_data):
    """Create test harness with embedded real image data."""

    input_c_array = ", ".join(str(x) for x in input_data.flatten())

    harness_code = f"""
#include <stdint.h>
#include <stddef.h>
#include "gemmini.h"
#include "tvmgen_default.h"

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static inline void print_char(char c) {{
    volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
    magic_mem[0] = 64;
    magic_mem[1] = 1;
    magic_mem[2] = (uintptr_t)&c;
    magic_mem[3] = 1;
    __sync_synchronize();
    tohost = (uintptr_t)magic_mem;
    while (fromhost == 0);
    fromhost = 0;
    __sync_synchronize();
}}

static inline void print_str(const char* s) {{
    while (*s) print_char(*s++);
}}

static inline void print_dec(uint64_t val) {{
    char buf[21];
    int i = 20;
    buf[i] = 0;
    do {{
        buf[--i] = '0' + (val % 10);
        val /= 10;
    }} while (val && i > 0);
    print_str(&buf[i]);
}}

static inline uint64_t read_cycles() {{
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}}

static inline void spike_exit(int code) {{
    tohost = (code << 1) | 1;
    while (1);
}}

#define INPUT_SIZE (1 * 3 * 224 * 224)
#define OUTPUT_SIZE 1000

static const int8_t input_data[INPUT_SIZE] __attribute__((aligned(16))) = {{
    {input_c_array}
}};

static float output_data[OUTPUT_SIZE] __attribute__((aligned(16)));

int main() {{
    print_str("\\n========================================\\n");
    print_str("I-ViT Real Image Inference\\n");
    print_str("Model: {model_name}\\n");
    print_str("========================================\\n\\n");
    
    struct tvmgen_default_inputs inputs;
    inputs.data = (void*)input_data;
    
    struct tvmgen_default_outputs outputs;
    outputs.output = output_data;
    
    print_str("Running inference...\\n");
    gemmini_flush(0);
    
    uint64_t start = read_cycles();
    tvmgen_default_run(&inputs, &outputs);
    gemmini_fence();
    uint64_t end = read_cycles();
    
    uint64_t cycles = end - start;
    
    print_str("\\n========================================\\n");
    print_str("Results\\n");
    print_str("========================================\\n");
    print_str("Cycles: ");
    print_dec(cycles);
    print_str("\\n\\n");
    
    print_str("Top-5 Predictions:\\n");
    for (int rank = 0; rank < 5; rank++) {{
        int max_idx = 0;
        float max_val = output_data[0];
        for (int j = 1; j < OUTPUT_SIZE; j++) {{
            if (output_data[j] > max_val) {{
                max_val = output_data[j];
                max_idx = j;
            }}
        }}
        print_str("  ");
        print_dec(rank + 1);
        print_str(". Class ");
        print_dec(max_idx);
        print_str("\\n");
        output_data[max_idx] = -1e9f;
    }}
    
    print_str("\\nDone!\\n");
    spike_exit(0);
    return 0;
}}
"""

    harness_path = output_dir / "main.c"
    with open(harness_path, "w") as f:
        f.write(harness_code)

    return harness_path


def fix_generated_code(output_dir):
    """Apply pointer type fix to generated TVM code."""
    lib0_path = output_dir / "codegen" / "host" / "src" / "default_lib0.c"

    with open(lib0_path, "r") as f:
        content = f.read()

    old_pattern = "&global_const_workspace,&global_workspace)"
    new_pattern = "(uint8_t*)&global_const_workspace,global_workspace)"

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        with open(lib0_path, "w") as f:
            f.write(content)
        print("[Fix] Applied pointer type fix to generated code")


def create_errno_stub(output_dir):
    """Create syscalls.c with __errno support."""
    fixed_dir = output_dir / "fixed_syscalls"
    fixed_dir.mkdir(exist_ok=True)

    syscalls_content = """
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <limits.h>

#define SYS_write 64

static int __errno_value = 0;
int* __errno(void) { return &__errno_value; }
int errno;

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

static uintptr_t syscall(uintptr_t which, uint64_t arg0, uint64_t arg1, uint64_t arg2) {
    volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
    magic_mem[0] = which;
    magic_mem[1] = arg0;
    magic_mem[2] = arg1;
    magic_mem[3] = arg2;
    __sync_synchronize();
    tohost = (uintptr_t)magic_mem;
    while (fromhost == 0);
    fromhost = 0;
    __sync_synchronize();
    return magic_mem[0];
}

void __attribute__((noreturn)) tohost_exit(uintptr_t code) {
    tohost = (code << 1) | 1;
    while (1);
}

void exit(int code) { tohost_exit(code); }
void abort() { exit(128 + 6); }

void printstr(const char* s) {
    const char* p = s;
    while (*p) p++;
    syscall(SYS_write, 1, (uintptr_t)s, p - s);
}

void __attribute__((weak)) thread_entry(int cid, int nc) {
    while (cid != 0);
}

int __attribute__((weak)) main(int argc, char** argv) {
    printstr("Implement main()!\\n");
    return -1;
}

void* memcpy(void* dest, const void* src, size_t len) {
    volatile char* d = (volatile char*)dest;
    volatile const char* s = (volatile const char*)src;
    while (len-- > 0) *d++ = *s++;
    return dest;
}

void* memset(void* dest, int byte, size_t len) {
    volatile char* d = (volatile char*)dest;
    while (len-- > 0) *d++ = (char)byte;
    return dest;
}

size_t strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}

size_t strnlen(const char *s, size_t n) {
    const char *p = s;
    while (n-- && *p) p++;
    return p - s;
}

int strcmp(const char* s1, const char* s2) {
    unsigned char c1, c2;
    do { c1 = *s1++; c2 = *s2++; } while (c1 != 0 && c1 == c2);
    return c1 - c2;
}

char* strcpy(char* dest, const char* src) {
    char* d = dest;
    while ((*d++ = *src++));
    return dest;
}

static void init_tls() {}

void _init(int cid, int nc) {
    init_tls();
    thread_entry(cid, nc);
    int ret = main(0, 0);
    exit(ret);
}

#undef putchar
int putchar(int ch) {
    static char buf[64] __attribute__((aligned(64)));
    static int buflen = 0;
    buf[buflen++] = ch;
    if (ch == '\\n' || buflen == sizeof(buf)) {
        syscall(SYS_write, 1, (uintptr_t)buf, buflen);
        buflen = 0;
    }
    return 0;
}

int printf(const char* fmt, ...) { printstr(fmt); return 0; }
int sprintf(char* str, const char* fmt, ...) { strcpy(str, fmt); return strlen(str); }

uintptr_t __attribute__((weak)) handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
    tohost_exit(1337);
}
"""

    with open(fixed_dir / "syscalls.c", "w") as f:
        f.write(syscalls_content)

    return fixed_dir


def fix_gemmini_includes(output_dir):
    """Fix Gemmini header include paths."""
    import re

    tvm_home = os.environ.get("TVM_HOME", "/root/flexi/third-party/tvm-gemmini")
    gemmini_rocc_tests = f"{tvm_home}/3rdparty/gemmini/software/gemmini-rocc-tests"
    gemmini_include = pathlib.Path(gemmini_rocc_tests) / "include"

    fixed_include_dir = output_dir / "fixed_include"
    fixed_include_dir.mkdir(exist_ok=True)

    for header_file in gemmini_include.glob("*.h"):
        with open(header_file, "r") as f:
            content = f.read()
        content = re.sub(r'#include\s*"include/([^"]+)"', r'#include "\1"', content)
        content = re.sub(
            r'#include\s*"rocc-software/src/([^"]+)"', r'#include "\1"', content
        )
        with open(fixed_include_dir / header_file.name, "w") as f:
            f.write(content)

    rocc_src = pathlib.Path(gemmini_rocc_tests) / "rocc-software" / "src"
    if rocc_src.exists():
        for header_file in rocc_src.glob("*.h"):
            shutil.copy(header_file, fixed_include_dir / header_file.name)

    return fixed_include_dir


def create_tvm_stubs(output_dir):
    """Create TVM runtime stubs."""
    stub_dir = output_dir / "tvm_stubs" / "tvm" / "runtime"
    stub_dir.mkdir(parents=True, exist_ok=True)

    with open(stub_dir / "c_runtime_api.h", "w") as f:
        f.write("""
#ifndef TVM_RUNTIME_C_RUNTIME_API_H_
#define TVM_RUNTIME_C_RUNTIME_API_H_
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef TVM_DLL
#define TVM_DLL
#endif
typedef int32_t tvm_index_t;
typedef void* TVMValue;
typedef int32_t TVMArrayHandle;
#define TVM_ASSERT(x) ((void)0)
#ifdef __cplusplus
}
#endif
#endif
""")

    with open(stub_dir / "c_backend_api.h", "w") as f:
        f.write("""
#ifndef TVM_RUNTIME_C_BACKEND_API_H_
#define TVM_RUNTIME_C_BACKEND_API_H_
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef TVM_DLL
#define TVM_DLL
#endif
static char __tvm_workspace[16 * 1024 * 1024];
static size_t __tvm_workspace_offset = 0;
static inline void* TVMBackendAllocWorkspace(int device_type, int device_id,
                                              uint64_t nbytes, int dtype_code_hint,
                                              int dtype_bits_hint) {
    void* ptr = &__tvm_workspace[__tvm_workspace_offset];
    __tvm_workspace_offset += ((nbytes + 15) / 16) * 16;
    return ptr;
}
static inline int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
    return 0;
}
#ifdef __cplusplus
}
#endif
#endif
""")

    return stub_dir


def compile_for_spike(output_dir, test_name="ivit_real"):
    """Compile for Spike."""
    tvm_home = os.environ.get("TVM_HOME", "/root/flexi/third-party/tvm-gemmini")
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")

    gemmini_rocc_tests = f"{tvm_home}/3rdparty/gemmini/software/gemmini-rocc-tests"
    riscv_tests = f"{gemmini_rocc_tests}/riscv-tests"
    bench_common = f"{riscv_tests}/benchmarks/common"

    cc = f"{riscv}/bin/riscv64-unknown-elf-gcc"

    codegen_dir = output_dir / "codegen" / "host"
    fixed_include = fix_gemmini_includes(output_dir)
    create_tvm_stubs(output_dir)
    fixed_syscalls_dir = create_errno_stub(output_dir)

    cflags = [
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-std=gnu99",
        "-O2",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        "-fno-builtin-memset",
        "-fno-builtin-memcpy",
        "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        f"-T{bench_common}/test.ld",
        "-DBAREMETAL=1",
        "-DTVM_RUNTIME_ALLOC",
        f"-I{riscv_tests}",
        f"-I{riscv_tests}/env",
        f"-I{fixed_include}",
        f"-I{gemmini_rocc_tests}/include",
        f"-I{bench_common}",
        f"-I{codegen_dir}/src",
        f"-I{codegen_dir}/include",
        f"-I{output_dir}/tvm_stubs",
        "-DPRINT_TILE=0",
    ]

    source_files = [str(output_dir / "main.c")]
    source_files += [str(fixed_syscalls_dir / "syscalls.c")]
    source_files += [str(f) for f in pathlib.Path(bench_common).glob("*.S")]
    source_files += [str(f) for f in (codegen_dir / "src").glob("*.c")]

    build_dir = output_dir / "build"
    build_dir.mkdir(exist_ok=True)
    output_binary = build_dir / f"{test_name}-baremetal"

    cmd = [cc] + cflags + source_files + ["-o", str(output_binary), "-lm", "-lgcc"]

    print(f"[Compile] Building {test_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Compilation failed:")
        print(result.stderr)
        return None

    print(f"[OK] Binary: {output_binary}")
    return output_binary


def run_spike(binary_path, timeout=600):
    """Run on Spike."""
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")
    spike = f"{riscv}/bin/spike"
    chipyard_lib = "/root/flexi/chipyard/.conda-env/lib"

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{chipyard_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    if "LD_PRELOAD" in env:
        del env["LD_PRELOAD"]

    cmd = [spike, "--extension=gemmini", str(binary_path)]

    print(f"\n[Spike] Running inference...")

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded {timeout}s")
        return None, None


def run_verilator(
    binary_path,
    timeout=600,
    chipyard_dir="/root/flexi/chipyard",
    verilator_config="BigRocketSaturnGemminiConfig",
    max_cycles=20000000000,
    dramsim=True,
):
    """Run on Chipyard Verilator simulator."""
    simulator = (
        pathlib.Path(chipyard_dir)
        / "sims"
        / "verilator"
        / f"simulator-chipyard.harness-{verilator_config}"
    )
    if not simulator.exists():
        print(f"[ERROR] Verilator simulator not found: {simulator}")
        return None, None

    cmd = [str(simulator), "+permissive"]

    if dramsim:
        dramsim_ini_dir = (
            pathlib.Path(chipyard_dir)
            / "generators"
            / "testchipip"
            / "src"
            / "main"
            / "resources"
            / "dramsim2_ini"
        )
        cmd += ["+dramsim", f"+dramsim_ini_dir={dramsim_ini_dir}"]

    cmd += [
        f"+max-cycles={max_cycles}",
        f"+loadmem={binary_path}",
        "+permissive-off",
        str(binary_path),
    ]

    print(f"\n[Verilator] Running inference...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded {timeout}s")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Run I-ViT on real image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint", type=str, default="/root/checkpoint_last.pth.tar"
    )
    parser.add_argument("--output-dir", type=str, default="ivit_real_image_project")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--simulator",
        type=str,
        default="spike",
        choices=["spike", "verilator"],
        help="Simulator backend",
    )
    parser.add_argument(
        "--chipyard-dir",
        type=str,
        default="/root/flexi/chipyard",
        help="Chipyard root path (used for Verilator)",
    )
    parser.add_argument(
        "--verilator-config",
        type=str,
        default="BigRocketSaturnGemminiConfig",
        help="Chipyard config name for Verilator simulator",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20000000000,
        help="Verilator +max-cycles limit",
    )
    parser.add_argument(
        "--no-dramsim",
        action="store_true",
        help="Disable +dramsim when running Verilator",
    )
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return 1

    print("=" * 60)
    print("I-ViT Real Image Inference on Gemmini")
    print("=" * 60)
    print(f"Image: {image_path}")

    gemmini.Environment.init_overwrite(
        dim=16,
        acc_rows=1024,
        bank_rows=4096,
        inp_dtype="int8",
        wgt_dtype="int8",
        acc_dtype="int32",
    )

    print("\n[1/6] Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    convert_model.load_qconfig(ckpt, 12)

    input_scale = ckpt["qact_input.act_scaling_factor"].item()
    print(f"       Input quantization scale: {input_scale}")

    print("\n[2/6] Preprocessing image...")
    input_data = preprocess_image(image_path, input_scale)
    print(f"       Input shape: {input_data.shape}, dtype: {input_data.dtype}")
    print(f"       Value range: [{input_data.min()}, {input_data.max()}]")

    print("\n[3/6] Building TVM model...")
    mod, _ = get_workload("deit_tiny_patch16_224", batch_size=1)

    params = {}
    params["embed_conv_weight"] = (
        ckpt["patch_embed.proj.weight_integer"].numpy().astype("int8")
    )
    params["embed_conv_bias"] = (
        ckpt["patch_embed.proj.bias_integer"]
        .numpy()
        .astype("int32")
        .reshape(1, -1, 1, 1)
    )

    for i in range(12):
        params[f"block_{i}_attn_qkv_weight"] = (
            ckpt[f"blocks.{i}.attn.qkv.weight_integer"].numpy().astype("int8")
        )
        params[f"block_{i}_attn_qkv_bias"] = (
            ckpt[f"blocks.{i}.attn.qkv.bias_integer"].numpy().astype("int32")
        )
        params[f"block_{i}_attn_proj_weight"] = (
            ckpt[f"blocks.{i}.attn.proj.weight_integer"].numpy().astype("int8")
        )
        params[f"block_{i}_attn_proj_bias"] = (
            ckpt[f"blocks.{i}.attn.proj.bias_integer"].numpy().astype("int32")
        )
        params[f"block_{i}_mlp_fc1_weight"] = (
            ckpt[f"blocks.{i}.mlp.fc1.weight_integer"].numpy().astype("int8")
        )
        params[f"block_{i}_mlp_fc1_bias"] = (
            ckpt[f"blocks.{i}.mlp.fc1.bias_integer"].numpy().astype("int32")
        )
        params[f"block_{i}_mlp_fc2_weight"] = (
            ckpt[f"blocks.{i}.mlp.fc2.weight_integer"].numpy().astype("int8")
        )
        params[f"block_{i}_mlp_fc2_bias"] = (
            ckpt[f"blocks.{i}.mlp.fc2.bias_integer"].numpy().astype("int32")
        )
        params[f"block_{i}_norm1_bias"] = (
            ckpt[f"blocks.{i}.norm1.bias_integer"].numpy().astype("int32")
        )
        params[f"block_{i}_norm2_bias"] = (
            ckpt[f"blocks.{i}.norm2.bias_integer"].numpy().astype("int32")
        )

    params["norm_bias"] = ckpt["norm.bias_integer"].numpy().astype("int32")
    params["head_weight"] = ckpt["head.weight_integer"].numpy().astype("int8")
    params["head_bias"] = ckpt["head.bias_integer"].numpy().astype("int32")
    params["cls_token_weight"] = ckpt["cls_token"].numpy()
    params["pos_embed_weight"] = ckpt["pos_embed"].numpy()

    tvm_params = {k: tvm.nd.array(v) for k, v in params.items()}

    mod = gemmini.preprocess_pass(mod)

    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    TARGET = tvm.target.target.Target({"kind": "c", "device": "gemmini"})
    EXECUTOR = tvm.relay.backend.Executor(
        "aot", options={"interface-api": "c", "unpacked-api": 1}
    )

    with gemmini.build_config(
        usmp_alg="hill_climb", opt_level=3, disabled_pass=["AlterOpLayout"]
    ):
        module = relay.build(
            mod, executor=EXECUTOR, runtime=RUNTIME, target=TARGET, params=tvm_params
        )

    print("\n[4/6] Exporting C code...")
    output_dir = pathlib.Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    mlf_path = output_dir / "model.tar"
    tvm.micro.export_model_library_format(module, mlf_path)
    with tarfile.open(mlf_path, "r:") as tar:
        tar.extractall(output_dir)

    fix_generated_code(output_dir)
    create_real_image_harness(output_dir, "deit_tiny_patch16_224", 192, input_data)

    print("\n[5/6] Compiling for Spike...")
    binary = compile_for_spike(output_dir, "ivit_real")
    if binary is None:
        return 1

    if args.simulator == "spike":
        print("\n[6/6] Running on Spike...")
        stdout, stderr = run_spike(binary, timeout=args.timeout)
    else:
        print("\n[6/6] Running on Verilator...")
        stdout, stderr = run_verilator(
            binary,
            timeout=args.timeout,
            chipyard_dir=args.chipyard_dir,
            verilator_config=args.verilator_config,
            max_cycles=args.max_cycles,
            dramsim=not args.no_dramsim,
        )

    if stdout is None:
        return 1

    print("\n" + "=" * 60)
    print(f"Simulation Output ({args.simulator}):")
    print("=" * 60)
    print(stdout)

    classes = load_imagenet_classes()

    print("\n" + "=" * 60)
    print("Class Labels:")
    print("=" * 60)

    import re

    for match in re.finditer(r"Class (\d+)", stdout):
        class_id = int(match.group(1))
        if class_id < len(classes):
            print(f"  Class {class_id}: {classes[class_id]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
