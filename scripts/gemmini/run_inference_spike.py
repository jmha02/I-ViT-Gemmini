#!/usr/bin/env python3
"""
Run I-ViT inference on a REAL IMAGE using Gemmini.

This script:
1. Loads a real image (JPEG/PNG)
2. Applies ImageNet preprocessing + quantization
3. Embeds the preprocessed image into the test harness
4. Runs on Spike or Verilator and reports the predicted class

Usage:
    python run_inference_spike.py --image /path/to/image.jpg --checkpoint /path/to/checkpoint.pth.tar
"""

import os
import sys
import argparse
import subprocess
import pathlib
import shutil
import tarfile
import bisect
import re
import time
import threading
from collections import defaultdict, deque
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
from tvm.contrib.gemmini.legalize import LegalizeGemmini

from models.build_model import get_workload
import pytorch_to_tvm_params as convert_model

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGENET_CLASSES = None
MODEL_SPECS = {
    "deit_tiny_patch16_224": {"embed_dim": 192, "depth": 12},
    "swin_tiny_patch4_window7_224": {"embed_dim": 768, "depth": 12},
}


def preprocess_for_gemmini(mod, model_name):
    """Apply Gemmini preprocess with a lighter pipeline for large Swin graphs."""
    if model_name.startswith("swin_"):
        pattern = relay.op.contrib.get_pattern_table("gemmini")
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.ConvertLayout({"qnn.conv2d": ["NHWC", "HWIO"]})(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.MergeComposite(pattern)(mod)
        mod = relay.transform.InferType()(mod)
        mod = LegalizeGemmini()(mod)
        mod = relay.transform.InferType()(mod)
        # Avoid build-time QNN canonicalization failures after ANF by lowering
        # remaining standalone qnn ops up front while constants are still direct.
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.InferType()(mod)
        return mod

    return gemmini.preprocess_pass(mod)


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


def create_real_image_harness(
    output_dir,
    model_name,
    embed_dim,
    input_data,
    classification_output=True,
    debug_unit=None,
):
    """Create test harness with embedded real image data."""

    input_c_array = ", ".join(str(x) for x in input_data.flatten())
    run_mode = "classification" if classification_output else "debug"
    debug_label = debug_unit if debug_unit is not None else "full_model"
    prologue_code = ""
    pre_run_code = ""
    post_run_code = ""
    status_error_code = ""
    epilogue_code = ""
    result_code = ""
    if classification_output:
        prologue_code = f"""
    print_str("\\n========================================\\n");
    print_str("I-ViT Real Image Inference\\n");
    print_str("Model: {model_name}\\n");
    print_str("Run mode: {run_mode}\\n");
    print_str("Debug unit: {debug_label}\\n");
    print_str("========================================\\n\\n");
"""
        pre_run_code = '    print_str("Running inference...\\n");'
        post_run_code = """
    print_str("\\n========================================\\n");
    print_str("Results\\n");
    print_str("========================================\\n");
    print_str("Cycles: ");
    print_dec(cycles);
    print_str("\\n\\n");
"""
        status_error_code = """
    if (status != 0) {
        print_str("TVM run failed with status: ");
        print_dec((uint64_t)status);
        print_str("\\n");
        spike_exit(1);
    }
"""
        epilogue_code = '    print_str("\\nDone!\\n");'
        result_code = """
    print_str("Top-5 Predictions:\\n");
    float* output_logits = (float*)output_data;
    for (int rank = 0; rank < 5; rank++) {
        int max_idx = 0;
        float max_val = output_logits[0];
        for (int j = 1; j < 1000; j++) {
            if (output_logits[j] > max_val) {
                max_val = output_logits[j];
                max_idx = j;
            }
        }
        print_str("  ");
        print_dec(rank + 1);
        print_str(". Class ");
        print_dec(max_idx);
        print_str("\\n");
        output_logits[max_idx] = -1e9f;
    }
"""
    else:
        status_error_code = """
    if (status != 0) {
        spike_exit(1);
    }
"""
        result_code = """
    uint64_t checksum = 0;
    for (int i = 0; i < TVMGEN_DEFAULT_OUTPUT_SIZE; ++i) {
        checksum = checksum * 131 + output_data[i];
    }
    g_checksum = checksum;
"""

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

static const int8_t input_data[INPUT_SIZE] __attribute__((aligned(16))) = {{
    {input_c_array}
}};

static uint8_t output_data[TVMGEN_DEFAULT_OUTPUT_SIZE] __attribute__((aligned(16)));
volatile uint64_t g_cycles = 0;
volatile uint64_t g_checksum = 0;

int main() {{
{prologue_code}
    
    struct tvmgen_default_inputs inputs;
    inputs.data = (void*)input_data;
    
    struct tvmgen_default_outputs outputs;
    outputs.output = output_data;
    
{pre_run_code}
    gemmini_flush(0);
    
    uint64_t start = read_cycles();
    int32_t status = tvmgen_default_run(&inputs, &outputs);
    gemmini_fence();
    uint64_t end = read_cycles();
    
    uint64_t cycles = end - start;
    g_cycles = cycles;
    
{post_run_code}
{status_error_code}
{result_code}
    
{epilogue_code}
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
    timeout_arg = timeout if timeout and timeout > 0 else None

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout_arg
        )
        if result.returncode != 0:
            print(f"[ERROR] Spike exited with code {result.returncode}")
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
    verbose=False,
    log_dir=None,
    log_tail_lines=20000,
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
        return None, None, None, None

    cmd = [str(simulator), "+permissive"]
    if verbose:
        cmd.append("+verbose")

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

    if max_cycles and max_cycles > 0:
        cmd.append(f"+max-cycles={max_cycles}")
    else:
        print("[Info] Verilator +max-cycles is disabled")

    cmd += [
        f"+loadmem={binary_path}",
        "+permissive-off",
        str(binary_path),
    ]

    print(f"\n[Verilator] Running inference...")
    stdout_path = None
    stderr_path = None
    timeout_arg = timeout if timeout and timeout > 0 else None

    def _run_with_tailed_logs(cmd_args, timeout_sec, out_path, err_path, tail_lines):
        out_tail = deque(maxlen=tail_lines)
        err_tail = deque(maxlen=tail_lines)

        def _reader(pipe, sink):
            try:
                for line in iter(pipe.readline, ""):
                    sink.append(line)
            finally:
                pipe.close()

        proc = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            errors="replace",
        )
        out_t = threading.Thread(target=_reader, args=(proc.stdout, out_tail), daemon=True)
        err_t = threading.Thread(target=_reader, args=(proc.stderr, err_tail), daemon=True)
        out_t.start()
        err_t.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()

        out_t.join()
        err_t.join()

        with open(out_path, "w") as fout:
            fout.writelines(out_tail)
        with open(err_path, "w") as ferr:
            ferr.writelines(err_tail)
        return proc.returncode, timed_out

    try:
        if log_dir is not None:
            log_dir = pathlib.Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / "verilator_stdout.log"
            stderr_path = log_dir / "verilator_stderr.log"

            if log_tail_lines and log_tail_lines > 0:
                print(
                    f"[Info] Saving only last {log_tail_lines} lines of Verilator logs "
                    "(set --verilator-log-tail-lines 0 for full logs)"
                )
                _, timed_out = _run_with_tailed_logs(
                    cmd, timeout_arg, stdout_path, stderr_path, log_tail_lines
                )
                if timed_out:
                    print(f"[TIMEOUT] Exceeded {timeout}s")
                    return None, None, stdout_path, stderr_path
                return "", "", stdout_path, stderr_path

            with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
                subprocess.run(cmd, stdout=fout, stderr=ferr, text=True, timeout=timeout_arg)
            return "", "", stdout_path, stderr_path

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_arg)
        return result.stdout, result.stderr, None, None
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded {timeout}s")
        return None, None, stdout_path, stderr_path


def _resolve_riscv_tool(tool_name):
    riscv = os.environ.get("RISCV", "/root/flexi/chipyard/.conda-env/riscv-tools")
    return pathlib.Path(riscv) / "bin" / tool_name


def decode_trace_with_spike_dasm(trace_path, decoded_path):
    """Decode verbose trace with spike-dasm for readability."""
    spike_dasm = _resolve_riscv_tool("spike-dasm")
    if not spike_dasm.exists():
        print(f"[WARN] spike-dasm not found: {spike_dasm}")
        return False
    with open(trace_path, "r") as fin, open(decoded_path, "w") as fout:
        result = subprocess.run([str(spike_dasm)], stdin=fin, stdout=fout, text=True)
    if result.returncode != 0:
        print("[WARN] spike-dasm decoding failed")
        return False
    print(f"[OK] Decoded trace: {decoded_path}")
    return True


def _load_function_ranges(binary_path):
    """Load text symbol ranges from ELF for PC->function lookup."""
    nm = _resolve_riscv_tool("riscv64-unknown-elf-nm")
    if not nm.exists():
        raise RuntimeError(f"nm not found: {nm}")
    result = subprocess.run(
        [str(nm), "-n", str(binary_path)], capture_output=True, text=True, check=True
    )
    symbols = []
    for line in result.stdout.splitlines():
        m = re.match(r"^([0-9a-fA-F]+)\s+([tT])\s+(\S+)$", line.strip())
        if not m:
            continue
        addr = int(m.group(1), 16)
        name = m.group(3)
        symbols.append((addr, name))
    if not symbols:
        raise RuntimeError("No text symbols found in binary")
    ranges = []
    for idx, (addr, name) in enumerate(symbols):
        end = symbols[idx + 1][0] if idx + 1 < len(symbols) else addr + 1
        if end > addr:
            ranges.append((addr, end, name))
    return ranges


def profile_kernels_from_trace(trace_path, binary_path, report_txt_path, report_csv_path, topk=40):
    """Attribute verbose trace cycles to kernels based on PC ranges."""
    ranges = _load_function_ranges(binary_path)
    starts = [r[0] for r in ranges]

    def find_func(pc):
        idx = bisect.bisect_right(starts, pc) - 1
        if idx < 0:
            return None
        start, end, name = ranges[idx]
        if start <= pc < end:
            return name
        return None

    cycle_re = re.compile(r"^C\d+:\s+(\d+)\s+\[\d+\]\s+pc=\[([0-9a-fA-F]+)\]")
    per_func_cycles = defaultdict(int)
    per_func_samples = defaultdict(int)

    prev_cycle = None
    prev_pc = None
    trace_points = 0
    with open(trace_path, "r") as f:
        for line in f:
            m = cycle_re.match(line)
            if not m:
                continue
            cycle = int(m.group(1))
            pc = int(m.group(2), 16)
            trace_points += 1
            if prev_cycle is not None:
                delta = cycle - prev_cycle
                if delta < 0:
                    delta = 0
                func = find_func(prev_pc)
                if func:
                    per_func_cycles[func] += delta
                    per_func_samples[func] += 1
            prev_cycle = cycle
            prev_pc = pc

    kernel_items = []
    other_cycles = 0
    for name, cycles in per_func_cycles.items():
        if name.startswith("tvmgen_default_fused_"):
            kernel_items.append((name, cycles, per_func_samples[name]))
        else:
            other_cycles += cycles
    kernel_items.sort(key=lambda x: x[1], reverse=True)
    total_kernel_cycles = sum(x[1] for x in kernel_items)
    total_cycles = total_kernel_cycles + other_cycles

    with open(report_txt_path, "w") as f:
        f.write("Kernel cycle profile from Verilator +verbose trace\n")
        f.write(f"Trace points: {trace_points}\n")
        f.write(f"Total attributed cycles: {total_cycles}\n")
        f.write(f"Kernel-attributed cycles: {total_kernel_cycles}\n")
        f.write(f"Non-kernel cycles: {other_cycles}\n")
        f.write("\nTop kernels by cycles:\n")
        f.write("rank,cycles,percent_of_kernels,samples,kernel\n")
        for rank, (name, cycles, samples) in enumerate(kernel_items[:topk], start=1):
            pct = (100.0 * cycles / total_kernel_cycles) if total_kernel_cycles else 0.0
            f.write(f"{rank},{cycles},{pct:.3f},{samples},{name}\n")

    with open(report_csv_path, "w") as f:
        f.write("kernel,cycles,samples\n")
        for name, cycles, samples in kernel_items:
            f.write(f"{name},{cycles},{samples}\n")

    return {
        "trace_points": trace_points,
        "total_cycles": total_cycles,
        "total_kernel_cycles": total_kernel_cycles,
        "other_cycles": other_cycles,
        "top": kernel_items[:topk],
    }


def main():
    parser = argparse.ArgumentParser(description="Run I-ViT on real image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint", type=str, default="/root/checkpoint_last.pth.tar"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="auto",
        choices=["auto", "deit_tiny_patch16_224", "swin_tiny_patch4_window7_224"],
        help="Model name (auto detects from checkpoint keys)",
    )
    parser.add_argument("--output-dir", type=str, default="ivit_real_image_project")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Host-side timeout in seconds (0 to disable)",
    )
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
        default=0,
        help="Verilator +max-cycles limit (0 to disable simulator-side timeout)",
    )
    parser.add_argument(
        "--no-dramsim",
        action="store_true",
        help="Disable +dramsim when running Verilator",
    )
    parser.add_argument(
        "--debug-unit",
        type=str,
        default=None,
        help=(
            "Relay debug cut point (e.g. post_block0, block_0_pre_softmax, "
            "post_stage0_block0, post_stem, pre_head, head_int)"
        ),
    )
    parser.add_argument(
        "--verilator-verbose",
        action="store_true",
        help="Pass +verbose to Verilator and save raw trace logs",
    )
    parser.add_argument(
        "--decode-dasm",
        action="store_true",
        help="Decode verbose trace via spike-dasm into a readable .dasm file",
    )
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Profile per-kernel cycles from Verilator verbose trace",
    )
    parser.add_argument(
        "--profile-topk",
        type=int,
        default=40,
        help="How many kernels to show in text report",
    )
    parser.add_argument(
        "--verilator-log-tail-lines",
        type=int,
        default=20000,
        help=(
            "When saving Verilator logs, keep only last N lines (default: 20000). "
            "Set 0 to save full logs."
        ),
    )
    parser.add_argument(
        "--usmp-alg",
        type=str,
        default=None,
        choices=["hill_climb", "greedy_by_size", "greedy_by_conflicts", "none"],
        help="USMP algorithm (default: auto by model)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Relay build opt level (default: auto by model)",
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
    if args.debug_unit:
        print(f"Debug unit: {args.debug_unit}")

    gemmini.Environment.init_overwrite(
        dim=16,
        acc_rows=1024,
        bank_rows=4096,
        inp_dtype="int8",
        wgt_dtype="int8",
        acc_dtype="int32",
    )

    print("\n[1/6] Loading checkpoint...")
    checkpoint_path = pathlib.Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return 1

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    requested_model_name = None if args.model_name == "auto" else args.model_name
    model_name = convert_model.resolve_model_name(ckpt, requested_model_name)
    if model_name not in MODEL_SPECS:
        print(f"[ERROR] Unsupported model for this runner: {model_name}")
        return 1

    depth = MODEL_SPECS[model_name]["depth"]
    convert_model.load_qconfig(ckpt, depth=depth, model_name=model_name)
    print(f"       Checkpoint: {checkpoint_path}")
    print(f"       Model: {model_name}")

    input_scale = ckpt["qact_input.act_scaling_factor"].item()
    print(f"       Input quantization scale: {input_scale}")

    print("\n[2/6] Preprocessing image...")
    input_data = preprocess_image(image_path, input_scale)
    print(f"       Input shape: {input_data.shape}, dtype: {input_data.dtype}")
    print(f"       Value range: [{input_data.min()}, {input_data.max()}]")

    print("\n[3/6] Building TVM model...")
    t_build_start = time.time()
    mod, _ = get_workload(model_name, batch_size=1, debug_unit=args.debug_unit)
    params = convert_model.build_param_dict(ckpt, depth=depth, model_name=model_name)

    tvm_params = {k: tvm.nd.array(v) for k, v in params.items()}

    print("       Applying Gemmini preprocess pass...")
    mod = preprocess_for_gemmini(mod, model_name)
    print("       Preprocess pass done")

    RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": False})
    TARGET = tvm.target.target.Target({"kind": "c", "device": "gemmini"})
    EXECUTOR = tvm.relay.backend.Executor(
        "aot", options={"interface-api": "c", "unpacked-api": 1}
    )

    usmp_alg = args.usmp_alg
    if usmp_alg is None:
        usmp_alg = "greedy_by_size" if model_name.startswith("swin_") else "hill_climb"
    if usmp_alg == "none":
        usmp_alg = ""
    opt_level = args.opt_level
    if opt_level is None:
        opt_level = 2 if model_name.startswith("swin_") else 3
    disabled_passes = ["AlterOpLayout"]
    print(
        f"       relay.build 시작 (usmp_alg={usmp_alg}, opt_level={opt_level}) "
        f"- Swin은 수십 분 걸릴 수 있음"
    )
    if disabled_passes:
        print(f"       disabled_pass={disabled_passes}")

    with gemmini.build_config(
        usmp_alg=usmp_alg, opt_level=opt_level, disabled_pass=disabled_passes
    ):
        module = relay.build(
            mod, executor=EXECUTOR, runtime=RUNTIME, target=TARGET, params=tvm_params
        )
    t_build_end = time.time()
    print(f"       relay.build 완료 ({t_build_end - t_build_start:.1f}s)")

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
    classification_output = args.debug_unit is None
    create_real_image_harness(
        output_dir,
        model_name,
        MODEL_SPECS[model_name]["embed_dim"],
        input_data,
        classification_output=classification_output,
        debug_unit=args.debug_unit,
    )

    print("\n[5/6] Compiling for Spike...")
    binary = compile_for_spike(output_dir, "ivit_real")
    if binary is None:
        return 1

    if args.simulator == "spike":
        print("\n[6/6] Running on Spike...")
        stdout, stderr = run_spike(binary, timeout=args.timeout)
        ver_stdout_path = None
        ver_stderr_path = None
    else:
        verilator_verbose = args.verilator_verbose or args.profile_kernels
        if args.profile_kernels and not args.verilator_verbose:
            print("[Info] --profile-kernels requested; enabling Verilator +verbose.")
        log_tail_lines = args.verilator_log_tail_lines
        if (args.decode_dasm or args.profile_kernels) and log_tail_lines > 0:
            print(
                "[Info] --decode-dasm/--profile-kernels needs full trace; "
                "disabling log tail truncation for this run."
            )
            log_tail_lines = 0
        print("\n[6/6] Running on Verilator...")
        stdout, stderr, ver_stdout_path, ver_stderr_path = run_verilator(
            binary,
            timeout=args.timeout,
            chipyard_dir=args.chipyard_dir,
            verilator_config=args.verilator_config,
            max_cycles=args.max_cycles,
            dramsim=not args.no_dramsim,
            verbose=verilator_verbose,
            log_dir=output_dir if verilator_verbose else None,
            log_tail_lines=log_tail_lines,
        )

    if stdout is None:
        return 1

    print("\n" + "=" * 60)
    print(f"Simulation Output ({args.simulator}):")
    print("=" * 60)
    if stdout:
        print(stdout)
    elif args.simulator == "verilator" and ver_stdout_path is not None:
        print(f"[Info] Verilator stdout saved to: {ver_stdout_path}")
        with open(ver_stdout_path, "r") as f:
            snippet = f.read(4000)
        if snippet:
            print(snippet)

    if args.simulator == "verilator" and ver_stderr_path is not None:
        print(f"[Info] Verilator stderr trace saved to: {ver_stderr_path}")
        if args.decode_dasm:
            decoded_path = pathlib.Path(output_dir) / "verilator_stderr.dasm"
            decode_trace_with_spike_dasm(ver_stderr_path, decoded_path)
        if args.profile_kernels:
            report_txt = pathlib.Path(output_dir) / "kernel_profile.txt"
            report_csv = pathlib.Path(output_dir) / "kernel_profile.csv"
            profile = profile_kernels_from_trace(
                ver_stderr_path,
                binary,
                report_txt,
                report_csv,
                topk=args.profile_topk,
            )
            print("\nKernel Profile Summary (top kernels):")
            for idx, (name, cycles, samples) in enumerate(profile["top"][:10], start=1):
                pct = (
                    100.0 * cycles / profile["total_kernel_cycles"]
                    if profile["total_kernel_cycles"]
                    else 0.0
                )
                print(f"  {idx:2d}. {name}: {cycles} cycles ({pct:.2f}%), samples={samples}")
            print(f"[Info] Kernel profile report: {report_txt}")
            print(f"[Info] Kernel profile CSV: {report_csv}")

    if classification_output:
        classes = load_imagenet_classes()

        print("\n" + "=" * 60)
        print("Class Labels:")
        print("=" * 60)

        output_text = stdout
        if (not output_text) and args.simulator == "verilator" and ver_stdout_path is not None:
            with open(ver_stdout_path, "r") as f:
                output_text = f.read()

        for match in re.finditer(r"Class (\d+)", output_text):
            class_id = int(match.group(1))
            if class_id < len(classes):
                print(f"  Class {class_id}: {classes[class_id]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
