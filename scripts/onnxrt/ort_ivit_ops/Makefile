# Makefile for ivit custom ops — RISC-V Linux cross-compilation
#
# Usage (from scripts/onnxrt/ort_ivit_ops/):
#   make ORT_RISCV_DIR=<path>
#
# ORT_RISCV_DIR defaults to the standard onnxruntime-riscv location
# inside the flexi repository.

REPO_ROOT ?= $(abspath ../../..)
ORT_RISCV_DIR ?= $(REPO_ROOT)/tvm-gemmini/3rdparty/gemmini/software/onnxruntime-riscv

CC ?= riscv64-unknown-linux-gnu-gcc
AR ?= riscv64-unknown-linux-gnu-ar

CFLAGS = \
	-O2 \
	-march=rv64imafdc \
	-mabi=lp64d \
	-static \
	-I$(ORT_RISCV_DIR)/include/onnxruntime/core/session \
	-I$(ORT_RISCV_DIR)/include/onnxruntime

SRCS = ivit_ops.c
OBJS = ivit_ops.o
LIB  = libivit_ops.a

all: $(LIB)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

clean:
	rm -f $(OBJS) $(LIB)

.PHONY: all clean
