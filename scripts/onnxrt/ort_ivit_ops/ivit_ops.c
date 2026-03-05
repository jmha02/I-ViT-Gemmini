/*
 * ivit_ops.c — ORT custom ops for I-ViT integer-only operations.
 *
 * Implements three custom ops in domain "ivit":
 *
 *   ivit.Shiftmax   [int8 x] → [int8]    attr: x0 (int64)
 *   ivit.ShiftGELU  [int8 x] → [int32]   attr: x0 (int64)
 *   ivit.QLayernorm [int8 x, int32 bias] → [int32]
 *
 * All algorithms match the I-ViT integer reference implementations.
 *
 * Build (RISC-V Linux):
 *   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64imafdc -mabi=lp64d \
 *       -I<ort_riscv>/include/onnxruntime/core/session \
 *       -c ivit_ops.c -o ivit_ops.o
 *   ar rcs libivit_ops.a ivit_ops.o
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

/* ── ORT API global ───────────────────────────────────────────────────────── */
static const OrtApi *g_ort = NULL;

/* ── Integer shift-exp (shared by Shiftmax and ShiftGELU) ────────────────── */
/*
 * shift_exp(data, x0, n):
 *   data = data + (data >> 1) - (data >> 4)
 *   data = max(data, n * x0)
 *   q    = data / x0   (truncating toward zero)
 *   r    = data - q * x0
 *   e    = (r >> 1) - x0
 *   return e << (n - q)         [n - q is always >= 0]
 */
static int32_t shift_exp(int32_t data, int32_t x0, int32_t n)
{
    /* polynomial adjustment */
    data = data + (data >> 1) - (data >> 4);

    int32_t floor_val = n * x0;   /* x0 < 0, so floor_val < 0 */
    if (data < floor_val) data = floor_val;

    /* truncated integer division: C99 truncates toward zero */
    int32_t q = data / x0;
    int32_t r = data - q * x0;

    int32_t e = (r >> 1) - x0;
    int32_t shift = n - q;
    if (shift < 0) shift = 0;   /* safety clamp */
    if (shift > 31) return 0;   /* underflow */
    return e << shift;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.Shiftmax  —  integer softmax
 *  Input : int8  [*, N]
 *  Output: int8  [*, N]
 *  Attr  : x0 (int64) = floor(-1/input_scale)
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const OrtApi *ort;
    int32_t       x0;  /* x0 for shift_exp */
} ShiftmaxKernel;

static void *ORT_API_CALL Shiftmax_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    ShiftmaxKernel *k = (ShiftmaxKernel *)malloc(sizeof(ShiftmaxKernel));
    k->ort = api;
    int64_t x0_attr = 0;
    api->KernelInfoGetAttribute_int64(info, "x0", &x0_attr);
    k->x0 = (int32_t)x0_attr;
    return k;
}

static void ORT_API_CALL Shiftmax_Destroy(void *op_kernel)
{
    free(op_kernel);
}

static void ORT_API_CALL Shiftmax_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    ShiftmaxKernel *k = (ShiftmaxKernel *)op_kernel;
    const OrtApi *ort = k->ort;
    const int32_t x0 = k->x0;
    const int32_t n  = 15;  /* Shiftmax uses n=15 */

    const OrtValue *in_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &in_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(in_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_val, (void **)&x);

    /* last dimension = softmax axis */
    int64_t last = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int8_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * last;
        int8_t       *row_out = y + b * last;

        /* find max */
        int32_t mx = (int32_t)row_in[0];
        for (int64_t i = 1; i < last; i++)
            if ((int32_t)row_in[i] > mx) mx = (int32_t)row_in[i];

        /* compute exp_int for each element + sum */
        int64_t sum = 0;
        int32_t exp_buf[1024];  /* max seq dimension we expect */
        int64_t cap = last < 1024 ? last : 1024;
        for (int64_t i = 0; i < cap; i++) {
            int32_t c = (int32_t)row_in[i] - mx;
            exp_buf[i] = shift_exp(c, x0, n);
            sum += (int64_t)exp_buf[i];
        }
        if (sum == 0) sum = 1;

        int64_t factor = (int64_t)0x7FFFFFFF;  /* 2^31 - 1 */
        int64_t scale  = factor / sum;

        for (int64_t i = 0; i < cap; i++) {
            int64_t v = (scale * (int64_t)exp_buf[i]) >> 24;
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            row_out[i] = (int8_t)v;
        }
        /* handle overflow beyond buffer */
        for (int64_t i = cap; i < last; i++) row_out[i] = 0;
    }
}

static const char *ORT_API_CALL Shiftmax_GetName(const OrtCustomOp *op)
{ (void)op; return "Shiftmax"; }

static const char *ORT_API_CALL Shiftmax_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static ONNXTensorElementDataType ORT_API_CALL Shiftmax_GetInputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL Shiftmax_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static ONNXTensorElementDataType ORT_API_CALL Shiftmax_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL Shiftmax_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
Shiftmax_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
Shiftmax_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_shiftmax_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = Shiftmax_CreateKernel,
    .GetName                    = Shiftmax_GetName,
    .GetExecutionProviderType   = Shiftmax_GetExecutionProviderType,
    .GetInputType               = Shiftmax_GetInputType,
    .GetInputTypeCount          = Shiftmax_GetInputTypeCount,
    .GetOutputType              = Shiftmax_GetOutputType,
    .GetOutputTypeCount         = Shiftmax_GetOutputTypeCount,
    .KernelCompute              = Shiftmax_Compute,
    .KernelDestroy              = Shiftmax_Destroy,
    .GetInputCharacteristic     = Shiftmax_GetInputCharacteristic,
    .GetOutputCharacteristic    = Shiftmax_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.ShiftGELU  —  integer GELU approximation
 *  Input : int8  [*, C]
 *  Output: int32 [*, C]   (natural scale = input_scale / 128)
 *  Attr  : x0 (int64) = floor(-1 / (scaling_factor * 1.702))
 * ══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const OrtApi *ort;
    int32_t       x0;
} ShiftGELUKernel;

static void *ORT_API_CALL ShiftGELU_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op;
    ShiftGELUKernel *k = (ShiftGELUKernel *)malloc(sizeof(ShiftGELUKernel));
    k->ort = api;
    int64_t x0_attr = 0;
    api->KernelInfoGetAttribute_int64(info, "x0", &x0_attr);
    k->x0 = (int32_t)x0_attr;
    return k;
}

static void ORT_API_CALL ShiftGELU_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL ShiftGELU_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    ShiftGELUKernel *k = (ShiftGELUKernel *)op_kernel;
    const OrtApi *ort = k->ort;
    const int32_t x0 = k->x0;
    const int32_t n  = 23;  /* ShiftGELU uses n=23 */

    const OrtValue *in_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &in_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(in_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)in_val, (void **)&x);

    int64_t last  = dims[ndim - 1];
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int32_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * last;
        int32_t      *row_out = y + b * last;

        /* find max for numerical stability */
        int32_t mx = (int32_t)row_in[0];
        for (int64_t i = 1; i < last; i++)
            if ((int32_t)row_in[i] > mx) mx = (int32_t)row_in[i];

        /* exp of (x[i] - mx) for sigmoid numerator */
        int64_t exp_buf[2048];
        int64_t cap = last < 2048 ? last : 2048;
        for (int64_t i = 0; i < cap; i++) {
            int32_t c = (int32_t)row_in[i] - mx;
            exp_buf[i] = (int64_t)shift_exp(c, x0, n);
        }

        /* exp of (-mx) for sigmoid denominator term */
        int64_t exp_max_neg = (int64_t)shift_exp(-mx, x0, n);

        int64_t factor = (int64_t)0x7FFFFFFF;

        for (int64_t i = 0; i < cap; i++) {
            int64_t exp_sum = exp_buf[i] + exp_max_neg;
            if (exp_sum == 0) exp_sum = 1;
            int64_t sig = (factor / exp_sum * exp_buf[i]) >> 24;
            /* sig ∈ [0, 127], represents sigmoid * 128 approximately */
            row_out[i] = (int32_t)((int64_t)(int32_t)row_in[i] * sig);
        }
        for (int64_t i = cap; i < last; i++) row_out[i] = 0;
    }
}

static const char *ORT_API_CALL ShiftGELU_GetName(const OrtCustomOp *op)
{ (void)op; return "ShiftGELU"; }

static const char *ORT_API_CALL ShiftGELU_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static ONNXTensorElementDataType ORT_API_CALL ShiftGELU_GetInputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; }

static size_t ORT_API_CALL ShiftGELU_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static ONNXTensorElementDataType ORT_API_CALL ShiftGELU_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL ShiftGELU_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
ShiftGELU_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
ShiftGELU_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_shiftgelu_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = ShiftGELU_CreateKernel,
    .GetName                    = ShiftGELU_GetName,
    .GetExecutionProviderType   = ShiftGELU_GetExecutionProviderType,
    .GetInputType               = ShiftGELU_GetInputType,
    .GetInputTypeCount          = ShiftGELU_GetInputTypeCount,
    .GetOutputType              = ShiftGELU_GetOutputType,
    .GetOutputTypeCount         = ShiftGELU_GetOutputTypeCount,
    .KernelCompute              = ShiftGELU_Compute,
    .KernelDestroy              = ShiftGELU_Destroy,
    .GetInputCharacteristic     = ShiftGELU_GetInputCharacteristic,
    .GetOutputCharacteristic    = ShiftGELU_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  ivit.QLayernorm  —  integer-only Layer Normalization
 *  Input 0: int8  [*, C]   activations
 *  Input 1: int32 [C]      bias_integer (per-channel)
 *  Output : int32 [*, C]   (scale = norm_scaling_factor)
 *
 *  Algorithm (matches I-ViT IntLayerNorm integer core):
 *    x32 = cast(x, int32)
 *    mean = round(sum(x32) / C)
 *    xc[c] = x32[c] - mean
 *    var = sum(xc * xc)             // scalar per token
 *    std = sqrt_newton(var, 10)     // integer Newton-Raphson
 *    scale = (2^31 - 1) / std
 *    out[c] = (scale * xc[c]) >> 1 + bias[c]
 * ══════════════════════════════════════════════════════════════════════════ */

/* Integer square root via Newton-Raphson (10 iterations, uint32) */
static uint32_t isqrt32(uint64_t val)
{
    if (val == 0) return 1;  /* avoid divide-by-zero */
    uint32_t s = 1u << 16;
    for (int i = 0; i < 10; i++) {
        uint32_t ns = (s + (uint32_t)(val / (uint64_t)s)) / 2u;
        s = ns;
    }
    return s == 0 ? 1 : s;
}

typedef struct {
    const OrtApi *ort;
} QLayernormKernel;

static void *ORT_API_CALL QLayernorm_CreateKernel(
        const OrtCustomOp *op, const OrtApi *api, const OrtKernelInfo *info)
{
    (void)op; (void)info;
    QLayernormKernel *k = (QLayernormKernel *)malloc(sizeof(QLayernormKernel));
    k->ort = api;
    return k;
}

static void ORT_API_CALL QLayernorm_Destroy(void *op_kernel) { free(op_kernel); }

static void ORT_API_CALL QLayernorm_Compute(void *op_kernel, OrtKernelContext *ctx)
{
    QLayernormKernel *k = (QLayernormKernel *)op_kernel;
    const OrtApi *ort = k->ort;

    /* input 0: int8 activations */
    const OrtValue *x_val = NULL;
    ort->KernelContext_GetInput(ctx, 0, &x_val);

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    ort->GetTensorTypeAndShape(x_val, &shape_info);
    size_t ndim = 0;
    ort->GetDimensionsCount(shape_info, &ndim);
    int64_t dims[8] = {0};
    ort->GetDimensions(shape_info, dims, ndim);
    ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int8_t *x = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)x_val, (void **)&x);

    /* input 1: int32 bias */
    const OrtValue *bias_val = NULL;
    ort->KernelContext_GetInput(ctx, 1, &bias_val);
    const int32_t *bias = NULL;
    ort->GetTensorMutableData((OrtValue *)(uintptr_t)bias_val, (void **)&bias);

    int64_t C     = dims[ndim - 1];  /* feature dimension */
    int64_t outer = 1;
    for (size_t i = 0; i < ndim - 1; i++) outer *= dims[i];

    OrtValue *out_val = NULL;
    ort->KernelContext_GetOutput(ctx, 0, dims, ndim, &out_val);
    int32_t *y = NULL;
    ort->GetTensorMutableData(out_val, (void **)&y);

    const int64_t INTMAX = (int64_t)0x7FFFFFFF;

    for (int64_t b = 0; b < outer; b++) {
        const int8_t *row_in  = x + b * C;
        int32_t      *row_out = y + b * C;

        /* integer mean (rounded) */
        int64_t sum = 0;
        for (int64_t c = 0; c < C; c++) {
            sum += (int64_t)(int32_t)row_in[c];
        }
        int32_t mean_int;
        if (sum >= 0) {
            mean_int = (int32_t)((sum + (C / 2)) / C);
        } else {
            mean_int = (int32_t)((sum - (C / 2)) / C);
        }

        /* sum of squares for centered variance */
        uint64_t var = 0;
        for (int64_t c = 0; c < C; c++) {
            int32_t xi = (int32_t)row_in[c] - mean_int;
            var += (uint64_t)(xi * xi);
        }

        /* integer sqrt via Newton-Raphson */
        uint32_t std_int = isqrt32(var);

        /* normalization scale: (2^31 - 1) / std */
        int64_t norm_scale = INTMAX / (int64_t)std_int;

        /* normalize + add bias */
        for (int64_t c = 0; c < C; c++) {
            int64_t xc = (int64_t)((int32_t)row_in[c] - mean_int);
            int64_t v = (norm_scale * xc) >> 1;
            row_out[c] = (int32_t)v + bias[c];
        }
    }
}

/* QLayernorm: input 0 = int8, input 1 = int32 */
static ONNXTensorElementDataType ORT_API_CALL QLayernorm_GetInputType(
        const OrtCustomOp *op, size_t idx)
{
    (void)op;
    return idx == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

static const char *ORT_API_CALL QLayernorm_GetName(const OrtCustomOp *op)
{ (void)op; return "QLayernorm"; }

static const char *ORT_API_CALL QLayernorm_GetExecutionProviderType(const OrtCustomOp *op)
{ (void)op; return "CPUExecutionProvider"; }

static size_t ORT_API_CALL QLayernorm_GetInputTypeCount(const OrtCustomOp *op)
{ (void)op; return 2; }

static ONNXTensorElementDataType ORT_API_CALL QLayernorm_GetOutputType(
        const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; }

static size_t ORT_API_CALL QLayernorm_GetOutputTypeCount(const OrtCustomOp *op)
{ (void)op; return 1; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernorm_GetInputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOpInputOutputCharacteristic ORT_API_CALL
QLayernorm_GetOutputCharacteristic(const OrtCustomOp *op, size_t idx)
{ (void)op; (void)idx; return INPUT_OUTPUT_REQUIRED; }

static OrtCustomOp g_qlayernorm_op = {
    .version                    = ORT_API_VERSION,
    .CreateKernel               = QLayernorm_CreateKernel,
    .GetName                    = QLayernorm_GetName,
    .GetExecutionProviderType   = QLayernorm_GetExecutionProviderType,
    .GetInputType               = QLayernorm_GetInputType,
    .GetInputTypeCount          = QLayernorm_GetInputTypeCount,
    .GetOutputType              = QLayernorm_GetOutputType,
    .GetOutputTypeCount         = QLayernorm_GetOutputTypeCount,
    .KernelCompute              = QLayernorm_Compute,
    .KernelDestroy              = QLayernorm_Destroy,
    .GetInputCharacteristic     = QLayernorm_GetInputCharacteristic,
    .GetOutputCharacteristic    = QLayernorm_GetOutputCharacteristic,
};

/* ══════════════════════════════════════════════════════════════════════════
 *  RegisterCustomOps — called by ort_test via -DUSE_CUSTOM_OP_LIBRARY
 * ══════════════════════════════════════════════════════════════════════════ */
OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                           const OrtApiBase *api_base)
{
    g_ort = api_base->GetApi(ORT_API_VERSION);
    if (!g_ort) return NULL;

    OrtCustomOpDomain *domain = NULL;
    OrtStatus *status;

    status = g_ort->CreateCustomOpDomain("ivit", &domain);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_shiftmax_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_shiftgelu_op);
    if (status) return status;

    status = g_ort->CustomOpDomain_Add(domain, &g_qlayernorm_op);
    if (status) return status;

    status = g_ort->AddCustomOpDomain(options, domain);
    return status;
}
