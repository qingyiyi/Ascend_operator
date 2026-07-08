/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "matmul_nz_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
MatMulDescParams ExtractMatMulParams(const OpParam::MatMul &opParam, const LaunchParam &launchParam) {
    MatMulDescParams params = {
        launchParam.GetInTensor(0).desc, launchParam.GetInTensor(1).desc, launchParam.GetOutTensor(0).desc,
        opParam.oriShape[0], opParam.oriShape[1], opParam.oriShape[2]
    };
    return params;
}

Status MatMulNzTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                      const BinHandle &binHandle)
{
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    MKI_CHECK(opParam.oriShape.size() == 3, "size of oriShape is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MatMulDescParams params = ExtractMatMulParams(opParam, launchParam);
    SVector<int64_t> oriShapeA = {params.m, params.k};
    if (opParam.transposeA) {
        oriShapeA = {params.k, params.m};
    }
    SVector<int64_t> oriShapeB = {params.k, params.n};
    if (opParam.transposeB) {
        oriShapeB = {params.n, params.k};
    }
    SVector<int64_t> oriShapeOut = {params.m, params.n};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetName("MatMulV2")
                      .SetKernelName(kernelName)
                      .AddInput(params.tensorDescA.dtype, params.tensorDescA.format, oriShapeA)
                      .AddInput(params.tensorDescB.dtype, params.tensorDescB.format, oriShapeB)
                      .AddOutput(params.tensorDescOut.dtype, params.tensorDescOut.format, oriShapeOut)
                      .AddAttrBool(opParam.transposeA)
                      .AddAttrBool(opParam.transposeB)
                      .AddAttrInt64(0); // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status BatchMatMulNzTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                           const BinHandle &binHandle)
{
    auto opParam = AnyCast<OpParam::MatMul>(launchParam.GetParam());
    MKI_CHECK(opParam.oriShape.size() == 3, "size of oriShape is invalid",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    MatMulDescParams params = ExtractMatMulParams(opParam, launchParam);
    SVector<int64_t> oriShapeA = {params.tensorDescA.dims[0], params.m, params.k};
    if (opParam.transposeA) {
        oriShapeA = {params.tensorDescA.dims[0], params.k, params.m};
    }
    SVector<int64_t> oriShapeB = {params.tensorDescB.dims[0], params.k, params.n};
    if (opParam.transposeB) {
        oriShapeB = {params.tensorDescB.dims[0], params.n, params.k};
    }
    SVector<int64_t> oriShapeOut = {params.m, params.n};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetName("BatchMatMulV2")
                      .SetKernelName(kernelName)
                      .AddInput(params.tensorDescA.dtype, params.tensorDescA.format, oriShapeA)
                      .AddInput(params.tensorDescB.dtype, params.tensorDescB.format, oriShapeB)
                      .AddOutput(params.tensorDescOut.dtype, params.tensorDescOut.format, oriShapeOut)
                      .AddAttrBool(opParam.transposeA)
                      .AddAttrBool(opParam.transposeB)
                      .AddAttrInt64(0); // 0x40 is high precision
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps