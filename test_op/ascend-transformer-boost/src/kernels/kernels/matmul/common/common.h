/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMMON_H
#define COMMMON_H

#include <mki/launch_param.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "common_tiling.h"

namespace AsdOps {
inline bool CheckInputsStart(const LaunchParam &launchParam, size_t size) {
    MKI_CHECK(launchParam.GetInTensorCount() == size, "inTensor count invalid", return false);
    MKI_CHECK(launchParam.GetOutTensorCount() == 1, "outTensor count invalid", return false);
    MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::MatMul), "check param type failed!", return false);
    return true;
}

inline bool CheckFirstInput(const LaunchParam &launchParam) {
    const auto &inTensor0 = launchParam.GetInTensor(0);
    MKI_CHECK(inTensor0.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(inTensor0.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor0.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor0.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor0.desc.dims.size() == 2 || inTensor0.desc.dims.size() == 3, "tensor dims invalid", return false);
    return true;
}

// size: 输入参数数量
inline bool CheckAsdOpsND(const LaunchParam &launchParam, size_t size)
{
    MKI_CHECK(CheckInputsStart(launchParam, size), "initial check does not pass!", return false);
    MKI_CHECK(CheckFirstInput(launchParam), "There is something wrong with input x", return false);
    const auto &inTensor1 = launchParam.GetInTensor(1);
    const auto &outTensor = launchParam.GetOutTensor(0);
    MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor1.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor1.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor1.desc.dims.size() == 2 || inTensor1.desc.dims.size() == 3, "tensor dims invalid", return false);
    MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || outTensor.desc.dtype == TENSOR_DTYPE_BF16 ||
                  outTensor.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3, "tensor dims invalid", return false);
    return true;
}

inline bool CheckAsdOpsWeightNZ(const LaunchParam &launchParam, size_t size)
{
    MKI_CHECK(CheckInputsStart(launchParam, size), "initial check does not pass!", return false);
    MKI_CHECK(CheckFirstInput(launchParam), "There is something wrong with input x", return false);
    const auto &inTensor1 = launchParam.GetInTensor(1);
    const auto &outTensor = launchParam.GetOutTensor(0);
    MKI_CHECK(inTensor1.desc.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
    MKI_CHECK(inTensor1.desc.dtype == TENSOR_DTYPE_FLOAT16 || inTensor1.desc.dtype == TENSOR_DTYPE_BF16 ||
                  inTensor1.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(inTensor1.desc.dims.size() == 4, "tensor dims invalid", return false);
    MKI_CHECK(outTensor.desc.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
    MKI_CHECK(outTensor.desc.dtype == TENSOR_DTYPE_FLOAT16 || outTensor.desc.dtype == TENSOR_DTYPE_BF16 ||
                  outTensor.desc.dtype == TENSOR_DTYPE_INT8,
              "tensor dtype invalid", return false);
    MKI_CHECK(outTensor.desc.dims.size() == 2 || outTensor.desc.dims.size() == 3, "tensor dims invalid", return false);
    return true;
}

inline bool CheckPlatformAndTwoInputs(const LaunchParam &launchParam) {
    MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "Platform not support.",
                    return false);
    MKI_CHECK(launchParam.GetInTensorCount() == 3, "Check inTensor count failed.", return false);
    MKI_CHECK(launchParam.GetOutTensorCount() == 1, "Check outTensor count failed.", return false);

    const auto &descA = launchParam.GetInTensor(0).desc;
    const auto &descB = launchParam.GetInTensor(1).desc;

    MKI_CHECK(descA.format == TENSOR_FORMAT_ND, "Tensor format is invalid.", return false);
    MKI_CHECK(descA.dtype == TENSOR_DTYPE_BF16 || descA.dtype == TENSOR_DTYPE_FLOAT16,
                    "Tensor dtype is invalid.", return false);
    MKI_CHECK(descB.format == TENSOR_FORMAT_ND, "Tensor format is invalid.", return false);
    MKI_CHECK(descB.dtype == TENSOR_DTYPE_BF16 || descB.dtype == TENSOR_DTYPE_FLOAT16,
                    "Tensor dtype is invalid.", return false);
    MKI_CHECK(descA.dtype == descB.dtype, "Input dtype must be the same.", return false);
    return true;
}

template <typename T>
inline uint64_t GetMatmulTilingSizeCommon(const LaunchParam &launchParam)
{
    (void)launchParam;
    constexpr uint32_t CONST_256 = 256;
    return Round<CONST_256>(sizeof(T));
}
} // namespace AsdOps

#endif
