/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernels/matmul/common/common_include.h"
#include "kernels/matmul/pp_matmul_f16_kernel/pp_matmul_f16_kernel_base.h"

namespace AsdOps {
using namespace Mki;
class PpMatMulF16OptKernel : public PpMatMulF16KernelBase {
public:
    explicit PpMatMulF16OptKernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulF16KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B, "platform not support",
                     return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "check inTensor count failed", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "check outTensor count failed", return false);

        const auto &descA = launchParam.GetInTensor(0).desc;
        const auto &descB = launchParam.GetInTensor(1).desc;

        MKI_CHECK(descA.format == TENSOR_FORMAT_ND, "tensor format invalid", return false);
        MKI_CHECK(descA.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);

        MKI_CHECK(descB.format == TENSOR_FORMAT_FRACTAL_NZ, "tensor format invalid", return false);
        MKI_CHECK(descB.dims.size() == 4, "tensor dims invalid", return false);
        MKI_CHECK(descB.dtype == TENSOR_DTYPE_FLOAT16, "tensor dtype invalid", return false);

        if (descA.dims.size() == 2) { // 2: The first input is a two-dimensional tensor while batch size is one.
            MKI_CHECK(descB.dims[0] == 1, "tensor dims invalid", return false);
        } else {
            MKI_CHECK(descA.dims.size() == 3, "tensor dims invalid", return false);
            MKI_CHECK(descA.dims[0] == descB.dims[0], "tensor dims invalid", return false);
        }

        return true;
    }
};
REG_KERNEL_BASE(PpMatMulF16OptKernel);
} // namespace AsdOps
