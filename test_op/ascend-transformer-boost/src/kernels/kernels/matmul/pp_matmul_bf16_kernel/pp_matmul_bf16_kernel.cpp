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

namespace AsdOps {
using namespace Mki;
class PpMatMulBf16Kernel : public KernelBase {
public:
    explicit PpMatMulBf16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B,
            "platformtype is invalid", return false);
        MKI_CHECK(launchParam.GetInTensorCount() == 2, "in tensor num invalid", return false);
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "out tensor num invalid", return false);

        const auto &descA = launchParam.GetInTensor(0).desc;
        const auto &descB = launchParam.GetInTensor(1).desc;

        MKI_CHECK(descA.format == TENSOR_FORMAT_ND, "in tensor0 format invalid", return false);
        MKI_CHECK(descA.dtype == TENSOR_DTYPE_BF16, "in tensor0 dtype invalid", return false);

        MKI_CHECK(descB.format == TENSOR_FORMAT_ND, "in tensor1 format invalid", return false);
        MKI_CHECK(descB.dtype == TENSOR_DTYPE_BF16, "in tensor1 dtype invalid", return false);

        return true;
    }

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override
    {
        (void)launchParam;
        constexpr uint32_t CONST_256 = 256;
        return RoundUp(sizeof(PpMatmulTilingData), CONST_256);
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return PpMatmulTiling(launchParam, kernelInfo_);
    }
};
REG_KERNEL_BASE(PpMatMulBf16Kernel);
} // namespace AsdOps