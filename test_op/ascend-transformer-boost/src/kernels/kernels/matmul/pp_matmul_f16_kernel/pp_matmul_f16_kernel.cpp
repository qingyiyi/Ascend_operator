/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pp_matmul_f16_kernel_base.h"

namespace AsdOps {
class PpMatMulF16Kernel : public PpMatMulF16KernelBase {
public:
    explicit PpMatMulF16Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : PpMatMulF16KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return CheckAsdOpsND(launchParam, 2); // 输入参数数量为2
    }
};
REG_KERNEL_BASE(PpMatMulF16Kernel);
} // namespace AsdOps