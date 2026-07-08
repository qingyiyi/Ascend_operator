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
// 实现 GetTilingSize
uint64_t PpMatMulF16KernelBase::GetTilingSize(const LaunchParam &launchParam) const {
    (void)launchParam;
    constexpr uint32_t CONST_256 = 256;
    return Round<CONST_256>(sizeof(PpMatmulTilingData));
}

// 实现 InitImpl
Status PpMatMulF16KernelBase::InitImpl(const LaunchParam &launchParam) {
    return PpMatmulTiling(launchParam, kernelInfo_);
}
}