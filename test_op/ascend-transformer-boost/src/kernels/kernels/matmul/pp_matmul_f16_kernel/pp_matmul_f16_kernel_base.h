/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PP_MAT_MUL_F16_KERNEL_BASE_H // 头文件保护宏
#define PP_MAT_MUL_F16_KERNEL_BASE_H

#include <mki_loader/op_register.h>
#include <mki/utils/log/log.h>
#include <mki/base/kernel_base.h>
#include "kernels/matmul/common/common.h"
#include "kernels/matmul/tiling/pp_matmul_tiling.h"
#include "kernels/matmul/tiling/tiling_data.h"
#include "kernels/matmul/common/common_tiling.h"
#include "asdops/params/params.h"

namespace AsdOps {
class PpMatMulF16KernelBase : public KernelBase {
public:
    explicit PpMatMulF16KernelBase(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle) {}
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override;
    Status InitImpl(const LaunchParam &launchParam) override;
};
}

#endif // PP_MAT_MUL_F16_KERNEL_BASE_H