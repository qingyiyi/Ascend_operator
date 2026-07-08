/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PP_MATMUL_ND_TILING_H
#define PP_MATMUL_ND_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/status/status.h>
#include "matmul_info_common.h"

namespace AsdOps {
using namespace Mki;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t L0C_SIZE = 128 * 1024;

template <typename T = uint32_t> inline T Max(const T a, const T b) { return a > b ? a : b; }

template <typename T = uint32_t> inline T Min(const T a, const T b) { return a < b ? a : b; }

struct MatMulInfoNd : public MatMulInfoCommon {
};

struct OpShapeNd {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
};

struct PpTilingDataNd {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t mLoop{1};
    uint32_t kLoop{1};
    uint32_t nLoop{1};
    uint32_t coreLoop{1};
    uint32_t tilingKey{0};

    void SetTilingKey(bool transB);
};

Status PpTilingNd(const LaunchParam &launchParam, KernelInfo &kernelInfo);
}
#endif // PP_MATMUL_ND_TILING_H
