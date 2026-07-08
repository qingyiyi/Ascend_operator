/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_RMSNORMFORWARD_TILING_DATA_H
#define ASCEND_OPS_RMSNORMFORWARD_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
constexpr uint32_t UB_SIZE = 192 * 1024;

struct RmsNormForwardTilingData {
    uint32_t numRow;
    uint32_t numCol;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    float epsilon;
    float avgFactor;
};

}  // namespace AsdOps

#endif