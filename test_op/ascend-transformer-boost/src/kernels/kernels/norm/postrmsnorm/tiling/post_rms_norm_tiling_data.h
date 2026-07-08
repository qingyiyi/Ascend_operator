/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_RMS_POST_NORM_TILING_DATA_H
#define ASCEND_OPS_RMS_POST_NORM_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
constexpr uint32_t BRCB_BYTE = 256;

struct PostRmsNormTilingData {
    uint32_t numCore{1};
    uint32_t numCol{1};
    uint32_t numRow{1};
    uint32_t avgFactor{1};
    uint32_t epsilon{0};
    uint32_t sliceSize{1};
    uint32_t precisionMode{0};
};
} // namespace AsdOps
#endif