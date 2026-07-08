/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef MATMUL_INFO_COMMON_H
#define MATMUL_INFO_COMMON_H

namespace AsdOps {
struct MatMulInfoCommon {
    uint32_t batchSize{0};
    uint32_t m{0};                  // 实际输入的 m
    uint32_t n{0};                  // 实际输入的 n
    uint32_t k{0};                  // 实际输入的 k
    bool transA{0};                 // false: 0, true: 1
    bool transB{0};                 // false: 0, true: 1
    bool biasFlag{0};               // false: 0, true: 1
    bool isInt8{0};                 // 是否 int8融合
    float inDtype{2};
    float outDtype{4};
    uint32_t formatSema{0};         // "FRACTAL_NZ": 0, "ND": 1
};
}
#endif