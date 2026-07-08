/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/utils_internal.h"
#include "self_attention_runner_utils.h"

namespace atb {
bool IsParamEqual(const infer::SelfAttentionParam &left, const infer::SelfAttentionParam &right)
{
    return left.headNum == right.headNum && left.kvHeadNum == right.kvHeadNum &&
           UtilsInternal::IsFloatEqual(left.qScale, right.qScale) &&
           UtilsInternal::IsFloatEqual(left.qkScale, right.qkScale) &&
           left.batchRunStatusEnable == right.batchRunStatusEnable && left.isTriuMask == right.isTriuMask &&
           left.calcType == right.calcType && left.kernelType == right.kernelType &&
           left.clampType == right.clampType && UtilsInternal::IsFloatEqual(left.clampMin, right.clampMin) &&
           UtilsInternal::IsFloatEqual(left.maskType, right.maskType) && left.kvcacheCfg == right.kvcacheCfg &&
           left.scaleType == right.scaleType && left.mlaVHeadSize == right.mlaVHeadSize &&
           left.windowSize == right.windowSize && left.cacheType == right.cacheType;
}

bool NeedElewiseMulsQScale(const infer::SelfAttentionParam &param) {
    return !UtilsInternal::IsFloatEqual(param.qScale, 1);
}
} // namespace atb
