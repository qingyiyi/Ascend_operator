/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATBOPS_PARAMS_MLA_PRE_H
#define ATBOPS_PARAMS_MLA_PRE_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AtbOps {
namespace OpParam {
struct MlaPreprocess {
    enum class QuantMode : int32_t {
        PER_TENSOR_ASYMM_QUANT = 0,
        PER_TOKEN_SYMM_QUANT,
        PER_TOKEN_ASYMM_QUANT,
        NO_QUANT,
    };
    uint32_t N = 128;
    uint32_t headNum = 0;
    uint32_t cacheMode = 0;
    QuantMode quantMode = QuantMode::PER_TENSOR_ASYMM_QUANT;
    bool operator==(const MlaPreprocess &other) const
    {
        return N == other.N && headNum == other.headNum && cacheMode == other.cacheMode && quantMode == other.quantMode;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_MLA_PRE_H