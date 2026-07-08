/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_FILL_H
#define ASDOPS_PARAMS_FILL_H

#include <string>
#include <sstream>
#include <mki/types.h>
#include <mki/utils/SVector/SVector.h>

namespace AsdOps {
namespace OpParam {
struct Fill {
    bool withMask = false;
    Mki::SVector<float> value;   // for fill/maskedfill
    Mki::SVector<int64_t> outDim; // for fill
    Mki::TensorDType outTensorType = Mki::TENSOR_DTYPE_FLOAT16; // for fill

    bool operator==(const Fill &other) const
    {
        return this->withMask == other.withMask && this->value == other.value &&
            this->outDim == other.outDim && this->outTensorType == other.outTensorType;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif // ASDOPS_PARAMS_FILL_H