/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reshape_and_cache_ops_runner_310p.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/operation_register.h"

namespace atb {
ReshapeAndCacheOpsRunner310P::ReshapeAndCacheOpsRunner310P(const infer::ReshapeAndCacheParam &param)
    : OpsRunner("ReshapeAndCacheOpsRunner310P"), param_(param)
{
    const std::size_t intensorSize = 5;
    const std::size_t outtensorSize = 2;
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(outtensorSize);

    int inTensorStart = 0;
    Mki::Tensor &keyTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &valueCacheTensor = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &slotsTensor = kernelGraph_.inTensors.at(inTensorStart++);

    int outTensorStart = 0;
    Mki::Tensor &outKeyCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);
    Mki::Tensor &outValueCacheTensor = kernelGraph_.outTensors.at(outTensorStart++);

    kernelGraph_.nodes.resize(1);
    auto &reshapeAndCacheNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::ReshapeAndCache reshapeAndCacheParam;
    reshapeAndCacheParam.type = AtbOps::OpParam::ReshapeAndCache::RESHAPE_AND_CACHE_NZ;
    reshapeAndCacheNode.opDesc = {0, "ReshapeAndCacheOperation", reshapeAndCacheParam};

    reshapeAndCacheNode.inTensors = {&keyTensor, &valueTensor, &keyCacheTensor, &valueCacheTensor, &slotsTensor};
    reshapeAndCacheNode.outTensors = {&outKeyCacheTensor, &outValueCacheTensor};

    reshapeAndCacheNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            if (i == 2 || i == 3) { // 2, 3: intensor index
                launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_FRACTAL_NZ;
            } else {
                launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
            }
        }
    };
}

ReshapeAndCacheOpsRunner310P::~ReshapeAndCacheOpsRunner310P() {}

REG_RUNNER_TYPE(ReshapeAndCacheOpsRunner310P);
} // namespace atb
