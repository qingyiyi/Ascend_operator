/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cohere_layernorm_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/param_compare.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT_TWO = 2;
static const uint64_t OUT_TENSOR_COUNT_ONE = 1;
static const uint64_t NODE_SIZE_ONE = 1;

CohereLayerNormRunner::CohereLayerNormRunner(const infer::CohereLayerNormParam &param)
    : OpsRunner("CohereLayerNormRunner"), param_(param)
{
    AsdOps::OpParam::Norm layerNormParam = {AsdOps::OpParam::Norm::LAYER_NORM};
    layerNormParam.inGamma = true;
    layerNormParam.inBeta = false;
    layerNormParam.inRes = false;
    layerNormParam.outRes = false;
    layerNormParam.outMean = false;
    layerNormParam.outVarience = false;
    layerNormParam.outResQuant = false;
    layerNormParam.epsilon = param.epsilon;

    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_TWO);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_ONE);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(NODE_SIZE_ONE);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &gammaTensor};
    layerNormNode.outTensors = {&resultTensor};
}

CohereLayerNormRunner::~CohereLayerNormRunner() {}
REG_RUNNER_TYPE(CohereLayerNormRunner);
REG_OP_PARAM(AsdOps::OpParam::Norm);
} // namespace atb