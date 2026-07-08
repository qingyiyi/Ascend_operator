/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "activation_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/runner_util.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/param_compare.h"

static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t SWIGLU_BACKWARD_IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;

namespace atb {
ActivationOpsRunner::ActivationOpsRunner(const infer::ActivationParam &param)
    : OpsRunner("ActivationOpsRunner"), param_(param)
{
    ATB_LOG(INFO) << "ActivationOpsRunner::ActivationOpsRunner called, param_.activationType:" << param_.activationType;
    kernelGraph_.nodes.resize(1);
    auto &activationNode = kernelGraph_.nodes.at(0);
    activationNode.opDesc = RunnerUtil::GetActivationNodeOpDesc(param_);

    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);
    activationNode.outTensors = {&operationOutTensor};
    if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD) {
        kernelGraph_.inTensors.resize(SWIGLU_BACKWARD_IN_TENSOR_NUM);

        int inTensorStart = 0; // 0: y_grad, 1: x
        Mki::Tensor &yGradInTensor = kernelGraph_.inTensors.at(inTensorStart++);
        Mki::Tensor &xInTensor = kernelGraph_.inTensors.at(inTensorStart++);

        activationNode.inTensors = {&yGradInTensor, &xInTensor};
    } else {
        kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
        Mki::Tensor &operationInTensor = kernelGraph_.inTensors.at(0); // 0: x
        activationNode.inTensors = {&operationInTensor};
    }
}

ActivationOpsRunner::~ActivationOpsRunner() {}

REG_RUNNER_TYPE(ActivationOpsRunner);
REG_OP_PARAM(AsdOps::OpParam::Activation);
} // namespace atb
