/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_q_concat_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/param_compare.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;

RopeQConcatOpsRunner::RopeQConcatOpsRunner(const infer::RopeQConcatParam &param)
    : OpsRunner("RopeQConcatOpsRunner"), param_(param)
{
}

RopeQConcatOpsRunner::~RopeQConcatOpsRunner() {}

Status RopeQConcatOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);

    int64_t inTensorNum = 0;
    Mki::Tensor &q = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &cos = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &sin = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &concatInput = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    Mki::Tensor &ropeQConcat = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &ropeNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::RopeQConcat ropeQConcatParam;
    ropeNode.opDesc = {0, "RopeQConcatOperation", ropeQConcatParam};
    ropeNode.inTensors = {&q, &cos, &sin, &concatInput};
    ropeNode.outTensors = {&ropeQConcat};

    return NO_ERROR;
}

void RopeQConcatOpsRunner::SetParam(const Mki::Any &param)
{
    infer::RopeQConcatParam newParam = Mki::AnyCast<infer::RopeQConcatParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "RopeQConcatOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

REG_RUNNER_TYPE(RopeQConcatOpsRunner);
REG_OP_PARAM(AtbOps::OpParam::RopeQConcat);
} // namespace atb