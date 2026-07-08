

/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/param_compare.h"

namespace atb {
GatherOpsRunner::GatherOpsRunner(const infer::GatherParam &param)
    : OpsRunner("GatherOpsRunner"), param_(param)
{
    ATB_LOG(INFO) << "GatherOpsRunner::GatherOpsRunner called";
    kernelGraph_.inTensors.resize(2); // intersorNum:2
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &yTensor = kernelGraph_.inTensors.at(1);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &gatherNode = kernelGraph_.nodes[0];

    AsdOps::OpParam::Gather gatherNodeParam = {param_.batchDims, {param_.axis}};

    gatherNode.opDesc = {0, "GatherOperation", gatherNodeParam};
    gatherNode.inTensors = {&xTensor, &yTensor};
    gatherNode.outTensors = {&outTensor};
}

GatherOpsRunner::~GatherOpsRunner() {}

REG_RUNNER_TYPE(GatherOpsRunner);
REG_OP_PARAM(AsdOps::OpParam::Gather);
} // namespace atb
