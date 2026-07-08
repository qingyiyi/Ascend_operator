/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "faupdate_ops_runner.h"
#include <asdops/params/params.h>
#include <atb/utils/log.h>
#include "atb/utils/operation_register.h"
#include "atb/utils/param_compare.h"

namespace atb {
FaUpdateOpsRunner::FaUpdateOpsRunner(const infer::FaUpdateParam &param)
    : OpsRunner("FaUpdateOpsRunner"), param_(param)
{
    ATB_LOG(INFO) << "FaUpdateOpsRunner::FaUpdateOpsRunner";
}

FaUpdateOpsRunner::~FaUpdateOpsRunner() {}

Status FaUpdateOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    size_t inTensorNum = 2;
    size_t outTensorNum = 1;
    size_t internalTensorNum = 0;
    size_t nodeNum = 1;
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);
    kernelGraph_.internalTensors.resize(internalTensorNum);
    kernelGraph_.nodes.resize(nodeNum);

    if (param_.faUpdateType == infer::FaUpdateParam::DECODE_UPDATE) {
        return SetupKernelGraphDecodeUpdate();
    }
    return NO_ERROR;
}

Status FaUpdateOpsRunner::SetupKernelGraphDecodeUpdate()
{
    ATB_LOG(INFO) << GetLogPrefix() << "FaUpdateOpsRunner::SetupKernelGraphDecodeUpdate";

    size_t inTensorId = 0;
    Mki::Tensor &lseTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &goTensor = kernelGraph_.inTensors.at(inTensorId++);

    size_t outTensorId = 0;
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(outTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &decodeUpdateNode = kernelGraph_.nodes.at(nodeId++);

    AsdOps::OpParam::FaUpdate faUpdateParam;
    faUpdateParam.faUpdateType = AsdOps::OpParam::FaUpdate::DECODE_UPDATE;
    faUpdateParam.sp = param_.sp;
    decodeUpdateNode.opDesc = {0, "FaUpdateOperation", faUpdateParam};
    decodeUpdateNode.inTensors = {&lseTensor, &goTensor};
    decodeUpdateNode.outTensors = {&outTensor};

    return NO_ERROR;
}

REG_RUNNER_TYPE(FaUpdateOpsRunner);
REG_OP_PARAM(AsdOps::OpParam::FaUpdate);
} // namespace atb
