/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transpose_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace atb {
TransposeOpsRunner::TransposeOpsRunner(const infer::TransposeParam &param)
    : OpsRunner("TransposeOpsRunner"), param_(param)
{
    ATB_LOG(INFO) << "TransposeOpsRunner::TransposeOpsRunner called, param_.perm:" << param_.perm;
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &operationInTensor = kernelGraph_.inTensors.at(0);

    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &transposeNode = kernelGraph_.nodes.at(0);

    Mki::SVector<int32_t> perms;
    TensorUtil::AtbSVector2OpsSVector(param_.perm, perms);
    AsdOps::OpParam::Transpose transposeNodeParam = {perms};

    transposeNode.opDesc = {0, "TransposeOperation", transposeNodeParam};
    transposeNode.inTensors = {&operationInTensor};
    transposeNode.outTensors = {&operationOutTensor};
}

TransposeOpsRunner::~TransposeOpsRunner() {}

REG_RUNNER_TYPE(TransposeOpsRunner);
} // namespace atb
