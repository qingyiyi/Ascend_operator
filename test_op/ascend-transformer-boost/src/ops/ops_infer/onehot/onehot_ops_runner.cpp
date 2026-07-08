/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onehot_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/param_compare.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 3;
static const int32_t OUT_TENSOR_NUM = 1;
static const int32_t TENSOR_ONE_INDEX = 1;
static const int32_t TENSOR_TWO_INDEX = 2;

OnehotOpsRunner::OnehotOpsRunner(const infer::OnehotParam &param)
    : OpsRunner("OnehotOpsRunner"), param_(param)
{
    ATB_LOG(INFO) << "OnehotOpsRunner::OnehotOpsRunner called";
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);
    Mki::Tensor &x = kernelGraph_.inTensors.at(0);
    Mki::Tensor &oneTensor = kernelGraph_.inTensors.at(TENSOR_ONE_INDEX);
    Mki::Tensor &zeroTensor = kernelGraph_.inTensors.at(TENSOR_TWO_INDEX);
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &onehotNode = kernelGraph_.nodes.at(0);

    Mki::SVector<int64_t> depth;
    depth.push_back(param_.depth);
    AsdOps::OpParam::Onehot opParam = {param_.axis, depth};
    onehotNode.opDesc = {0, "OnehotOperation", opParam};
    onehotNode.inTensors = {&x, &oneTensor, &zeroTensor};
    onehotNode.outTensors = {&operationOutTensor};
}

OnehotOpsRunner::~OnehotOpsRunner() {}

REG_RUNNER_TYPE(OnehotOpsRunner);
REG_OP_PARAM(AsdOps::OpParam::Onehot);
} // namespace atb