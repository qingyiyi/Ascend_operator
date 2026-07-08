/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatterv_operation.h"
#include "atb/utils/config.h"
#include "reduce_scatterv_hccl_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace {
static constexpr int NUM_2 = 2;

static constexpr uint32_t IN_TENSOR_NUM = 5;
static constexpr uint32_t OUT_TENSOR_NUM = 1;

static constexpr uint32_t IN_TENSOR_0 = 0;
static constexpr uint32_t IN_TENSOR_1 = 1;
static constexpr uint32_t IN_TENSOR_2 = 2;
static constexpr uint32_t IN_TENSOR_3 = 3;
static constexpr uint32_t IN_TENSOR_4 = 4;

static constexpr uint32_t OUT_TENSOR_0 = 0;

static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;

static constexpr uint32_t DIM_NUM_1 = 1;
static constexpr uint32_t DIM_NUM_2 = 2;

static constexpr int32_t RECVCOUNT_LENGTH_1 = 1;

bool ParamCheck(const atb::infer::ReduceScatterVParam &opParam)
{
    if (opParam.backend != "hccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must be hccl";
        return false;
    }
    if (opParam.rankSize < NUM_2) {
        ATB_LOG(ERROR) << "ReduceScatterVParam ranksize must be larger than 1, current ranksize: " << opParam.rankSize;
        return atb::ERROR_INVALID_PARAM;
    }
    if (opParam.reduceType != "sum" && opParam.reduceType != "max" && opParam.reduceType != "min") {
        ATB_LOG(ERROR) << "reduceType is " << opParam.reduceType
                       << ", ReduceScatterVParam reduceType must be one of the following sum, max, min";
        return false;
    }
    if (opParam.reduceType != "sum" && atb::GetSingleton<atb::Config>().Is310P()) {
        ATB_LOG(ERROR) << "reduceType is " << opParam.reduceType
                       << ", At Altas 300I, ReduceScatterVParam reduceType must be sum";
        return false;
    }
    if (atb::OperationUtil::DistributedInitCheck<atb::infer::ReduceScatterVParam>(opParam) != atb::NO_ERROR) {
        ATB_LOG(ERROR) << "ReduceScatterVOperation DistributedInitCheck failed";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(ReduceScatterVOperation, infer::ReduceScatterVParam)

ReduceScatterVOperation::ReduceScatterVOperation(const infer::ReduceScatterVParam &param)
    : OperationBase("ReduceScatterVOperation"), param_(param)
{
    if (GetSingleton<Config>().Is310P()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceScatterVOperation310p");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceScatterVOperation");
    }
}

ReduceScatterVOperation::~ReduceScatterVOperation() {}

uint32_t ReduceScatterVOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t ReduceScatterVOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status ReduceScatterVOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                               SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(OUT_TENSOR_0) = inTensorDescs.at(IN_TENSOR_0);
    outTensorDescs.at(OUT_TENSOR_0).shape.dims[DIM_0] = inTensorDescs.at(IN_TENSOR_4).shape.dims[DIM_0];
    ATB_LOG(INFO) << GetLogPrefix() << " inTensorDescs Size:" << inTensorDescs.size()
                  << " outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

Status ReduceScatterVOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_0).shape.dimNum != DIM_NUM_2 ||
        inTensorDescs.at(IN_TENSOR_1).shape.dimNum != DIM_NUM_1 ||
        inTensorDescs.at(IN_TENSOR_2).shape.dimNum != DIM_NUM_1 ||
        inTensorDescs.at(IN_TENSOR_3).shape.dimNum != DIM_NUM_1 ||
        inTensorDescs.at(IN_TENSOR_4).shape.dimNum != DIM_NUM_1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid inTensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(IN_TENSOR_1).shape.dims[DIM_0] != param_.rankSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "sendcountslength must be equal to ranksize";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_2).shape.dims[DIM_0] != param_.rankSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "sdispls length must be equal to ranksize";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_3).shape.dims[DIM_0] != RECVCOUNT_LENGTH_1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "recvCount length must be equal to 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status ReduceScatterVOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                               const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (outTensors.at(OUT_TENSOR_0).desc.shape.dimNum != DIM_NUM_2) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid outTensor dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    Dims expectShape = inTensors.at(IN_TENSOR_0).desc.shape;
    expectShape.dims[DIM_0] = inTensors.at(IN_TENSOR_4).desc.shape.dims[DIM_0];
    if (!TensorUtil::TensorShapeEqual(outTensors.at(OUT_TENSOR_0).desc.shape, expectShape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dim0 must be equal to inTensor4 dim0, "
                       << "outTensor0 dim1 must be equal to inTensor0 dim1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> ReduceScatterVOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<ReduceScatterVHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<ReduceScatterVHcclRunner>(param_, param_.hcclComm);
        }
    }

    ATB_LOG(ERROR) << "ReduceScatterVOperation::ReduceScatterVOperation backend " << param_.backend << "is not exist.";
    return std::shared_ptr<Runner>();
}

infer::ReduceScatterVParam ReduceScatterVOperation::GetParam() const
{
    return param_;
}

void ReduceScatterVOperation::SetParam(const infer::ReduceScatterVParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

nlohmann::json ReduceScatterVOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb