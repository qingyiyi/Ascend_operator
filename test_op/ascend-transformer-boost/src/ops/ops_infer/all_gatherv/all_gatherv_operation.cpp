/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gatherv_operation.h"
#include "atb/utils/config.h"
#include "all_gatherv_hccl_runner.h"
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

static constexpr int32_t SENDCOUNT_LENGTH_1 = 1;

bool ParamCheck(const atb::infer::AllGatherVParam &opParam)
{
    if (opParam.backend != "hccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must be hccl";
        return false;
    }
    if (opParam.rankSize < NUM_2) {
        ATB_LOG(ERROR) << "AllGatherV ranksize must be larger than 1, current ranksize: " << opParam.rankSize;
        return false;
    }
    if (atb::OperationUtil::DistributedInitCheck<atb::infer::AllGatherVParam>(opParam) != atb::NO_ERROR) {
        ATB_LOG(ERROR) << "AllGatherVOperation DistributedInitCheck failed";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(AllGatherVOperation, infer::AllGatherVParam)

AllGatherVOperation::AllGatherVOperation(const infer::AllGatherVParam &param)
    : OperationBase("AllGatherVOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("AllGatherVOperation");
}

AllGatherVOperation::~AllGatherVOperation() {}

uint32_t AllGatherVOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t AllGatherVOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status AllGatherVOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                           SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(OUT_TENSOR_0) = inTensorDescs.at(IN_TENSOR_0);
    outTensorDescs.at(OUT_TENSOR_0).shape.dims[DIM_0] = inTensorDescs.at(IN_TENSOR_4).shape.dims[DIM_0];
    return NO_ERROR;
}

Status AllGatherVOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(IN_TENSOR_1).shape.dimNum != DIM_NUM_1 ||
        inTensorDescs.at(IN_TENSOR_2).shape.dimNum != DIM_NUM_1 ||
        inTensorDescs.at(IN_TENSOR_3).shape.dimNum != DIM_NUM_1 ||
        inTensorDescs.at(IN_TENSOR_4).shape.dimNum != DIM_NUM_1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "invalid inTensor dimNum, inTensor 1~4 dimNum should be euqal to 1";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(IN_TENSOR_1).shape.dims[DIM_0] != SENDCOUNT_LENGTH_1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "sendCount length must be equal to 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_2).shape.dims[DIM_0] != param_.rankSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "recvCounts length must be equal to ranksize";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(IN_TENSOR_3).shape.dims[DIM_0] != param_.rankSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "rdispls length must be equal to ranksize";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

int64_t CalculateTensorSize(const SVector<Tensor> &tensors)
{
    std::size_t size = tensors.at(0).desc.shape.dimNum;
    int64_t result = 1;
    for (std::size_t i = 0; i < size; ++i) {
        result *= tensors.at(0).desc.shape.dims[i];
    }
    return result;
}

Status AllGatherVOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    if (outTensors.at(OUT_TENSOR_0).desc.shape.dimNum != inTensors.at(IN_TENSOR_0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dimNum must equal to inTensor0 dimNum";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    Dims expectShape = inTensors.at(IN_TENSOR_0).desc.shape;
    expectShape.dims[DIM_0] = inTensors.at(IN_TENSOR_4).desc.shape.dims[DIM_0];
    if (!TensorUtil::TensorShapeEqual(outTensors.at(OUT_TENSOR_0).desc.shape, expectShape)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dim0 must be equal to inTensor4 dim0, "
                       << "outTensor0 dim1~n must be equal to inTensor0 dim1~n";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t count = 0;
    int64_t* recvCounts = static_cast<int64_t*>(inTensors[2].hostData);
    int64_t* rdispls = static_cast<int64_t*>(inTensors[3].hostData);
    if (*(static_cast<int64_t *>(inTensors[1].hostData)) > (CalculateTensorSize(inTensors))) {
        ATB_LOG(ERROR) << "sendcount should be less than intensor total data length";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if ((param_.rankSize != inTensors.at(2).desc.shape.dims[0]) || // 2: recvcount
        (param_.rankSize != inTensors.at(3).desc.shape.dims[0])) { // 3: rdispls
        ATB_LOG(ERROR) << "recvcounts or recvdis length must be equal to ranksize";
        return ERROR_INVALID_PARAM;
    }
    for (int i = 0; i < param_.rankSize; ++i) {
        if (count > std::numeric_limits<int64_t>::max() - recvCounts[i]) {
            ATB_LOG(ERROR) << " ,AllGatherVOperation sum(recvCounts) will overflow ";
            return ERROR_INVALID_PARAM;
        }
        count += recvCounts[i];
        if (recvCounts[i] < 0) {
            ATB_LOG(ERROR) << "recvcounts must more than zero";
            return ERROR_INVALID_PARAM;
        }
        if (recvCounts[i] + rdispls[i] > CalculateTensorSize(outTensors)) {
            ATB_LOG(ERROR) << "The sum of recvconts and recvdisp should be less than output length";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (recvCounts[i] > std::numeric_limits<int64_t>::max() - rdispls[i] || recvCounts[i] + rdispls[i] > count) {
            ATB_LOG(ERROR) << "AllGatherVOperation recvCounts + rdispls is out of bounds";
            return ERROR_INVALID_PARAM;
        }
    }
    if (count <= 0) {
        ATB_LOG(ERROR) << "AllGatherVOperation sum(recvCounts) should be more than 0";
        return ERROR_INVALID_PARAM;
    }
    if (recvCounts[param_.rank] !=  *(static_cast<int64_t *>(inTensors[1].hostData))) {
        ATB_LOG(ERROR) << param_.rank << "sendcount must equal to recivecount";
        return ERROR_INVALID_PARAM;
    }
    if (outTensors.at(0).desc.shape.dims[0] != inTensors.at(4).desc.shape.dims[0]) { // 4: y
        ATB_LOG(ERROR) << "outTensor first dimension does not match y";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> AllGatherVOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<AllGatherVHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<AllGatherVHcclRunner>(param_, param_.hcclComm);
        }
    }
    ATB_LOG(FATAL) << "AllGatherVOperation::AllGatherVOperation backend " << param_.backend << "does not exist.";
    return std::shared_ptr<Runner>();
}

infer::AllGatherVParam AllGatherVOperation::GetParam() const
{
    return param_;
}

void AllGatherVOperation::SetParam(const infer::AllGatherVParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

nlohmann::json AllGatherVOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb