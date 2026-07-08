/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <aclnn/opdev/op_errno.h>
#include <atb/utils/log.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/operation_register.h"
#include "reduce_aclnn_runner.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {
ReduceAclnnSumGetWorkspaceSizeFunc ReduceAclnnRunner::aclnnReduceSumGetWorkspaceSizeFunc_ = nullptr;
ReduceAclnnRunner::ExecuteFuncType ReduceAclnnRunner::aclnnReduceSumExecuteFunc_ = nullptr;

ReduceAclnnAmaxGetWorkspaceSizeFunc ReduceAclnnRunner::aclnnAmaxGetWorkspaceSizeFunc_ = nullptr;
ReduceAclnnRunner::ExecuteFuncType ReduceAclnnRunner::aclnnAmaxExecuteFunc_ = nullptr;

ReduceAclnnAminGetWorkspaceSizeFunc ReduceAclnnRunner::aclnnAminGetWorkspaceSizeFunc_ = nullptr;
ReduceAclnnRunner::ExecuteFuncType ReduceAclnnRunner::aclnnAminExecuteFunc_ = nullptr;

ReduceAclnnRunner::ReduceAclnnRunner(const infer::ReduceParam &param) : AclnnRunner("ReduceAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReduceAclnnRunner::ReduceAclnnRunner created";
}

ReduceAclnnRunner::~ReduceAclnnRunner()
{
    if (dims_) {
        aclnnStatus ret = aclDestroyIntArray(dims_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "failed to destroy intArray, dim, with return value: " << ret;
        }
        dims_ = nullptr;
    }
}

Status ReduceAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // input
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create inTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // output
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "outTensor index: " << i;
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create outTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclnnStatus ReduceAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn reduce setup start.";
    ATB_LOG(INFO) << GetLogPrefix()
                  << "aclnn reduce, aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    std::shared_ptr<AclNNTensor> xAclnnTensorPtr = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++);
    aclTensor *x = xAclnnTensorPtr->tensor; // input
    aclDataType dtype = xAclnnTensorPtr->atbTensor.desc.dtype;
    // dims
    std::vector<int64_t> dimsValue(param_.axis.size(), 0);
    for (size_t i = 0; i < param_.axis.size(); ++i) {
        dimsValue[i] = param_.axis[i];
    }

    for (size_t i = 0; i < param_.axis.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "dimsValue i " << i << " " << dimsValue[i];
    }
    aclnnStatus ret;
    if (dims_ != nullptr) {
        ret = aclDestroyIntArray(dims_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "failed to destrory intArray, dim";
            return ret;
        }
        dims_ = nullptr;
    }
    dims_ = aclCreateIntArray(dimsValue.data(), static_cast<uint64_t>(param_.axis.size()));
    if (dims_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "failed to create intArray, dim";
        return ACLNN_ERR_PARAM_INVALID;
    }
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // out
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    bool keepDims = false;
    switch (param_.reduceType) {
        case atb::infer::ReduceParam::ReduceType::REDUCE_SUM:
            ret = ReduceAclnnRunner::aclnnReduceSumGetWorkspaceSizeFunc_(
                x, dims_, keepDims, dtype, output, &(this->atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
            break;
        case atb::infer::ReduceParam::ReduceType::REDUCE_MAX:
            ret = ReduceAclnnRunner::aclnnAmaxGetWorkspaceSizeFunc_(
                x, dims_, keepDims, output, &(this->atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
            break;
        case atb::infer::ReduceParam::ReduceType::REDUCE_MIN:
            ret = ReduceAclnnRunner::aclnnAminGetWorkspaceSizeFunc_(
                x, dims_, keepDims, output, &(this->atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
            break;
        default:
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "expect reduceType to be one of REDUCE_MAX, REDUCE_MIN or REDUCE_SUM, but got: "
                           << param_.reduceType;
            return ACLNN_ERR_PARAM_INVALID;
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set AclOpExecutorRepeatable failed!";
        return ret;
    }
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ReduceAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    Status status = GetFunc();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "get aclnn func failed!";
        return status;
    }
    if (executeFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnn ExecuteFunc is null!";
        return ERROR_CANN_ERROR;
    }
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = executeFunc_(this->atbVariantPack_.workspaceBuffer, this->atbVariantPack_.workspaceBufferSize,
                                   this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status ReduceAclnnRunner::GetFunc()
{
    switch (param_.reduceType) {
        case atb::infer::ReduceParam::ReduceType::REDUCE_SUM:
            executeFunc_ = ReduceAclnnRunner::aclnnReduceSumExecuteFunc_;
            return NO_ERROR;
        case atb::infer::ReduceParam::ReduceType::REDUCE_MAX:
            executeFunc_ = ReduceAclnnRunner::aclnnAmaxExecuteFunc_;
            return NO_ERROR;
        case atb::infer::ReduceParam::ReduceType::REDUCE_MIN:
            executeFunc_ = ReduceAclnnRunner::aclnnAminExecuteFunc_;
            return NO_ERROR;
        default:
            executeFunc_ = nullptr;
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "expect reduceType to be one of REDUCE_MAX, REDUCE_MIN or REDUCE_SUM, but got: "
                           << param_.reduceType;
    }
    return ERROR_INVALID_PARAM;
}

bool ReduceAclnnRunner::useCache()
{
    return false;
}

Status ReduceAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "ReduceAclnnRunner LoadMethod";
    Status status = NO_ERROR;
    // reduce sum
    if (ReduceAclnnRunner::aclnnReduceSumGetWorkspaceSizeFunc_ != nullptr &&
        ReduceAclnnRunner::aclnnReduceSumExecuteFunc_ != nullptr &&
        ReduceAclnnRunner::aclnnAmaxGetWorkspaceSizeFunc_ != nullptr &&
        ReduceAclnnRunner::aclnnAmaxExecuteFunc_ != nullptr &&
        ReduceAclnnRunner::aclnnAminGetWorkspaceSizeFunc_ != nullptr &&
        ReduceAclnnRunner::aclnnAminExecuteFunc_ != nullptr) {
        return status;
    }
    status = LoadFromSharedObjectFile("aclnnReduceSumGetWorkspaceSize", "aclnnReduceSum",
                                      ReduceAclnnRunner::aclnnReduceSumGetWorkspaceSizeFunc_,
                                      ReduceAclnnRunner::aclnnReduceSumExecuteFunc_);
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << "ReduceAclnnRunner load aclnnReduceSum failed!";
        return status;
    }
    // reduce max
    status = LoadFromSharedObjectFile("aclnnAmaxGetWorkspaceSize", "aclnnAmax",
                                      ReduceAclnnRunner::aclnnAmaxGetWorkspaceSizeFunc_,
                                      ReduceAclnnRunner::aclnnAmaxExecuteFunc_);
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << "ReduceAclnnRunner load aclnnAmax failed!";
        return status;
    }
    // reduce min
    status = LoadFromSharedObjectFile("aclnnAminGetWorkspaceSize", "aclnnAmin",
                                      ReduceAclnnRunner::aclnnAminGetWorkspaceSizeFunc_,
                                      ReduceAclnnRunner::aclnnAminExecuteFunc_);
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << "ReduceAclnnRunner load aclnnAmin failed!";
    }
    return status;
}

REG_RUNNER_TYPE(ReduceAclnnRunner);
} // namespace atb
