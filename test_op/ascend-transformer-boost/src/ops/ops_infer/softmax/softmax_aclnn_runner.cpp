/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "softmax_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const size_t IN_X_INDEX = 0;
} // namespace

namespace atb {
aclnnStatus (*SoftmaxAclnnRunner::aclnnGetWorkspaceSizeFunc_)(const aclTensor *, int64_t, aclTensor *, uint64_t *,
                                                              aclOpExecutor **) = nullptr;
aclnnStatus (*SoftmaxAclnnRunner::aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

SoftmaxAclnnRunner::SoftmaxAclnnRunner(const infer::SoftmaxParam &param)
    : AclnnRunner("SoftmaxAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SoftmaxAclnnRunner::SoftmaxAclnnRunner created";
}

SoftmaxAclnnRunner::~SoftmaxAclnnRunner() {}

Status SoftmaxAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    // self
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    Status ret = RecordDims(runnerVariantPack.inTensors, 0, param_.axes);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "record shape by axis failed!";
        return ret;
    }
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        // convert shape to flattened shape
        Dims shape = atbTensor.desc.shape;
        if (i == IN_X_INDEX) {
            shape = targetDims_;
        }
        aclnnTensorPtr->strides = GetCopyTensorStride(shape);
        ret = CallAclCreateTensor(shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
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
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        Dims shape = atbTensor.desc.shape;
        if (i == IN_X_INDEX) {
            shape = targetDims_;
        }
        aclnnTensorPtr->strides = GetCopyTensorStride(shape);
        ret = CallAclCreateTensor(shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

Status SoftmaxAclnnRunner::RecordDims(SVector<Tensor> tensors, const size_t id, SVector<int64_t> &axes)
{
    Dims originDims_ = tensors.at(id).desc.shape;
    if (axes.size() == 1 && axes.at(0) == -1) { // 1: 1维, -1：只归一化最后一维
        targetDims_ = originDims_;
        return NO_ERROR;
    }
    if (axes.size() > originDims_.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << " softmax axes's dimNum[" << axes.size()
                       << "] should be less than x's dimNum[" << originDims_.dimNum << "].";
        return ERROR_INVALID_PARAM;
    }
    int64_t start = axes.at(0);
    int64_t end = axes.at(axes.size() - 1);
    int64_t dimNum = static_cast<int64_t>(originDims_.dimNum);
    for (int64_t i = 0; i <= start; ++i) {
        targetDims_.dims[i] = originDims_.dims[i];
    }
    if (axes.size() > 1) {
        for (int64_t i = axes.at(1); i <= end; ++i) {
            targetDims_.dims[start] *= originDims_.dims[i];
        }
    }
    for (int64_t i = end + 1; i < dimNum; ++i) {
        targetDims_.dims[++start] = originDims_.dims[i];
    }
    targetDims_.dimNum = static_cast<uint64_t>(start + 1);
    return NO_ERROR;
}

aclnnStatus SoftmaxAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Softmax setup start.";
    if (SoftmaxAclnnRunner::aclnnGetWorkspaceSizeFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix()
                  << "aclnn Softmax, aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();


    int64_t axis = param_.axes[0];
    size_t inTensorStart = 0;
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor; // self
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // out
    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    aclnnStatus ret = SoftmaxAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x, axis, output, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status SoftmaxAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (SoftmaxAclnnRunner::aclnnExecuteFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = SoftmaxAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                            this->atbVariantPack_.workspaceBufferSize,
                                                            this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status SoftmaxAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SoftmaxAclnnRunner LoadMethod";
    if (SoftmaxAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && SoftmaxAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnSoftmaxGetWorkspaceSize", "aclnnSoftmax",
                                    SoftmaxAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                    SoftmaxAclnnRunner::aclnnExecuteFunc_);
}

REG_RUNNER_TYPE(SoftmaxAclnnRunner);
} // namespace atb