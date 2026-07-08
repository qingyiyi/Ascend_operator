/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "repeat_aclnn_runner.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace atb {
// 初始化类函数指针
// 对应aclnnop/aclnn_repeat.h中的两段式接口
RepeatAclnnGetWorkspaceSizeFunc RepeatAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
RepeatAclnnExecuteFunc RepeatAclnnRunner::aclnnExecuteFunc_ = nullptr;

RepeatAclnnRunner::RepeatAclnnRunner(const infer::RepeatParam &param) : AclnnRunner("RepeatAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RepeatAclnnRunner::RepeatAclnnRunner called";
}

RepeatAclnnRunner::~RepeatAclnnRunner() {}

Status RepeatAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(1);
    this->aclnnVariantPack_.aclInTensors.resize(1);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    this->aclnnVariantPack_.aclOutTensors.reserve(1);
    this->aclnnVariantPack_.aclOutTensors.resize(1);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr);
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

aclnnStatus RepeatAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn setup start.";

    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    aclTensor *self = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    size_t outTensorStart = 0;
    aclTensor *out = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    size_ = aclCreateIntArray(this->param_.multiples.data(), this->param_.multiples.size());
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    aclnnStatus ret = RepeatAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        self, size_, out, &(this->atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    aclDestroyIntArray(size_);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status RepeatAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (!RepeatAclnnRunner::aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = RepeatAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                           this->atbVariantPack_.workspaceBufferSize,
                                                           this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status RepeatAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "RepeatAclnnRunner LoadMethod";
    if (RepeatAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && RepeatAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status =
        LoadFromSharedObjectFile("aclnnRepeatGetWorkspaceSize", "aclnnRepeat",
                                 RepeatAclnnRunner::aclnnGetWorkspaceSizeFunc_, RepeatAclnnRunner::aclnnExecuteFunc_);

    return status;
}

REG_RUNNER_TYPE(RepeatAclnnRunner);
} // namespace atb
