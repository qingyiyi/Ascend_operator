/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_dynamic_quant_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "acl/acl.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 2;

} // namespace

namespace atb {
// 初始化类函数指针
aclnnStatus (*AclnnDynamicQuantRunner::aclnnGetWorkspaceSizeFunc_)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*AclnnDynamicQuantRunner::aclnnExecuteFunc_)( void *, uint64_t , aclOpExecutor *, const aclrtStream) = nullptr;

AclnnDynamicQuantRunner::AclnnDynamicQuantRunner(const infer::ElewiseParam &param)
    : AclnnRunner("AclnnDynamicQuantRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AclnnDynamicQuantRunner::AclnnDynamicQuantRunner called";
}

AclnnDynamicQuantRunner::~AclnnDynamicQuantRunner() {}

Status AclnnDynamicQuantRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    
    Status ret = NO_ERROR;

    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "AclnnDynamicQuantRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);

        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "AclnnDynamicQuantRunner::BuildAclnnVariantPack outTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
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

aclnnStatus AclnnDynamicQuantRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnnDynamicQuant setup start.";

    ATB_LOG(INFO) << GetLogPrefix() << "aclnn dynamicQuant: "
                  << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();

    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    aclnnStatus ret = AclnnDynamicQuantRunner::aclnnGetWorkspaceSizeFunc_(
        this->aclnnVariantPack_.aclInTensors.at(0)->tensor,    // x
        nullptr,    // smoothScaleOptional
        this->aclnnVariantPack_.aclOutTensors.at(0)->tensor,    // yOut
        this->aclnnVariantPack_.aclOutTensors.at(1)->tensor,    // scaleOut
        &(this->atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr);
    
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status AclnnDynamicQuantRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";

    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = AclnnDynamicQuantRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status AclnnDynamicQuantRunner::LoadMethod()
{
    ATB_LOG(INFO) << "AclnnDynamicQuantRunner LoadMethod";
    if (AclnnDynamicQuantRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        AclnnDynamicQuantRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnDynamicQuantGetWorkspaceSize", "aclnnDynamicQuant",
                                            AclnnDynamicQuantRunner::aclnnGetWorkspaceSizeFunc_,
                                            AclnnDynamicQuantRunner::aclnnExecuteFunc_);
    return status;
}

REG_RUNNER_TYPE(AclnnDynamicQuantRunner);
} // namespace atb
