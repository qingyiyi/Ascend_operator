/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {
AclnnGatherV3GetWorkspaceSizeFunc GatherAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnGatherV3Func GatherAclnnRunner::aclnnExecuteFunc_ = nullptr;

GatherAclnnRunner::GatherAclnnRunner(const infer::GatherParam &param) : AclnnRunner("GatherAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "GatherAclnnRunner::GatherAclnnRunner created";
}

GatherAclnnRunner::~GatherAclnnRunner() {}

Status GatherAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "GatherAclnnRunner LoadAclnnFuncs";
    if (GatherAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && GatherAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnGatherV3GetWorkspaceSize", "aclnnGatherV3",
                                    GatherAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                    GatherAclnnRunner::aclnnExecuteFunc_);
}

Status GatherAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // input
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclInTensors.size(); ++i) {
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
        aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // output
    aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack outTensor index: " << i;
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
        aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclnnStatus GatherAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Gather setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Gather, aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *index = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(0)->tensor;
    int64_t mode = 1; // 1：索引聚集场景性能优化
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    aclnnStatus ret = GatherAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        self, param_.axis, index, param_.batchDims, mode, out, &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status GatherAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    void *executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = GatherAclnnRunner::aclnnExecuteFunc_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

REG_RUNNER_TYPE(GatherAclnnRunner);
} // namespace atb