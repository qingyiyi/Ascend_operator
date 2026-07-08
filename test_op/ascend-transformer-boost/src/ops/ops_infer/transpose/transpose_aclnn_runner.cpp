/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transpose_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {
AclnnPermuteGetWorkspaceSizeFunc TransposeAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnPermuteFunc TransposeAclnnRunner::aclnnExecuteFunc_ = nullptr;

TransposeAclnnRunner::TransposeAclnnRunner(const infer::TransposeParam &param)
    : AclnnRunner("TransposeAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "TransposeAclnnRunner::TransposeAclnnRunner created";
}

TransposeAclnnRunner::~TransposeAclnnRunner() {}

Status TransposeAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "TransposeAclnnRunner LoadAclnnFuncs";
    if (TransposeAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        TransposeAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnPermuteGetWorkspaceSize", "aclnnPermute",
                                    TransposeAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                    TransposeAclnnRunner::aclnnExecuteFunc_);
}

Status TransposeAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // self
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

aclnnStatus TransposeAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Permute setup start.";
    aclTensor *x = aclnnVariantPack_.aclInTensors.at(0)->tensor;       // self
    aclTensor *output = aclnnVariantPack_.aclOutTensors.at(0)->tensor; // out

    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();

    SVector<int32_t> perm = param_.perm;
    int64_t dims[perm.size()];
    for (size_t i = 0; i < perm.size(); ++i) {
        dims[i] = static_cast<int64_t>(perm[i]);
    }
    aclnnStatus ret = ACL_SUCCESS;
    if (permIntArray_) {
        ret = aclDestroyIntArray(permIntArray_);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        permIntArray_ = nullptr;
    }
    permIntArray_ = aclCreateIntArray(dims, perm.size());
    ret = TransposeAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x, permIntArray_, output, &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status TransposeAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    void *executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = TransposeAclnnRunner::aclnnExecuteFunc_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    if (permIntArray_) {
        ret = aclDestroyIntArray(permIntArray_);
        if (ret != ACL_SUCCESS) {
            return ERROR_CANN_ERROR;
        }
        permIntArray_ = nullptr;
    }
    return NO_ERROR;
}

REG_RUNNER_TYPE(TransposeAclnnRunner);
} // namespace atb
