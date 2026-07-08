/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fill_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include <atb/utils/log.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 0;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {
AclnnInplaceFillScalarGetWorkspaceSizeFunc FillAclnnRunner::aclnnInplaceFillScalarGetWorkspaceSizeFunc_ = nullptr;
AclnnInplaceFillScalarExecuteFunc FillAclnnRunner::aclnnInplaceFillScalarExecuteFunc_ = nullptr;

FillAclnnRunner::FillAclnnRunner(const infer::FillParam &param) : AclnnRunner("FillAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "FillAclnnRunner::FillAclnnRunner created";
}

FillAclnnRunner::~FillAclnnRunner() {}

Status FillAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // input: none
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    // output
    aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    size_t outTensorIndex = 0;
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    ATB_LOG(INFO) << GetLogPrefix() << "outTensor index: " << outTensorIndex;
    atb::Tensor atbTensor = runnerVariantPack.outTensors.at(outTensorIndex);
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                              atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create outTensor by aclCreateTensor failed!";
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(outTensorIndex);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclOutTensors[outTensorIndex] = aclnnTensorPtr;
    return atb::NO_ERROR;
}

aclnnStatus FillAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn inplaceFillScalar setup start.";
    if (aclnnInplaceFillScalarGetWorkspaceSizeFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix()
                  << "aclnn inplaceFillScalar, aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();

    size_t outTensorStart = 0;
    aclTensor *output = aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // selfRef
    float value = param_.value.at(0);                                                       // 0: value
    aclScalar *valueScalarPtr = aclCreateScalar(&value, ACL_FLOAT);

    aclOpExecutor *raw_executor_ptr = aclnnExecutor_.get();
    aclnnStatus ret = aclnnInplaceFillScalarGetWorkspaceSizeFunc_(
        output, valueScalarPtr, &(atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnInplaceFillScalarGetWorkspaceSize failed";
        return ret;
    }
    ret = aclDestroyScalar(valueScalarPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "destroy scalar value failed";
    }
    valueScalarPtr = nullptr;
    return ret;
}

Status FillAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnInplaceFillScalarExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                                         atbVariantPack_.workspaceBufferSize,
                                                         aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status FillAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "FillAclnnRunner LoadMethod";
    if (aclnnInplaceFillScalarGetWorkspaceSizeFunc_ != nullptr &&
        aclnnInplaceFillScalarExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnInplaceFillScalarGetWorkspaceSize", "aclnnInplaceFillScalar",
                                     aclnnInplaceFillScalarGetWorkspaceSizeFunc_,
                                     aclnnInplaceFillScalarExecuteFunc_);
}

REG_RUNNER_TYPE(FillAclnnRunner);
} // namespace atb
