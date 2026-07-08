/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transdata_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t INDEX_0 = 0;
} // namespace

namespace atb {
aclnnStatus (*TransdataAclnnRunner::aclnnGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                                aclOpExecutor **) = nullptr;
aclnnStatus (*TransdataAclnnRunner::aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

TransdataAclnnRunner::TransdataAclnnRunner(const infer::TransdataParam &param)
    : AclnnRunner("TransdataAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "TransdataAclnnRunner::TransdataAclnnRunner created";
}

TransdataAclnnRunner::~TransdataAclnnRunner() {}

Status TransdataAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // self
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = runnerVariantPack.inTensors.at(INDEX_0);
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                              atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(INDEX_0);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclInTensors[INDEX_0] = aclnnTensorPtr;

    // output
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atbTensor = runnerVariantPack.outTensors.at(INDEX_0);
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                              atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(INDEX_0);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclOutTensors[INDEX_0] = aclnnTensorPtr;
    return atb::NO_ERROR;
}

aclnnStatus TransdataAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnnNpuFormatCast setup start.";
    if (TransdataAclnnRunner::aclnnGetWorkspaceSizeFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(INDEX_0)->tensor;  // srcTensor
    aclTensor *y = this->aclnnVariantPack_.aclOutTensors.at(INDEX_0)->tensor; // dstTensor
    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();

    aclnnStatus ret = TransdataAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x, y, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
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

Status TransdataAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (TransdataAclnnRunner::aclnnExecuteFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = TransdataAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                              this->atbVariantPack_.workspaceBufferSize,
                                                              this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status TransdataAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "TransdataAclnnRunner LoadMethod";
    if (TransdataAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        TransdataAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnNpuFormatCastGetWorkspaceSize", "aclnnNpuFormatCast",
                                    TransdataAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                    TransdataAclnnRunner::aclnnExecuteFunc_);
}

REG_RUNNER_TYPE(TransdataAclnnRunner);
} // namespace atb
