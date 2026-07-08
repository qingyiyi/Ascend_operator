/*
 * Copyright (c) 205 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "as_strided_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {

// 初始化类函数指针
aclnnStatus (*AsStridedAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
    aclTensor* selfRef,
    const aclTensor* src, 
    uint64_t* workspaceSize,
    aclOpExecutor** executor) = nullptr;

aclnnStatus (*AsStridedAclnnRunner::aclnnExecuteFunc_)(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream) = nullptr;

AsStridedAclnnRunner::AsStridedAclnnRunner(const infer::AsStridedParam &param)
    : AclnnRunner("AsStridedAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AsStridedAclnnRunner::AsStridedAclnnRunner called";
}

static atb::Dims SVector2Dims(const atb::SVector<int64_t>& vector) {
    atb::Dims dims;
    dims.dimNum = static_cast<uint64_t>(vector.size());
    for (size_t i = 0; i < dims.dimNum; ++i) {
        dims.dims[i] = vector[i];
    }
    return dims;
}

Status AsStridedAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;

    size_t num_inputs = runnerVariantPack.inTensors.size();
    if (num_inputs != IN_TENSOR_NUM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "AsStrided requires exactly 1 input, but got " << num_inputs;
        return ErrorType::ERROR_INVALID_PARAM;
    }

    // 创建输入ACL tensor
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    
    for (size_t i = 0; i < IN_TENSOR_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "AsStridedAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = param_.stride;

        atb::Dims viewDims = SVector2Dims(param_.size);

        ret = CallAclCreateTensor(viewDims, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype, param_.offset.at(0));
        
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // 构建输出tensor
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "AsStridedAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
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

aclnnStatus AsStridedAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn as_strided setup start.";
    if (!AsStridedAclnnRunner::aclnnGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    // 调用aclnn获取workspace大小
    // 注意：根据函数签名，第一个参数是selfRef（输出），第二个参数是src（输入）
    aclnnStatus ret = AsStridedAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        this->aclnnVariantPack_.aclOutTensors.at(0)->tensor,       // 输出tensor (selfRef)
        this->aclnnVariantPack_.aclInTensors.at(0)->tensor,        // 输入tensor (src)
        &(this->atbVariantPack_.workspaceBufferSize),              // 输出的workspace大小
        &raw_executor_ptr);                                        // 输出的executor

    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status AsStridedAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (!AsStridedAclnnRunner::aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;

    }

    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = AsStridedAclnnRunner::aclnnExecuteFunc_(
        this->atbVariantPack_.workspaceBuffer,
        this->atbVariantPack_.workspaceBufferSize,
        this->aclnnExecutor_.get(),
        executeStream);

    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status AsStridedAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "AsStridedAclnnRunner LoadMethod";
    if (AsStridedAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        AsStridedAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnInplaceCopyGetWorkspaceSize", "aclnnInplaceCopy",
                                             AsStridedAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                             AsStridedAclnnRunner::aclnnExecuteFunc_);
    return status;
}

REG_RUNNER_TYPE(AsStridedAclnnRunner);
} // namespace atb
