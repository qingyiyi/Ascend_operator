/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cumsum_aclnn_runner.h"

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
aclnnStatus (*CumsumAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
    const aclTensor* input,
    int64_t dim,
    bool exclusive,
    bool reverse,
    aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) = nullptr;

aclnnStatus (*CumsumAclnnRunner::aclnnExecuteFunc_)(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream) = nullptr;

CumsumAclnnRunner::CumsumAclnnRunner(const infer::CumsumParam &param)
    : AclnnRunner("CumsumAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "CumsumAclnnRunner::CumsumAclnnRunner called";
}

Status CumsumAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;

    size_t num_inputs = runnerVariantPack.inTensors.size();
    if (num_inputs != IN_TENSOR_NUM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Cumsum requires exactly 1 input, but got " << num_inputs;
        return ErrorType::ERROR_INVALID_PARAM;
    }

    // 构建输入tensor
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    
    for (size_t i = 0; i < IN_TENSOR_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "CumsumAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
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

    // 构建输出tensor
    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "CumsumAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
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

aclnnStatus CumsumAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn cumsum setup start.";
    if (!CumsumAclnnRunner::aclnnGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }

    // 注意：cumsum支持多个轴，这里取第一个轴，因为aclnn接口只支持单个维度
    int64_t dim = param_.axes.empty() ? 0 : param_.axes[0];

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    // 调用aclnn获取workspace大小
    aclnnStatus ret = CumsumAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        this->aclnnVariantPack_.aclInTensors.at(0)->tensor,              // 输入tensor
        dim,                                                             // cumsum维度
        param_.exclusive,                                                // exclusive参数
        param_.reverse,                                                  // reverse参数
        this->aclnnVariantPack_.aclOutTensors.at(0)->tensor,             // 输出tensor
        &(this->atbVariantPack_.workspaceBufferSize),                    // 输出的workspace大小
        &raw_executor_ptr);                                              // 输出的executor

    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });

    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status CumsumAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (!CumsumAclnnRunner::aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }

    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = CumsumAclnnRunner::aclnnExecuteFunc_(
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

Status CumsumAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "CumsumAclnnRunner LoadMethod";
    if (CumsumAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        CumsumAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }

    Status status = LoadFromSharedObjectFile("aclnnCumsumV2GetWorkspaceSize", "aclnnCumsumV2",
                                             CumsumAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                             CumsumAclnnRunner::aclnnExecuteFunc_);
    return status;
}

REG_RUNNER_TYPE(CumsumAclnnRunner);
} // namespace atb