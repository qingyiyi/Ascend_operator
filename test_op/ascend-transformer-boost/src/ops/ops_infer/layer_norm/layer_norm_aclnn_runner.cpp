/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "layer_norm_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t OUT_TENSOR_NUM = 1;
} // namespace

namespace atb {
aclnnStatus (*LayerNormAclnnRunner::aclnnGetWorkspaceSizeFunc_)(const aclTensor *, const aclIntArray *,
                                                                const aclTensor *, const aclTensor *, double eps,
                                                                aclTensor *, aclTensor *, aclTensor *, uint64_t *,
                                                                aclOpExecutor **) = nullptr;
aclnnStatus (*LayerNormAclnnRunner::aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

LayerNormAclnnRunner::LayerNormAclnnRunner(const infer::LayerNormParam &param)
    : AclnnRunner("LayerNormAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LayerNormAclnnRunner::LayerNormAclnnRunner created";
}

LayerNormAclnnRunner::~LayerNormAclnnRunner() {}

Status LayerNormAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // input: x, gamma, beta, without quant inTensors: scale, offset
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

aclnnStatus LayerNormAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn indexAdd setup start.";
    if (LayerNormAclnnRunner::aclnnGetWorkspaceSizeFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix()
                  << "aclnn indexAdd, aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    size_t inTensorStart = 0;
    std::shared_ptr<AclNNTensor> xAclnnTensorPtr = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++);
    aclTensor *x = xAclnnTensorPtr->tensor; // input
    // normalizedShape
    Dims &xDims = xAclnnTensorPtr->atbTensor.desc.shape;
    int32_t beginNormAxis = param_.normParam.beginNormAxis;
    if (beginNormAxis < 0) {
        beginNormAxis += static_cast<int32_t>(xDims.dimNum);
    }

    int64_t totalNormAxes = static_cast<int64_t>(xDims.dimNum) - beginNormAxis;
    int64_t shapeHostData[totalNormAxes];
    for (int i = 0; i < totalNormAxes; ++i) {
        shapeHostData[i] = xDims.dims[beginNormAxis + i];
    }
    aclIntArray *normalizedShape = aclCreateIntArray(shapeHostData, totalNormAxes);
    aclTensor *gamma = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor; // weightOptional
    aclTensor *beta = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;  // biasOptional
    double epsilon = static_cast<double>(param_.normParam.epsilon);                      // eps
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // out
    aclTensor *emptyTensorPtr = nullptr;

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    ATB_LOG(DEBUG) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
                   << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                   << ", raw ptr from it: " << raw_executor_ptr;

    ATB_LOG(DEBUG) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    aclnnStatus ret = LayerNormAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        x, normalizedShape, gamma, beta, epsilon, output, emptyTensorPtr, emptyTensorPtr,
        &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ret = aclDestroyIntArray(normalizedShape);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray shape failed!";
        return ret;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status LayerNormAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (LayerNormAclnnRunner::aclnnExecuteFunc_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = LayerNormAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                              this->atbVariantPack_.workspaceBufferSize,
                                                              this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status LayerNormAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "LayerNormAclnnRunner LoadMethod";
    if (LayerNormAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        LayerNormAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnLayerNormGetWorkspaceSize", "aclnnLayerNorm",
                                    LayerNormAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                    LayerNormAclnnRunner::aclnnExecuteFunc_);
}

REG_RUNNER_TYPE(LayerNormAclnnRunner);
} // namespace atb
