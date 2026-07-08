/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"
#include "sort_aclnn_runner.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t INDEX_ZERO = 0;
static const uint32_t INDEX_ONE = 1;
static const uint32_t INDEX_TWO = 2;
static const uint32_t SIZE_TWO = 2;
} // namespace

namespace atb {
AclnnGetWorkspaceSizeFunc SortAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnExecuteFunc SortAclnnRunner::aclnnExecuteFunc_ = nullptr;
AclnnCastGetWorkspaceSizeFunc SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ = nullptr;
AclnnCastExecuteFunc SortAclnnRunner::aclnnCastExecuteFunc_ = nullptr;

SortAclnnRunner::SortAclnnRunner(const infer::SortParam &param) : AclnnRunner("SortAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SortAclnnRunner::SortAclnnRunner created";
}

void SortAclnnRunner::CleanUp()
{
    aclnnStatus ret = 0;
    if (indices_ != nullptr) {
        ret = aclDestroyTensor(indices_);
        if (ret != ACL_SUCCESS)
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy indices_->tensor failed with return value: " << ret;
        indices_ = nullptr;
    }
}

SortAclnnRunner::~SortAclnnRunner()
{
    CleanUp();
}

Status SortAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // self
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


    // temp indices
    indicesBufferSize_ = this->atbVariantPack_.outTensors.at(INDEX_ONE).dataSize * SIZE_TWO;
    atb::SVector<int64_t> strides = GetCopyTensorStride(this->atbVariantPack_.outTensors.at(INDEX_ONE).desc.shape);
    Dims viewDims;

    viewDims.dimNum = this->atbVariantPack_.outTensors.at(INDEX_ONE).desc.shape.dimNum;
    for (size_t i = 0; i < viewDims.dimNum; i++) {
        viewDims.dims[i] = this->atbVariantPack_.outTensors.at(INDEX_ONE).desc.shape.dims[i];
    }
    if (indices_ != nullptr) {
        ret = aclDestroyTensor(indices_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy indices_->tensor failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
        indices_ = nullptr;
    }
    indices_ = aclCreateTensor(viewDims.dims, viewDims.dimNum, ACL_INT64, strides.data(), 0,
                               this->atbVariantPack_.outTensors.at(INDEX_ONE).desc.format, viewDims.dims,
                               viewDims.dimNum, nullptr);
    if (indices_ == nullptr) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create int64 indices by aclCreateTensor failed!";
        return ERROR_CANN_ERROR;
    }
    return atb::NO_ERROR;
}

aclnnStatus SortAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Topk setup start.";

    size_t inTensorStart = 0;
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor; // self
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor; // valueOut

    int64_t k = static_cast<int64_t>(param_.num.at(0));
    int64_t dim = -1;
    bool largest = true;
    bool sorted = true;
    aclnnStatus ret = 0;

    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ret = SortAclnnRunner::aclnnGetWorkspaceSizeFunc_(x, k, dim, largest, sorted, output, indices_,
                                                      &(this->topkWorkspaceSize_), &rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set Topk AclOpExecutorRepeatable failed!";
        return ret;
    }
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    // indices_->tensor holds the aclnnTopK return in INT64, we need aclnnCast to turn aclnn
    aclOpExecutor *rawCastExecutorPtr = this->aclnnCastExecutor_.get();
    aclTensor *out = this->aclnnVariantPack_.aclOutTensors.at(INDEX_ONE)->tensor; // indicesOut
    ret = SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_(indices_, ACL_INT32, out, &(this->castWorkspaceSize_),
                                                          &rawCastExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnCastGetWorkspaceSize failed!";
        return ret;
    }

    // setCastExecutorRepeatable same as the topkExecutorRepeatable, this is essential for cache to work
    ret = aclSetAclOpExecutorRepeatable(rawCastExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set Cast AclOpExecutorRepeatable failed!";
        return ret;
    }

    this->aclnnCastExecutor_ = std::shared_ptr<aclOpExecutor>(rawCastExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });

    this->atbVariantPack_.workspaceBufferSize =
        this->topkWorkspaceSize_ + this->castWorkspaceSize_ + this->indicesBufferSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status SortAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = ACL_SUCCESS;
    ret = aclSetOutputTensorAddr(this->aclnnExecutor_.get(), INDEX_ONE, this->indices_,
                                 this->atbVariantPack_.workspaceBuffer + this->topkWorkspaceSize_ +
                                     this->castWorkspaceSize_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclSetOutputTensorAddr failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ret = aclSetInputTensorAddr(this->aclnnCastExecutor_.get(), INDEX_ZERO, this->indices_,
                                this->atbVariantPack_.workspaceBuffer + this->topkWorkspaceSize_ +
                                    this->castWorkspaceSize_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclSetInputTensorAddr failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ret = SortAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer, this->topkWorkspaceSize_,
                                             this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ret =
        SortAclnnRunner::aclnnCastExecuteFunc_(this->atbVariantPack_.workspaceBuffer + this->topkWorkspaceSize_,
                                               this->castWorkspaceSize_, this->aclnnCastExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

bool SortAclnnRunner::useCache()
{
    return false;
}

Status SortAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SortAclnnRunner LoadMethod";
    if (SortAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && SortAclnnRunner::aclnnExecuteFunc_ != nullptr &&
        SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ != nullptr &&
        SortAclnnRunner::aclnnCastExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status ret =
        LoadFromSharedObjectFile("aclnnTopkGetWorkspaceSize", "aclnnTopk", SortAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                 SortAclnnRunner::aclnnExecuteFunc_);
    if (ret != NO_ERROR) {
        return ret;
    }

    return LoadFromSharedObjectFile("aclnnCastGetWorkspaceSize", "aclnnCast",
                                    SortAclnnRunner::aclnnCastGetWorkspaceSizeFunc_,
                                    SortAclnnRunner::aclnnCastExecuteFunc_);
}

REG_RUNNER_TYPE(SortAclnnRunner);
} // namespace atb
