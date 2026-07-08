/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_ascend_quant_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_check.h"
#include "acl/acl.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t IN_TENSOR_NUM_WITHOUT_OFFSET = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t INDEX_0 = 0;
static const uint32_t INDEX_1 = 1;
static const uint32_t INDEX_2 = 2;
static const uint32_t MULTIPLE_2 = 2;
static const uint32_t DIM_0 = 0;

void ReshapeTailDims(atb::Dims &shape, size_t targetDims)
{
    for (size_t i = targetDims + 1; i < shape.dimNum; ++i) {
        shape.dims[targetDims] *= shape.dims[i];
        shape.dims[i] = 0;
    }
    shape.dimNum = targetDims + 1;
}
} // namespace

namespace atb {
// 初始化类函数指针
aclnnStatus (*AclnnAscendQuantRunner::aclnnGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *,
                                                                  const aclTensor *, bool, char *, int32_t, int32_t,
                                                                  const aclTensor *, uint64_t *,
                                                                  aclOpExecutor **) = nullptr;

aclnnStatus (*AclnnAscendQuantRunner::aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *,
                                                         const aclrtStream) = nullptr;

aclnnStatus (*AclnnAscendQuantRunner::aclnnReciprocalGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                                            aclOpExecutor **) = nullptr;

aclnnStatus (*AclnnAscendQuantRunner::aclnnReciprocalExecuteFunc_)(void *, uint64_t, aclOpExecutor *,
                                                                   aclrtStream) = nullptr;

aclnnStatus (*AclnnAscendQuantRunner::aclnnCastGetWorkspaceSizeFunc_)(const aclTensor *, const aclDataType, aclTensor *,
                                                                      uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*AclnnAscendQuantRunner::aclnnCastExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

AclnnAscendQuantRunner::AclnnAscendQuantRunner(const infer::ElewiseParam &param)
    : AclnnRunner("AclnnAscendQuantRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AclnnAscendQuantRunner::AclnnAscendQuantRunner called";
}

void AclnnAscendQuantRunner::CleanUp()
{
    aclnnStatus ret = 0;
    if (scale_ != nullptr) {
        ret = aclDestroyTensor(scale_);
        if (ret != ACL_SUCCESS)
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy scale_->tensor failed with return value: " << ret;
        scale_ = nullptr;
    }
    if (offset_ != nullptr) {
        ret = aclDestroyTensor(offset_);
        if (ret != ACL_SUCCESS)
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy offset_->tensor failed with return value: " << ret;
        offset_ = nullptr;
    }
}

AclnnAscendQuantRunner::~AclnnAscendQuantRunner()
{
    CleanUp();
}

Status AclnnAscendQuantRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    is_offset_empty_ = TensorCheck::IsEmptyTensor(runnerVariantPack.inTensors.at(INDEX_2));
    this->aclnnVariantPack_.aclInTensors.reserve(is_offset_empty_ ? IN_TENSOR_NUM_WITHOUT_OFFSET : IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(is_offset_empty_ ? IN_TENSOR_NUM_WITHOUT_OFFSET : IN_TENSOR_NUM);
    const bool is_scale_all_one = [](const Dims &shape) {
        for (size_t i = 0; i < shape.dimNum; ++i) {
            if (shape.dims[i] != 1) {
                return false;
            }
        }
        return true;
    }(runnerVariantPack.inTensors.at(INDEX_1).desc.shape);

    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "AclnnAscendQuantRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        if (i != INDEX_0 || !is_scale_all_one) {
            ReshapeTailDims(atbTensor.desc.shape, runnerVariantPack.inTensors.at(i).desc.shape.dimNum -
                                                      runnerVariantPack.inTensors.at(INDEX_1).desc.shape.dimNum);
        }
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
        ATB_LOG(INFO) << GetLogPrefix() << "AclnnAscendQuantRunner::BuildAclnnVariantPack outTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        if (i != INDEX_0 || !is_scale_all_one) {
            ReshapeTailDims(atbTensor.desc.shape, runnerVariantPack.outTensors.at(i).desc.shape.dimNum -
                                                      runnerVariantPack.inTensors.at(INDEX_1).desc.shape.dimNum);
        }
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
    aclnnStatus status = ACL_SUCCESS;
    // temp scale
    scaleBufferSize_ = 0;
    {
        scaleBufferSize_ = runnerVariantPack.inTensors.at(INDEX_1).dataSize;
        ATB_LOG(INFO) << GetLogPrefix() << "scaleBufferSize_: " << scaleBufferSize_;
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(INDEX_1);
        ReshapeTailDims(atbTensor.desc.shape, DIM_0);
        atb::SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
        Dims viewDims = atbTensor.desc.shape;

        if (scale_ != nullptr) {
            status = aclDestroyTensor(scale_);
            if (status != ACL_SUCCESS) {
                ATB_LOG(ERROR) << GetLogPrefix() << "destroy scale_->tensor failed with return value: " << status;
                CleanUp();
                return ERROR_CANN_ERROR;
            }
            scale_ = nullptr;
        }
        scale_ = aclCreateTensor(viewDims.dims, viewDims.dimNum, atbTensor.desc.dtype, strides.data(), 0,
                                 atbTensor.desc.format, viewDims.dims, viewDims.dimNum, nullptr);
        if (scale_ == nullptr) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create int64 indices by aclCreateTensor failed!";
            return ERROR_CANN_ERROR;
        }
    }

    scaleDatatype_ = runnerVariantPack.inTensors.at(INDEX_0).desc.dtype;
    ATB_LOG(INFO) << GetLogPrefix() << "scaleDatatype_: " << scaleDatatype_;

    // temp offset
    offsetBufferSize_ = 0;
    if (!is_offset_empty_) {
        offsetBufferSize_ = runnerVariantPack.inTensors.at(INDEX_2).dataSize * MULTIPLE_2;
        ATB_LOG(INFO) << GetLogPrefix() << "offsetBufferSize_: " << offsetBufferSize_;
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(INDEX_2);
        ReshapeTailDims(atbTensor.desc.shape, DIM_0);
        atb::SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
        Dims viewDims = atbTensor.desc.shape;

        if (offset_ != nullptr) {
            status = aclDestroyTensor(offset_);
            if (status != ACL_SUCCESS) {
                ATB_LOG(ERROR) << GetLogPrefix() << "destroy offset_->tensor failed with return value: " << status;
                CleanUp();
                return ERROR_CANN_ERROR;
            }
            offset_ = nullptr;
        }
        offset_ = aclCreateTensor(viewDims.dims, viewDims.dimNum, scaleDatatype_, strides.data(), 0,
                                  atbTensor.desc.format, viewDims.dims, viewDims.dimNum, nullptr);
        if (offset_ == nullptr) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create int64 indices by aclCreateTensor failed!";
            return ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

aclnnStatus AclnnAscendQuantRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnnAscendQuantV3 setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << " aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();

    size_t inTensorStart = 0;
    aclTensor *x = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *scale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    size_t outTensorStart = 0;
    aclTensor *output = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

    aclnnStatus ret = ACL_SUCCESS;

    this->reciprocalWorkspaceSize_ = 0;
    aclOpExecutor *rawReciprocalExecutorPtr = this->aclnnReciprocalExecutor_.get();
    ret = AclnnAscendQuantRunner::aclnnReciprocalGetWorkspaceSizeFunc_(scale,  // self
                                                                       scale_, // out
                                                                       &(this->reciprocalWorkspaceSize_),
                                                                       &rawReciprocalExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnReciprocalGetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawReciprocalExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set Reciprocal AclOpExecutorRepeatable failed!";
        return ret;
    }
    this->aclnnReciprocalExecutor_ =
        std::shared_ptr<aclOpExecutor>(rawReciprocalExecutorPtr, [this](aclOpExecutor *ptr) {
            if (ptr) { // 可复用时才手动销毁aclOpExecutor
                aclDestroyAclOpExecutor(ptr);
            }
        });

    this->castWorkspaceSize_ = 0;
    if (!is_offset_empty_) {
        aclTensor *offset = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
        aclOpExecutor *rawCastExecutorPtr = this->aclnnCastExecutor_.get();
        ret = AclnnAscendQuantRunner::aclnnCastGetWorkspaceSizeFunc_(offset,         // self
                                                                     scaleDatatype_, // dtype
                                                                     offset_,        // out
                                                                     &(this->castWorkspaceSize_), &rawCastExecutorPtr);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclnnCastGetWorkspaceSize failed!";
            return ret;
        }
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
    }

    this->quantWorkspaceSize_ = 0;
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ret = AclnnAscendQuantRunner::aclnnGetWorkspaceSizeFunc_(x,                                      // x
                                                             scale_,                                 // scale
                                                             is_offset_empty_ ? nullptr : offset_,   // offset
                                                             false,                                  // sqrtMode
                                                             (char *)(std::string("round").c_str()), // roundMode
                                                             ACL_INT8,                               // dstType
                                                             -1,                                     // axis
                                                             output,                                 // y
                                                             &(this->quantWorkspaceSize_), &rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnAscendQuantV3GetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set AscendQuantV3 AclOpExecutorRepeatable failed!";
        return ret;
    }
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    this->atbVariantPack_.workspaceBufferSize = this->reciprocalWorkspaceSize_ + this->castWorkspaceSize_ +
                                                this->quantWorkspaceSize_ + this->scaleBufferSize_ +
                                                this->offsetBufferSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "reciprocalWorkspaceSize_: " << this->reciprocalWorkspaceSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "castWorkspaceSize_: " << this->castWorkspaceSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "quantWorkspaceSize_: " << this->quantWorkspaceSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status AclnnAscendQuantRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = ACL_SUCCESS;
    ret = aclSetOutputTensorAddr(this->aclnnReciprocalExecutor_.get(), INDEX_0, this->scale_,
                                 this->atbVariantPack_.workspaceBuffer + this->reciprocalWorkspaceSize_ +
                                     this->castWorkspaceSize_ + this->quantWorkspaceSize_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclSetOutputTensorAddr failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ret = aclSetInputTensorAddr(this->aclnnExecutor_.get(), INDEX_1, this->scale_,
                                this->atbVariantPack_.workspaceBuffer + this->reciprocalWorkspaceSize_ +
                                    this->castWorkspaceSize_ + this->quantWorkspaceSize_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclSetInputTensorAddr failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    if (!is_offset_empty_) {
        ret = aclSetOutputTensorAddr(this->aclnnCastExecutor_.get(), INDEX_0, this->offset_,
                                     this->atbVariantPack_.workspaceBuffer + this->reciprocalWorkspaceSize_ +
                                         this->castWorkspaceSize_ + this->quantWorkspaceSize_ + this->scaleBufferSize_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclSetOutputTensorAddr failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
        ret = aclSetInputTensorAddr(this->aclnnExecutor_.get(), INDEX_2, this->offset_,
                                    this->atbVariantPack_.workspaceBuffer + this->reciprocalWorkspaceSize_ +
                                        this->castWorkspaceSize_ + this->quantWorkspaceSize_ + this->scaleBufferSize_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclSetInputTensorAddr failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    ret = AclnnAscendQuantRunner::aclnnReciprocalExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                              this->reciprocalWorkspaceSize_,
                                                              this->aclnnReciprocalExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    if (!is_offset_empty_) {
        ret = AclnnAscendQuantRunner::aclnnCastExecuteFunc_(
            this->atbVariantPack_.workspaceBuffer + this->reciprocalWorkspaceSize_, this->castWorkspaceSize_,
            this->aclnnCastExecutor_.get(), executeStream);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    ret = AclnnAscendQuantRunner::aclnnExecuteFunc_(
        this->atbVariantPack_.workspaceBuffer + this->reciprocalWorkspaceSize_ + this->castWorkspaceSize_,
        this->quantWorkspaceSize_, this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

bool AclnnAscendQuantRunner::useCache()
{
    return false;
}

Status AclnnAscendQuantRunner::LoadMethod()
{
    ATB_LOG(INFO) << "AclnnAscendQuantRunner LoadMethod";
    if (AclnnAscendQuantRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        AclnnAscendQuantRunner::aclnnExecuteFunc_ != nullptr &&
        AclnnAscendQuantRunner::aclnnReciprocalGetWorkspaceSizeFunc_ != nullptr &&
        AclnnAscendQuantRunner::aclnnReciprocalExecuteFunc_ != nullptr &&
        AclnnAscendQuantRunner::aclnnCastGetWorkspaceSizeFunc_ != nullptr &&
        AclnnAscendQuantRunner::aclnnCastExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnAscendQuantV3GetWorkspaceSize", "aclnnAscendQuantV3",
                                             AclnnAscendQuantRunner::aclnnGetWorkspaceSizeFunc_,
                                             AclnnAscendQuantRunner::aclnnExecuteFunc_);
    if (status != NO_ERROR) {
        return status;
    }
    status = LoadFromSharedObjectFile("aclnnReciprocalGetWorkspaceSize", "aclnnReciprocal",
                                      AclnnAscendQuantRunner::aclnnReciprocalGetWorkspaceSizeFunc_,
                                      AclnnAscendQuantRunner::aclnnReciprocalExecuteFunc_);
    if (status != NO_ERROR) {
        return status;
    }
    status = LoadFromSharedObjectFile("aclnnCastGetWorkspaceSize", "aclnnCast",
                                      AclnnAscendQuantRunner::aclnnCastGetWorkspaceSizeFunc_,
                                      AclnnAscendQuantRunner::aclnnCastExecuteFunc_);
    return status;
}

REG_RUNNER_TYPE(AclnnAscendQuantRunner);
} // namespace atb
