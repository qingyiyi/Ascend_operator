/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/tensor_util.h"
#include "acl/acl.h"
namespace atb {
static const uint32_t ROPE_IN_NUM = 4;
static const uint32_t ROPE_OUT_NUM = 2;
static const uint32_t ROPE_QUERY_INDEX = 0;
static const uint32_t ROPE_KEY_INDEX = 1;
static const uint32_t ROPE_COS_INDEX = 2;
static const uint32_t ROPE_SIN_INDEX = 3;
static const uint32_t ROPE_SEQLEN_INDEX = 4;
static const uint32_t ROTARY_COEFF_HALF = 2;
static const uint32_t ROTARY_COEFF_QUARTER = 4;
static const uint32_t ACLNN_BSND_DIM_NUM = 4;
static const uint32_t ACLNN_TND_DIM_NUM = 3;
static const uint32_t ATB_TND_DIM_NUM = 2;
static const uint32_t DIM_B = 0;
static const uint32_t DIM_S = 1;
static const uint32_t DIM_N = 2;
static const uint32_t DIM_D = 3;
static const uint32_t DIM_ONE = 1;
static const uint32_t COS_SIN_NUM = 2;

enum class RotaryLayout : int {
    BSND = 1,
    SBND = 2,
    BNSD = 3,
    TND = 4,
};

aclnnGetWorkspaceSizeFuncPtr RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
aclnnExecuteFuncPtr RopeAclnnRunner::aclnnExecuteFunc_ = nullptr;

RopeAclnnRunner::RopeAclnnRunner(const infer::RopeParam &param)
    : AclnnRunner("RopeAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::RopeAclnnRunner called";
}

RopeAclnnRunner::~RopeAclnnRunner() {}

Status RopeAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "RopeAclnnRunner LoadMethod";
    if (RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        RopeAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnApplyRotaryPosEmbV2GetWorkspaceSize", "aclnnApplyRotaryPosEmbV2",
                                             RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                             RopeAclnnRunner::aclnnExecuteFunc_);
    return status;
}

Status RopeAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    this->aclnnVariantPack_.aclInTensors.reserve(ROPE_IN_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(ROPE_IN_NUM);
    this->aclnnVariantPack_.aclOutTensors.reserve(ROPE_OUT_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(ROPE_OUT_NUM);
    Status ret = NO_ERROR;
    bool isBSND4D = runnerVariantPack.inTensors.at(ROPE_QUERY_INDEX).desc.shape.dimNum == ACLNN_BSND_DIM_NUM;
    int64_t headDim = runnerVariantPack.inTensors.at(ROPE_COS_INDEX).desc.shape.dims[1]; // 1: headDim dim

    //key and query reshape
    for (size_t i = 0; i < ROPE_IN_NUM - COS_SIN_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        if (!isBSND4D) { // 2: [ntoken, hiddenSize]
            // [ntoken, hiddenSize] -> [ntoken, headNum, headDim]
            atbTensor.desc.shape.dimNum = ACLNN_TND_DIM_NUM; // tnd: [ntoken, headDim, headDim]
            atbTensor.desc.shape.dims[2] = headDim; // 2: d
            atbTensor.desc.shape.dims[1] = atbTensor.desc.shape.dims[1] / headDim; // 1: aclnn n dim, 1: atb nd dim, nd / d
            atbTensor.desc.shape.dims[0] = atbTensor.desc.shape.dims[0]; // 1: bs dim
        } else {
            atbTensor.desc.shape.dims[DIM_N] = atbTensor.desc.shape.dims[DIM_N] * atbTensor.desc.shape.dims[DIM_D] / headDim;
            atbTensor.desc.shape.dims[DIM_D] = headDim;
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
        aclnnTensorPtr->needUpdateTensorDataPtr = false;
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    //cos sin reshape
    for (size_t i = ROPE_IN_NUM - COS_SIN_NUM; i < ROPE_IN_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        if (!isBSND4D) {
            // tnd: [bs, 1*d] -> [bs, 1, d]
            atbTensor.desc.shape.dimNum = ACLNN_TND_DIM_NUM;
            atbTensor.desc.shape.dims[2] = atbTensor.desc.shape.dims[1]; // 2: aclnn d dim, 1: atb nd dim
            atbTensor.desc.shape.dims[1] = DIM_ONE; // 1: aclnn n dim
        } else {
            atbTensor.desc.shape.dimNum = ACLNN_BSND_DIM_NUM;
            Dims qDims = runnerVariantPack.inTensors.at(0).desc.shape;
            atbTensor.desc.shape.dims[DIM_D] = headDim; // 3: aclnn d dim
            atbTensor.desc.shape.dims[DIM_N] = DIM_ONE; // 2: aclnn n dim
            atbTensor.desc.shape.dims[DIM_S] = qDims.dims[DIM_S];
            atbTensor.desc.shape.dims[DIM_B] = qDims.dims[DIM_B];
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
        aclnnTensorPtr->needUpdateTensorDataPtr = false;
        this->aclnnVariantPack_.aclInTensors.at(i) = aclnnTensorPtr;
    }
    //create output
    for (size_t i = 0; i < ROPE_OUT_NUM; ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "RopeAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = this->aclnnVariantPack_.aclInTensors.at(i)->atbTensor;
        atbTensor.deviceData = runnerVariantPack.outTensors.at(i).deviceData;
        auto memRet = aclrtMemcpy(atbTensor.deviceData, atbTensor.dataSize,
                    this->aclnnVariantPack_.aclInTensors.at(i)->atbTensor.deviceData, atbTensor.dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (memRet != ACL_SUCCESS) {
            return ERROR_CANN_ERROR;
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
        aclnnTensorPtr->needUpdateTensorDataPtr = false;
        this->aclnnVariantPack_.aclOutTensors.at(i) = aclnnTensorPtr;
    }
    return NO_ERROR;
}

Status RopeAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = RopeAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

aclnnStatus RopeAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn rope setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    aclTensor *queryRef = aclnnVariantPack_.aclOutTensors.at(ROPE_QUERY_INDEX)->tensor;
    aclTensor *keyRef = aclnnVariantPack_.aclOutTensors.at(ROPE_KEY_INDEX)->tensor;
    aclTensor *cos = aclnnVariantPack_.aclInTensors.at(ROPE_COS_INDEX)->tensor;
    aclTensor *sin = aclnnVariantPack_.aclInTensors.at(ROPE_SIN_INDEX)->tensor;
    aclOpExecutor *rawExecutorPtr = this->aclnnExecutor_.get();
    ATB_LOG(INFO) << GetLogPrefix() << "&(this->aclnnExecutor_): " << &(this->aclnnExecutor_)
#ifdef _DEBUG
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
#endif
                  << ", addr of this->aclnnExecutor_: " << this->aclnnExecutor_
                  << ", raw ptr from it: " << rawExecutorPtr
                  << ", then take the address of the raw ptr: " << &rawExecutorPtr;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize addr: " << &(this->atbVariantPack_.workspaceBufferSize);

    for (size_t i = 0; i < aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "index " << i << TensorUtil::TensorToString(aclnnVariantPack_.aclInTensors.at(i)->atbTensor);
    }
    std::string rotaryMode = "half";
    if (param_.rotaryCoeff == ROTARY_COEFF_HALF) {
        rotaryMode = "half";
    } else if (param_.rotaryCoeff == ROTARY_COEFF_QUARTER) {
        rotaryMode = "quarter";
    } else {
        rotaryMode = "interleave";
    }
    // 4d -> bsnd, 2d -> tnd
    RotaryLayout layout = RotaryLayout::BSND;
    if (aclnnVariantPack_.aclInTensors.at(ROPE_QUERY_INDEX)->atbTensor.desc.shape.dimNum == ACLNN_TND_DIM_NUM) {
        layout = RotaryLayout::TND;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "layout: " << static_cast<int64_t>(layout);
    aclnnStatus ret = RopeAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        queryRef,
        keyRef,
        cos,
        sin,
        static_cast<int64_t>(layout),
        (char *)rotaryMode.c_str(),
        &(this->atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr
    );
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

REG_RUNNER_TYPE(RopeAclnnRunner);
} // namespace atb
