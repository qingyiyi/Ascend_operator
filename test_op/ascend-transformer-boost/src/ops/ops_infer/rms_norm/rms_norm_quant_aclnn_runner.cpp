/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_quant_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "acl/acl.h"
#include "atbops/params/params.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const int X_ACLNN_TENSOR_IDX = 0;
static const int GAMMA_ACLNN_TENSOR_IDX = 1;
static const int BETA_ACLNN_TENSOR_IDX = 2;
static const int SCALE_ACLNN_TENSOR_IDX = 3;
static const int OFFSET_ACLNN_TENSOR_IDX = 4;
static const int Y_ACLNN_TENSOR_IDX = 0;
}  // namespace

namespace atb {
AclnnRmsNormQuantGetWorkspaceSizeFunc RmsNormQuantAclnnRunner::aclnnRmsNormQuantGetWorkspaceSizeFunc_ = nullptr;
AclnnRmsNormQuantFunc RmsNormQuantAclnnRunner::aclnnRmsNormQuantFunc_ = nullptr;

RmsNormQuantAclnnRunner::RmsNormQuantAclnnRunner(const infer::RmsNormParam &param)
    : AclnnRunner("RmsNormQuantAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::RmsNormQuantAclnnRunner";
}

RmsNormQuantAclnnRunner::~RmsNormQuantAclnnRunner()
{}

Status RmsNormQuantAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "RmsNormQuantAclnnRunner::LoadAclnnFuncs";
    if (aclnnRmsNormQuantGetWorkspaceSizeFunc_ && aclnnRmsNormQuantFunc_) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnRmsNormQuantGetWorkspaceSize",
        "aclnnRmsNormQuant",
        aclnnRmsNormQuantGetWorkspaceSizeFunc_,
        aclnnRmsNormQuantFunc_);
}

Status RmsNormQuantAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    GetTensorNum();
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    Status st = CreateXAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateGammaAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateBetaAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateScaleAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateOffsetAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    return CreateYAclnnTensor();
}

aclnnStatus RmsNormQuantAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::SetAclNNWorkspaceExecutor";
    aclTensor *x = aclnnVariantPack_.aclInTensors.at(xAclTensorIndex_)->tensor;
    aclTensor *gamma = aclnnVariantPack_.aclInTensors.at(gammaAclTensorIndex_)->tensor;
    aclTensor *beta = aclnnVariantPack_.aclInTensors.at(betaAclTensorIndex_)->tensor;
    aclTensor *scale = aclnnVariantPack_.aclInTensors.at(scaleAclTensorIndex_)->tensor;
    aclTensor *offset = aclnnVariantPack_.aclInTensors.at(offsetAclTensorIndex_)->tensor;
    aclTensor *y = aclnnVariantPack_.aclOutTensors.at(yAclTensorIndex_)->tensor;
    double epsilon = static_cast<double>(param_.normParam.epsilon);
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    aclnnStatus ret = aclnnRmsNormQuantGetWorkspaceSizeFunc_(
        x, gamma, beta, scale, offset, epsilon, y, &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret == ACLNN_SUCCESS) {
        ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetAclNNWorkspaceExecutor failed, ret: " << ret;
    }
    return ret;
}

Status RmsNormQuantAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::LaunchAclnnKernel";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnRmsNormQuantFunc_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (ret == ACLNN_SUCCESS) {
        ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::LaunchAclnnKernel success";
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "RmsNormQuantAclnnRunner::LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void RmsNormQuantAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 5;   // 5: x, gamma, beta, scale, offset
    aclOutTensorNum_ = 1;  // 1: y
}

void RmsNormQuantAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;

    xAclTensorIndex_ = 0;
    gammaAclTensorIndex_ = 0;
    betaAclTensorIndex_ = 0;
    scaleAclTensorIndex_ = 0;
    offsetAclTensorIndex_ = 0;
    yAclTensorIndex_ = 0;
}

Status RmsNormQuantAclnnRunner::CreateXAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::CreateXAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, X_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "x aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    xAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status RmsNormQuantAclnnRunner::CreateGammaAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::CreateGammaAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    Dims viewShape;
    viewShape.dimNum = 1;
    viewShape.dims[0] = atbTensor.desc.shape.dims[atbTensor.desc.shape.dimNum - 1];
    SVector<int64_t> strides = GetCopyTensorStride(viewShape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, GAMMA_ACLNN_TENSOR_IDX, viewShape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    gammaAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status RmsNormQuantAclnnRunner::CreateBetaAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::CreateBetaAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    Dims viewShape;
    viewShape.dimNum = 1;
    viewShape.dims[0] = atbTensor.desc.shape.dims[atbTensor.desc.shape.dimNum - 1];
    SVector<int64_t> strides = GetCopyTensorStride(viewShape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, BETA_ACLNN_TENSOR_IDX, viewShape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "beta aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    betaAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status RmsNormQuantAclnnRunner::CreateScaleAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::CreateScaleAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, SCALE_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "scale aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    scaleAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status RmsNormQuantAclnnRunner::CreateOffsetAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::CreateOffsetAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, OFFSET_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "offset aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    offsetAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status RmsNormQuantAclnnRunner::CreateYAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "RmsNormQuantAclnnRunner::CreateYAclnnTensor";

    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, Y_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "y aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    yAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

REG_RUNNER_TYPE(RmsNormQuantAclnnRunner);
}  // namespace atb