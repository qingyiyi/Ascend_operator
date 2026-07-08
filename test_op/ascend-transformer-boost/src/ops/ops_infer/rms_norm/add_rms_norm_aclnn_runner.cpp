/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "add_rms_norm_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "acl/acl.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"
#include "atbops/params/params.h"

namespace {
static const int X1_ACLNN_TENSOR_IDX = 0;
static const int X2_ACLNN_TENSOR_IDX = 1;
static const int GAMMA_ACLNN_TENSOR_IDX = 2;
static const int Y_OUT_ACLNN_TENSOR_IDX = 0;
static const int RSTD_OUT_ACLNN_TENSOR_IDX = 1;
static const int X_OUT_ACLNN_TENSOR_IDX = 2;

static const size_t FLOAT_SIZE = 4;
}  // namespace

namespace atb {
AclnnAddRmsNormGetWorkspaceSizeFunc AddRmsNormAclnnRunner::aclnnAddRmsNormGetWorkspaceSizeFunc_ = nullptr;
AclnnAddRmsNormFunc AddRmsNormAclnnRunner::aclnnAddRmsNormFunc_ = nullptr;

AddRmsNormAclnnRunner::AddRmsNormAclnnRunner(const infer::RmsNormParam &param)
    : AclnnRunner("AddRmsNormAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::AddRmsNormAclnnRunner";
}

AddRmsNormAclnnRunner::~AddRmsNormAclnnRunner()
{
    if (rstdDeviceData_) {
        aclError ret = aclrtFree(rstdDeviceData_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclrtFree rstdDeviceData_ failed, ret: " << ret;
        }
        rstdDeviceData_ = nullptr;
    }
}

Status AddRmsNormAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "AddRmsNormAclnnRunner::LoadAclnnFuncs";
    if (aclnnAddRmsNormGetWorkspaceSizeFunc_ && aclnnAddRmsNormFunc_) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnAddRmsNormGetWorkspaceSize",
        "aclnnAddRmsNorm",
        aclnnAddRmsNormGetWorkspaceSizeFunc_,
        aclnnAddRmsNormFunc_);
}

Status AddRmsNormAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    GetTensorNum();
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    Status st = CreateX1AclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateX2AclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateGammaAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateYOutAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateXOutAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    return CreateRstdOutAclnnTensor();
}

aclnnStatus AddRmsNormAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::SetAclNNWorkspaceExecutor";

    aclTensor *x1 = aclnnVariantPack_.aclInTensors.at(x1AclTensorIndex_)->tensor;
    aclTensor *x2 = aclnnVariantPack_.aclInTensors.at(x2AclTensorIndex_)->tensor;
    aclTensor *gamma = aclnnVariantPack_.aclInTensors.at(gammaAclTensorIndex_)->tensor;
    double epsilon = static_cast<double>(param_.normParam.epsilon);
    aclTensor *yOut = aclnnVariantPack_.aclOutTensors.at(yOutAclTensorIndex_)->tensor;
    aclTensor *rstdOut = aclnnVariantPack_.aclOutTensors.at(rstdOutAclTensorIndex_)->tensor;
    aclTensor *xOut = aclnnVariantPack_.aclOutTensors.at(xOutAclTensorIndex_)->tensor;
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    aclnnStatus ret = aclnnAddRmsNormGetWorkspaceSizeFunc_(
        x1, x2, gamma, epsilon, yOut, rstdOut, xOut, &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
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

Status AddRmsNormAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::LaunchAclnnKernel";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnAddRmsNormFunc_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (ret == ACLNN_SUCCESS) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void AddRmsNormAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 3;   // x1, x2, gamma
    aclOutTensorNum_ = 3;  // yOut, rstdOut, xOut
}

void AddRmsNormAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;

    x1AclTensorIndex_ = 0;
    x2AclTensorIndex_ = 0;
    gammaAclTensorIndex_ = 0;
    yOutAclTensorIndex_ = 0;
    rstdOutAclTensorIndex_ = 0;
    xOutAclTensorIndex_ = 0;
}

Status AddRmsNormAclnnRunner::CreateX1AclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::CreateX1AclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, X1_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "x1 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    x1AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status AddRmsNormAclnnRunner::CreateX2AclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::CreateX2AclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, X2_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "x2 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    x2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status AddRmsNormAclnnRunner::CreateGammaAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::CreateGammaAclnnTensor";

    Tensor &atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    Dims xShape = aclnnVariantPack_.aclInTensors.at(x1AclTensorIndex_)->atbTensor.desc.shape;
    size_t startDim = 0;
    for (uint64_t i = 0; i < atbTensor.desc.shape.dimNum; i++) {
        if (atbTensor.desc.shape.dims[atbTensor.desc.shape.dimNum - i - 1] != xShape.dims[xShape.dimNum - i - 1]) {
            startDim = atbTensor.desc.shape.dimNum - i;
            break;
        }
    }
    if (startDim != 0) {
        atbTensor.desc.shape.dimNum -= startDim;
        for (uint64_t i = 0; i < atbTensor.desc.shape.dimNum; i++) {
            atbTensor.desc.shape.dims[i] = atbTensor.desc.shape.dims[i + startDim];
        }
    }
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, GAMMA_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    gammaAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status AddRmsNormAclnnRunner::CreateYOutAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::CreateYOutAclnnTensor";

    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, Y_OUT_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "yOut aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    yOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

Status AddRmsNormAclnnRunner::CreateRstdOutAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::CreateRstdOutAclnnTensor";

    uint64_t xDimNum = aclnnVariantPack_.aclInTensors.at(x1AclTensorIndex_)->atbTensor.desc.shape.dimNum;
    uint64_t gammaDimNum = aclnnVariantPack_.aclInTensors.at(gammaAclTensorIndex_)->atbTensor.desc.shape.dimNum;
    uint64_t xGammaDimNumDiff = (xDimNum < gammaDimNum) ? 0 : (xDimNum - gammaDimNum);
    Dims rstdOutAtbShape;
    rstdOutAtbShape.dimNum = xDimNum;
    for (uint64_t i = 0; i < rstdOutAtbShape.dimNum; i++) {
        if (i >= xGammaDimNumDiff) {
            rstdOutAtbShape.dims[i] = 1;
        } else {
            rstdOutAtbShape.dims[i] =
                aclnnVariantPack_.aclInTensors.at(x1AclTensorIndex_)->atbTensor.desc.shape.dims[i];
        }
    }
    SVector<int64_t> strides = GetCopyTensorStride(rstdOutAtbShape);
    size_t dataSize = 1;
    for (size_t i = 0; i < rstdOutAtbShape.dimNum; i++) {
        dataSize *= rstdOutAtbShape.dims[i];
    }
    dataSize *= FLOAT_SIZE;
    aclError ret = ACL_SUCCESS;
    if (rstdDeviceData_) {
        ret = aclrtFree(rstdDeviceData_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclrtFree rstdDeviceData_ failed, ret: " << ret;
            return ERROR_INTERNAL_ERROR;
        }
        rstdDeviceData_ = nullptr;
    }
    ret = aclrtMalloc(&rstdDeviceData_, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtMalloc rstdDeviceData_ failed, ret: " << ret;
        return ERROR_INTERNAL_ERROR;
    }
    aclTensor *rstdOutAclTensor = aclCreateTensor(rstdOutAtbShape.dims,
        rstdOutAtbShape.dimNum,
        aclDataType::ACL_FLOAT,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        rstdOutAtbShape.dims,
        rstdOutAtbShape.dimNum,
        rstdDeviceData_);
    if (!rstdOutAclTensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "rstdOut aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->tensorIdx = RSTD_OUT_ACLNN_TENSOR_IDX;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnTensorPtr->strides = strides;
    aclnnTensorPtr->tensor = rstdOutAclTensor;
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    rstdOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

Status AddRmsNormAclnnRunner::CreateXOutAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "AddRmsNormAclnnRunner::CreateXOutAclnnTensor";

    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, X_OUT_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "xOut aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    xOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

REG_RUNNER_TYPE(AddRmsNormAclnnRunner);
}  // namespace atb