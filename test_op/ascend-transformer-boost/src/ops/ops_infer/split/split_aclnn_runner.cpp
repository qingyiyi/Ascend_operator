/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "split_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const int SELF_ACLNN_TENSOR_IDX = 0;
static const int OUT_ACLNN_TENSOR_LIST_IDX = 0;
} // namespace

namespace atb {
AclnnSplitTensorGetWorkspaceSizeFunc SplitAclnnRunner::aclnnSplitTensorGetWorkspaceSizeFunc_ = nullptr;
AclnnSplitTensorFunc SplitAclnnRunner::aclnnSplitTensorFunc_ = nullptr;
AclnnSplitWithSizeGetWorkspaceSizeFunc SplitAclnnRunner::aclnnSplitWithSizeGetWorkspaceSizeFunc_ = nullptr;
AclnnSplitWithSizeFunc SplitAclnnRunner::aclnnSplitWithSizeFunc_ = nullptr;

SplitAclnnRunner::SplitAclnnRunner(const infer::SplitParam &param) : AclnnRunner("SplitAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::SplitAclnnRunner";

    splitWithSize_ = param_.splitSizes.size() > 0;
    GetTensorNum();
}

SplitAclnnRunner::~SplitAclnnRunner()
{
    if (splitSize_) {
        aclnnStatus ret = aclDestroyIntArray(splitSize_);
        if (ret != ACLNN_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "splitSize_ aclDestroyIntArray failed, ret: " << ret;
        }
        splitSize_ = nullptr;
    }
}

Status SplitAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "SplitAclnnRunner::LoadAclnnFuncs";

    Status st = NO_ERROR;
    if (!aclnnSplitWithSizeGetWorkspaceSizeFunc_ || !aclnnSplitWithSizeFunc_) {
        st = LoadFromSharedObjectFile("aclnnSplitWithSizeGetWorkspaceSize", "aclnnSplitWithSize",
                                      aclnnSplitWithSizeGetWorkspaceSizeFunc_, aclnnSplitWithSizeFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnSplitTensorGetWorkspaceSizeFunc_ || !aclnnSplitTensorFunc_) {
        st = LoadFromSharedObjectFile("aclnnSplitTensorGetWorkspaceSize", "aclnnSplitTensor",
                                      aclnnSplitTensorGetWorkspaceSizeFunc_, aclnnSplitTensorFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status SplitAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix()
                  << "SplitAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    Status st = CreateSelfAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    aclnnVariantPack_.aclOutTensorList.reserve(aclOutTensorListNum_);
    aclnnVariantPack_.aclOutTensorList.resize(aclOutTensorListNum_);
    return CreateOutAclnnTensorList();
}

aclnnStatus SplitAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::SetAclNNWorkspaceExecutor";
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(selfAclTensorIndex_)->tensor;
    aclTensorList *out = aclnnVariantPack_.aclOutTensorList.at(outAclTensorListIndex_);
    aclnnStatus ret = ACLNN_SUCCESS;
    int64_t dim = static_cast<int64_t>(param_.splitDim);
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    if (splitWithSize_) {
        ret = CreateSplitSizeAclIntArray();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = aclnnSplitWithSizeGetWorkspaceSizeFunc_(self, splitSize_, dim, out,
                                                      &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    } else {
        int32_t splitNum = param_.splitNum;
        Dims selfShape = aclnnVariantPack_.aclInTensors.at(selfAclTensorIndex_)->atbTensor.desc.shape;
        if (dim < 0) {
            dim += selfShape.dimNum;
        }
        uint64_t splitSections = static_cast<uint64_t>(selfShape.dims[dim] / splitNum);
        ret = aclnnSplitTensorGetWorkspaceSizeFunc_(self, splitSections, dim, out,
                                                    &(atbVariantPack_.workspaceBufferSize), &rawExecutorPtr);
    }

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

Status SplitAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::LaunchAclnnKernel";
    void *executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = ACLNN_SUCCESS;
    if (splitWithSize_) {
        ret = aclnnSplitWithSizeFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                      aclnnExecutor_.get(), executeStream);
    } else {
        ret = aclnnSplitTensorFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                    aclnnExecutor_.get(), executeStream);
    }
    if (ret == ACLNN_SUCCESS) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void SplitAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 1;      // self
    aclOutTensorListNum_ = 1; // out
}

void SplitAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorListIndex_ = 0;

    selfAclTensorIndex_ = 0;
    outAclTensorListIndex_ = 0;
}

Status SplitAclnnRunner::CreateSelfAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::CreateSelfAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, SELF_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "self aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    selfAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status SplitAclnnRunner::CreateOutAclnnTensorList()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::CreateOutAclnnTensorList";

    std::vector<aclTensor *> outTensors;
    size_t outTensorNum = atbVariantPack_.outTensors.size();
    outTensors.reserve(outTensorNum);
    outTensors.resize(outTensorNum);

    for (size_t i = 0; i < outTensorNum; i++) {
        Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
        SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
        std::shared_ptr<AclNNTensor> aclnnTensorPtr =
            CreateAclnnTensor(atbTensor, OUT_ACLNN_TENSOR_LIST_IDX, atbTensor.desc.shape, strides);
        if (!aclnnTensorPtr->tensor) {
            ATB_LOG(ERROR) << GetLogPrefix() << "out aclCreateTensor failed";
            return ERROR_INTERNAL_ERROR;
        }
        outTensors.at(i) = aclnnTensorPtr->tensor;
    }
    aclTensorList *outTensorList = aclCreateTensorList(outTensors.data(), outTensors.size());
    if (outTensorList) {
        aclnnVariantPack_.aclOutTensorList.at(aclOutTensorListIndex_) = outTensorList;
        outAclTensorListIndex_ = aclOutTensorListIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "out aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

aclnnStatus SplitAclnnRunner::CreateSplitSizeAclIntArray()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SplitAclnnRunner::CreateSplitSizeAclIntArray";

    if (splitSize_) {
        aclnnStatus ret = aclDestroyIntArray(splitSize_);
        if (ret != ACLNN_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "splitSize_ aclDestroyIntArray failed, ret: " << ret;
            return ret;
        }
        splitSize_ = nullptr;
    }
    std::vector<int64_t> splitSizeVec;
    splitSizeVec.reserve(param_.splitSizes.size());
    splitSizeVec.resize(param_.splitSizes.size());
    for (size_t i = 0; i < splitSizeVec.size(); i++) {
        splitSizeVec.at(i) = static_cast<int64_t>(param_.splitSizes.at(i));
    }
    splitSize_ = aclCreateIntArray(splitSizeVec.data(), splitSizeVec.size());
    if (splitSize_) {
        return ACLNN_SUCCESS;
    }
    ATB_LOG(ERROR) << "splitSize_ aclCreateIntArray failed!";
    return ACLNN_ERR_INNER;
}

REG_RUNNER_TYPE(SplitAclnnRunner);
} // namespace atb
