/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reshape_and_cache_siso_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "acl/acl.h"
#include "atbops/params/params.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const int KEY_ACLNN_TENSOR_IDX = 0;
static const int KEY_CACHE_ACLNN_TENSOR_IDX = 1;
static const int SLOT_MAPPING_ACLNN_TENSOR_IDX = 2;
}  // namespace

namespace atb {
AclnnScatterPaCacheGetWorkspaceSizeFunc ReshapeAndCacheSisoAclnnRunner::aclnnScatterPaCacheGetWorkspaceSizeFunc_ =
    nullptr;
AclnnScatterPaCacheFunc ReshapeAndCacheSisoAclnnRunner::aclnnScatterPaCacheFunc_ = nullptr;

ReshapeAndCacheSisoAclnnRunner::ReshapeAndCacheSisoAclnnRunner(const infer::ReshapeAndCacheParam &param)
    : AclnnRunner("ReshapeAndCacheSisoAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::ReshapeAndCacheSisoAclnnRunner";
}

ReshapeAndCacheSisoAclnnRunner::~ReshapeAndCacheSisoAclnnRunner()
{}

Status ReshapeAndCacheSisoAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "ReshapeAndCacheSisoAclnnRunner::LoadAclnnFuncs";

    if (aclnnScatterPaCacheGetWorkspaceSizeFunc_ && aclnnScatterPaCacheFunc_) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnScatterPaCacheGetWorkspaceSize",
        "aclnnScatterPaCache",
        aclnnScatterPaCacheGetWorkspaceSizeFunc_,
        aclnnScatterPaCacheFunc_);
}

Status ReshapeAndCacheSisoAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    GetTensorNum();
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    Status st = CreateKeyAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateKeyCacheAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    return CreateSlotMappingAclnnTensor();
}

aclnnStatus ReshapeAndCacheSisoAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::SetAclNNWorkspaceExecutor";
    aclTensor *key = aclnnVariantPack_.aclInTensors.at(keyAclTensorIndex_)->tensor;
    aclTensor *keyCacheRef = aclnnVariantPack_.aclInTensors.at(keyCacheRefAclTensorIndex_)->tensor;
    aclTensor *slotMapping = aclnnVariantPack_.aclInTensors.at(slotMappingAclTensorIndex_)->tensor;
    aclTensor *compressLensOptional = nullptr;
    aclTensor *compressSeqOffsetOptional = nullptr;
    aclTensor *seqLensOptional = nullptr;
    char *cacheMode = nullptr;
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnScatterPaCacheGetWorkspaceSizeFunc_(key,
        keyCacheRef,
        slotMapping,
        compressLensOptional,
        compressSeqOffsetOptional,
        seqLensOptional,
        cacheMode,
        &(atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr);
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

Status ReshapeAndCacheSisoAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::LaunchAclnnKernel";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnScatterPaCacheFunc_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (ret == ACLNN_SUCCESS) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void ReshapeAndCacheSisoAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 3;  // key, keyCacheRef, slotMapping
}

void ReshapeAndCacheSisoAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;

    keyAclTensorIndex_ = 0;
    keyCacheRefAclTensorIndex_ = 0;
    slotMappingAclTensorIndex_ = 0;
}

Status ReshapeAndCacheSisoAclnnRunner::CreateKeyAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::CreateKeyAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, KEY_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    keyAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status ReshapeAndCacheSisoAclnnRunner::CreateKeyCacheAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::CreateKeyCacheAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, KEY_CACHE_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "keyCache aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    keyCacheRefAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status ReshapeAndCacheSisoAclnnRunner::CreateSlotMappingAclnnTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "ReshapeAndCacheSisoAclnnRunner::CreateSlotMappingAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, SLOT_MAPPING_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "slotMapping aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    slotMappingAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

REG_RUNNER_TYPE(ReshapeAndCacheSisoAclnnRunner);
}  // namespace atb
