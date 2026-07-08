/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "paged_attention_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include <securec.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const int64_t INT_MAX_VALUE = 2147483647;

static const int QUERY_ACLNN_TENSOR_IDX = 0;
static const int KEY_ACLNN_TENSOR_IDX = 0;   // V5: key passed via tensorList, tensorIdx is always 0
static const int VALUE_ACLNN_TENSOR_IDX = 0; // V5: value passed via tensorList, tensorIdx is always 0
static const int BLOCK_TABLE_ACLNN_TENSOR_IDX = 14;
static const int ATTENTION_OUT_ACLNN_TENSOR_IDX = 0;
} // namespace

namespace atb {
AclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc
    PagedAttentionAclnnRunner::aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_ = nullptr;
AclnnFusedInferAttentionScoreV5Func PagedAttentionAclnnRunner::aclnnFusedInferAttentionScoreV5Func_ = nullptr;

PagedAttentionAclnnRunner::PagedAttentionAclnnRunner(const infer::PagedAttentionParam &param)
    : AclnnRunner("PagedAttentionAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::PagedAttentionAclnnRunner";
}

PagedAttentionAclnnRunner::~PagedAttentionAclnnRunner() {}

Status PagedAttentionAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::LoadAclnnFuncs";
    if (aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_ && aclnnFusedInferAttentionScoreV5Func_) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile(
        "aclnnFusedInferAttentionScoreV5GetWorkspaceSize", "aclnnFusedInferAttentionScoreV5",
        aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_, aclnnFusedInferAttentionScoreV5Func_);
}

Status PagedAttentionAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    GetTensorNum();
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclInTensorList.reserve(aclInTensorListNum_);
    aclnnVariantPack_.aclInTensorList.resize(aclInTensorListNum_);
    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    Status st = CreateQueryAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateKeyAclnnTensorList();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateValueAclnnTensorList();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateBlockTableAclnnTensor();
    if (st != NO_ERROR) {
        return st;
    }
    st = CreateActualSeqLengthsKvAclIntArray();
    if (st != NO_ERROR) {
        return st;
    }
    return CreateAttentionOutAclnnTensor();
}

aclnnStatus PagedAttentionAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::SetAclNNWorkspaceExecutor";

    aclTensor *query = aclnnVariantPack_.aclInTensors.at(queryAclTensorIndex_)->tensor;
    aclTensorList *key = aclnnVariantPack_.aclInTensorList.at(keyAclTensorListIndex_);
    aclTensorList *value = aclnnVariantPack_.aclInTensorList.at(valueAclTensorListIndex_);
    aclTensor *pseShiftOptional = nullptr;
    aclTensor *attenMaskOptional = nullptr;
    aclIntArray *actualSeqLengthsOptional = nullptr;
    aclTensor *deqScale1Optional = nullptr;
    aclTensor *quantScale1Optional = nullptr;
    aclTensor *deqScale2Optional = nullptr;
    aclTensor *quantScale2Optional = nullptr;
    aclTensor *quantOffset2Optional = nullptr;
    aclTensor *antiquantScaleOptional = nullptr;
    aclTensor *antiquantOffsetOptional = nullptr;
    aclTensor *blockTableOptional = aclnnVariantPack_.aclInTensors.at(blockTableAclTensorIndex_)->tensor;
    aclTensor *queryPaddingSizeOptional = nullptr;
    aclTensor *kvPaddingSizeOptional = nullptr;
    aclTensor *keyAntiquantScaleOptional = nullptr;
    aclTensor *keyAntiquantOffsetOptional = nullptr;
    aclTensor *valueAntiquantScaleOptional = nullptr;
    aclTensor *valueAntiquantOffsetOptional = nullptr;
    aclTensor *keySharedPrefixOptional = nullptr;
    aclTensor *valueSharedPrefixOptional = nullptr;
    aclIntArray *actualSharedPrefixLenOptional = nullptr;
    aclTensor *queryRopeOptional = nullptr;
    aclTensor *keyRopeOptional = nullptr;
    aclTensor *keyRopeAntiquantScaleOptional = nullptr;
    aclTensor *dequantScaleQueryOptional = nullptr;
    aclTensor *learnableSinkOptional = nullptr;
    aclIntArray *qStartIdxOptional = nullptr;
    aclIntArray *kvStartIdxOptional = nullptr;
    int64_t numHeads = param_.headNum;
    double scaleValue = static_cast<double>(param_.qkScale);
    int64_t preTokens = INT_MAX_VALUE;
    int64_t nextTokens = INT_MAX_VALUE;
    char inputLayout[] = "BSND";
    int64_t numKeyValueHeads = param_.kvHeadNum;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    int64_t queryQuantMode = 0;
    int64_t pseType = 0;
    aclTensor *attentionOut = aclnnVariantPack_.aclOutTensors.at(attentionOutAclTensorIndex_)->tensor;
    aclTensor *softmaxLse = nullptr;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_(
        query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKv_,
        deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional, quantOffset2Optional,
        antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional,
        kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional,
        valueAntiquantOffsetOptional, keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenOptional,
        queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
        learnableSinkOptional, qStartIdxOptional, kvStartIdxOptional, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize_, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, attentionOut, softmaxLse,
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
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

Status PagedAttentionAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "PagedAttentionAclnnRunner::LaunchAclnnKernel";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnFusedInferAttentionScoreV5Func_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (actualSeqLengthsKv_) {
        aclnnStatus destroyRet = aclDestroyIntArray(actualSeqLengthsKv_);
        if (destroyRet != ACLNN_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "actualSeqLengthsKv aclDestroyIntArray failed";
            return ERROR_INTERNAL_ERROR;
        }
        actualSeqLengthsKv_ = nullptr;
    }
    if (ret == ACLNN_SUCCESS) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void PagedAttentionAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 4; // 4: query, key, value, blockTable
    aclOutTensorNum_ = 1;
    aclInTensorListNum_ = 3; // 3: query (place holder), key, value
}

void PagedAttentionAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    aclInTensorListIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;

    queryAclTensorIndex_ = 0;
    keyAclTensorListIndex_ = 0;
    valueAclTensorListIndex_ = 0;
    blockTableAclTensorIndex_ = 0;
    attentionOutAclTensorIndex_ = 0;
}

Status PagedAttentionAclnnRunner::CreateQueryAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateQueryAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 4; // 4: [B, S, N, D]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = 1;
    viewShape.dims[2] = storageShape.dims[1]; // 2: N(head_num)
    viewShape.dims[3] = storageShape.dims[2]; // 3: D(head_size); 2: head_size
    SVector<int64_t> strides = GetCopyTensorStride(viewShape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, QUERY_ACLNN_TENSOR_IDX, viewShape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "query aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    queryAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status PagedAttentionAclnnRunner::CreateKeyAclnnTensorList()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateKeyAclnnTensorList";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    Dims storageShape = atbTensor.desc.shape;
    blockSize_ = storageShape.dims[1];
    Dims viewShape;
    viewShape.dimNum = 3; // 3: [B, S, H]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = storageShape.dims[1];
    viewShape.dims[2] = storageShape.dims[2] * storageShape.dims[3]; // 2: H(hidden_size); 2: head_num; 3: head_size
    SVector<int64_t> strides = GetCopyTensorStride(viewShape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, KEY_ACLNN_TENSOR_IDX, viewShape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclInTensorListIndex_++;
    aclnnTensorPtr->tensorListidx = aclInTensorListIndex_;
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_++) = aclnnTensorPtr;
    aclTensor *keyAclTensors[1] = {aclnnTensorPtr->tensor};
    auto keyAclTensorList = aclCreateTensorList(keyAclTensors, 1);
    if (keyAclTensorList) {
        aclnnVariantPack_.aclInTensorList.at(aclInTensorListIndex_) = keyAclTensorList;
        keyAclTensorListIndex_ = aclInTensorListIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

Status PagedAttentionAclnnRunner::CreateValueAclnnTensorList()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateValueAclnnTensorList";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 3; // 3: [B, S, H]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = storageShape.dims[1];
    viewShape.dims[2] = storageShape.dims[2] * storageShape.dims[3]; // 2: H(hidden_size); 2: head_num; 3: head_size
    SVector<int64_t> strides = GetCopyTensorStride(viewShape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, VALUE_ACLNN_TENSOR_IDX, viewShape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "value aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnTensorPtr->tensorListidx = aclInTensorListIndex_;
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_++) = aclnnTensorPtr;
    aclTensor *valueAclTensors[1] = {aclnnTensorPtr->tensor};
    auto valueAclTensorList = aclCreateTensorList(valueAclTensors, 1);
    if (valueAclTensorList) {
        aclnnVariantPack_.aclInTensorList.at(aclInTensorListIndex_) = valueAclTensorList;
        valueAclTensorListIndex_ = aclInTensorListIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "value aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

Status PagedAttentionAclnnRunner::CreateBlockTableAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateBlockTableAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    SVector<int64_t> strides = GetCopyTensorStride(atbTensor.desc.shape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, BLOCK_TABLE_ACLNN_TENSOR_IDX, atbTensor.desc.shape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "blockTable aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    blockTableAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status PagedAttentionAclnnRunner::CreateActualSeqLengthsKvAclIntArray()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateActualSeqLengthsKvAclIntArray";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    if (atbTensor.hostData == nullptr) {
        ATB_LOG(ERROR) << "contextLen tensor host data is null";
        return ERROR_INVALID_TENSOR_ADDR;
    }
    if (actualSeqLengthsKv_) {
        aclError ret = aclDestroyIntArray(actualSeqLengthsKv_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "actualSeqLengthsKv aclDestroyIntArray failed";
            return ERROR_INTERNAL_ERROR;
        }
        actualSeqLengthsKv_ = nullptr;
    }
    SVector<int32_t> contextLensInt32;
    uint64_t dataSize = atbTensor.dataSize / sizeof(int32_t);
    batch_ = dataSize;
    contextLensInt32.reserve(dataSize);
    contextLensInt32.resize(dataSize);
    if (memcpy_s(contextLensInt32.data(), atbTensor.dataSize, atbTensor.hostData, atbTensor.dataSize) != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "contextLens memcpy_s failed";
        return ERROR_INTERNAL_ERROR;
    }
    std::vector<int64_t> contextLensInt64;
    contextLensInt64.reserve(dataSize);
    contextLensInt64.resize(dataSize);
    for (uint64_t i = 0; i < dataSize; i++) {
        contextLensInt64.at(i) = static_cast<int64_t>(contextLensInt32.at(i));
    }

    ATB_LOG(INFO) << GetLogPrefix() << "kv seqlen, batch: " << batch_;

    actualSeqLengthsKv_ = aclCreateIntArray(contextLensInt64.data(), dataSize);
    if (actualSeqLengthsKv_) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << "actualSeqLengthsKv aclCreateIntArray failed!";
    return ERROR_INTERNAL_ERROR;
}

Status PagedAttentionAclnnRunner::CreateAttentionOutAclnnTensor()
{
    ATB_LOG(INFO) << "PagedAttentionAclnnRunner::CreateAttentionOutAclnnTensor";
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 4; // 4: [B, S, N, D]
    viewShape.dims[0] = storageShape.dims[0];
    viewShape.dims[1] = 1;
    viewShape.dims[2] = storageShape.dims[1]; // 2: N(head_num)
    viewShape.dims[3] = storageShape.dims[2]; // 3: D(head_size); 2: head_size

    SVector<int64_t> strides = GetCopyTensorStride(viewShape);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr =
        CreateAclnnTensor(atbTensor, ATTENTION_OUT_ACLNN_TENSOR_IDX, viewShape, strides);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "attentionOut aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    attentionOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

REG_RUNNER_TYPE(PagedAttentionAclnnRunner);
} // namespace atb
