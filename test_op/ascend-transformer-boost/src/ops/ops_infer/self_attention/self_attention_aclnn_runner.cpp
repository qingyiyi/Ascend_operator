/*
 * Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "self_attention_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include <securec.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"

namespace {
static const int64_t INT_MAX_VALUE = 2147483647;

static const int QUERY_ACLNN_TENSOR_IDX = 0;
static const int KEY_ACLNN_TENSOR_IDX = 1;
static const int VALUE_ACLNN_TENSOR_IDX = 2;
static const int PSE_SHIFT_ACLNN_TENSOR_IDX = 3;
static const int ATTEN_MASK_ACLNN_TENSOR_IDX = 4;
static const int ATTENTION_OUT_ACLNN_TENSOR_IDX = 0;
}  // namespace

namespace atb {

AclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc
    SelfAttentionAclnnRunner::aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_ = nullptr;
AclnnFusedInferAttentionScoreV5Func SelfAttentionAclnnRunner::aclnnFusedInferAttentionScoreV5Func_ = nullptr;

SelfAttentionAclnnRunner::SelfAttentionAclnnRunner(const infer::SelfAttentionParam &opParam)
    : AclnnRunner("SelfAttentionAclnnRunner"), param_(opParam)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::SelfAttentionAclnnRunner";

    GetTensorNum();
    InitAclnnParam();
}

SelfAttentionAclnnRunner::~SelfAttentionAclnnRunner()
{}

Status SelfAttentionAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::LoadMethod";

    if (aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_ && aclnnFusedInferAttentionScoreV5Func_) {
        return NO_ERROR;
    }
    return LoadFromSharedObjectFile("aclnnFusedInferAttentionScoreV5GetWorkspaceSize",
        "aclnnFusedInferAttentionScoreV5",
        aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_,
        aclnnFusedInferAttentionScoreV5Func_);
}

Status SelfAttentionAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: "
                  << runnerVariantPack.ToString();

    atbVariantPack_ = runnerVariantPack;
    InitTensorIndex();
    size_t seqLenIndex = 0;
    if (param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        seqLenIndex = 4;
    } else {
        seqLenIndex = 3;
    }
    batch_ = atbVariantPack_.inTensors.at(seqLenIndex)
                 .desc.shape.dims[atbVariantPack_.inTensors.at(seqLenIndex).desc.shape.dimNum - 1];
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclInTensorList.reserve(aclInTensorListNum_);
    aclnnVariantPack_.aclInTensorList.resize(aclInTensorListNum_);
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
    if (!isDecoder_) {
        // Prefill need mask
        switch (param_.maskType) {
            case infer::SelfAttentionParam::MASK_TYPE_ALIBI:
                st = CreatePseShiftAclnnTensor();
                if (st != NO_ERROR) {
                    return st;
                }
                break;
            case infer::SelfAttentionParam::MASK_TYPE_NORM:
            case infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS:
                st = CreateAttenMaskAclnnTensor();
                if (st != NO_ERROR) {
                    return st;
                }
                break;
            default:
                break;
        }
    } else if (param_.maskType != infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        atbInTensorIndex_++;
    }
    st = CreateActualSeqLengthsAclIntArray();
    if (st != NO_ERROR) {
        return st;
    }

    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensorList.reserve(aclOutTensorListNum_);
    aclnnVariantPack_.aclOutTensorList.resize(aclOutTensorListNum_);
    return CreateAttentionOutAclnnTensor();
}

aclnnStatus SelfAttentionAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::SetAclNNWorkspaceExecutor";

    aclTensor *query = aclnnVariantPack_.aclInTensors.at(queryAclTensorIndex_)->tensor;
    aclTensorList *key = aclnnVariantPack_.aclInTensorList.at(keyAclTensorListIndex_);
    aclTensorList *value = aclnnVariantPack_.aclInTensorList.at(valueAclTensorListIndex_);
    aclTensor *pseShift = nullptr;
    aclTensor *attenMask = nullptr;
    if (!isDecoder_) {
        if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI) {
            pseShift = aclnnVariantPack_.aclInTensors.at(pseShiftAclTensorIndex_)->tensor;
        }
        if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM ||
            param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
            attenMask = aclnnVariantPack_.aclInTensors.at(attenMaskAclTensorIndex_)->tensor;
        }
    }

    aclTensor *attentionOut = aclnnVariantPack_.aclOutTensors.at(attentionOutAclTensorIndex_)->tensor;

    std::string inputLayoutStr = aclnnParam_.inputLayoutStr;
    auto inputLayout = std::make_unique<char[]>(inputLayoutStr.length() + 1);
    errno_t err = strcpy_s(inputLayout.get(), inputLayoutStr.length() + 1, inputLayoutStr.c_str());
    if (err != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inputLayout strcpy_s failed";
        return ERROR_INVALID_PARAM;
    }
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_(
        query, key, value, pseShift, attenMask, actualSeqLengths_,
        actualSeqLengths_, // actualSeqLengthsKv
        nullptr,           // deqScale1
        nullptr,           // quantScale1
        nullptr,           // deqScale2
        nullptr,           // quantScale2
        nullptr,           // quantOffset2
        nullptr,           // antiquantScale
        nullptr,           // antiquantOffset
        nullptr,           // blockTable
        nullptr,           // queryPaddingSize
        nullptr,           // kvPaddingSize
        nullptr,           // keyAntiquantScale
        nullptr,           // keyAntiquantOffset
        nullptr,           // valueAntiquantScale
        nullptr,           // valueAntiquantOffset
        nullptr,           // keySharedPrefix
        nullptr,           // valueSharedPrefix
        nullptr,           // actualSharedPrefixLen
        nullptr,           // queryRope
        nullptr,           // keyRope
        nullptr,           // keyRopeAntiquantScale
        nullptr,           // dequantScaleQuery
        nullptr,           // learnableSink
        nullptr,           // qStartIdx
        nullptr,           // kvStartIdx
        aclnnParam_.numHeads, aclnnParam_.scaleValue, aclnnParam_.preTokens, aclnnParam_.nextTokens, inputLayout.get(),
        aclnnParam_.numKeyValueHeads, aclnnParam_.sparseMode, aclnnParam_.innerPrecise, aclnnParam_.blockSize,
        aclnnParam_.antiquantMode, aclnnParam_.softmaxLseFlag, aclnnParam_.keyAntiquantMode,
        aclnnParam_.valueAntiquantMode, aclnnParam_.queryQuantMode, aclnnParam_.pseType, attentionOut,
        nullptr, // softmaxLse
        &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    if (ret == ACL_SUCCESS) {
        ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "SetAclNNWorkspaceExecutor failed, ret: " << ret;
    }
    return ret;
}

Status SelfAttentionAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "SelfAttentionAclnnRunner::LaunchAclnnKernel";

    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnFusedInferAttentionScoreV5Func_(
        atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(), executeStream);
    if (actualSeqLengths_ != nullptr) {
        aclnnStatus ret = aclDestroyIntArray(actualSeqLengths_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "actualSeqLengths_ aclDestroyIntArray failed";
            return ERROR_INTERNAL_ERROR;
        }
        actualSeqLengths_ = nullptr;
    }
    if (ret == ACL_SUCCESS) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "LaunchAclnnKernel failed, ret: " << ret;
    return ERROR_CANN_ERROR;
}

void SelfAttentionAclnnRunner::GetTensorNum()
{
    aclInTensorNum_ = 3;  // 1: query, key, value
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI) {
        aclInTensorNum_ += 1;  // 1: pseShift
    }
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM ||
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
        aclInTensorNum_ += 1;  // 1: attenMask
    }
    aclOutTensorNum_ = 1;
    aclInTensorListNum_ = 2;  // 2: key, value
}

void SelfAttentionAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    aclInTensorListIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;
    aclOutTensorListIndex_ = 0;

    queryAclTensorIndex_ = 0;
    keyAclTensorListIndex_ = 0;
    valueAclTensorListIndex_ = 0;
    pseShiftAclTensorIndex_ = 0;
    attenMaskAclTensorIndex_ = 0;
    attentionOutAclTensorIndex_ = 0;
}

void SelfAttentionAclnnRunner::InitAclnnParam()
{
    aclnnParam_.numHeads = param_.headNum;
    aclnnParam_.scaleValue = param_.qkScale;
    aclnnParam_.preTokens = INT_MAX_VALUE;
    aclnnParam_.nextTokens = INT_MAX_VALUE;
    if (param_.inputLayout == infer::TYPE_BSND) {
        aclnnParam_.inputLayoutStr = "TND";
    } else if (param_.inputLayout == infer::TYPE_BNSD) {
        aclnnParam_.inputLayoutStr = "BNSD";
    }
    aclnnParam_.numKeyValueHeads = param_.kvHeadNum == 0 ? param_.headNum : param_.kvHeadNum;
    aclnnParam_.sparseMode = 0;
}

Status SelfAttentionAclnnRunner::CreateQueryAclnnTensor()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreateQueryAclnnTensor";

    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, QUERY_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    // [B*S, N, D] reduced to [B*1, N, D]
    isDecoder_ = storageShape.dims[0] == batch_;
    Dims viewShape;
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI) {
        viewShape.dimNum = 4;  // 4: [B, S, N, D]
        viewShape.dims[0] = batch_;
        viewShape.dims[1] = storageShape.dims[0] / batch_;
        viewShape.dims[2] = param_.headNum;
        if (storageShape.dimNum == 2) {
            viewShape.dims[3] = storageShape.dims[1] / param_.headNum;
        } else if (storageShape.dimNum == 3) {
            viewShape.dims[3] = storageShape.dims[2];  // 3: D(head_size); 2: head_size
        }
        aclnnParam_.inputLayoutStr = "BSND";
    } else {
        viewShape.dimNum = storageShape.dimNum;
        for (uint64_t i = 0; i < viewShape.dimNum; i++) {
            viewShape.dims[i] = storageShape.dims[i];
        }
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        storageShape.dims,
        storageShape.dimNum,
        atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "query aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    queryAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status SelfAttentionAclnnRunner::CreateKeyAclnnTensorList()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreateKeyAclnnTensorList";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, KEY_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI) {
        viewShape.dimNum = 4;  // 4: [B, S, N, D]
        viewShape.dims[0] = batch_;
        viewShape.dims[1] = storageShape.dims[0] / batch_;
        viewShape.dims[2] = param_.kvHeadNum;
        if (storageShape.dimNum == 2) {
            viewShape.dims[3] = storageShape.dims[1] / param_.kvHeadNum;
        } else if (storageShape.dimNum == 3) {
            viewShape.dims[3] = storageShape.dims[2];  // 3: D(head_size); 2: head_size
        }
    } else {
        viewShape.dimNum = storageShape.dimNum;
        for (uint64_t i = 0; i < viewShape.dimNum; i++) {
            viewShape.dims[i] = storageShape.dims[i];
        }
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        storageShape.dims,
        storageShape.dimNum,
        atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    aclTensor *keyTensors[1] = {aclnnTensorPtr->tensor};
    auto keyTensorList = aclCreateTensorList(keyTensors, 1);
    if (keyTensorList) {
        aclnnVariantPack_.aclInTensorList.at(aclInTensorListIndex_) = keyTensorList;
        keyAclTensorListIndex_ = aclInTensorListIndex_++;
        aclInTensorIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "key aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

Status SelfAttentionAclnnRunner::CreateValueAclnnTensorList()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreateValueAclnnTensorList";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, VALUE_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI) {
        viewShape.dimNum = 4;  // 4: [B, S, N, D]
        viewShape.dims[0] = batch_;
        viewShape.dims[1] = storageShape.dims[0] / batch_;
        viewShape.dims[2] = param_.kvHeadNum;
        if (storageShape.dimNum == 2) {
            viewShape.dims[3] = storageShape.dims[1] / param_.kvHeadNum;
        } else if (storageShape.dimNum == 3) {
            viewShape.dims[3] = storageShape.dims[2];  // 3: D(head_size); 2: head_size
        }
    } else {
        viewShape.dimNum = storageShape.dimNum;
        for (uint64_t i = 0; i < viewShape.dimNum; i++) {
            viewShape.dims[i] = storageShape.dims[i];
        }
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        storageShape.dims,
        storageShape.dimNum,
        atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "value aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    aclTensor *valueTensors[1] = {aclnnTensorPtr->tensor};
    auto valueTensorList = aclCreateTensorList(valueTensors, 1);
    if (valueTensorList) {
        aclnnVariantPack_.aclInTensorList.at(aclInTensorListIndex_) = valueTensorList;
        valueAclTensorListIndex_ = aclInTensorListIndex_++;
        aclInTensorIndex_++;
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << GetLogPrefix() << "value aclCreateTensorList failed";
    return ERROR_INTERNAL_ERROR;
}

Status SelfAttentionAclnnRunner::CreatePseShiftAclnnTensor()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreatePseShiftAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, PSE_SHIFT_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    viewShape.dimNum = 4;  // 4: [batch, headNum, maxSeqLen, maxSeqLen]
    if (storageShape.dimNum == 4) {
        viewShape.dims[0] = storageShape.dims[0];
        viewShape.dims[1] = storageShape.dims[1];
        viewShape.dims[2] = storageShape.dims[2];  // 2: maxSeqLen
        viewShape.dims[3] = storageShape.dims[3];  // 3: maxSeqLen
    }
    if (storageShape.dimNum == 3) {
        viewShape.dims[0] = 1;
        viewShape.dims[1] = storageShape.dims[0];
        viewShape.dims[2] = storageShape.dims[1];  // 2: maxSeqLen
        viewShape.dims[3] = storageShape.dims[2];  // 3: maxSeqLen
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        storageShape.dims,
        storageShape.dimNum,
        atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "pseShift aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    pseShiftAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status SelfAttentionAclnnRunner::CreateAttenMaskAclnnTensor()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreateAttenMaskAclnnTensor";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, ATTEN_MASK_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    // only prefill need mask
    if (!isDecoder_ && storageShape.dims[1] == 2048 && param_.isTriuMask) {
        aclnnParam_.sparseMode = 2;  // 2: leftUpCausal mask
    } else if (storageShape.dimNum == 2) { // 2: [maxQSeqlen, maxKVSeqlen]
        storageShape.dimNum = 3; // 3: [1, maxQSeqlen, maxKVSeqlen]
        for (int64_t i = storageShape.dimNum; i > 0; --i) {
            storageShape.dims[i] = storageShape.dims[i-1];
        }
        storageShape.dims[0] = 1;
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(storageShape);
    aclnnTensorPtr->tensor = aclCreateTensor(storageShape.dims,
        storageShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        storageShape.dims,
        storageShape.dimNum,
        atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "attenMask aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    attenMaskAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status SelfAttentionAclnnRunner::CreateActualSeqLengthsAclIntArray()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreateActualSeqLengthsAclIntArray";
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    if (atbTensor.hostData == nullptr) {
        ATB_LOG(ERROR) << "contextLen tensor host data is null";
        return ERROR_INVALID_TENSOR_ADDR;
    }
    if (actualSeqLengths_) {
        aclnnStatus ret = aclDestroyIntArray(actualSeqLengths_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "actualSeqLengths aclDestroyIntArray failed";
            return ERROR_INTERNAL_ERROR;
        }
        actualSeqLengths_ = nullptr;
    }
    std::vector<int32_t> contextLensInt32;
    uint64_t dataSize = atbTensor.dataSize / 4;  // 4: int32 size
    contextLensInt32.reserve(dataSize);
    contextLensInt32.resize(dataSize);
    if (memcpy_s(contextLensInt32.data(), atbTensor.dataSize, atbTensor.hostData, atbTensor.dataSize) != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "contextLens memcpy_s failed";
        return ERROR_INTERNAL_ERROR;
    }
    std::vector<int64_t> contextLensInt64;
    contextLensInt64.reserve(dataSize);
    contextLensInt64.resize(dataSize);
    contextLensInt64.at(0) = static_cast<int64_t>(contextLensInt32.at(0));
    for (uint64_t i = 1; i < dataSize; i++) {
        if (aclnnParam_.inputLayoutStr == "TND") {
            contextLensInt64.at(i) = static_cast<int64_t>(contextLensInt32.at(i)) + contextLensInt64.at(i - 1);
        } else {
            contextLensInt64.at(i) = static_cast<int64_t>(contextLensInt32.at(i));
        }
    }
    actualSeqLengths_ = aclCreateIntArray(contextLensInt64.data(), dataSize);
    if (actualSeqLengths_) {
        return NO_ERROR;
    }
    ATB_LOG(ERROR) << "actualSeqLengths_ aclCreateIntArray failed!";
    return ERROR_INTERNAL_ERROR;
}

Status SelfAttentionAclnnRunner::CreateAttentionOutAclnnTensor()
{
    ATB_LOG(INFO) << "SelfAttentionAclnnRunner::CreateAttentionOutAclnnTensor";
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, ATTENTION_OUT_ACLNN_TENSOR_IDX);
    Dims storageShape = atbTensor.desc.shape;
    Dims viewShape;
    if (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI) {
        viewShape.dimNum = 4;  // 4: [B, S, N, D]
        viewShape.dims[0] = batch_;
        viewShape.dims[1] = storageShape.dims[0] / batch_;
        viewShape.dims[2] = param_.headNum;
        if (storageShape.dimNum == 2) {
            viewShape.dims[3] = storageShape.dims[1] / param_.headNum;
        } else if (storageShape.dimNum == 3) {
            viewShape.dims[3] = storageShape.dims[2];  // 3: D(head_size); 2: head_size
        }
    } else {
        viewShape.dimNum = storageShape.dimNum;
        for (uint64_t i = 0; i < viewShape.dimNum; i++) {
            viewShape.dims[i] = storageShape.dims[i];
        }
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(viewShape.dims,
        viewShape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensorPtr->strides.data(),
        0,
        atbTensor.desc.format,
        storageShape.dims,
        storageShape.dimNum,
        atbTensor.deviceData);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "attentionOut aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    attentionOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

std::shared_ptr<AclNNTensor> SelfAttentionAclnnRunner::InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = aclnnTensorIndex;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(SelfAttentionAclnnRunner);
}  // namespace atb
