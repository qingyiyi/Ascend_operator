/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "self_attention_fusion_bypass_ops_runner_BNSD.h"
#include <cmath>
#include "asdops/params/params.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/tensor_util.h"
#include "param.h"
#include "self_attention_runner_utils.h"

namespace {
static constexpr uint32_t VALUE_TENSOR_POS = 2;
static constexpr uint32_t INTENSOR_BASE_SIZE = 6;
static constexpr uint32_t KERNEL_GRAPH_BASE_SIZE = 1;
static constexpr uint32_t FA_NODE_BASE_IDX = 0;
} // namespace

namespace atb {

void SelfAttentionFusionViewBypassFuncDecoder(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    newDims = {oldDims.at(0), oldDims.at(1) * oldDims.at(2) * oldDims.at(3)}; // 2, 3: 设置新张量形状
}

void FlashAttentionInferShapeBypassBNSDPreFunc(Mki::LaunchParam &launchParam)
{
    launchParam.GetInTensor(3).desc.dtype = Mki::TENSOR_DTYPE_UINT32; // 3: 设置第四个输入张量的dtype
}

SelfAttentionFusionBypassOpsRunnerBNSD::SelfAttentionFusionBypassOpsRunnerBNSD(const infer::SelfAttentionParam &param)
    : OpsRunner("SelfAttentionFusionBypassOpsRunnerBNSD"), param_(param)
{
    skipSetUpKernelGraphWhenCacheHit_ = false;
    needKernelGraphModify_ = true;
    ATB_LOG(INFO) << "SelfAttentionFusionBypassOpsRunnerBNSD::SelfAttentionFusionBypassOpsRunnerBNSD called";

    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED);
    bool isMaskAlibiCompress = (param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
                                param_.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    bool needQScale = NeedElewiseMulsQScale(param_);
    size_t intensorSize = INTENSOR_BASE_SIZE;
    if (needMask) {
        intensorSize++;
    }
    if (param_.batchRunStatusEnable) {
        intensorSize++;
    }
    if (isMaskAlibiCompress) {
        intensorSize++;
    }
    kernelGraph_.inTensors.resize(intensorSize);
    size_t tensorId = 0;
    Mki::Tensor *mixedQuery = &kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheK = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor &cacheV = kernelGraph_.inTensors.at(tensorId++);
    Mki::Tensor *attentionMask = needMask ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;
    Mki::Tensor &tokenOffset = kernelGraph_.inTensors.at(tensorId++);
    (void)tokenOffset;
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(tensorId++);
    (void)seqLen;
    Mki::Tensor &layerId = kernelGraph_.inTensors.at(tensorId++);
    if (param_.batchRunStatusEnable) {
        tensorId++;
    }
    Mki::Tensor *slopes = isMaskAlibiCompress ? &kernelGraph_.inTensors.at(tensorId++) : &nullTensor_;

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &context = kernelGraph_.outTensors.at(0);
    Mki::Tensor *divOut = mixedQuery;
    kernelGraph_.nodes.resize(needQScale ? KERNEL_GRAPH_BASE_SIZE + 1 : KERNEL_GRAPH_BASE_SIZE); // 1: ElewiseMuls
    size_t nodeId = 0;

    if (needQScale) {
        kernelGraph_.internalTensors.resize(1);       // 1: divOut
        divOut = &kernelGraph_.internalTensors.at(0); // 0: qScale ElewiseMuls
        // muls
        auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
        mulsQNode.opDesc = {0, "ElewiseOperation",
                            AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.qScale})};
        mulsQNode.inTensors = {mixedQuery};
        mulsQNode.outTensors = {divOut};
    }

    // mix.h type=UNPAD_FLASH_ATTENTION
    auto &flashAttentionNode = kernelGraph_.nodes.at(nodeId++);
    AtbOps::OpParam::UnpadFlashAttention flashAttentionQParam;
    SetFAParam(flashAttentionQParam);

    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", flashAttentionQParam};
    ATB_LOG(INFO) << "flashAttentionQParam.type: " << flashAttentionQParam.type;
    flashAttentionNode.inTensors = {divOut,        &cacheK,      &cacheV,      &layerId,
                                    attentionMask, slopes,       &nullTensor_, &nullTensor_,
                                    &nullTensor_,  &nullTensor_, &nullTensor_, &nullTensor_};
    flashAttentionNode.outTensors = {&context};
    if (param_.calcType == atb::infer::SelfAttentionParam::DECODER) {
        flashAttentionNode.inTensorViewFuncs.resize(flashAttentionNode.inTensors.size());
        flashAttentionNode.inTensorViewFuncs[0] = &SelfAttentionFusionViewBypassFuncDecoder;
    }
    flashAttentionNode.inferShapePreFunc = &FlashAttentionInferShapeBypassBNSDPreFunc;
    newParam_.tokenOffset.reserve(128);    // 128: 预留大小
    newParam_.seqLen.reserve(128);         // 128: 预留大小
    newParam_.batchRunStatus.reserve(128); // 128: 预留大小
}

Status SelfAttentionFusionBypassOpsRunnerBNSD::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    bool needMask = (param_.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED);
    bool hasKV = (param_.kvcacheCfg != atb::infer::SelfAttentionParam::K_BYPASS_V_BYPASS);
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, param_.batchRunStatusEnable, needMask, hasKV);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    size_t faNodeId = FA_NODE_BASE_IDX;
    if (NeedElewiseMulsQScale(param_)) {
        faNodeId++; // 1: qScale Elewise
    }
    auto &flashAttentionNode = kernelGraph_.nodes.at(faNodeId);
    ATB_LOG(INFO) << GetLogPrefix() << "kernelGraph_.nodes.size: " << kernelGraph_.nodes.size();
    AtbOps::OpParam::UnpadFlashAttention flashAttentionQParam;
    SetFAParam(flashAttentionQParam);
    ATB_LOG(INFO) << GetLogPrefix() << "SetFAParam";
    flashAttentionQParam.qSeqLen = newParam_.seqLen;
    flashAttentionQParam.kvSeqLen = newParam_.tokenOffset;
    flashAttentionQParam.batchRunStatus = newParam_.batchRunStatus;
    if (newParam_.isUseTensorList) { // kvcache使用tensorList
        Status st = SetKVCacheTensorList(flashAttentionQParam, flashAttentionNode);
        if (st != NO_ERROR) {
            return st;
        }
    }

    flashAttentionNode.opDesc = {0, "UnpadFlashAttentionOperation", flashAttentionQParam};
    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::UnpadFlashAttention.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHead:" << param_.kvHeadNum
                  << ", calcType: " << param_.calcType << ", maskType: " << param_.maskType
                  << ", kernelTypeType: " << param_.kernelType;
    return NO_ERROR;
}

Status SelfAttentionFusionBypassOpsRunnerBNSD::SetKVCacheTensorList(AtbOps::OpParam::UnpadFlashAttention &param,
                                                                    KernelGraphNode &flashAttentionNode)
{
    if (param_.calcType != atb::infer::SelfAttentionParam::DECODER) {
        ATB_LOG(ERROR) << "param_.calcType should be DECODER!";
        return ERROR_INVALID_PARAM;
    }
    // 1:kcache Tensor 0:layer维度
    uint64_t kLayer = static_cast<uint64_t>(flashAttentionNode.inTensors.at(1)->desc.dims.at(0));
    // 2:vcache Tensor 0:layer维度
    uint64_t vLayer = static_cast<uint64_t>(flashAttentionNode.inTensors.at(2)->desc.dims.at(0));
    if (kLayer != 1 || vLayer != 1) {
        ATB_LOG(ERROR) << "Layer dim should be 1 when use tensorList in kCache and vCache.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    uint64_t batchSize = newParam_.seqLen.size();
    if (newParam_.kCache.size() != batchSize || newParam_.vCache.size() != batchSize) {
        ATB_LOG(ERROR) << "TensorListSize of kCache or vCache is not correct! kCacheSize:" << newParam_.kCache.size()
                       << " vCacheSize:" << newParam_.vCache.size() << " batchSize:" << batchSize;
        return ERROR_INVALID_TENSOR_DIM;
    }
    param.kTensorList.resize(newParam_.kCache.size());
    param.vTensorList.resize(newParam_.vCache.size());
    for (std::size_t i = 0; i < newParam_.kCache.size(); i++) {
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.kCache.at(i), param.kTensorList.at(i));
        TensorUtil::ConvertAtbTensor2OpsTensor(newParam_.vCache.at(i), param.vTensorList.at(i));
    }
    flashAttentionNode.inTensors.at(1) = &nullTensor_; // 1:kcache Tensor
    flashAttentionNode.inTensors.at(2) = &nullTensor_; // 2:vcache Tensor
    return NO_ERROR;
}

AtbOps::OpParam::UnpadFlashAttention::Type SelfAttentionFusionBypassOpsRunnerBNSD::GetFaType()
{
    AtbOps::OpParam::UnpadFlashAttention::Type faType =
        (param_.calcType == atb::infer::SelfAttentionParam::DECODER) ?
            AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_DECODER_ND :
            (param_.kernelType == atb::infer::SelfAttentionParam::KERNELTYPE_HIGH_PRECISION ?
                 AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_FP32_ND :
                 AtbOps::OpParam::UnpadFlashAttention::UNPAD_FLASH_ATTENTION_ND);
    return faType;
}

void SelfAttentionFusionBypassOpsRunnerBNSD::SetFAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam)
{
    flashAttentionParam.type = GetFaType();

    flashAttentionParam.isClamp = (param_.clampType == infer::SelfAttentionParam::CLAMP_TYPE_MIN_MAX);
    flashAttentionParam.clampMin = param_.clampMin;
    flashAttentionParam.clampMax = param_.clampMax;

    flashAttentionParam.headSize = param_.headNum;
    flashAttentionParam.kvHead = param_.kvHeadNum;
    flashAttentionParam.tor = param_.qkScale;
    flashAttentionParam.isTriuMask = param_.isTriuMask;

    flashAttentionParam.maskType =
        param_.maskType == infer::SelfAttentionParam::MASK_TYPE_NORM_COMPRESS ?
            static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(1) : // 1: norm mask
            ((param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) ?
                 static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(2) : // 2: alibi mask
                 static_cast<AtbOps::OpParam::UnpadFlashAttention::MaskType>(param_.maskType));
    flashAttentionParam.isAlibiMaskSqrt = (param_.maskType == infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT);
    flashAttentionParam.dataShapeType =
        static_cast<AtbOps::OpParam::UnpadFlashAttention::DataShapeType>(param_.inputLayout);
}

SelfAttentionFusionBypassOpsRunnerBNSD::~SelfAttentionFusionBypassOpsRunnerBNSD() {}

REG_RUNNER_TYPE(SelfAttentionFusionBypassOpsRunnerBNSD);
} // namespace atb