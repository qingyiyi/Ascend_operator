/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "paged_attention_ops_runner.h"
#include "asdops/params/params.h"
#include "atbops/params/params.h"
#include "atb/utils/log.h"
#include "paged_attention_operation.h"
#include "paged_attention_runner_utils.h"

namespace {
static constexpr uint32_t KERNEL_GRAPH_NODE_BASE_SIZE = 1;
static constexpr uint32_t PA_NODE_BASE_IDX = 0;
} // namespace

namespace atb {

std::size_t PagedAttentionOpsRunner::IntensorSizeGenerate()
{
    std::size_t intensorSize = 5;
    if (param_.maskType != atb::infer::PagedAttentionParam::UNDEFINED) {
        intensorSize += 1; // need to input mask
    }
    if (param_.batchRunStatusEnable) {
        intensorSize += 1; // need to input batchRunStatus
    }
    if (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION) {
        intensorSize += 2; // 2: kDescale, vDescale
        if (param_.hasQuantOffset) {
            intensorSize += 2; // 2: kOffset, vOffset
        }
    }
    bool needQLens = (param_.calcType == atb::infer::PagedAttentionParam::CALC_TYPE_SPEC);
    if (needQLens) {
        intensorSize += 1;
    }
    bool needRazorOffset = (param_.compressType == atb::infer::PagedAttentionParam::COMPRESS_TYPE_KVHEAD_ROPE);
    if (needRazorOffset) {
        intensorSize += 1; // razorOffset
    }
    if (isMla_) {
        intensorSize--;
    }
    bool needQKVQuant = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE);
    if (needQKVQuant) {
        intensorSize += 2; // 2: kDescale, vDescale
        if (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE) {
            intensorSize += 1; // pScale
        }
    }
    bool needLogN = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN);
    if (needLogN) {
        intensorSize += 1; // logN
    }
    return intensorSize;
}

PagedAttentionOpsRunner::PagedAttentionOpsRunner(const infer::PagedAttentionParam &param)
    : OpsRunner("PagedAttentionOpsRunner"), param_(param)
{
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
    ATB_LOG(INFO) << "PagedAttentionOpsRunner::PagedAttentionOpsRunner called";
    isMla_ = param_.mlaVHeadSize > 0;
    needQScale_ = NeedElewiseMulsQScale(param_);

    std::size_t intensorSize = IntensorSizeGenerate();
    kernelGraph_.inTensors.resize(intensorSize);
    kernelGraph_.outTensors.resize(1);

    int inTensorStart = 0;
    Mki::Tensor *query = &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &keyCache = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor *valueCache = isMla_ ? &nullTensor_ : &kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &blockTables = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &contextLens = kernelGraph_.inTensors.at(inTensorStart++);
    (void)contextLens;
    Mki::Tensor *mask = param_.maskType == atb::infer::PagedAttentionParam::MaskType::UNDEFINED ?
                            &nullTensor_ :
                            &kernelGraph_.inTensors.at(inTensorStart++);

    if (param_.batchRunStatusEnable) {
        inTensorStart++;
    }
    bool needQLens = (param_.calcType == atb::infer::PagedAttentionParam::CALC_TYPE_SPEC);
    bool needRazorOffset = (param_.compressType == atb::infer::PagedAttentionParam::COMPRESS_TYPE_KVHEAD_ROPE);
    bool needQKVQuant = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE ||
                         param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE);
    bool needLogN = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN);
    Mki::Tensor *kDescale = nullptr;
    Mki::Tensor *kOffset = nullptr;
    Mki::Tensor *vDescale = nullptr;
    Mki::Tensor *vOffset = nullptr;
    if (param_.quantType == infer::PagedAttentionParam::TYPE_QUANT_UNQUANT) {
        kDescale = &nullTensor_;
        kOffset = &nullTensor_;
        vDescale = &nullTensor_;
        vOffset = &nullTensor_;
    } else if (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION) {
        kDescale = &kernelGraph_.inTensors.at(inTensorStart++);
        if (param_.hasQuantOffset) {
            kOffset = &kernelGraph_.inTensors.at(inTensorStart++);
        } else {
            kOffset = &nullTensor_;
        }
        vDescale = &kernelGraph_.inTensors.at(inTensorStart++);
        if (param_.hasQuantOffset) {
            vOffset = &kernelGraph_.inTensors.at(inTensorStart++);
        } else {
            vOffset = &nullTensor_;
        }
    } else if (needQKVQuant) {
        kDescale = &kernelGraph_.inTensors.at(inTensorStart++);
        vDescale = &kernelGraph_.inTensors.at(inTensorStart++);
        kOffset = &nullTensor_;
        vOffset = &nullTensor_;
    }
    Mki::Tensor *qLens = needQLens ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    (void)qLens;
    Mki::Tensor *razorOffset = needRazorOffset ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *pScale = (param_.quantType == atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_OFFLINE) ?
                              &kernelGraph_.inTensors.at(inTensorStart++) :
                              &nullTensor_;
    Mki::Tensor *logN = needLogN ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    // 用于ElewiseMuls
    Mki::Tensor *divOut = query;

    size_t nodeSize = KERNEL_GRAPH_NODE_BASE_SIZE;
    if (needQScale_) {
        kernelGraph_.internalTensors.resize(1); // 1: qScale ElewiseMuls
        divOut = &kernelGraph_.internalTensors.at(0);
        nodeSize++;
    }
    kernelGraph_.nodes.resize(nodeSize);
    size_t nodeId = 0;
    if (needQScale_) {
        auto &mulsQNode = kernelGraph_.nodes.at(nodeId++);
        mulsQNode.opDesc = {0, "ElewiseOperation",
                            AsdOps::OpParam::Elewise({AsdOps::OpParam::Elewise::ELEWISE_MULS, param_.qScale})};
        mulsQNode.inTensors = {query};
        mulsQNode.outTensors = {divOut};
    }

    auto &pagedAttentionNode = kernelGraph_.nodes.at(nodeId++);

    AtbOps::OpParam::PagedAttention inPagedAttention;
    inPagedAttention.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    inPagedAttention.headSize = param_.headNum;
    inPagedAttention.tor = param_.qkScale;
    inPagedAttention.kvHead = param_.kvHeadNum;
    inPagedAttention.maskType = GetMaskType();
    inPagedAttention.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                     AtbOps::OpParam::PagedAttention::SCALE_LOGN_FP32 :
                                     AtbOps::OpParam::PagedAttention::SCALE_TOR;
    inPagedAttention.compressHead =
        (param_.compressType == infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD);
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", inPagedAttention};

    if (!isMla_) {
        pagedAttentionNode.inTensors = {divOut,  &keyCache, valueCache, &blockTables, mask,   kDescale,
                                        kOffset, vDescale,  vOffset,    razorOffset,  pScale, logN};
    } else if (needQLens) {
        pagedAttentionNode.inTensors = {divOut, &keyCache, &blockTables, mask};
    } else {
        pagedAttentionNode.inTensors = {divOut,  &keyCache, &blockTables, mask,  kDescale,
                                        kOffset, vDescale,  vOffset,      pScale};
    }

    pagedAttentionNode.outTensors = {&output};
    pagedAttentionNode.inTensorViewFuncs.resize(pagedAttentionNode.inTensors.size()); // view
    pagedAttentionNode.inferShapePreFunc = [](Mki::LaunchParam &launchParam) {        // format, dtype 设置
        for (size_t i = 0; i < launchParam.GetInTensorCount(); i++) {
            launchParam.GetInTensor(i).desc.format = Mki::TENSOR_FORMAT_ND;
        }
    };
    newParam_.batchRunStatus.reserve(128); // 128: 预留大小
    newParam_.contextLens.reserve(128);    // 128: 预留大小
    newParam_.qLens.reserve(128);          // 128: 预留大小
}

PagedAttentionOpsRunner::~PagedAttentionOpsRunner() {}

AtbOps::OpParam::PagedAttention::MaskType PagedAttentionOpsRunner::GetMaskType() const
{
    return static_cast<AtbOps::OpParam::PagedAttention::MaskType>(param_.maskType);
}

Status PagedAttentionOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    size_t batchRunStatusIndex = 5; // 5: batchRunStatus 位置
    if (param_.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        batchRunStatusIndex++;
    }
    size_t qLensIndex = 5; // 5: batchRunStatus 位置
    if (param_.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        qLensIndex++;
    }
    if (param_.batchRunStatusEnable) {
        qLensIndex++;
    }
    if (param_.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        qLensIndex += 2; // 2: kv descale
    }
    if (param_.hasQuantOffset) {
        qLensIndex += 2; // 2: kv offset
    }
    size_t contextLensTensorId = 4;
    if (isMla_) {
        qLensIndex--;
        batchRunStatusIndex--;
        contextLensTensorId--;
    }
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, param_.batchRunStatusEnable, batchRunStatusIndex,
                                         param_.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC,
                                         qLensIndex, contextLensTensorId);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    size_t paNodeId = needQScale_ ? PA_NODE_BASE_IDX + 1 : PA_NODE_BASE_IDX; // 1: qScale ElewiseMuls
    auto &pagedAttentionNode = kernelGraph_.nodes.at(paNodeId);
    AtbOps::OpParam::PagedAttention inPagedAttention;
    SetPaParam(inPagedAttention);
    pagedAttentionNode.opDesc = {0, "PagedAttentionOperation", inPagedAttention};
    ATB_LOG(INFO) << GetLogPrefix() << " update AtbOps::OpParam::PagedAttention.headNum:" << param_.headNum
                  << ", qkScale:" << param_.qkScale << ", kvHead:" << param_.kvHeadNum
                  << ", batchRunStatus: " << newParam_.batchRunStatus.size() << ", quantType: " << param_.quantType
                  << ", hasQuantOffset: " << param_.hasQuantOffset << ", compressHead: "
                  << (param_.compressType == infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD)
                  << ", qLens: " << newParam_.qLens.size();
    return NO_ERROR;
}

void PagedAttentionOpsRunner::SetPaParam(AtbOps::OpParam::PagedAttention &inPagedAttention)
{
    if (isMla_) {
        inPagedAttention.type =
            param_.calcType == infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC ?
                AtbOps::OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_MULTI_TOKEN_PREDICTION_MASK_ND :
                AtbOps::OpParam::PagedAttention::PAGED_MULTI_LATENT_ATTENTION_COMBINE_CACHE_MASK_ND;
    } else {
        inPagedAttention.type = AtbOps::OpParam::PagedAttention::PAGED_ATTENTION_MASK_ND;
    }
    inPagedAttention.headSize = param_.headNum;
    inPagedAttention.tor = param_.qkScale;
    inPagedAttention.kvHead = param_.kvHeadNum;
    inPagedAttention.maskType = GetMaskType();
    inPagedAttention.scaleType = (param_.scaleType == atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN) ?
                                     AtbOps::OpParam::PagedAttention::SCALE_LOGN_FP32 :
                                     AtbOps::OpParam::PagedAttention::SCALE_TOR;
    inPagedAttention.kvSeqLen = newParam_.contextLens;
    inPagedAttention.batchRunStatus = newParam_.batchRunStatus;
    inPagedAttention.qSeqLen = newParam_.qLens;
    inPagedAttention.headDimV = static_cast<int32_t>(param_.mlaVHeadSize);
    inPagedAttention.dataShapeType = static_cast<AtbOps::OpParam::PagedAttention::DataShapeType>(param_.inputLayout);
    inPagedAttention.quantType = static_cast<AtbOps::OpParam::PagedAttention::QuantType>(param_.quantType);
    inPagedAttention.outDataType = static_cast<Mki::TensorDType>(param_.outDataType);
    inPagedAttention.compressHead =
        (param_.compressType != infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_UNDEFINED);
    if (param_.quantType == infer::PagedAttentionParam::TYPE_DEQUANT_FUSION) {
        inPagedAttention.identityM.resize(32 * 32); // 32: 单位阵大小
        for (size_t i = 0; i < 32; ++i) {           // 32: 单位阵大小
            inPagedAttention.identityM[i * 32 + i] = 1;
        }
    }
    needQScale_ = NeedElewiseMulsQScale(param_);
}
} // namespace atb
