/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_latent_attention_ops_runner.h"
#include <atb/utils/log.h>
#include <atbops/params/params.h>

namespace {
static const uint32_t IN_TENSOR_NUM_BASE = 6;
static const uint32_t OUT_TENSOR_NUM_BASE = 1;
static const uint32_t CONTEXTLENS_INDEX = 5;
static const uint32_t QSEQLEN_INDEX = 6;
static const int32_t MTP_TP1_HEAD_NUM = 128;
} // namespace
namespace atb {
MultiLatentAttentionOpsRunner::MultiLatentAttentionOpsRunner(const infer::MultiLatentAttentionParam &param)
    : OpsRunner("MultiLatentAttentionOpsRunner"), param_(param)
{
    ATB_LOG(INFO) << "MultiLatentAttentionOpsRunner::MultiLatentAttentionOpsRunner called";
    needKernelGraphModify_ = true;
    skipSetUpKernelGraphWhenCacheHit_ = false;
}

MultiLatentAttentionOpsRunner::~MultiLatentAttentionOpsRunner() {}

Status MultiLatentAttentionOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    std::size_t inTensorSize = IN_TENSOR_NUM_BASE;
    std::size_t outTensorSize = OUT_TENSOR_NUM_BASE;
    bool needMask = param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED;
    bool needQLens = param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC ||
                     param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING;
    bool isInt8Nz = param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE;
    bool isRing = param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING ||
                  param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING;
    bool needMaskUseStatus = param_.maskUseStatusType ==
                             infer::MultiLatentAttentionParam::MaskUseStatusType::MASK_USE_STATUS_TYPE_BATCH_MASK;

    inTensorSize += needMask ? 1 : 0;
    inTensorSize += needQLens ? 1 : 0;
    inTensorSize += isInt8Nz ? 2 : 0; // 2: qDescale kDescale
    inTensorSize += needMaskUseStatus ? 1 : 0; // 1: maskUseStatus
    kernelGraph_.inTensors.resize(inTensorSize);
    int inTensorStart = 0;
    Mki::Tensor &query = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &queryRope = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &kvCache = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &kvCacheRope = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &blockTables = kernelGraph_.inTensors.at(inTensorStart++);
    Mki::Tensor &contextLens = kernelGraph_.inTensors.at(inTensorStart++);
    (void)contextLens;
    Mki::Tensor *mask = needMask ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *qLens = needQLens ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    (void)qLens;
    Mki::Tensor *qkDescale = isInt8Nz ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *pvDescale = isInt8Nz ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    Mki::Tensor *MaskUseStatus = needMaskUseStatus ? &kernelGraph_.inTensors.at(inTensorStart++) : &nullTensor_;
    (void)MaskUseStatus;

    outTensorSize += isRing ? 1 : 0;
    kernelGraph_.outTensors.resize(outTensorSize);
    Mki::Tensor &output = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &mlaNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::MLA asdParam;
    asdParam.type = AtbOps::OpParam::MLA::SPLIT_CACHE;
    asdParam.headSize = param_.headNum;
    asdParam.tor = param_.qkScale;
    asdParam.kvHead = param_.kvHeadNum;
    asdParam.isRing = isRing ? 1 : 0;
    asdParam.windowSize = static_cast<int32_t>(param_.windowSize);
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SPEC) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_LOOK_AHEAD;
    } else if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_MASK_FREE;
    } else if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_CAUSAL_MASK;
    }
    mlaNode.opDesc = {0, "MLAOperation", asdParam};
    mlaNode.inTensors = {&query, &queryRope, &kvCache, &kvCacheRope, &blockTables, mask, qkDescale, pvDescale};
    if (!isRing) {
        mlaNode.outTensors = {&output, &nullTensor_};
    } else {
        Mki::Tensor &outputLse = kernelGraph_.outTensors.at(1);
        mlaNode.outTensors = {&output, &outputLse};
    }
    return NO_ERROR;
}

Status MultiLatentAttentionOpsRunner::ModifyKernelGraph(const OpsTensorPack &opsTensorPack)
{
    auto &mlaNode = kernelGraph_.nodes.at(0);
    size_t qSeqlenIdxBase = QSEQLEN_INDEX;
    if (param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED) {
        qSeqlenIdxBase++;
    }
    bool needQLens = param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC ||
                     param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING;
    bool needMask = param_.maskType != infer::MultiLatentAttentionParam::MaskType::UNDEFINED &&
                    param_.maskType != infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK;
    bool needMaskUseStatus = param_.maskUseStatusType ==
                             infer::MultiLatentAttentionParam::MaskUseStatusType::MASK_USE_STATUS_TYPE_BATCH_MASK;
    bool ret = newParam_.BuildFromTensor(opsTensorPack.inTensors, CONTEXTLENS_INDEX, qSeqlenIdxBase, needQLens);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    size_t idx = 6;
    if (needMask) {
        idx++; // 1: mask
    }
    if (needQLens) {
        idx++; // 1: qSeqlen
    }
    if (param_.cacheMode == infer::MultiLatentAttentionParam::CacheMode::INT8_NZCACHE) {
        idx += 2; // 2: qDescale, kDescale
    }
    size_t batch = opsTensorPack.inTensors.at(CONTEXTLENS_INDEX).desc.dims[0];
    ret = newParam_.BuildMaskUseStatusFromTensor(opsTensorPack.inTensors, idx, needMaskUseStatus, batch);
    if (!ret) {
        ATB_LOG(ERROR) << GetLogPrefix() << " build maskUseStatus param from host tensor fail";
        return ERROR_INVALID_PARAM;
    }
    AtbOps::OpParam::MLA asdParam;
    asdParam.type = AtbOps::OpParam::MLA::SPLIT_CACHE;
    asdParam.headSize = param_.headNum;
    asdParam.tor = param_.qkScale;
    asdParam.kvHead = param_.kvHeadNum;
    asdParam.kvSeqLen = newParam_.contextLens;
    asdParam.qSeqLen = newParam_.qSeqlen;
    asdParam.maskUseStatus = newParam_.maskUseStatus;
    asdParam.isRing = param_.calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_RING || param_.
                      calcType == infer::MultiLatentAttentionParam::CalcType::CALC_TYPE_SPEC_AND_RING;
    asdParam.windowSize = static_cast<int32_t>(param_.windowSize);
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_SPEC) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_LOOK_AHEAD;
    }
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_MASK_FREE) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_MASK_FREE;
    }
    if (param_.maskType == infer::MultiLatentAttentionParam::MaskType::MASK_TYPE_CAUSAL_MASK) {
        asdParam.maskType = AtbOps::OpParam::MLA::MaskType::MASK_TYPE_CAUSAL_MASK;
    }
    mlaNode.opDesc = {0, "MLAOperation", asdParam};
    return NO_ERROR;
}
} // namespace atb