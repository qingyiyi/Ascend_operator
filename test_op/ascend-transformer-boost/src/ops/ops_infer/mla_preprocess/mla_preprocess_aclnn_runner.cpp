/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mla_preprocess_aclnn_runner.h"

#include <aclnn/opdev/op_errno.h>

#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 24;
static const uint32_t OUT_TENSOR_NUM = 4;
static const uint32_t INPUT_INDEX = 0;
static const uint32_t Q_OUT1_INDEX = 2;
static const uint32_t KV_CACHE_OUT1_INDEX = 3;
static const uint32_t KV_CACHE_ROPE_INDEX = 20;
static const uint32_t CTKV_SCALE_INDEX = 22;
static const uint32_t Q_NOPE_SCALE_INDEX = 23;

} // namespace

namespace atb {
// 初始化类函数指针
aclnnStatus (*MlaPreprocessAclnnRunner::aclnnGetWorkspaceSizeFunc_)(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    int64_t, int64_t, int64_t, double, int64_t, int64_t, bool, bool, bool, int64_t, int64_t, bool, int64_t,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **) = nullptr;

aclnnStatus (*MlaPreprocessAclnnRunner::aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

MlaPreprocessAclnnRunner::MlaPreprocessAclnnRunner(const infer::MlaPreprocessParam &param)
    : AclnnRunner("MlaPreprocessAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "MlaPreprocessAclnnRunner::MlaPreprocessAclnnRunner called";
}

MlaPreprocessAclnnRunner::MlaPreprocessAclnnRunner(const infer::MlaPreprocessParam &param, bool doRmsNorm)
    : AclnnRunner("MlaPreprocessAclnnRunner"), param_(param), doRmsNorm_(doRmsNorm)
{
    ATB_LOG(INFO) << GetLogPrefix()
                  << "MlaPreprocessAclnnRunnecr::MlaPreprocessAclnnRunner called, set doRmsNorm: " << doRmsNorm;
}

MlaPreprocessAclnnRunner::~MlaPreprocessAclnnRunner() {}

Status MlaPreprocessAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    bool isRopeCache = param_.cacheMode != infer::MlaPreprocessParam::CacheMode::KVCACHE;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    aclDataType inputDataType = runnerVariantPack.inTensors.at(INPUT_INDEX).desc.dtype;
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "MlaPreprocessAclnnRunner::BuildAclnnVariantPack inTensor index: " << i;
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        if (i == KV_CACHE_ROPE_INDEX && !isRopeCache) {
            // kvCache不带rope转置时kvCacheRope为空tensor
            TensorDesc desc = {};
            desc.dtype = runnerVariantPack.inTensors.at(INPUT_INDEX).desc.dtype;
            desc.format = runnerVariantPack.inTensors.at(INPUT_INDEX).desc.format;
            atbTensor.desc = desc;
        }
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        if (param_.cacheMode != infer::MlaPreprocessParam::CacheMode::INT8_NZCACHE &&
            (i == CTKV_SCALE_INDEX || i == Q_NOPE_SCALE_INDEX)) {
            // 非KROPE_CTKV场景，不传入ctkvScale和qNopeScale，使用inputDataType绕过空tensor的dtype检测
            ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      inputDataType);
        } else {
            ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);
        }
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "MlaPreprocessAclnnRunner::BuildAclnnVariantPack outTensor index: " << i;
        atb::Tensor atbTensor = {};
        if ((i != Q_OUT1_INDEX && i != KV_CACHE_OUT1_INDEX) || isRopeCache) {
            atbTensor = runnerVariantPack.outTensors.at(i);
        } else {
            // kvCache不带rope转置时不生成2个rope分量
            // 使用input的dtype和format填补空tensor
            TensorDesc desc = {};
            desc.dtype = runnerVariantPack.inTensors.at(INPUT_INDEX).desc.dtype;
            desc.format = runnerVariantPack.inTensors.at(INPUT_INDEX).desc.format;
            atbTensor.desc = desc;
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
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        this->aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    return atb::NO_ERROR;
}

aclnnStatus MlaPreprocessAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn mlaPreprocess setup start.";
    if (!aclnnGetWorkspaceSizeFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    size_t inTensorStart = 0;
    bool isRopeCache = param_.cacheMode != infer::MlaPreprocessParam::CacheMode::KVCACHE;
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn mlaPreprocess isRopeCache: " << isRopeCache
                  << ", aclInTensors size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << this->aclnnVariantPack_.aclOutTensors.size();
    aclTensor *input = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *gamma0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *beta0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantScale0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantOffset0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *wdqkv = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *deScale0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *bias0 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *gamma1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *beta1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantScale1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *quantOffset1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *wuq = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *deScale1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *bias1 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *gamma2 = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *cos = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *sin = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *wuk = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *kvCache = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *kRope = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *slotmapping = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *ctkvScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    aclTensor *qNopeScale = this->aclnnVariantPack_.aclInTensors.at(inTensorStart++)->tensor;
    size_t outTensorStart = 0;
    aclTensor *qOut0 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *kvCacheOut0 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    aclTensor *qOut1 = nullptr;
    aclTensor *kvCacheOut1 = nullptr;
    qOut1 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;
    kvCacheOut1 = this->aclnnVariantPack_.aclOutTensors.at(outTensorStart++)->tensor;

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();
    int64_t wdkvSplitCount = 1;
    aclnnStatus ret = MlaPreprocessAclnnRunner::aclnnGetWorkspaceSizeFunc_(
        input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0, bias0, gamma1, beta1, quantScale1,
        quantOffset1, wuq, deScale1, bias1, gamma2, cos, sin, wuk, kvCache, kRope, slotmapping, ctkvScale, qNopeScale,
        param_.wdqDim, param_.qRopeDim, param_.kRopeDim, param_.epsilon, param_.qRotaryCoeff, param_.kRotaryCoeff,
        param_.transposeWdq, param_.transposeWuq, param_.transposeWuk, param_.cacheMode, param_.quantMode, doRmsNorm_,
        wdkvSplitCount, qOut0, kvCacheOut0, qOut1, kvCacheOut1, &(this->atbVariantPack_.workspaceBufferSize),
        &raw_executor_ptr);
    this->aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status MlaPreprocessAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    if (!aclnnExecuteFunc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Aclnn ExecuteFunc is null!";
        return ERROR_INVALID_PARAM;
    }
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = MlaPreprocessAclnnRunner::aclnnExecuteFunc_(this->atbVariantPack_.workspaceBuffer,
                                                                  this->atbVariantPack_.workspaceBufferSize,
                                                                  this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status MlaPreprocessAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "MlaPreprocessAclnnRunner LoadMethod";
    if (MlaPreprocessAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr &&
        MlaPreprocessAclnnRunner::aclnnExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status status = LoadFromSharedObjectFile("aclnnMlaPreprocessGetWorkspaceSize", "aclnnMlaPreprocess",
                                             MlaPreprocessAclnnRunner::aclnnGetWorkspaceSizeFunc_,
                                             MlaPreprocessAclnnRunner::aclnnExecuteFunc_);
    return status;
}
} // namespace atb
