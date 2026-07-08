/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "activation_aclnn_runner.h"

#include <aclnn/opdev/op_errno.h>

#include "atb/utils/aclnn_util.h"
#include "atb/utils/operation_register.h"

static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t IN_TENSOR_IDX = 0;
static const uint32_t SWIGLU_BACKWARD_IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_IDX = 0;

namespace atb {
aclnnStatus (*ActivationAclnnRunner::fastGeluGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                                    aclOpExecutor **) = nullptr;
aclnnStatus (*ActivationAclnnRunner::fastGeluExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

aclnnStatus (*ActivationAclnnRunner::geluGetWorkspaceSizeFunc_)(const aclTensor *, int64_t *, aclTensor *, uint64_t *,
                                                                aclOpExecutor **) = nullptr;
aclnnStatus (*ActivationAclnnRunner::geluExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream) = nullptr;

aclnnStatus (*ActivationAclnnRunner::logGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                               aclOpExecutor **) = nullptr;
aclnnStatus (*ActivationAclnnRunner::logExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

aclnnStatus (*ActivationAclnnRunner::reluGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                                aclOpExecutor **) = nullptr;
aclnnStatus (*ActivationAclnnRunner::reluExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream) = nullptr;

aclnnStatus (*ActivationAclnnRunner::sigmoidGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                                   aclOpExecutor **) = nullptr;
aclnnStatus (*ActivationAclnnRunner::sigmoidExecuteFunc_)(void *, uint64_t, aclOpExecutor *,
                                                          const aclrtStream) = nullptr;

aclnnStatus (*ActivationAclnnRunner::swishGetWorkspaceSizeFunc_)(const aclTensor *, const aclScalar *, aclTensor *,
                                                                 uint64_t *, aclOpExecutor **) = nullptr;
aclnnStatus (*ActivationAclnnRunner::swishExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream) = nullptr;

inline aclTensor *In(const AclNNVariantPack &variantPack) { return variantPack.aclInTensors.at(IN_TENSOR_IDX)->tensor; }
inline aclTensor *Out(const AclNNVariantPack &variantPack) { return variantPack.aclOutTensors.at(OUT_TENSOR_IDX)->tensor; }

ActivationAclnnRunner::KernelAdapters 
ActivationAclnnRunner::MakeAdaptersByType(const infer::ActivationParam &param) {
    switch (param.activationType) {
        case infer::ActivationType::ACTIVATION_FAST_GELU:
        return {
            "fast_gelu",
            [](const AclNNVariantPack &vp, uint64_t *wsSize, aclOpExecutor **executor) {
                return ActivationAclnnRunner::fastGeluGetWorkspaceSizeFunc_(In(vp), Out(vp), wsSize, executor);
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                return ActivationAclnnRunner::fastGeluExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        case infer::ActivationType::ACTIVATION_GELU:
        return {
            "gelu",
            [mode = (param.geluMode == infer::ActivationParam::GeLUMode::NONE_MODE) ? 0 : 1](
                const AclNNVariantPack &vp, uint64_t *wsSize, aclOpExecutor **executor) {
                int64_t modeTmp = mode;
                return ActivationAclnnRunner::geluGetWorkspaceSizeFunc_(In(vp), &modeTmp, Out(vp), wsSize,
                                                                        executor);
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                return ActivationAclnnRunner::geluExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        case infer::ActivationType::ACTIVATION_LOG:
        return {
            "log",
            [](const AclNNVariantPack &vp, uint64_t *wsSize, aclOpExecutor **executor) {
                return ActivationAclnnRunner::logGetWorkspaceSizeFunc_(In(vp), Out(vp), wsSize, executor);
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                return ActivationAclnnRunner::logExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        case infer::ActivationType::ACTIVATION_RELU:
        return {
            "relu",
            [](const AclNNVariantPack &vp, uint64_t *wsSize, aclOpExecutor **executor) {
                return ActivationAclnnRunner::reluGetWorkspaceSizeFunc_(In(vp), Out(vp), wsSize, executor);
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                return ActivationAclnnRunner::reluExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        case infer::ActivationType::ACTIVATION_SIGMOID:
        return {
            "sigmoid",
            [](const AclNNVariantPack &vp, uint64_t *wsSize, aclOpExecutor **executor) {
                return ActivationAclnnRunner::sigmoidGetWorkspaceSizeFunc_(In(vp), Out(vp), wsSize, executor);
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                return ActivationAclnnRunner::sigmoidExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        case infer::ActivationType::ACTIVATION_SWISH:
        return {
            "swish",
            [scale = param.scale](const AclNNVariantPack &vp, uint64_t *wsSize, aclOpExecutor **executor) {
                float scaleTmp = scale;
                aclScalar *scalar = aclCreateScalar(&scaleTmp, aclDataType::ACL_FLOAT);
                aclnnStatus aclnnstatus =
                    ActivationAclnnRunner::swishGetWorkspaceSizeFunc_(In(vp), scalar, Out(vp), wsSize, executor);
                aclnnStatus aclnnDestroyStatus = aclDestroyScalar(scalar);
                if (aclnnDestroyStatus != ACL_SUCCESS) {
                    ATB_LOG(ERROR) << "Failed to destroy scalar, ret = " << aclnnDestroyStatus;
                    return aclnnDestroyStatus;
                }
                return aclnnstatus;
            },
            [](void *ws, uint64_t wsSize, aclOpExecutor *executor, aclrtStream stream) {
                return ActivationAclnnRunner::swishExecuteFunc_(ws, wsSize, executor, stream);
            }
        };

        default:
        return {
            "unsupported",
            [](const AclNNVariantPack &, uint64_t *, aclOpExecutor **) {
                return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
            },
            [](void *, uint64_t, aclOpExecutor *, aclrtStream) {
                return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
            }
        };
    }
}

ActivationAclnnRunner::ActivationAclnnRunner(const infer::ActivationParam &param)
    : AclnnRunner("ActivationAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ActivationAclnnRunner called, activation type: " << param_.activationType;
}

ActivationAclnnRunner::~ActivationAclnnRunner() {}

Status ActivationAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack called.";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    this->atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    this->aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    this->aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    ATB_LOG(INFO) << GetLogPrefix() << "Processing inTensor[0]...";
    std::shared_ptr<AclNNTensor> aclnnInTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbInTensor = runnerVariantPack.inTensors.at(IN_TENSOR_IDX);

    aclnnInTensorPtr->atbTensor = atbInTensor;
    aclnnInTensorPtr->strides = GetCopyTensorStride(atbInTensor.desc.shape);

    ret = CallAclCreateTensor(atbInTensor.desc.shape, atbInTensor.desc.shape, atbInTensor, aclnnInTensorPtr,
                              atbInTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Create aclTensor using AclCreateTensor failed.";
        return ret;
    }
    aclnnInTensorPtr->tensorIdx = static_cast<int>(IN_TENSOR_IDX);
    aclnnInTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclInTensors[IN_TENSOR_IDX] = aclnnInTensorPtr;

    this->aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    this->aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    std::shared_ptr<AclNNTensor> aclnnOutTensorPtr = std::make_shared<AclNNTensor>();
    ATB_LOG(INFO) << GetLogPrefix() << "Processing outTensor[0]...";
    atb::Tensor atbOutTensor = {};
    atbOutTensor = runnerVariantPack.outTensors.at(OUT_TENSOR_IDX);
    aclnnOutTensorPtr->atbTensor = atbOutTensor;
    aclnnOutTensorPtr->strides = GetCopyTensorStride(atbOutTensor.desc.shape);
    ret = CallAclCreateTensor(atbOutTensor.desc.shape, atbOutTensor.desc.shape, atbOutTensor, aclnnOutTensorPtr,
                              atbOutTensor.desc.dtype);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Create aclTensor using AclCreateTensor failed.";
        return ret;
    }
    aclnnOutTensorPtr->tensorIdx = static_cast<int>(OUT_TENSOR_IDX);
    aclnnOutTensorPtr->needUpdateTensorDataPtr = true;
    this->aclnnVariantPack_.aclOutTensors[OUT_TENSOR_IDX] = aclnnOutTensorPtr;

    return NO_ERROR;
}

aclnnStatus ActivationAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "Aclnn activation setup start.";
    KernelAdapters kernelAdapter = MakeAdaptersByType(param_);
    this->getWs_ = std::move(kernelAdapter.getWs);
    this->execFn_ = std::move(kernelAdapter.exec);
    if (!getWs_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Alcnn GetWorkspaceSizeFunc is null!";
        return ACLNN_ERR_INNER_FIND_KERNEL_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn activation type: " << param_.activationType
                  << ", aclInTensor size: " << this->aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensor size: " << this->aclnnVariantPack_.aclOutTensors.size();

    aclOpExecutor *raw_executor_ptr = this->aclnnExecutor_.get();

    aclnnStatus ret =
        this->getWs_(this->aclnnVariantPack_, &(this->atbVariantPack_.workspaceBufferSize), &raw_executor_ptr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSizeFunc failed, error: " << ret;
        return ret;
    }
    this->aclnnExecutor_.reset(raw_executor_ptr, [this](aclOpExecutor *ptr) {
        if (ptr && this->executorRepeatable_)
            aclDestroyAclOpExecutor(ptr); // 可复用时需要手动销毁aclOpExecutor
    });
    ATB_LOG(INFO) << GetLogPrefix() << "WorkspaceSize: " << this->atbVariantPack_.workspaceBufferSize;
    return ret;
}

Status ActivationAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel called.";
    void *executeStream = GetExecuteStream(this->atbVariantPack_.context);
    aclnnStatus ret = this->execFn_(this->atbVariantPack_.workspaceBuffer,
                                    this->atbVariantPack_.workspaceBufferSize,
                                    this->aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Launch kernel failed with error: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel success.";
    return NO_ERROR;
}

Status ActivationAclnnRunner::LoadAclnnFunctions()
{
    ATB_LOG(INFO) << "ActivationAclnnRunner LoadAclnnFunctions...";

    static std::once_flag flag;
    static Status once_status = NO_ERROR;
    std::call_once(flag, [&]() {
        Status status;

        status = LoadFromSharedObjectFile("aclnnFastGeluGetWorkspaceSize", "aclnnFastGelu",
                                          ActivationAclnnRunner::fastGeluGetWorkspaceSizeFunc_,
                                          ActivationAclnnRunner::fastGeluExecuteFunc_);
        if (status != NO_ERROR) { once_status = status; return; }

        status = LoadFromSharedObjectFile("aclnnGeluV2GetWorkspaceSize", "aclnnGeluV2",
                                          ActivationAclnnRunner::geluGetWorkspaceSizeFunc_,
                                          ActivationAclnnRunner::geluExecuteFunc_);
        if (status != NO_ERROR) { once_status = status; return; }

        status = LoadFromSharedObjectFile("aclnnLogGetWorkspaceSize", "aclnnLog",
                                          ActivationAclnnRunner::logGetWorkspaceSizeFunc_,
                                          ActivationAclnnRunner::logExecuteFunc_);
        if (status != NO_ERROR) { once_status = status; return; }

        status = LoadFromSharedObjectFile("aclnnReluGetWorkspaceSize", "aclnnRelu",
                                          ActivationAclnnRunner::reluGetWorkspaceSizeFunc_,
                                          ActivationAclnnRunner::reluExecuteFunc_);
        if (status != NO_ERROR) { once_status = status; return; }

        status = LoadFromSharedObjectFile("aclnnSigmoidGetWorkspaceSize", "aclnnSigmoid",
                                          ActivationAclnnRunner::sigmoidGetWorkspaceSizeFunc_,
                                          ActivationAclnnRunner::sigmoidExecuteFunc_);
        if (status != NO_ERROR) { once_status = status; return; }

        status = LoadFromSharedObjectFile("aclnnSwishGetWorkspaceSize", "aclnnSwish",
                                          ActivationAclnnRunner::swishGetWorkspaceSizeFunc_,
                                          ActivationAclnnRunner::swishExecuteFunc_);
        if (status != NO_ERROR) { once_status = status; return; }

        once_status = NO_ERROR;
    });

    return once_status;
}

REG_RUNNER_TYPE(ActivationAclnnRunner);
} // namespace atb