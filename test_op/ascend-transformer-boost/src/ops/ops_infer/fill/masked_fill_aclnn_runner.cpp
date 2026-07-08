/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "masked_fill_aclnn_runner.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"
#include "acl/acl.h"

namespace atb {
const uint32_t MASK_FILL_IN_TENSOR_NUM = 2;
const uint32_t MASK_FILL_OUT_TENSOR_NUM = 1;
const uint32_t MASK_FILL_OUT_INDEX_ZERO = 0;
const uint32_t MASK_FILL_IN_INDEX_ZERO = 0;
const uint32_t MASK_FILL_IN_INDEX_ONE = 1;
const uint32_t MASK_FILL_PARAM_VALUE_INDEX = 0;

AclnnMaskFillGetWorkspaceSizeFunc MaskedFillAclnnRunner::aclnnInplaceMaskedFillScalarGetWorkspaceSizeFunc_ = nullptr;
AclnnMaskFillExecuteFunc MaskedFillAclnnRunner::aclnnInplaceMaskedFillScalarFunc_ = nullptr;

MaskedFillAclnnRunner::MaskedFillAclnnRunner(const infer::FillParam &param) : AclnnRunner("MaskedFillAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "MaskedFillAclnnRunner::MaskedFillAclnnRunner called";
    valueScalarPtr_ = nullptr;
}
MaskedFillAclnnRunner::~MaskedFillAclnnRunner() {}
Status MaskedFillAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "MaskedFillAclnnRunner LoadMethod";
    Status status = NO_ERROR;
    if (aclnnInplaceMaskedFillScalarGetWorkspaceSizeFunc_ == nullptr ||
        aclnnInplaceMaskedFillScalarFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnInplaceMaskedFillScalarGetWorkspaceSize",
                                          "aclnnInplaceMaskedFillScalar",
                                          aclnnInplaceMaskedFillScalarGetWorkspaceSizeFunc_,
                                          aclnnInplaceMaskedFillScalarFunc_);
    }
    return status;
}
Status MaskedFillAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    Status ret = NO_ERROR;
    atbVariantPack_ = runnerVariantPack;
    aclnnVariantPack_.aclInTensors.reserve(MASK_FILL_IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(MASK_FILL_IN_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.reserve(MASK_FILL_OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(MASK_FILL_OUT_TENSOR_NUM);
    ret = BuildXTensor();
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor X by aclCreateTensor failed!";
        return ret;
    }
    ret = BuildMaskTensor();
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor Mask by aclCreateTensor failed!";
        return ret;
    }
    ret = BuildOutputTensor();
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor output by aclCreateTensor failed!";
        return ret;
    }
    return ret;

}
Status MaskedFillAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = aclnnInplaceMaskedFillScalarFunc_(atbVariantPack_.workspaceBuffer,
                                                        atbVariantPack_.workspaceBufferSize,
                                                        aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";

    ret = aclDestroyScalar(valueScalarPtr_);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "destroy scalar value failed";
    }
    valueScalarPtr_ = nullptr;
    return NO_ERROR;
}
aclnnStatus MaskedFillAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn MaskedFillAclnnRunner setup start.";
    ATB_LOG(INFO) << GetLogPrefix()
                  << "aclnn MaskedFillScalar, aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    aclTensor *selfRef = aclnnVariantPack_.aclOutTensors.at(MASK_FILL_OUT_INDEX_ZERO)->tensor;
    aclTensor *mask = aclnnVariantPack_.aclInTensors.at(MASK_FILL_IN_INDEX_ONE)->tensor;
    
    //create value for aclnn interface
    float value = param_.value.at(MASK_FILL_PARAM_VALUE_INDEX);
    if (valueScalarPtr_ != nullptr) {
        auto res = aclDestroyScalar(valueScalarPtr_);
        if (res != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy scalar value failed";
            return res;
        }
        valueScalarPtr_ = nullptr;
    }
    valueScalarPtr_ = aclCreateScalar(&value, ACL_FLOAT);

    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    aclnnStatus ret = aclnnInplaceMaskedFillScalarGetWorkspaceSizeFunc_(
        selfRef,
        mask,
        (const aclScalar *)valueScalarPtr_,
        &(atbVariantPack_.workspaceBufferSize),
        &rawExecutorPtr
    );

    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnInplaceMaskedFillScalarGetWorkspaceSize failed";
        return ret;
    }
    return ret;
}

Status MaskedFillAclnnRunner::BuildXTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "MaskedFillAclnnRunner::BuildAclnnVariantPack inTensor index: " << MASK_FILL_IN_INDEX_ZERO;
    Status ret = NO_ERROR;
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = atbVariantPack_.inTensors.at(MASK_FILL_IN_INDEX_ZERO);
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                      atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(MASK_FILL_IN_INDEX_ZERO);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclInTensors.at(MASK_FILL_IN_INDEX_ZERO) = aclnnTensorPtr;
    return atb::NO_ERROR;
}

Status MaskedFillAclnnRunner::BuildMaskTensor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "MaskedFillAclnnRunner::BuildAclnnVariantPack inTensor index: " << MASK_FILL_IN_INDEX_ONE;
    Status ret = NO_ERROR;
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    atb::Tensor atbTensor = atbVariantPack_.inTensors.at(MASK_FILL_IN_INDEX_ONE);
    //transform int8 to bool for aclnn interface
    if (atbTensor.desc.dtype == ACL_INT8) {
        atbTensor.desc.dtype = ACL_BOOL;
    }
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                    atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(MASK_FILL_IN_INDEX_ONE);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclInTensors.at(MASK_FILL_IN_INDEX_ONE) = aclnnTensorPtr;
    return atb::NO_ERROR;
}

Status MaskedFillAclnnRunner::BuildOutputTensor()
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    ATB_LOG(INFO) << GetLogPrefix() << "MaskedFillAclnnRunner::BuildAclnnVariantPack outTensor index: " << MASK_FILL_OUT_INDEX_ZERO;
    Status ret = NO_ERROR;
    atb::Tensor atbTensor = atbVariantPack_.outTensors.at(MASK_FILL_OUT_INDEX_ZERO);
    //move the selfRef into output for aclnn interface
    auto memRet = aclrtMemcpy(atbTensor.deviceData, atbTensor.dataSize,
                              atbVariantPack_.inTensors.at(MASK_FILL_IN_INDEX_ZERO).deviceData,
                              atbVariantPack_.inTensors.at(MASK_FILL_IN_INDEX_ZERO).dataSize,
                              ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (memRet != ACL_SUCCESS) {
        return ERROR_CANN_ERROR;
    }
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                atbTensor.desc.dtype);
    if (ret != NO_ERROR) {
        return ret;
    }
    aclnnTensorPtr->tensorIdx = static_cast<int>(MASK_FILL_OUT_INDEX_ZERO);
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    aclnnVariantPack_.aclOutTensors.at(MASK_FILL_IN_INDEX_ZERO) = aclnnTensorPtr;
    return atb::NO_ERROR;
}
REG_RUNNER_TYPE(MaskedFillAclnnRunner);
} // namespace atb