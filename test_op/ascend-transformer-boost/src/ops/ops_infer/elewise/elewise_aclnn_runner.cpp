/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include <aclnn/opdev/bfloat16.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"
#include "atb/utils/utils_internal.h"
#include "elewise_aclnn_runner.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM_1 = 1;
static const uint32_t IN_TENSOR_NUM_2 = 2;
static const uint32_t IN_TENSOR_NUM_3 = 3;
static const bool SQRT_MODE = false;
static const std::string ROUND_MODE = "round";
static const int AXIS = -1;
static int DST_TYPE = 2;

// 初始化类函数指针
AclnnCastGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ = nullptr;
AclnnCastExecuteFunc ElewiseAclnnRunner::aclnnCastExecuteFunc_ = nullptr;

AclnnMulsGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnMulsGetWorkspaceSizeFunc_ = nullptr;
AclnnMulsExecuteFunc ElewiseAclnnRunner::aclnnMulsExecuteFunc_ = nullptr;

AclnnCosGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnCosGetWorkspaceSizeFunc_ = nullptr;
AclnnCosExecuteFunc ElewiseAclnnRunner::aclnnCosExecuteFunc_ = nullptr;

AclnnSinGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnSinGetWorkspaceSizeFunc_ = nullptr;
AclnnSinExecuteFunc ElewiseAclnnRunner::aclnnSinExecuteFunc_ = nullptr;

AclnnLogicalNotGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnLogicalNotGetWorkspaceSizeFunc_ = nullptr;
AclnnLogicalNotExecuteFunc ElewiseAclnnRunner::aclnnLogicalNotExecuteFunc_ = nullptr;

AclnnAddGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnAddGetWorkspaceSizeFunc_ = nullptr;
AclnnAddExecuteFunc ElewiseAclnnRunner::aclnnAddExecuteFunc_ = nullptr;

AclnnMulGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnMulGetWorkspaceSizeFunc_ = nullptr;
AclnnMulExecuteFunc ElewiseAclnnRunner::aclnnMulExecuteFunc_ = nullptr;

AclnnDivGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnDivGetWorkspaceSizeFunc_ = nullptr;
AclnnDivExecuteFunc ElewiseAclnnRunner::aclnnDivExecuteFunc_ = nullptr;

AclnnLtTensorGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnLtTensorGetWorkspaceSizeFunc_ = nullptr;
AclnnLtTensorExecuteFunc ElewiseAclnnRunner::aclnnLtTensorExecuteFunc_ = nullptr;

AclnnGtTensorGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnGtTensorGetWorkspaceSizeFunc_ = nullptr;
AclnnGtTensorExecuteFunc ElewiseAclnnRunner::aclnnGtTensorExecuteFunc_ = nullptr;

AclnnAscendQuantGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnAscendQuantGetWorkspaceSizeFunc_ = nullptr;
AclnnAscendQuantExecuteFunc ElewiseAclnnRunner::aclnnAscendQuantExecuteFunc_ = nullptr;

AclnnSubGetWorkspaceSizeFunc ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_ = nullptr;
AclnnSubExecuteFunc ElewiseAclnnRunner::aclnnSubExecuteFunc_ = nullptr;

ElewiseAclnnRunner::ElewiseAclnnRunner(const infer::ElewiseParam &param)
    : AclnnRunner("ElewiseAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "ElewiseAclnnRunner::ElewiseAclnnRunner called";
    std::unordered_map<infer::ElewiseParam::ElewiseType, uint32_t> inTensorNumMap = {
        {infer::ElewiseParam::ElewiseType::ELEWISE_CAST, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MULS, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_COS, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SIN, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, IN_TENSOR_NUM_1},
        {infer::ElewiseParam::ElewiseType::ELEWISE_ADD, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MUL, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LESS, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, IN_TENSOR_NUM_2},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT, IN_TENSOR_NUM_3},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SUB, IN_TENSOR_NUM_2},
    };
    std::unordered_map<infer::ElewiseParam::ElewiseType, uint32_t>::iterator it =
        inTensorNumMap.find(param.elewiseType);
    if (it != inTensorNumMap.end()) {
        inTensorNum_ = it->second;
    }
}

ElewiseAclnnRunner::~ElewiseAclnnRunner() {
    aclnnStatus ret = ACL_SUCCESS;
    if (scaleTensor_ != nullptr) {
        ret = aclDestroyTensor(scaleTensor_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy scale tensor ERROR: " << ret;
        }
        scaleTensor_ = nullptr;
    }
    if (offsetTensor_ != nullptr) {
        ret = aclDestroyTensor(offsetTensor_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy offset tensor ERROR: " << ret;
        }
        offsetTensor_ = nullptr;
    }
    aclError err = ACL_SUCCESS;
    if (scaleDeviceAddr_ != nullptr) {
        err = aclrtFree(scaleDeviceAddr_);
        if (err != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "free scale device addr ERROR: " << err;
        }
        scaleDeviceAddr_ = nullptr;
    }
    if (offsetDeviceAddr_ != nullptr) {
        err = aclrtFree(offsetDeviceAddr_);
        if (err != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "free offset device addr ERROR: " << err;
        }
        offsetDeviceAddr_ = nullptr;
    }
}

template <typename T>
aclnnStatus ElewiseAclnnRunner::CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                                                void** deviceAddr, aclDataType dataType, aclTensor** tensor,
                                                uint64_t size)
{
    aclnnStatus ret = ACL_SUCCESS;
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtMalloc failed. ERROR: " << ret;
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclrtMemcpy failed. ERROR: " << ret;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return ret;
}

Status ElewiseAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();

    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;

    aclnnVariantPack_.aclInTensors.reserve(inTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(inTensorNum_);

    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_QUANT) {
        ret = ProcessQuantTensors(runnerVariantPack);
        if (ret != NO_ERROR) return ret;
    } else {
        ret = ProcessNormalTensors(runnerVariantPack, aclnnVariantPack_.aclInTensors, false);
        if (ret != NO_ERROR) return ret;
    }

    aclnnVariantPack_.aclOutTensors.reserve(1);
    aclnnVariantPack_.aclOutTensors.resize(1);
    ret = ProcessNormalTensors(runnerVariantPack, aclnnVariantPack_.aclOutTensors, true);

    return ret;
}

Status ElewiseAclnnRunner::ProcessNormalTensors(const RunnerVariantPack &runnerVariantPack,
                                               atb::SVector<std::shared_ptr<AclNNTensor>> &tensors,
                                               bool isOutput)
{
    const SVector<Tensor>& srcTensors = isOutput ? runnerVariantPack.outTensors : runnerVariantPack.inTensors;

    for (size_t i = 0; i < tensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        Status ret = CreateAclNNTensorByAtbTensor(srcTensors.at(i), static_cast<int>(i), aclnnTensorPtr);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        tensors[i] = aclnnTensorPtr;
    }
    return NO_ERROR;
}

Status ElewiseAclnnRunner::ProcessQuantTensors(const RunnerVariantPack &runnerVariantPack)
{
    std::shared_ptr<AclNNTensor> xTensorPtr = std::make_shared<AclNNTensor>();
    Status ret = CreateAclNNTensorByAtbTensor(runnerVariantPack.inTensors.at(0), 0, xTensorPtr);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
        return ret;
    }
    aclnnVariantPack_.aclInTensors[0] = xTensorPtr;

    std::shared_ptr<AclNNTensor> scaleTensorPtr = std::make_shared<AclNNTensor>();
    aclnnStatus aclRet = CreateQuantParamTensor(xTensorPtr->atbTensor, scaleDeviceAddr_, scaleTensor_, param_.quantParam.inputScale, scaleTensorPtr);
    if (aclRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create scale tensor failed!";
        return aclRet;
    }
    aclnnVariantPack_.aclInTensors[1] = scaleTensorPtr;

    std::shared_ptr<AclNNTensor> offsetTensorPtr = std::make_shared<AclNNTensor>();
    aclRet = CreateQuantParamTensor(xTensorPtr->atbTensor, offsetDeviceAddr_, offsetTensor_, param_.quantParam.inputOffset, offsetTensorPtr);
    if (aclRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "create offset tensor failed!";
        return aclRet;
    }
    aclnnVariantPack_.aclInTensors[2] = offsetTensorPtr;

    return NO_ERROR;
}

Status ElewiseAclnnRunner::CreateAclNNTensorByAtbTensor(atb::Tensor atbTensor, int index, std::shared_ptr<AclNNTensor>& tensorPtr)
{
    tensorPtr->atbTensor = atbTensor;
    tensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
    tensorPtr->tensorIdx = index;
    tensorPtr->needUpdateTensorDataPtr = true;

    return CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, tensorPtr);
}

aclnnStatus ElewiseAclnnRunner::CreateQuantParamTensor(atb::Tensor baseTensor,
                                                  void* deviceAddr,
                                                  aclTensor *paramTensor,
                                                  float paramValue,
                                                  std::shared_ptr<AclNNTensor>& tensorPtr)
{
    std::vector<int64_t> shape = {1};
    aclnnStatus ret = ACL_SUCCESS;
    uint64_t size = UtilsInternal::GetDataTypeSize(baseTensor.desc.dtype);

    switch (baseTensor.desc.dtype) {
        case ACL_FLOAT16: {
            std::vector<aclFloat16> hostData = {aclFloatToFloat16(paramValue)};
            ret = CreateAclTensor<aclFloat16>(hostData, shape, &deviceAddr, aclDataType::ACL_FLOAT16, &paramTensor, size);
            break;
        }
        case ACL_FLOAT: {
            std::vector<float> hostData = {paramValue};
            CreateAclTensor<float>(hostData, shape, &deviceAddr, aclDataType::ACL_FLOAT, &paramTensor, size);
            break;
        }
        case ACL_BF16: {
            std::vector<op::bfloat16> hostData = {static_cast<op::bfloat16>(paramValue)};
            CreateAclTensor<op::bfloat16>(hostData, shape, &deviceAddr, aclDataType::ACL_BF16, &paramTensor, size);
            break;
        }
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "invalid inTensor dtype!";
            return ERROR_INTERNAL_ERROR;
    }

    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "CreateAclTensor failed for dtype "
                       << baseTensor.desc.dtype << ". Error: " << ret;
        return ret;
    }

    tensorPtr->tensor = paramTensor;
    return ret;
}

aclnnStatus ElewiseAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << ", aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    aclnnStatus ret = ACL_SUCCESS;
    switch (param_.elewiseType) {
        case infer::ElewiseParam::ElewiseType::ELEWISE_CAST:
            ret = HandleCast(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_MULS:
            ret = HandleMuls(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_COS:
            ret = HandleCos(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SIN:
            ret = HandleSin(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT:
            ret = HandleLogicalNot(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_ADD:
            ret = HandleAdd(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_MUL:
            ret = HandleMul(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV:
            ret = HandleRealDiv(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LESS:
            ret = HandleLess(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_GREATER:
            ret = HandleGreater(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_QUANT:
            ret = HandleQuant(&rawExecutorPtr);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SUB:
            ret = HandleSub(&rawExecutorPtr);
            break;
        default:
            break;
    }
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

aclnnStatus ElewiseAclnnRunner::HandleCast(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    return ElewiseAclnnRunner::aclnnCastGetWorkspaceSizeFunc_(self, param_.outTensorType, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleSub(aclOpExecutor** executor)
{   
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *other = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    float varAttrFloat_ = 1.0f;
    int32_t varAttrInt = 1;
    if (aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT ||
        aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT16 ||
        aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_BF16) {
        alpha_ = aclCreateScalar(&varAttrFloat_, aclDataType::ACL_FLOAT);
    } else {
        alpha_ = aclCreateScalar(&varAttrInt, aclDataType::ACL_INT32);
    };
    return aclnnSubGetWorkspaceSizeFunc_(self, other, alpha_, out, &(atbVariantPack_.workspaceBufferSize), executor);
}


aclnnStatus ElewiseAclnnRunner::HandleMuls(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    float varAttrFloat_ = static_cast<float>(param_.mulsParam.varAttr);
    int32_t varAttrInt = static_cast<int32_t>(param_.mulsParam.varAttr);
    if (aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT ||
        aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT16 ||
        aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_BF16) {
        alpha_ = aclCreateScalar(&varAttrFloat_, aclDataType::ACL_FLOAT);
    } else {
        alpha_ = aclCreateScalar(&varAttrInt, aclDataType::ACL_INT32);
    };
    return aclnnMulsGetWorkspaceSizeFunc_(self, alpha_, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleCos(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    return aclnnCosGetWorkspaceSizeFunc_(self, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleSin(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    return aclnnSinGetWorkspaceSizeFunc_(self, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleLogicalNot(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    return aclnnLogicalNotGetWorkspaceSizeFunc_(self, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleAdd(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *other = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    float varAttrFloat_ = 1.0f;
    int32_t varAttrInt = 1;
    if (aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT ||
        aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_FLOAT16 ||
        aclnnVariantPack_.aclInTensors.at(0)->atbTensor.desc.dtype == aclDataType::ACL_BF16) {
        alpha_ = aclCreateScalar(&varAttrFloat_, aclDataType::ACL_FLOAT);
    } else {
        alpha_ = aclCreateScalar(&varAttrInt, aclDataType::ACL_INT32);
    };
    return aclnnAddGetWorkspaceSizeFunc_(self, other, alpha_, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleMul(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *other = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    return aclnnMulGetWorkspaceSizeFunc_(self, other, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleRealDiv(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *other = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    return aclnnDivGetWorkspaceSizeFunc_(self, other, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleLess(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *other = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    return aclnnLtTensorGetWorkspaceSizeFunc_(self, other, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleGreater(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *other = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    return aclnnGtTensorGetWorkspaceSizeFunc_(self, other, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

aclnnStatus ElewiseAclnnRunner::HandleQuant(aclOpExecutor** executor)
{
    size_t inTensorIndex = 0;
    aclTensor *self = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    size_t outTensorIndex = 0;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(outTensorIndex++)->tensor;
    aclTensor *scale = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    aclTensor *offset = aclnnVariantPack_.aclInTensors.at(inTensorIndex++)->tensor;
    if (param_.outTensorType != ACL_DT_UNDEFINED) {
        DST_TYPE = param_.outTensorType;
    }
    return aclnnAscendQuantGetWorkspaceSizeFunc_(self, scale, offset, SQRT_MODE, ROUND_MODE.c_str(), DST_TYPE, AXIS, out, &(atbVariantPack_.workspaceBufferSize), executor);
}

Status ElewiseAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    aclnnStatus ret = ACL_SUCCESS;
    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);
    switch (param_.elewiseType) {
        case infer::ElewiseParam::ElewiseType::ELEWISE_CAST:
            ret = aclnnCastExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                        atbVariantPack_.workspaceBufferSize,
                                        aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_MULS:
            ret = aclnnMulsExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                        atbVariantPack_.workspaceBufferSize,
                                        aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_COS:
            ret = aclnnCosExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                       atbVariantPack_.workspaceBufferSize,
                                       aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SIN:
            ret = aclnnSinExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                       atbVariantPack_.workspaceBufferSize,
                                       aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT:
            ret = aclnnLogicalNotExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                              atbVariantPack_.workspaceBufferSize,
                                              aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_ADD:
            ret = aclnnAddExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                       atbVariantPack_.workspaceBufferSize,
                                       aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_SUB:
            ret = aclnnSubExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                       atbVariantPack_.workspaceBufferSize,
                                       aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_MUL:
            ret = aclnnMulExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                       atbVariantPack_.workspaceBufferSize,
                                       aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV:
            ret = aclnnDivExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                       atbVariantPack_.workspaceBufferSize,
                                       aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_LESS:
            ret = aclnnLtTensorExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                            atbVariantPack_.workspaceBufferSize,
                                            aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_GREATER:
            ret = aclnnGtTensorExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                            atbVariantPack_.workspaceBufferSize,
                                            aclnnExecutor_.get(), executeStream);
            break;
        case infer::ElewiseParam::ElewiseType::ELEWISE_QUANT:
            ret = aclnnAscendQuantExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                               atbVariantPack_.workspaceBufferSize,
                                               aclnnExecutor_.get(), executeStream);
            break;
        default:
            break;
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    if (alpha_ != nullptr) {
        ret = aclDestroyScalar(alpha_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy scalar failed: " << ret;
            return ERROR_CANN_ERROR;
        }
        alpha_ = nullptr;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

Status ElewiseAclnnRunner::LoadMethod()
{
    ATB_LOG(INFO) << "ElewiseAclnnRunner LoadMethod";
    Status status = NO_ERROR;
    if (ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_ == nullptr ||
        ElewiseAclnnRunner::aclnnSubExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnSubGetWorkspaceSize", "aclnnSub",
                                          ElewiseAclnnRunner::aclnnSubGetWorkspaceSizeFunc_,
                                          ElewiseAclnnRunner::aclnnSubExecuteFunc_);
    }
    if (aclnnCastGetWorkspaceSizeFunc_ == nullptr ||
        aclnnCastExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnCastGetWorkspaceSize", "aclnnCast",
                                          aclnnCastGetWorkspaceSizeFunc_,
                                          aclnnCastExecuteFunc_);
    }
    if (aclnnMulsGetWorkspaceSizeFunc_ == nullptr ||
        aclnnMulsExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnMulsGetWorkspaceSize", "aclnnMuls",
                                          aclnnMulsGetWorkspaceSizeFunc_,
                                          aclnnMulsExecuteFunc_);
    }
    if (aclnnCosGetWorkspaceSizeFunc_ == nullptr ||
        aclnnCosExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnCosGetWorkspaceSize", "aclnnCos",
                                          aclnnCosGetWorkspaceSizeFunc_,
                                          aclnnCosExecuteFunc_);
    }
    if (aclnnSinGetWorkspaceSizeFunc_ == nullptr ||
        aclnnSinExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnSinGetWorkspaceSize", "aclnnSin",
                                          aclnnSinGetWorkspaceSizeFunc_,
                                          aclnnSinExecuteFunc_);
    }
    if (aclnnLogicalNotGetWorkspaceSizeFunc_ == nullptr ||
        aclnnLogicalNotExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnLogicalNotGetWorkspaceSize", "aclnnLogicalNot",
                                          aclnnLogicalNotGetWorkspaceSizeFunc_,
                                          aclnnLogicalNotExecuteFunc_);
    }
    if (aclnnAddGetWorkspaceSizeFunc_ == nullptr ||
        aclnnAddExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnAddGetWorkspaceSize", "aclnnAdd",
                                          aclnnAddGetWorkspaceSizeFunc_,
                                          aclnnAddExecuteFunc_);
    }
    if (aclnnMulGetWorkspaceSizeFunc_ == nullptr ||
        aclnnMulExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnMulGetWorkspaceSize", "aclnnMul",
                                          aclnnMulGetWorkspaceSizeFunc_,
                                          aclnnMulExecuteFunc_);
    }
    if (aclnnDivGetWorkspaceSizeFunc_ == nullptr ||
        aclnnDivExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnDivGetWorkspaceSize", "aclnnDiv",
                                          aclnnDivGetWorkspaceSizeFunc_,
                                          aclnnDivExecuteFunc_);
    }
    if (aclnnLtTensorGetWorkspaceSizeFunc_ == nullptr ||
        aclnnLtTensorExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnLtTensorGetWorkspaceSize", "aclnnLtTensor",
                                          aclnnLtTensorGetWorkspaceSizeFunc_,
                                          aclnnLtTensorExecuteFunc_);
    }
    if (aclnnGtTensorGetWorkspaceSizeFunc_ == nullptr ||
        aclnnGtTensorExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnGtTensorGetWorkspaceSize", "aclnnGtTensor",
                                          aclnnGtTensorGetWorkspaceSizeFunc_,
                                          aclnnGtTensorExecuteFunc_);
    }
    if (aclnnAscendQuantGetWorkspaceSizeFunc_ == nullptr ||
        aclnnAscendQuantExecuteFunc_ == nullptr) {
        status = LoadFromSharedObjectFile("aclnnAscendQuantV3GetWorkspaceSize", "aclnnAscendQuantV3",
                                          aclnnAscendQuantGetWorkspaceSizeFunc_,
                                          aclnnAscendQuantExecuteFunc_);
        }
    return status;
}

REG_RUNNER_TYPE(ElewiseAclnnRunner);
} // namespace atb
