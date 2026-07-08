/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <aclnn/opdev/op_errno.h>
#include <unordered_set>
#include <mki/utils/platform/platform_info.h>
#include "elewise_aclnn_runner.h"
#include "elewise_ops_runner.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"
#include "aclnn_dynamic_quant_runner.h"
#include "aclnn_ascend_quant_runner.h"
#include "elewise_operation.h"

namespace atb {
const uint32_t TENSOR_NUM_ONE = 1;
const uint32_t TENSOR_NUM_TWO = 2;
const uint32_t TENSOR_NUM_THREE = 3;
const uint32_t TENSOR_IDX_ZERO = 0;
const uint32_t TENSOR_IDX_ONE = 1;
const uint32_t TENSOR_IDX_TWO = 2;
const uint32_t MAX_VALUE1 = 26624;
const uint32_t MAX_VALUE2 = 7552;
const uint32_t MAX_VALUE3 = 4096;
const uint32_t THIRTY_TWO = 32;

template <> Status CreateOperation(const infer::ElewiseParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }

    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        if (opParam.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT) {
            if (AclnnDynamicQuantRunner::LoadMethod() != NO_ERROR) {
                ATB_LOG(ERROR) << "Load aclnn function failed, please check your CANN version.";
                return ERROR_CANN_ERROR;
            }
        } else if (opParam.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL) {
            if (AclnnAscendQuantRunner::LoadMethod() != NO_ERROR) {
                ATB_LOG(ERROR) << "Load aclnn function failed, please check your CANN version.";
                return ERROR_CANN_ERROR;
            }
        }
    }

    OP_PARAM_RSV_CHECK(opParam);
    OP_PARAM_RSV_CHECK(opParam.mulsParam);
    OP_PARAM_RSV_CHECK(opParam.quantParam);
    if ((!GetSingleton<Config>().Is910B()) &&
        (opParam.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL)) {
        ATB_LOG(ERROR) << "Elewise dequant only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    if ((!GetSingleton<Config>().Is910B() &&
         Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_950) &&
        (opParam.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_QUANT)) {
        ATB_LOG(ERROR) << "Elewise quant is not supported on this product series.";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT &&
        opParam.quantParam.asymmetric) {
        ATB_LOG(ERROR) << "Elewise dynamic quant doesn't support asymmetric";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.elewiseType <= infer::ElewiseParam::ElewiseType::ELEWISE_UNDEFINED ||
        opParam.elewiseType >= infer::ElewiseParam::ElewiseType::ELEWISE_TYPE_MAX) {
        ATB_LOG(ERROR) << "Please enter the correct elewiseType";
        return ERROR_INVALID_PARAM;
    }
    Status status = ElewiseAclnnRunner::LoadMethod();
    if (status != NO_ERROR) {
        ATB_LOG(ERROR) << "Load Aclnn functions failed!";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) ElewiseOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

ElewiseOperation::ElewiseOperation(const infer::ElewiseParam &param) : OperationBase("ElewiseOperation"), param_(param)
{
    bool IS_310B = GetSingleton<Config>().Is310B();
    bool IS_950 = Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950;
    static std::map<infer::ElewiseParam::ElewiseType, std::string> opIniTable = {
        {infer::ElewiseParam::ElewiseType::ELEWISE_CAST,
         IS_310B ? "ElewiseOperationCastAtlas200I500A2" : "ElewiseOperationCast"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MULS,
         IS_310B ? "ElewiseOperationMulsAtlas200I500A2" : "ElewiseOperationMuls"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_COS, "ElewiseOperationCos"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SIN, "ElewiseOperationSin"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_NEG, "ElewiseOperationNeg"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT,
         IS_950 ? "ElewiseOperationQuantAscend950" : "ElewiseOperationQuant"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, "ElewiseOperationLogicalNot"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_ADD,
         IS_310B ? "ElewiseOperationAddAtlas200I500A2" : "ElewiseOperationAdd"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SUB, "ElewiseOperationSub"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MUL,
         IS_310B ? "ElewiseOperationMulAtlas200I500A2" : "ElewiseOperationMul"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, "ElewiseOperationRealDiv"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, "ElewiseOperationLogicalAnd"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, "ElewiseOperationLogicalOr"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LESS, "ElewiseOperationLess"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, "ElewiseOperationGreater"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, "ElewiseOperationEqual"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL, "ElewiseOperationQuantPerChannel"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL, "ElewiseOperationDequantPerChannel"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT,
         IS_950 ? "ElewiseOperationDynamicQuant950" : "ElewiseOperationDynamicQuant"},
        {infer::ElewiseParam::ElewiseType::ELEWISE_TANH, "ElewiseOperationTanh"},
    };
    std::map<infer::ElewiseParam::ElewiseType, std::string>::const_iterator it = opIniTable.find(param_.elewiseType);
    if (it != opIniTable.end()) {
        if (it->first == infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT && param_.quantParam.asymmetric) {
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(it->second + "Asymmetric");
        } else {
            operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(it->second);
        }
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "param_.elewiseType is invalid, type:" << param_.elewiseType;
    }
}

ElewiseOperation::~ElewiseOperation() {}

uint32_t ElewiseOperation::GetInputNum() const
{
    static std::map<infer::ElewiseParam::ElewiseType, uint32_t> inTensorNumTable = {
        {infer::ElewiseParam::ElewiseType::ELEWISE_CAST, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MULS, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_COS, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SIN, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_NEG, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_TANH, TENSOR_NUM_ONE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_ADD, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_SUB, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_MUL, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_AND, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_OR, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_LESS, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_GREATER, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_EQUAL, TENSOR_NUM_TWO},
        {infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL, TENSOR_NUM_THREE},
        {infer::ElewiseParam::ElewiseType::ELEWISE_DEQUANT_PER_CHANNEL, TENSOR_NUM_THREE},
    };
    std::map<infer::ElewiseParam::ElewiseType, uint32_t>::const_iterator it = inTensorNumTable.find(param_.elewiseType);
    if (it != inTensorNumTable.end()) {
        return it->second;
    }
    ATB_LOG(ERROR) << "param_.elewiseType is invalid, type:" << param_.elewiseType;
    return NO_ERROR;
}

uint32_t ElewiseOperation::GetOutputNum() const
{
    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT) {
        return param_.quantParam.asymmetric ? TENSOR_NUM_THREE : TENSOR_NUM_TWO;
    }
    return TENSOR_NUM_ONE;
}

Status ElewiseOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(TENSOR_IDX_ZERO) = inTensorDescs.at(TENSOR_IDX_ZERO);

    infer::ElewiseParam::ElewiseType type = param_.elewiseType;
    if (type == infer::ElewiseParam::ELEWISE_MULS || type == infer::ElewiseParam::ELEWISE_COS ||
        type == infer::ElewiseParam::ELEWISE_SIN || type == infer::ElewiseParam::ELEWISE_NEG ||
        type == infer::ElewiseParam::ELEWISE_LOGICAL_NOT || type == infer::ElewiseParam::ELEWISE_TANH) {
        return NO_ERROR;
    }
    switch (type) {
        case infer::ElewiseParam::ELEWISE_CAST:
            return InferShapeImplCast(inTensorDescs, outTensorDescs);
        case infer::ElewiseParam::ELEWISE_QUANT:
            if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
                return InferShapeImplQuantAscend950(inTensorDescs, outTensorDescs);
            } else {
                return InferShapeImplQuant(inTensorDescs, outTensorDescs);
            }
        case infer::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL:
            return InferShapeImplQuantChannel(inTensorDescs, outTensorDescs);
        case infer::ElewiseParam::ELEWISE_DEQUANT_PER_CHANNEL:
            return InferShapeImplDequantChannel(inTensorDescs, outTensorDescs);
        case infer::ElewiseParam::ELEWISE_ADD:
        case infer::ElewiseParam::ELEWISE_MUL:
        case infer::ElewiseParam::ELEWISE_REALDIV:
        case infer::ElewiseParam::ELEWISE_LESS:
        case infer::ElewiseParam::ELEWISE_GREATER:
        case infer::ElewiseParam::ELEWISE_SUB:
        case infer::ElewiseParam::ELEWISE_EQUAL:
            return InferShapeCommon(inTensorDescs, outTensorDescs);
        case infer::ElewiseParam::ELEWISE_LOGICAL_AND:
        case infer::ElewiseParam::ELEWISE_LOGICAL_OR:
            {
                Status status = InferShapeCommon(inTensorDescs, outTensorDescs);
                if (status == 0) {
                    outTensorDescs.at(0).dtype = ACL_INT8;
                }
                return status;
            }
        case infer::ElewiseParam::ELEWISE_DYNAMIC_QUANT:
            return InferShapeImplDynamicQuant(inTensorDescs, outTensorDescs);
        default:
            ATB_LOG(WARN) << "ElewiseOperation InferShapeImpl: no matched elewiseType.";
            return ERROR_INVALID_PARAM;
    }
}

Status ElewiseOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const infer::ElewiseParam::ElewiseType type = param_.elewiseType;
    const bool IS_950 = Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950;
    switch (type) {
        case infer::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL:
            if (IS_950) {
                // 此处校验如果IN_TENSOR_1的维度不是全1，就要和IN_TENSOR_0的尾部维度完全一致
                if ([](const Dims &shape) {
                        for (size_t i = 0; i < shape.dimNum; ++i) {
                            if (shape.dims[i] != 1) {
                                return true;
                            }
                        }
                        return false;
                    }(inTensorDescs[TENSOR_IDX_ONE].shape)) {
                    if (inTensorDescs[TENSOR_IDX_ZERO].shape.dimNum < inTensorDescs[TENSOR_IDX_ONE].shape.dimNum) {
                        ATB_LOG(ERROR) << GetLogPrefix()
                                       << "IN_TENSOR_0's dimNum should be greater than IN_TENSOR_1's dimNum";
                        return ERROR_INVALID_TENSOR_DIM_NUM;
                    }
                    for (size_t i = 1; i <= inTensorDescs[TENSOR_IDX_ONE].shape.dimNum; ++i) {
                        if (inTensorDescs[TENSOR_IDX_ZERO]
                                .shape.dims[inTensorDescs[TENSOR_IDX_ZERO].shape.dimNum - i] !=
                            inTensorDescs[TENSOR_IDX_ONE].shape.dims[inTensorDescs[TENSOR_IDX_ONE].shape.dimNum - i]) {
                            ATB_LOG(ERROR) << GetLogPrefix() << "IN_TENSOR_0's dims["
                                           << inTensorDescs[TENSOR_IDX_ZERO].shape.dimNum - i << "]: "
                                           << inTensorDescs[TENSOR_IDX_ZERO]
                                                  .shape.dims[inTensorDescs[TENSOR_IDX_ZERO].shape.dimNum - i]
                                           << " should match IN_TENSOR_1's dims["
                                           << inTensorDescs[TENSOR_IDX_ONE].shape.dimNum - i << "]: "
                                           << inTensorDescs[TENSOR_IDX_ONE]
                                                  .shape.dims[inTensorDescs[TENSOR_IDX_ONE].shape.dimNum - i];
                            return ERROR_INVALID_TENSOR_DIM;
                        }
                    }
                }
                if (!TensorCheck::IsEmptyTensor(inTensorDescs[TENSOR_IDX_TWO]) &&
                    !TensorUtil::TensorShapeEqual(inTensorDescs[TENSOR_IDX_ONE].shape,
                                                  inTensorDescs[TENSOR_IDX_TWO].shape)) {
                    ATB_LOG(ERROR) << GetLogPrefix() << "IN_TENSOR_1's shape should match IN_TENSOR_2's shape";
                    return ERROR_INVALID_TENSOR_DIM;
                }
            }
            break;

        default:
            return NO_ERROR;
    }
    return NO_ERROR;
}

Status ElewiseOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    inTensorDescs.reserve(inTensors.size());
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    const infer::ElewiseParam::ElewiseType type = param_.elewiseType;
    const bool IS_950 = Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950;
    switch (type) {
        case infer::ElewiseParam::ELEWISE_QUANT_PER_CHANNEL:
            if (IS_950) {
                if (!TensorUtil::TensorShapeEqual(inTensors[TENSOR_IDX_ZERO].desc.shape,
                                                  outTensors[TENSOR_IDX_ZERO].desc.shape)) {
                    ATB_LOG(ERROR) << GetLogPrefix() << "IN_TENSOR_1's shape should match IN_TENSOR_2's shape";
                    return ERROR_INVALID_TENSOR_DIM;
                }
            }
            break;
        default:
            return NO_ERROR;
    }
    return NO_ERROR;
}

Status ElewiseOperation::InferShapeImplCast(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    aclDataType dtype = inTensorDescs.at(TENSOR_IDX_ZERO).dtype;
    if (param_.outTensorType == ACL_DT_UNDEFINED) {
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = (dtype == ACL_FLOAT) ? ACL_FLOAT16 : ACL_FLOAT;
    } else if (param_.outTensorType == ACL_FLOAT16 || param_.outTensorType == ACL_FLOAT ||
               param_.outTensorType == ACL_INT32 || param_.outTensorType == ACL_INT64 ||
               param_.outTensorType == ACL_BF16) {
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = param_.outTensorType;
    } else {
        ATB_LOG(WARN) << "ElewiseOperation InferShapeImpl cast: no matched input desc.";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status ElewiseOperation::InferShapeImplQuant(const SVector<TensorDesc> &inTensorDescs,
                                             SVector<TensorDesc> &outTensorDescs) const
{
    uint64_t intensorZeroDimNum = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dimNum;
    int64_t intensorZeroLastDim = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dims[intensorZeroDimNum - 1];
    if (intensorZeroLastDim % THIRTY_TWO != 0) {
        ATB_LOG(ERROR) << "ElewiseOperation InferShapeImpl quant: last dim is not 32 bytes align.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    aclDataType dtype = inTensorDescs.at(TENSOR_IDX_ZERO).dtype;
    if (dtype == ACL_FLOAT16) {
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = ACL_INT8;
        return NO_ERROR;
    } else {
        ATB_LOG(WARN) << "ElewiseOperation InferShapeImpl quant: no matched input desc.";
        return ERROR_INVALID_PARAM;
    }
}

Status ElewiseOperation::InferShapeImplQuantAscend950(const SVector<TensorDesc> &inTensorDescs,
                                                      SVector<TensorDesc> &outTensorDescs) const
{
    aclDataType dtype0 = inTensorDescs.at(TENSOR_IDX_ZERO).dtype;
    if ((dtype0 == ACL_FLOAT) || (dtype0 == ACL_FLOAT16) || (dtype0 == ACL_BF16)) {
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype =
            param_.outTensorType == ACL_DT_UNDEFINED ? ACL_INT8 : param_.outTensorType;
        return NO_ERROR;
    } else {
        ATB_LOG(WARN) << "ElewiseOperation InferShapeImplQuantAscend950: no matched input desc.";
        return ERROR_INVALID_PARAM;
    }
}


Status ElewiseOperation::InferShapeImplDynamicQuant(const SVector<TensorDesc> &inTensorDescs,
                                                    SVector<TensorDesc> &outTensorDescs) const
{
    if (inTensorDescs.at(TENSOR_IDX_ZERO).shape.dimNum < 2) { // 2: min support dims
        ATB_LOG(ERROR) << "InferShapeImplDynamicQuant: input dims must greater than or equal to 2.";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    aclDataType dtype = inTensorDescs.at(TENSOR_IDX_ZERO).dtype;
    if (dtype == ACL_FLOAT16) {
        if (GetSingleton<Config>().Is910B() && !InferShapeCheckDynamicQuant(inTensorDescs)) {
            ATB_LOG(ERROR) << "In Atlas 800I A2 inference product, ElewiseOperation InferShapeImplDynamicQuant:"
                           << " when the dtype of the first tensor is float16,"
                           << " the size of the last dimension does not exceed 26624.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (!GetSingleton<Config>().Is910B() && !InferShapeCheckDynamicQuant310P(inTensorDescs)) {
            ATB_LOG(ERROR) << "In Atlas inference products, ElewiseOperation InferShapeImplDynamicQuant:"
                           << " when the dtype of the first tensor is float16,"
                           << " the size of the last dimension does not exceed 4096.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = ACL_INT8;
        outTensorDescs.at(TENSOR_IDX_ONE) = inTensorDescs.at(TENSOR_IDX_ZERO);
        outTensorDescs.at(TENSOR_IDX_ONE).dtype = ACL_FLOAT;
        outTensorDescs.at(TENSOR_IDX_ONE).shape.dimNum--;
        if (param_.quantParam.asymmetric) {
            outTensorDescs.at(TENSOR_IDX_TWO) = outTensorDescs.at(TENSOR_IDX_ONE);
        }
    } else if (dtype == ACL_BF16) {
        if (GetSingleton<Config>().Is910B() && !InferShapeCheckDynamicQuant(inTensorDescs)) {
            ATB_LOG(ERROR) << "In Atlas 800I A2 inference product, ElewiseOperation InferShapeImplDynamicQuant:"
                           << " when the dtype of the first tensor is BF16,"
                           << " the size of the last dimension does not exceed 7552.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = ACL_INT8;
        outTensorDescs.at(TENSOR_IDX_ONE) = inTensorDescs.at(TENSOR_IDX_ZERO);
        outTensorDescs.at(TENSOR_IDX_ONE).dtype = ACL_FLOAT;
        outTensorDescs.at(TENSOR_IDX_ONE).shape.dimNum--;
    } else {
        ATB_LOG(WARN) << "ElewiseOperation InferShapeImplDynamicQuant: inTensor only support FP16 and BF16 now.";
        return ERROR_INVALID_PARAM;
    }
    if (param_.outTensorType == ACL_HIFLOAT8 || param_.outTensorType == ACL_FLOAT8_E5M2 ||
        param_.outTensorType == ACL_FLOAT8_E4M3FN) {
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = param_.outTensorType;
    }
    return NO_ERROR;
}

Status ElewiseOperation::InferShapeCommon(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    infer::ElewiseParam::ElewiseType type = param_.elewiseType;
    aclDataType dtype0 = inTensorDescs.at(TENSOR_IDX_ZERO).dtype;
    aclDataType dtype1 = inTensorDescs.at(TENSOR_IDX_ONE).dtype;
    if (dtype0 != dtype1) {
        ATB_LOG(WARN) << "ElewiseOperation InferShapeCommon: unsupported input internsor.";
        return ERROR_INVALID_PARAM;
    }

    uint64_t dimNum0 = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dimNum;
    uint64_t dimNum1 = inTensorDescs.at(TENSOR_IDX_ONE).shape.dimNum;
    uint64_t largerSize = (dimNum0 >= dimNum1) ? dimNum0 : dimNum1;
    outTensorDescs.at(TENSOR_IDX_ZERO).shape.dimNum = largerSize;
    int64_t idx = 0;
    while (idx < static_cast<int64_t>(largerSize)) {
        auto dimIdxOut = static_cast<int64_t>(largerSize) - idx - 1;
        auto dimIdx0 = static_cast<int64_t>(dimNum0) - idx - 1;
        auto dimIdx1 = static_cast<int64_t>(dimNum1) - idx - 1;
        auto dim0 = dimIdx0 < 0 ? 1 : inTensorDescs.at(0).shape.dims[dimIdx0];
        auto dim1 = dimIdx1 < 0 ? 1 : inTensorDescs.at(1).shape.dims[dimIdx1];
        if (dim0 != 1 && dim1 != 1 && dim0 != dim1) {
            ATB_LOG(WARN) << "ElewiseOperation InferShapeCommon: cannot broadcast this dim.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        auto dimOut = dim0 >= dim1 ? dim0 : dim1;
        outTensorDescs.at(0).shape.dims[dimIdxOut] = dimOut;
        idx++;
    }

    if (type == infer::ElewiseParam::ELEWISE_LESS || type == infer::ElewiseParam::ELEWISE_GREATER ||
        type == infer::ElewiseParam::ELEWISE_EQUAL) {
        outTensorDescs.at(0).dtype = ACL_INT8;
    }

    return NO_ERROR;
}

Status ElewiseOperation::InferShapeImplQuantChannel(const SVector<TensorDesc> &inTensorDescs,
                                                    SVector<TensorDesc> &outTensorDescs) const
{
    (void)inTensorDescs;
    outTensorDescs.at(TENSOR_IDX_ZERO).dtype = ACL_INT8;
    return NO_ERROR;
}

Status ElewiseOperation::InferShapeImplDequantChannel(const SVector<TensorDesc> &inTensorDescs,
                                                      SVector<TensorDesc> &outTensorDescs) const
{
    aclDataType dtype0 = inTensorDescs.at(TENSOR_IDX_ZERO).dtype;
    aclDataType dtype1 = inTensorDescs.at(TENSOR_IDX_ONE).dtype;
    aclDataType dtype2 = inTensorDescs.at(TENSOR_IDX_TWO).dtype;
    if ((dtype0 == ACL_INT8) && (dtype1 == ACL_FLOAT16) && (dtype2 == ACL_INT8)) {
        outTensorDescs.at(TENSOR_IDX_ZERO).dtype = ACL_FLOAT16;
        return NO_ERROR;
    } else {
        ATB_LOG(WARN) << "ElewiseOperation InferShapeImpl dequant channel: no matched input desc.";
        return ERROR_INVALID_PARAM;
    }
}

SVector<bool> ElewiseOperation::GetEmptyInTensorPermissions() const
{
    SVector<bool> v;

    // quant_per_channel dequant_per_channel intensor[2] allows null
    if (GetInputNum() == TENSOR_NUM_THREE) {
        SVector<bool> emptyTensorPerms(GetInputNum(), false);
        emptyTensorPerms.at(TENSOR_NUM_THREE - 1) = true;
        return emptyTensorPerms;
    }

    return v;
}

bool ElewiseOperation::InferShapeCheckDynamicQuant(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t dimNumZeroTensor = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dimNum;
    int64_t lastDim = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dims[dimNumZeroTensor - 1];
    if (inTensorDescs.at(TENSOR_IDX_ZERO).dtype == ACL_FLOAT16 && lastDim > MAX_VALUE1) {
        return false;
    }

    if (inTensorDescs.at(TENSOR_IDX_ZERO).dtype == ACL_BF16 && lastDim > MAX_VALUE2) {
        return false;
    }
    return true;
}

bool ElewiseOperation::InferShapeCheckDynamicQuant310P(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t dimNumZeroTensor = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dimNum;
    int64_t lastDim = inTensorDescs.at(TENSOR_IDX_ZERO).shape.dims[dimNumZeroTensor - 1];
    if (lastDim > MAX_VALUE3) {
        return false;
    }
    return true;
}

std::shared_ptr<Runner> ElewiseOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_DYNAMIC_QUANT &&
        Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        return std::make_shared<AclnnDynamicQuantRunner>(param_);
    } else if (param_.elewiseType == infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL &&
               Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        return std::make_shared<AclnnAscendQuantRunner>(param_);
    }
    std::unordered_set<infer::ElewiseParam::ElewiseType> aclnnOpType = {
        infer::ElewiseParam::ElewiseType::ELEWISE_CAST,    infer::ElewiseParam::ElewiseType::ELEWISE_MULS,
        infer::ElewiseParam::ElewiseType::ELEWISE_COS,     infer::ElewiseParam::ElewiseType::ELEWISE_SIN,
        infer::ElewiseParam::ElewiseType::ELEWISE_ADD,     infer::ElewiseParam::ElewiseType::ELEWISE_LESS,
        infer::ElewiseParam::ElewiseType::ELEWISE_MUL,     infer::ElewiseParam::ElewiseType::ELEWISE_GREATER,
        infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV, infer::ElewiseParam::ElewiseType::ELEWISE_LOGICAL_NOT,
        infer::ElewiseParam::ElewiseType::ELEWISE_QUANT,   infer::ElewiseParam::ElewiseType::ELEWISE_SUB,
    };
    if (aclnnOpType.find(param_.elewiseType) != aclnnOpType.end() &&
        Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        return std::make_shared<ElewiseAclnnRunner>(param_);
    }
    return std::make_shared<ElewiseOpsRunner>(param_);
}

nlohmann::json ElewiseOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb