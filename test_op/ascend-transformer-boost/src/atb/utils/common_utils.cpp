/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/utils/common_utils.h"
#include <sstream>
#include <atb/utils/log.h>
#include <unordered_map>
#include "lccl.h"
namespace atb {
std::string GenerateOperationName(const std::string &opType, const std::vector<int64_t> &ids)
{
    std::string opName = opType;
    for (size_t i = 0; i < ids.size(); i++) {
        opName += "_";
        opName += std::to_string(ids.at(i));
    }
    return opName;
}

HcclDataType GetHcclDtype(const aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT:
            return HCCL_DATA_TYPE_FP32;
        case ACL_FLOAT16:
            return HCCL_DATA_TYPE_FP16;
        case ACL_INT8:
            return HCCL_DATA_TYPE_INT8;
        case ACL_INT32:
            return HCCL_DATA_TYPE_INT32;
        case ACL_UINT8:
            return HCCL_DATA_TYPE_UINT8;
        case ACL_INT16:
            return HCCL_DATA_TYPE_INT16;
        case ACL_UINT16:
            return HCCL_DATA_TYPE_UINT16;
        case ACL_UINT32:
            return HCCL_DATA_TYPE_UINT32;
        case ACL_INT64:
            return HCCL_DATA_TYPE_INT64;
        case ACL_BF16:
            return HCCL_DATA_TYPE_BFP16;
        default:
            ATB_LOG(ERROR) << "not support dtype:" << dtype;
            return static_cast<HcclDataType>(255); // RESERVED TYPE
    }
}

HcclReduceOp GetAllReduceType(const std::string &allReduceType)
{
    if (allReduceType == "sum") {
        return HCCL_REDUCE_SUM;
    } else if (allReduceType == "prod") {
        return HCCL_REDUCE_PROD;
    } else if (allReduceType == "max") {
        return HCCL_REDUCE_MAX;
    } else if (allReduceType == "min") {
        return HCCL_REDUCE_MIN;
    } else {
        return HCCL_REDUCE_RESERVED;
    }
}

const std::unordered_map<HcclResult, Status> hcclResultToStatusMap = {
    {HcclResult::HCCL_SUCCESS, NO_ERROR},
    {HcclResult::HCCL_E_PARA, ERROR_INVALID_PARAM},
    {HcclResult::HCCL_E_MEMORY, ERROR_OUT_OF_DEVICE_MEMORY},
    {static_cast<HcclResult>(24), ERROR_OUT_OF_DEVICE_MEMORY}, // 24: HcclResult::HCCL_E_OOM
};

Status ConvertHcclResultToStatus(const HcclResult hcclResult)
{
    auto it = hcclResultToStatusMap.find(hcclResult);
    if (it != hcclResultToStatusMap.end()) {
        return it->second;
    }
    return ERROR_HCCL_FAIL;
}

const std::unordered_map<int, Status> lcclResultToStatusMap = {
    {Lcal::LCAL_SUCCESS, NO_ERROR},
    {Lcal::LCAL_ERROR_PARA_CHECK_FAIL, ERROR_INVALID_SINGLE_OPERATION_PARAM},
    {Lcal::LCAL_ERROR_OUT_OF_MEMORY, ERROR_OUT_OF_DEVICE_MEMORY},
};

Status ConvertLcclResultToStatus(const int lcclResult)
{
    auto it = lcclResultToStatusMap.find(lcclResult);
    if (it != lcclResultToStatusMap.end()) {
        return it->second;
    }
    return ERROR_RT_FAIL;
}

} // namespace atb