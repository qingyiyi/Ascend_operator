/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Open Software License version 1.0
 * as published by the Open Source Initiative (OSI-Approved Open Software License).
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * Open Software License version 1.0 for more details.
 *
 * You should have received a copy of the Open Software License
 * version 1.0 along with this program; if not, you may obtain one at
 * https://opensource.org/licenses/OSL-1.0
 */
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_COMMON_UTIL_H
#define ATB_COMMON_UTIL_H
#include <acl/acl.h>
#include <hccl/hccl_types.h>
#include <vector>
#include <string>
#include "atb/svector.h"
#include "atb/types.h"
namespace atb {
template <typename T> std::vector<T> SVectorToVector(const SVector<T> &svector)
{
    std::vector<T> tmpVec;
    tmpVec.resize(svector.size());
    for (size_t i = 0; i < svector.size(); i++) {
        tmpVec.at(i) = svector.at(i);
    }
    return tmpVec;
}

std::string GenerateOperationName(const std::string &opType, const std::vector<int64_t> &ids);
HcclDataType GetHcclDtype(const aclDataType dtype);
HcclReduceOp GetAllReduceType(const std::string &allReduceType);
atb::Status ConvertHcclResultToStatus(const HcclResult hcclResult);
atb::Status ConvertLcclResultToStatus(const int lcclResult);
} // namespace atb
#endif