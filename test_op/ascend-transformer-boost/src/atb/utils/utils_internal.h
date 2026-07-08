/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_UTILS_INTERNAL_H
#define ATB_UTILS_INTERNAL_H

#include <cstdint>
#include "acl/acl.h"

namespace atb {
class UtilsInternal {
public:
    static bool IsFloatEqual(float lh, float rh);
    static bool IsDoubleEqual(double lh, double rh);
    static int32_t GetCurrentThreadId();
    static int32_t GetCurrentProcessId();
    static int64_t AlignUp(int64_t dim, int64_t align);
    //!
    //! \brief 返回aclDataType对应的数据类型大小。
    //!
    //! \param dType 传入数据类型
    //!
    //! \return 返回整数值对应aclDataType大小
    //!
    static uint64_t GetDataTypeSize(const aclDataType &dType);
};
} // namespace atb
#endif