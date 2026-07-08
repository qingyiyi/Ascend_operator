/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_STUB_FRAMEWORK_COMMON_UTIL_H
#define ASCEND_OPS_STUB_FRAMEWORK_COMMON_UTIL_H

#include <functional>

#include "debug/ge_log.h"
#include "debug/log.h"
#include "external/ge_common/ge_api_error_codes.h"

namespace ge {
} // namespace ge

#define GE_RETURN_IF_ERROR(expr)                \
    do {                                        \
        const ge::Status _chk_status = (expr);  \
        if (_chk_status != ge::SUCCESS) {       \
            return _chk_status;                 \
        }                                       \
    } while (false)

#endif // ASCEND_OPS_STUB_FRAMEWORK_COMMON_UTIL_H
