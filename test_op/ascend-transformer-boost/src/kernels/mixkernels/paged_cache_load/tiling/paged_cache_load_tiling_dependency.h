/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_PAGED_CACHE_LOAD_TILING_DEPENDENCY_H
#define ASCEND_OPS_PAGED_CACHE_LOAD_TILING_DEPENDENCY_H

#include <cstdint>

namespace AtbOps {

struct PagedCacheLoadTilingData {
    int32_t blockSize;
    int32_t numTokens;
    int32_t numblkTabCol;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    int32_t typeByteK;
    int32_t hasSeqStarts;
    int32_t cuSeqLens;
    int32_t typeByteV;
};
}

#endif // ASCEND_OPS_PAGED_CACHE_LOAD_TILING_DEPENDENCY_H
