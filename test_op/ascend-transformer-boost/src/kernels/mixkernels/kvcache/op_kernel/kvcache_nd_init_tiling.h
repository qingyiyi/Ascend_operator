/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef KVCACHE_ND_INIT_TILING_H
#define KVCACHE_ND_INIT_TILING_H

#include "kernel_operator.h"
#include "../tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::KVCacheTilingData *tilingData)
{
#if defined(__CCE_KT_TEST__) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220)
    tilingData->batch = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingData->maxSeqLen = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingData->hiddenSize = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::KVCacheTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingData->batch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingData->maxSeqLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingData->hiddenSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#endif