
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/nonzero/nonzero/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

namespace {
static constexpr int32_t BUFFER_NUM = 1;
static constexpr uint32_t ONE_LOOP_ELE = 16384;

class Nonzero {
public:
    __aicore__ inline Nonzero() {}
    __aicore__ inline void Init(__gm__ uint8_t *x,
                                __gm__ uint8_t *y,
                                __gm__ uint8_t *numTrues,
                                uint32_t *xDims,
                                uint32_t xdimLength,
                                uint32_t xNumel) {
        xDims_ = xDims;
        xdimLength_ = xdimLength;
        xNumel_ = xNumel;

        xGm_ = (__gm__ int64_t *)x;
        yGm_ = (__gm__ int64_t *)y;
        numTruesGm_ = (__gm__ int64_t *)numTrues;
        y_gm.SetGlobalBuffer(yGm_);
        pipe.InitBuffer(y_buff, ONE_LOOP_ELE);
    }
    __aicore__ inline void Process()
    {
        int64_t numTrues = 0;
        AscendC::LocalTensor<int64_t> y64_local = y_buff.Get<int64_t>();
        AscendC::LocalTensor<int32_t> y32_local = y_buff.Get<int32_t>();
        Duplicate(y32_local, (int32_t)0, ONE_LOOP_ELE / 4);
        y64_local = y32_local.ReinterpretCast<int64_t>();
        uint64_t copyTime = (static_cast<uint64_t>(xNumel_) * xdimLength_ * 8) / ONE_LOOP_ELE;
        uint64_t copyRemain = (static_cast<uint64_t>(xNumel_) * xdimLength_ * 8) % ONE_LOOP_ELE;
        for (uint64_t i = 0; i < copyTime; i++) {
            AscendC::DataCopyParams dataCopyParams{(uint16_t)1, (uint16_t)(ONE_LOOP_ELE), 0, 0};
            AscendC::DataCopyPad(y_gm[i * ONE_LOOP_ELE / 8], y64_local, dataCopyParams);
        }
        if(copyRemain != 0)
        {
            AscendC::DataCopyParams dataCopyParamsRemain{(uint16_t)1, (uint16_t)(copyRemain), 0, 0};
            AscendC::DataCopyPad(y_gm[copyTime * ONE_LOOP_ELE / 8], y64_local, dataCopyParamsRemain);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < xNumel_; i++) {
            if (xGm_[i]) {
                uint32_t numelLeft = xNumel_;
                uint32_t tmp = i;
                for (uint32_t j = 0; j < xdimLength_; j++) {
                    numelLeft /=  xDims_[j];
                    uint32_t idxThis = tmp / numelLeft;
                    y_gm.SetValue(j * xNumel_ + numTrues, idxThis);
                    tmp %= numelLeft;
                }

                numTrues++;
            }
        }
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(y_gm);
        numTruesGm_[0] = numTrues;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_buff;
    __gm__ int64_t *xGm_;
    __gm__ int64_t *yGm_;
    __gm__ int64_t *numTruesGm_;
    AscendC::GlobalTensor<int64_t> y_gm;
    uint32_t xdimLength_{1};
    uint32_t xNumel_{1};
    uint32_t *xDims_{nullptr};
};
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::NonzeroTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220)
    tilingdata->xdimLength = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->xNumel = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    for (uint32_t i = 0; i < tilingdata->xdimLength; i++) {
        tilingdata->xDims[i] = (*(const __gm__ uint32_t *)(p_tilingdata + 8 + i * sizeof(uint32_t)));
    }
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::NonzeroTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->xdimLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->xNumel = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    for (uint32_t i = 0; i < tilingdata->xdimLength; i++) {
        tilingdata->xDims[i] = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8 + i * sizeof(uint32_t)));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

extern "C" __global__ __aicore__ void nonzero(GM_ADDR x, GM_ADDR y, GM_ADDR numTrues,
                                                    GM_ADDR workspace, GM_ADDR tiling) {
    AsdOps::NonzeroTilingData tiling_data;
    InitTilingData(tiling, &(tiling_data));
    Nonzero op;
    op.Init(x, y, numTrues, tiling_data.xDims, tiling_data.xdimLength, tiling_data.xNumel);
    op.Process();
}