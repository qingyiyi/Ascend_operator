/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_PARAMS_H
#define ATBOPS_PARAMS_PARAMS_H
#include "atbops/params/common.h"
#include "atbops/params/fastsoftmax.h"
#include "atbops/params/fastsoftmax_grad.h"
#include "atbops/params/ffn.h"
#include "atbops/params/fused_add_topk_div.h"
#include "atbops/params/gating.h"
#include "atbops/params/genattentionmask.h"
#include "atbops/params/kvcache.h"
#include "atbops/params/laser_attention.h"
#include "atbops/params/laser_attention_grad.h"
#include "atbops/params/pad.h"
#include "atbops/params/pad_with_hidden_state.h"
#include "atbops/params/pagedattention.h"
#include "atbops/params/reshape_and_cache.h"
#include "atbops/params/paged_cache_load.h"
#include "atbops/params/rope_grad.h"
#include "atbops/params/rope.h"
#include "atbops/params/stridedbatchmatmul.h"
#include "atbops/params/toppsample.h"
#include "atbops/params/toppsample_rand.h"
#include "atbops/params/unpad_flash_attention_nz.h"
#include "atbops/params/unpad_flash_attention.h"
#include "atbops/params/unpad_with_hidden_state.h"
#include "atbops/params/unpad.h"
#include "atbops/params/blockcopy.h"
#include "atbops/params/moe_gmm.h"
#include "atbops/params/gmm_add.h"
#include "atbops/params/rope_q_concat.h"
#include "atbops/params/swiglu_quant.h"
#include "atbops/params/rms_norm_and_rope_and_reshape_and_cache.h"
#include "atbops/params/mla_preprocess.h"
#include "atbops/params/mla.h"
#include "atbops/params/fusion.h"
#include "atbops/params/ring_mla.h"
#include "atbops/params/gmm_deq_swiglu_quant_gmm_deq.h"
#include "atbops/params/mm_deq_swiglu_quant_mm_deq.h"

namespace AtbOps {
namespace OpParam {
class AllParams {
    size_t dim_0 = DIM_0;
    AtbOps::OpParam::FastSoftMax fastSoftMax;
    AtbOps::OpParam::FastSoftMaxGrad fastSoftMaxGrad;
    AtbOps::OpParam::FFN ffn;
    AtbOps::OpParam::FusedAddTopkDiv fusedAddTopkDiv;
    AtbOps::OpParam::Gating gating;
    AtbOps::OpParam::GenAttentionMask genAttentionMask;
    AtbOps::OpParam::KVCache kvCache;
    AtbOps::OpParam::LaserAttention laserAttention;
    AtbOps::OpParam::LaserAttentionGrad laserAttentionGrad;
    AtbOps::OpParam::Pad pad;
    AtbOps::OpParam::PadWithHiddenState padWithHiddenState;
    AtbOps::OpParam::PagedAttention pagedAttention;
    AtbOps::OpParam::ReshapeAndCache reshapeAndCache;
    AtbOps::OpParam::PagedCacheLoad pagedCacheLoad;
    AtbOps::OpParam::RopeGrad ropeGrad;
    AtbOps::OpParam::Rope rope;
    AtbOps::OpParam::StridedBatchMatmul stridedBatchMatmul;
    AtbOps::OpParam::Toppsample toppsample;
    AtbOps::OpParam::ToppsampleRand toppsampleRand;
    AtbOps::OpParam::UnpadFlashAttentionNz unpadFlashAttentionNz;
    AtbOps::OpParam::UnpadFlashAttention unpadFlashAttention;
    AtbOps::OpParam::UnpadWithHiddenState unpadWithHiddenState;
    AtbOps::OpParam::Unpad unpad;
    AtbOps::OpParam::BlockCopy blockCopy;
    AtbOps::OpParam::MoeGmm moeGmm;
    AtbOps::OpParam::GmmAdd gmmAdd;
    AtbOps::OpParam::RopeQConcat ropeQConcat;
    AtbOps::OpParam::SwigluQuant swigluQuant;
    AtbOps::OpParam::RmsNormAndRopeAndReshapeAndCache rmsNormAndRopeAndReshapeAndCache;
    AtbOps::OpParam::MlaPreprocess mlaPreprocess;
    AtbOps::OpParam::MLA mla;
    AtbOps::OpParam::Fusion fusion;
    AtbOps::OpParam::RINGMLA ringmla;
    AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq gmmDeqSwigluQuantGmmDeq;
    AtbOps::OpParam::MmDeqSwigluQuantMmDeq mmDeqSwigluQuantMmDeq;
};
} // namespace OpParam
} // namespace AtbOps

#endif
