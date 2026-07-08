/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CORE_PARAMS_PARAMS_H
#define CORE_PARAMS_PARAMS_H
#include "asdops/params/activation.h"
#include "asdops/params/asstrided.h"
#include "asdops/params/concat.h"
#include "asdops/params/copy.h"
#include "asdops/params/cumsum.h"
#include "asdops/params/elewise.h"
#include "asdops/params/expand.h"
#include "asdops/params/fill.h"
#include "asdops/params/gather.h"
#include "asdops/params/index.h"
#include "asdops/params/matmul.h"
#include "asdops/params/multinomial.h"
#include "asdops/params/norm.h"
#include "asdops/params/onehot.h"
#include "asdops/params/reduce.h"
#include "asdops/params/reverse.h"
#include "asdops/params/slice.h"
#include "asdops/params/sort.h"
#include "asdops/params/split.h"
#include "asdops/params/softmax.h"
#include "asdops/params/transdata.h"
#include "asdops/params/transpose.h"
#include "asdops/params/dynamic_ntk.h"
#include "asdops/params/group_topk.h"
#include "asdops/params/zeroslike.h"
#include "asdops/params/scatter_elements_v2.h"
#include "asdops/params/logprobs.h"
#include "asdops/params/faUpdate.h"
#include "asdops/params/moe_gate_corr.h"

namespace AsdOps {
namespace OpParam {
class AllParams {
    AsdOps::OpParam::Activation activation;
    AsdOps::OpParam::AsStrided asStrided;
    AsdOps::OpParam::Concat concat;
    AsdOps::OpParam::Copy copy;
    AsdOps::OpParam::Cumsum cumsum;
    AsdOps::OpParam::Elewise elewise;
    AsdOps::OpParam::Expand expand;
    AsdOps::OpParam::Fill fill;
    AsdOps::OpParam::Gather gather;
    AsdOps::OpParam::Index index;
    AsdOps::OpParam::MatMul matMul;
    AsdOps::OpParam::Multinomial multinomial;
    AsdOps::OpParam::Norm norm;
    AsdOps::OpParam::Onehot onehot;
    AsdOps::OpParam::Reduce reduce;
    AsdOps::OpParam::Reverse reverse;
    AsdOps::OpParam::Slice slice;
    AsdOps::OpParam::Sort sort;
    AsdOps::OpParam::Split split;
    AsdOps::OpParam::Softmax softmax;
    AsdOps::OpParam::Transdata transdata;
    AsdOps::OpParam::Transpose transpose;
    AsdOps::OpParam::DynamicNTK dynamicNTK;
    AsdOps::OpParam::GroupTopk groupTopk;
    AsdOps::OpParam::ZerosLike zerosLike;
    AsdOps::OpParam::ScatterElementsV2 scatterElementsV2;
    AsdOps::OpParam::LogprobsSample logprobsSample;
    AsdOps::OpParam::FaUpdate faUpdate;
    AsdOps::OpParam::MoeGateCorr moeGateCorr;
};
} // namespace OpParam
} // namespace AsdOps

#endif
