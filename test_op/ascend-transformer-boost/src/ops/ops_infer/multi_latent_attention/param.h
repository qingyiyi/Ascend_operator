/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_MULTI_LATENT_ATTENTION_PARAM_H
#define ATB_MULTI_LATENT_ATTENTION_PARAM_H

#include <mki/utils/SVector/SVector.h>
#include <mki/tensor.h>
#include <vector>
#include <acl/acl.h>
#include "atb/svector.h"

namespace atb {
struct MultiLatentAttentionVariantPackParam {
    std::vector<int32_t> contextLens;
    std::vector<int32_t> qSeqlen;
    std::vector<int32_t> maskUseStatus;
    bool BuildFromTensor(const SVector<Mki::Tensor> &inTensors, size_t contextLensTensorId, size_t qSeqlenTensorId,
                         bool needQLens);
    bool BuildMaskUseStatusFromTensor(const SVector<Mki::Tensor> &inTensors, size_t maskUseStatusId,
                                      bool needMaskUseStatus, size_t batch);
};
} // namespace atb
#endif