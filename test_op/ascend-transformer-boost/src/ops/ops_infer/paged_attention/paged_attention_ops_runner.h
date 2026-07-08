/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PAGED_ATTENTION_OPS_RUNNER_H
#define PAGED_ATTENTION_OPS_RUNNER_H

#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "param.h"

namespace atb {
class PagedAttentionOpsRunner : public OpsRunner {
public:
    std::size_t IntensorSizeGenerate();
    explicit PagedAttentionOpsRunner(const infer::PagedAttentionParam &param);
    ~PagedAttentionOpsRunner() override;
    void SetPaParam(AtbOps::OpParam::PagedAttention &inPagedAttention);

private:
    AtbOps::OpParam::PagedAttention::MaskType GetMaskType() const;
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::PagedAttentionParam param_;
    Mki::Tensor nullTensor_ = {}; // 空tensor，作为{true, 0}场景占位符
    PagedAttentionFusionVariantPackParam newParam_;
    bool isMla_ = false;
    bool needQScale_ = false;
};
} // namespace atb

#endif
