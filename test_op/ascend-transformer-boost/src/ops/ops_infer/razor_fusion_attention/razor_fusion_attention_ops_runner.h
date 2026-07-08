/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RAZOR_FUSION_ATTENTION_RUNNER_H
#define ATB_RAZOR_FUSION_ATTENTION_RUNNER_H

#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class RazorFusionAttentionOpsRunner : public OpsRunner {
public:
    explicit RazorFusionAttentionOpsRunner(const infer::RazorFusionAttentionParam &param);
    ~RazorFusionAttentionOpsRunner() override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;
    void SetRFAParam(AtbOps::OpParam::UnpadFlashAttention &attentionParam);
    infer::RazorFusionAttentionParam param_;
};
} // namespace atb
#endif