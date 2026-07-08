/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SELF_ATTENTION_ENCODER_FUSION_OPS_RUNNER_910A_H
#define SELF_ATTENTION_ENCODER_FUSION_OPS_RUNNER_910A_H
#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "param.h"


namespace atb {
class SelfAttentionEncoderFusionOpsRunner910A : public OpsRunner {
public:
    explicit SelfAttentionEncoderFusionOpsRunner910A(const infer::SelfAttentionParam &param);
    ~SelfAttentionEncoderFusionOpsRunner910A() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;
    void SetFAParam(AtbOps::OpParam::UnpadFlashAttentionNz &flashAttentionParam);
    bool NeedModifySlopes(const OpsTensorPack &opsTensorPack);

private:
    infer::SelfAttentionParam param_;
    SelfAttentionFusionVariantPackParam newParam_;
    // BSND (1, bs, nd) last two dim shapes: bs, nd
    // BNSD (bn, s, d) last two dim shapes: s, d
    int64_t qDim1_ = 0;
    int64_t qDim2_ = 0;
    int64_t vDim1_ = 0;
    int64_t vDim2_ = 0;

    bool isBNSD_ = false; // specify use either BSND/BNSD
    Mki::Tensor nullTensor_ = {}; // 空tensor，作为layerId
};
} // namespace atb
#endif // SELF_ATTENTION_ENCODER_FUSION_OPS_RUNNER_910A_H
