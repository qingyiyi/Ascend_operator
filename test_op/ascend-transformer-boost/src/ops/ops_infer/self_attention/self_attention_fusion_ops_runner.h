/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SELF_ATTENTION_FUSION_OPS_RUNNER_H
#define ATB_SELF_ATTENTION_FUSION_OPS_RUNNER_H

#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"
#include "param.h"


namespace atb {
class SelfAttentionFusionOpsRunner : public OpsRunner {
public:
    explicit SelfAttentionFusionOpsRunner(const infer::SelfAttentionParam &param);
    ~SelfAttentionFusionOpsRunner() override;

private:
    Status ModifyKernelGraph(const OpsTensorPack &opsTensorPack) override;
    AtbOps::OpParam::UnpadFlashAttention::Type GetFaType();
    void SetFAParam(AtbOps::OpParam::UnpadFlashAttention &flashAttentionParam);
    void SetKVCacheParam(AtbOps::OpParam::KVCache &kCacheParam, bool kvcacheWithParam);
    bool ModifyKVCacheNode(const size_t tokenOffsetTensorId);

private:
    infer::SelfAttentionParam param_;
    SelfAttentionFusionVariantPackParam newParam_;
    Mki::Tensor nullTensor_ = {}; // 空tensor
};
} // namespace atb
#endif // ATB_SELF_ATTENTION_FUSION_OPS_RUNNER_H
