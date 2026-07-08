/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_ROPE_ACLNN_RUNNER_H
#define ATB_ROPE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"
namespace atb {
using aclnnGetWorkspaceSizeFuncPtr = aclnnStatus(*)(
        const aclTensor *, // queryRef
        const aclTensor *, // keyRef
        const aclTensor *, // cos
        const aclTensor *, // sin
        int64_t,           // layout
        char *, // rotaryMode
        uint64_t *, //workspaceSize
        aclOpExecutor** //executor
    );
using aclnnExecuteFuncPtr = aclnnStatus(*)(
        void *,
        uint64_t,
        aclOpExecutor *,
        aclrtStream
    );
class RopeAclnnRunner : public AclnnRunner {
public:
    explicit RopeAclnnRunner(const infer::RopeParam &param);
    ~RopeAclnnRunner() override;

    static Status LoadMethod();
protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
private:
    infer::RopeParam param_;
    //两个函数指针分别对应对应aclnnop/aclnn_apply_rotary_pos_emb_v2.h中的两段式接口
    static aclnnGetWorkspaceSizeFuncPtr aclnnGetWorkspaceSizeFunc_;
    static aclnnExecuteFuncPtr aclnnExecuteFunc_;
};
}// namespace atb

#endif
