/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_TRANSPOSE_ACLNN_RUNNER_H
#define ATB_TRANSPOSE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnPermuteGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclIntArray *, aclTensor *, uint64_t *,
                                                     aclOpExecutor **);
using AclnnPermuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class TransposeAclnnRunner : public AclnnRunner {
public:
    explicit TransposeAclnnRunner(const infer::TransposeParam &param);
    ~TransposeAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    infer::TransposeParam param_;
    aclIntArray *permIntArray_ = nullptr;

    static AclnnPermuteGetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    static AclnnPermuteFunc aclnnExecuteFunc_;
};
} // namespace atb
#endif // ATB_TRANSPOSE_ACLNN_RUNNER_H
