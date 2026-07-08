/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GATHER_ACLNN_RUNNER_H
#define ATB_GATHER_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnGatherV3GetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, int64_t, const aclTensor *, int64_t,
                                                           int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnGatherV3Func = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class GatherAclnnRunner : public AclnnRunner {
public:
    explicit GatherAclnnRunner(const infer::GatherParam &param);
    ~GatherAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    infer::GatherParam param_;

    static AclnnGatherV3GetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    static AclnnGatherV3Func aclnnExecuteFunc_;
};
} // namespace atb
#endif // ATB_GATHER_ACLNN_RUNNER_H