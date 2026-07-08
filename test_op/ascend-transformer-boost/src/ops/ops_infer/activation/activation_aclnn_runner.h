/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_ACTIVATION_ACLNN_RUNNER_H
#define OPS_ACTIVATION_ACLNN_RUNNER_H

#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
using GetWsThunk = std::function<
    aclnnStatus(const AclNNVariantPack &,
                uint64_t * /*workspace size*/, 
                aclOpExecutor **)>;

using ExecThunk = std::function<
    aclnnStatus(void * /*workspace*/,
                uint64_t /*workspace size*/,
                aclOpExecutor *,
                aclrtStream)>;

class ActivationAclnnRunner : public AclnnRunner {
public:
    explicit ActivationAclnnRunner(const infer::ActivationParam &param);
    ~ActivationAclnnRunner() override;
    static Status LoadAclnnFunctions();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    struct KernelAdapters {
        const char *name;
        GetWsThunk getWs;
        ExecThunk exec;
    };
    static KernelAdapters MakeAdaptersByType(const infer::ActivationParam &param);
    infer::ActivationParam param_;
    GetWsThunk getWs_;
    ExecThunk execFn_;
    static aclnnStatus (*fastGeluGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*fastGeluExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    static aclnnStatus (*geluGetWorkspaceSizeFunc_)(const aclTensor *, int64_t *, aclTensor *, uint64_t *,
                                                    aclOpExecutor **);
    static aclnnStatus (*geluExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

    static aclnnStatus (*logGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*logExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    static aclnnStatus (*reluGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*reluExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

    static aclnnStatus (*sigmoidGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*sigmoidExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

    static aclnnStatus (*swishGetWorkspaceSizeFunc_)(const aclTensor *, const aclScalar *, aclTensor *, uint64_t *,
                                                     aclOpExecutor **);
    static aclnnStatus (*swishExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif