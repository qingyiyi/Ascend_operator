/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_REDUCE_ACLNN_RUNNER_H
#define ATB_REDUCE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using ReduceAclnnSumGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, const aclIntArray *, bool, aclDataType,
                                                              aclTensor *, uint64_t *, aclOpExecutor **);

using ReduceAclnnAmaxGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, const aclIntArray *, bool, aclTensor *,
                                                         uint64_t *, aclOpExecutor **);

using ReduceAclnnAminGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, const aclIntArray *, bool, aclTensor *,
                                                         uint64_t *, aclOpExecutor **);

namespace atb {
class ReduceAclnnRunner : public AclnnRunner {
public:
    explicit ReduceAclnnRunner(const infer::ReduceParam &param);
    ~ReduceAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status GetFunc();
    virtual bool useCache() override;

private:
    infer::ReduceParam param_;
    using ExecuteFuncType = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    ExecuteFuncType executeFunc_ = nullptr;
    aclIntArray *dims_ = nullptr;

    // reduceSum
    static ReduceAclnnSumGetWorkspaceSizeFunc aclnnReduceSumGetWorkspaceSizeFunc_;
    static ExecuteFuncType aclnnReduceSumExecuteFunc_;
    // reduceMax -> Amax
    static ReduceAclnnAmaxGetWorkspaceSizeFunc aclnnAmaxGetWorkspaceSizeFunc_;
    static ExecuteFuncType aclnnAmaxExecuteFunc_;
    // reduceMin -> Amin
    static ReduceAclnnAminGetWorkspaceSizeFunc aclnnAminGetWorkspaceSizeFunc_;
    static ExecuteFuncType aclnnAminExecuteFunc_;
};
} // namespace atb
#endif // ATB_REDUCE_ACLNN_RUNNER_H
