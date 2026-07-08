/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_LINEAR_PARALLEL_ACLNN_RUNNER_H
#define ATB_LINEAR_PARALLEL_ACLNN_RUNNER_H

#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"
#include "atb/runner/hccl_runner.h"

namespace atb {
class LinearParallelAclnnRunner : public AclnnRunner {
public:
    explicit LinearParallelAclnnRunner(const infer::LinearParallelParam &param);
    LinearParallelAclnnRunner(const infer::LinearParallelParam &param, bool useRankTableFile);
    LinearParallelAclnnRunner(const infer::LinearParallelParam &param, HcclComm hcclComm);
    ~LinearParallelAclnnRunner() override;
    static Status LoadMethodMatmulReduceScatter();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;
    bool useCache() override;

private:
    HcclRunner hcclRunner_;
    infer::LinearParallelParam param_;
    static aclnnStatus (*aclnnMatmulReduceScatterV2GetWorkspaceSizeFunc_)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, int64_t, const char *, const char *, int64_t, int64_t, int64_t, const char *,
        const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnMatmulReduceScatterV2Func_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif