/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SOFTMAX_ACLNN_RUNNER_H
#define ATB_SOFTMAX_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class SoftmaxAclnnRunner : public AclnnRunner {
public:
    explicit SoftmaxAclnnRunner(const infer::SoftmaxParam &param);
    ~SoftmaxAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status RecordDims(SVector<Tensor> tensors, const size_t id, SVector<int64_t> &axes);

private:
    infer::SoftmaxParam param_;
    Dims targetDims_{};
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(const aclTensor *, int64_t, aclTensor *, uint64_t *,
                                                     aclOpExecutor **);
    static aclnnStatus (*aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif // ATB_SOFTMAX_ACLNN_RUNNER_H