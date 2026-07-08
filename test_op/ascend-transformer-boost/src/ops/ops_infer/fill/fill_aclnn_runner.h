/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LAYER_NORM_ACLNN_RUNNER_H
#define ATB_LAYER_NORM_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnInplaceFillScalarGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclScalar *, uint64_t *,
                                                                   aclOpExecutor **);
using AclnnInplaceFillScalarExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class FillAclnnRunner : public AclnnRunner {
public:
    explicit FillAclnnRunner(const infer::FillParam &param);
    ~FillAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::FillParam param_;
    static AclnnInplaceFillScalarGetWorkspaceSizeFunc aclnnInplaceFillScalarGetWorkspaceSizeFunc_;
    static AclnnInplaceFillScalarExecuteFunc aclnnInplaceFillScalarExecuteFunc_;
};
} // namespace atb
#endif // ATB_LAYER_NORM_ACLNN_RUNNER_H
