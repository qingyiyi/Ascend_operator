/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_MASKED_FILL_ACLNN_RUNNER_H
#define ATB_MASKED_FILL_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"
namespace atb {

using AclnnMaskFillGetWorkspaceSizeFunc = aclnnStatus (*)(
    aclTensor *, // selfRef
    const aclTensor *, //mask
    const aclScalar *, //value
    uint64_t *, //workspaceSize
    aclOpExecutor** //executor
);

using AclnnMaskFillExecuteFunc = aclnnStatus (*)(
    void *, //workspace
    uint64_t , //workspaceSize
    aclOpExecutor *, //executor
    aclrtStream //stream
);

class MaskedFillAclnnRunner : public AclnnRunner {
public:
    explicit MaskedFillAclnnRunner(const infer::FillParam &param);
    ~MaskedFillAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    Status BuildXTensor();
    Status BuildMaskTensor();
    Status BuildOutputTensor();
private:
    infer::FillParam param_;
    aclScalar *valueScalarPtr_;
    static AclnnMaskFillGetWorkspaceSizeFunc aclnnInplaceMaskedFillScalarGetWorkspaceSizeFunc_;
    static AclnnMaskFillExecuteFunc aclnnInplaceMaskedFillScalarFunc_;
};
} // namespace atb
#endif