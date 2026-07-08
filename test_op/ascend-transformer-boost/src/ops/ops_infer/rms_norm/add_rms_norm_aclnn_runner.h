/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_ADD_RMS_NORM_ACLNN_RUNNER_H
#define ATB_ADD_RMS_NORM_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnAddRmsNormGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *x1, const aclTensor *x2,
    const aclTensor *gamma, double epsilon, const aclTensor *yOut, const aclTensor *rstdOut, const aclTensor *xOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
using AclnnAddRmsNormFunc = aclnnStatus (*)(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

namespace atb {
class AddRmsNormAclnnRunner : public AclnnRunner {
public:
    explicit AddRmsNormAclnnRunner(const infer::RmsNormParam &param);
    ~AddRmsNormAclnnRunner() override;

    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateX1AclnnTensor();
    Status CreateX2AclnnTensor();
    Status CreateGammaAclnnTensor();
    Status CreateYOutAclnnTensor();
    Status CreateRstdOutAclnnTensor();
    Status CreateXOutAclnnTensor();

private:
    infer::RmsNormParam param_;

    size_t aclInTensorNum_ = 0;
    size_t aclOutTensorNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;
    size_t atbOutTensorIndex_ = 0;
    size_t aclOutTensorIndex_ = 0;

    size_t x1AclTensorIndex_ = 0;
    size_t x2AclTensorIndex_ = 0;
    size_t gammaAclTensorIndex_ = 0;
    size_t yOutAclTensorIndex_ = 0;
    size_t rstdOutAclTensorIndex_ = 0;
    size_t xOutAclTensorIndex_ = 0;

    void *rstdDeviceData_ = nullptr;

    static AclnnAddRmsNormGetWorkspaceSizeFunc aclnnAddRmsNormGetWorkspaceSizeFunc_;
    static AclnnAddRmsNormFunc aclnnAddRmsNormFunc_;
};
}  // namespace atb
#endif  // ATB_ADD_RMS_NORM_ACLNN_RUNNER_H