/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SPLIT_ACLNN_RUNNER_H
#define ATB_SPLIT_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnSplitTensorGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *self, uint64_t splitSections, int64_t dim,
                                                             aclTensorList *out, uint64_t *workspaceSize,
                                                             aclOpExecutor **executor);
using AclnnSplitTensorFunc = aclnnStatus (*)(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream);
using AclnnSplitWithSizeGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *self, aclIntArray *splitSize,
                                                               int64_t dim, aclTensorList *out, uint64_t *workspaceSize,
                                                               aclOpExecutor **executor);
using AclnnSplitWithSizeFunc = aclnnStatus (*)(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

namespace atb {
class SplitAclnnRunner : public AclnnRunner {
public:
    explicit SplitAclnnRunner(const infer::SplitParam &param);
    ~SplitAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateSelfAclnnTensor();
    Status CreateOutAclnnTensorList();
    aclnnStatus CreateSplitSizeAclIntArray();

private:
    infer::SplitParam param_;

    bool splitWithSize_ = false;

    size_t aclInTensorNum_ = 0;
    size_t aclOutTensorListNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;
    size_t atbOutTensorIndex_ = 0;
    size_t aclOutTensorListIndex_ = 0;

    size_t selfAclTensorIndex_ = 0;
    size_t outAclTensorListIndex_ = 0;

    aclIntArray *splitSize_ = nullptr;

    static AclnnSplitTensorGetWorkspaceSizeFunc aclnnSplitTensorGetWorkspaceSizeFunc_;
    static AclnnSplitTensorFunc aclnnSplitTensorFunc_;
    static AclnnSplitWithSizeGetWorkspaceSizeFunc aclnnSplitWithSizeGetWorkspaceSizeFunc_;
    static AclnnSplitWithSizeFunc aclnnSplitWithSizeFunc_;
};
} // namespace atb
#endif // ATB_SPLIT_ACLNN_RUNNER_H
