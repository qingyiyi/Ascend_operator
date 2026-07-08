/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_REPEAT_ACLNN_RUNNER_H
#define ATB_REPEAT_ACLNN_RUNNER_H

#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using RepeatAclnnGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, const aclIntArray *, const aclTensor *,
                                                     uint64_t *, aclOpExecutor **);
using RepeatAclnnExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *,const  aclrtStream);


namespace atb {
class RepeatAclnnRunner : public AclnnRunner {
public:
    explicit RepeatAclnnRunner(const infer::RepeatParam &param);
    ~RepeatAclnnRunner() override;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::RepeatParam param_;
    aclIntArray *size_ = nullptr;
    // 对应aclnnop/aclnn_repeat.h中的两段式接口
    static RepeatAclnnGetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    static RepeatAclnnExecuteFunc aclnnExecuteFunc_;
};
} // namespace atb
#endif
