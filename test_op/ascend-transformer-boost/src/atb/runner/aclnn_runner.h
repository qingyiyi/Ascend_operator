/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACLNN_RUNNER_H
#define ATB_ACLNN_RUNNER_H
#include "runner.h"

namespace atb {
class AclnnRunner : public Runner {
public:
    explicit AclnnRunner(const std::string &name);
    ~AclnnRunner() override;
protected:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    virtual Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) = 0;
    virtual aclnnStatus SetAclNNWorkspaceExecutor() = 0;

    Status PreExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    virtual Status LaunchAclnnKernel() = 0;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    void UpdateWorkspace(const RunnerVariantPack &runnerVariantPack);
    virtual bool useCache();
    int64_t runnerTypeIdx_ = -1;
    bool executorRepeatable_ = false;
    std::shared_ptr<aclOpExecutor> aclnnExecutor_ = nullptr;
    AclNNVariantPack aclnnVariantPack_;
    RunnerVariantPack atbVariantPack_;
};
} // namespace atb
#endif
