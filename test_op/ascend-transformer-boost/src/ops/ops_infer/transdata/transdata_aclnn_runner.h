/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_TRANSDATA_ACLNN_RUNNER_H
#define ATB_TRANSDATA_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class TransdataAclnnRunner : public AclnnRunner {
public:
    explicit TransdataAclnnRunner(const infer::TransdataParam &param);
    ~TransdataAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::TransdataParam param_;
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif // ATB_TRANSDATA_ACLNN_RUNNER_H
