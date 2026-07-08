/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_CONCAT_ACLNN_RUNNER_H
#define ATB_CONCAT_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnConcatGetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensorList *inputs,         // 输入tensor数组
    int64_t dim,                         // concat维度
    aclTensor* output,                   // 输出tensor
    uint64_t* workspaceSize,             // 输出workspace大小
    aclOpExecutor** executor);           // 输出executor

using AclnnConcatFunc = aclnnStatus (*)(
    void* workspace,                     // workspace内存
    uint64_t workspaceSize,              // workspace大小
    aclOpExecutor* executor,             // executor
    aclrtStream stream);                 // 计算流

namespace atb {

class ConcatAclnnRunner : public AclnnRunner {
public:
    explicit ConcatAclnnRunner(const infer::ConcatParam &param);
    
    ~ConcatAclnnRunner() override = default;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::ConcatParam param_;

    // 对应aclnnop/aclnn_concat.h中的两段式接口
    static AclnnConcatGetWorkspaceSizeFunc aclnnConcatGetWorkspaceSizeFunc_;
    static AclnnConcatFunc aclnnConcatFunc_;
};
}

#endif
