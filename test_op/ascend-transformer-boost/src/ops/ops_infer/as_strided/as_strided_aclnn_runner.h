/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ASSTRIDED_ACLNN_RUNNER_H
#define ATB_ASSTRIDED_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"


namespace atb {

class AsStridedAclnnRunner : public AclnnRunner {
public:
    explicit AsStridedAclnnRunner(const infer::AsStridedParam &param);
    
    ~AsStridedAclnnRunner() override = default;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::AsStridedParam param_;

    // 对应aclnnop/aclnn_AsStrided.h中的两段式接口
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(
        aclTensor* selfRef,                  // 输入tensor数组
        const aclTensor* src,                // concat维度
        uint64_t* workspaceSize,             // 输出workspace大小
        aclOpExecutor** executor);           // 输出executor

    static aclnnStatus (*aclnnExecuteFunc_)(
        void* workspace,                     // workspace内存
        uint64_t workspaceSize,              // workspace大小
        aclOpExecutor* executor,             // executor
        aclrtStream stream);                 // 计算流
};
}

#endif
