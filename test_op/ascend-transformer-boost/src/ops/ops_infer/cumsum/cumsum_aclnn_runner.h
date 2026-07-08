/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_CUMSUM_ACLNN_RUNNER_H
#define ATB_CUMSUM_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {

class CumsumAclnnRunner : public AclnnRunner {
public:
    explicit CumsumAclnnRunner(const infer::CumsumParam &param);
    
    ~CumsumAclnnRunner() override = default;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::CumsumParam param_;

    // 对应aclnnop/aclnn_cumsum.h中的两段式接口
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(
        const aclTensor* input,              // 输入tensor
        int64_t dim,                         // cumsum维度
        bool exclusive,                      // 是否包含顶部元素的和
        bool reverse,                        // 是否反向求和
        aclTensor* output,                   // 输出tensor
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