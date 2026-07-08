/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACLNN_DYNAMIC_QUANT_RUNNER_H
#define ATB_ACLNN_DYNAMIC_QUANT_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class AclnnDynamicQuantRunner : public AclnnRunner {
public:
    explicit AclnnDynamicQuantRunner(const infer::ElewiseParam &param);
    ~AclnnDynamicQuantRunner() override;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::ElewiseParam param_;

    // 对应aclnnop/aclnn_dynamic_quant.h中的两段式接口
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

};
} // namespace atb
#endif
