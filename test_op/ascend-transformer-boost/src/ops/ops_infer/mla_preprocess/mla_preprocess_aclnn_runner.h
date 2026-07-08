/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_MLAPREPROCESS_ACLNN_RUNNER_H
#define ATB_MLAPREPROCESS_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class MlaPreprocessAclnnRunner : public AclnnRunner {
public:
    explicit MlaPreprocessAclnnRunner(const infer::MlaPreprocessParam &param);
    explicit MlaPreprocessAclnnRunner(const infer::MlaPreprocessParam &param, bool doRmsNorm = true);
    ~MlaPreprocessAclnnRunner() override;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;

private:
    infer::MlaPreprocessParam param_;
    bool doRmsNorm_ = true;

    // 对应aclnnop/aclnn_mla_preprocess.h中的两段式接口
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, int64_t, int64_t, int64_t, double,
        int64_t, int64_t, bool, bool, bool, int64_t, int64_t, bool, int64_t, const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
    static aclnnStatus (*aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif
