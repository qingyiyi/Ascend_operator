/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SOFTMAX_ACLNN_RUNNER_H
#define ATB_SOFTMAX_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"


using AclnnGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, int64_t, int64_t, bool, bool, aclTensor *,
                                                     aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnExecuteFunc = aclnnStatus(*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnCastGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, const aclDataType,
                                                        aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnCastExecuteFunc = aclnnStatus(*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class SortAclnnRunner : public AclnnRunner {
public:
    explicit SortAclnnRunner(const infer::SortParam &param);
    ~SortAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    virtual bool useCache() override;

private:
    infer::SortParam param_;

    static AclnnGetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    static AclnnExecuteFunc aclnnExecuteFunc_;
    static AclnnCastGetWorkspaceSizeFunc aclnnCastGetWorkspaceSizeFunc_;
    static AclnnCastExecuteFunc aclnnCastExecuteFunc_;

    void CleanUp();
    aclTensor *indices_ = nullptr;
    uint64_t topkWorkspaceSize_ = 0;
    uint64_t castWorkspaceSize_ = 0;
    uint64_t indicesBufferSize_ = 0;
    std::shared_ptr<aclOpExecutor> aclnnCastExecutor_;
};
} // namespace atb
#endif // ATB_SOFTMAX_ACLNN_RUNNER_H