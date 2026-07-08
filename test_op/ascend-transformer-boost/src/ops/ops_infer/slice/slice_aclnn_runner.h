/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SLICE_ACLNN_RUNNER_H
#define ATB_SLICE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnSliceV2GetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclIntArray *, const aclIntArray *,
                                                         const aclIntArray *, const aclIntArray *, aclTensor *,
                                                         uint64_t *, aclOpExecutor **);
using AclnnSliceV2Func = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AclnnCastGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclDataType, aclTensor *, uint64_t *,
                                                      aclOpExecutor **);
using AclnnCastExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
namespace atb {
class SliceAclnnRunner : public AclnnRunner {
public:
    explicit SliceAclnnRunner(const infer::SliceParam &param);
    ~SliceAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;
    virtual bool useCache() override;
    void CleanUp();

private:
    infer::SliceParam param_;
    aclIntArray *stepsArray_ = nullptr;
    aclIntArray *axesArray_ = nullptr;
    aclIntArray *startsArray_ = nullptr;
    aclIntArray *endsArray_ = nullptr;
    aclTensor *self_ = nullptr;
    aclTensor *out_ = nullptr;
    uint64_t selfBufferSize_ = 0;
    uint64_t outBufferSize_ = 0;
    uint64_t sliceWorkspaceSize_ = 0;
    uint64_t cast1stWorkspaceSize_ = 0;
    uint64_t cast2ndWorkspaceSize_ = 0;
    std::shared_ptr<aclOpExecutor> aclnnCastExecutor1st_;
    std::shared_ptr<aclOpExecutor> aclnnCastExecutor2nd_;

    static AclnnSliceV2GetWorkspaceSizeFunc aclnnGetWorkspaceSizeFunc_;
    static AclnnSliceV2Func aclnnExecuteFunc_;
    static AclnnCastGetWorkspaceSizeFunc aclnnCastGetWorkspaceSizeFunc_;
    static AclnnCastExecuteFunc aclnnCastExecuteFunc_;
};
} // namespace atb
#endif // ATB_SLICE_ACLNN_RUNNER_H
