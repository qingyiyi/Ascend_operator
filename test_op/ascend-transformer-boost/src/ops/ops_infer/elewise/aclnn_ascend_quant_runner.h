/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACLNN_ASCEND_QUANT_RUNNER_H
#define ATB_ACLNN_ASCEND_QUANT_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

namespace atb {
class AclnnAscendQuantRunner : public AclnnRunner {
public:
    explicit AclnnAscendQuantRunner(const infer::ElewiseParam &param);
    ~AclnnAscendQuantRunner() override;

    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    virtual bool useCache() override;

private:
    void CleanUp();
    infer::ElewiseParam param_;
    bool is_offset_empty_ = false;
    aclDataType scaleDatatype_ = ACL_FLOAT16;
    aclTensor *scale_ = nullptr;
    aclTensor *offset_ = nullptr;
    uint64_t scaleBufferSize_ = 0;
    uint64_t offsetBufferSize_ = 0;
    uint64_t reciprocalWorkspaceSize_ = 0;
    uint64_t castWorkspaceSize_ = 0;
    uint64_t quantWorkspaceSize_ = 0;
    std::shared_ptr<aclOpExecutor> aclnnReciprocalExecutor_;
    std::shared_ptr<aclOpExecutor> aclnnCastExecutor_;

    // 对应aclnnop/aclnn_ascend_quant.h中的两段式接口
    static aclnnStatus (*aclnnGetWorkspaceSizeFunc_)(const aclTensor *, const aclTensor *, const aclTensor *, bool,
                                                     char *, int32_t, int32_t, const aclTensor *, uint64_t *,
                                                     aclOpExecutor **);
    static aclnnStatus (*aclnnExecuteFunc_)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
    // 对应aclnnop/aclnn_reciprocal.h中的两段式接口
    static aclnnStatus (*aclnnReciprocalGetWorkspaceSizeFunc_)(const aclTensor *, aclTensor *, uint64_t *,
                                                               aclOpExecutor **);
    static aclnnStatus (*aclnnReciprocalExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    // 对应aclnnop/aclnn_cast.h中的两段式接口
    static aclnnStatus (*aclnnCastGetWorkspaceSizeFunc_)(const aclTensor *, const aclDataType, aclTensor *, uint64_t *,
                                                         aclOpExecutor **);
    static aclnnStatus (*aclnnCastExecuteFunc_)(void *, uint64_t, aclOpExecutor *, aclrtStream);
};
} // namespace atb
#endif
