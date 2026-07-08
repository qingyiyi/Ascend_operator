/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_ACLNN_RUNNER_H
#define ATB_LINEAR_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnMatmulGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, int8_t,
                                                        uint64_t *, aclOpExecutor **);
using AclnnMatmulExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AclnnAddmmGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                                       aclScalar *, aclScalar *, const aclTensor *, int8_t, uint64_t *,
                                                       aclOpExecutor **);
using AclnnAddmmExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AclnnMatmulWeightNzGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                                                int8_t, uint64_t *, aclOpExecutor **);
using AclnnMatmulWeightNzExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AclnnAddmmWeightNzGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *,
                                                               aclScalar *, aclScalar *, const aclTensor *, int8_t,
                                                               uint64_t *, aclOpExecutor **);
using AclnnAddmmWeightNzExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);
using AclnnBatchMatMulWeightNzGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *,
                                                                     const aclTensor *, int8_t, uint64_t *,
                                                                     aclOpExecutor **);
using AclnnBatchMatMulWeightNzExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class LinearAclnnRunner : public AclnnRunner {
public:
    explicit LinearAclnnRunner(const infer::LinearParam &param);
    ~LinearAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateMatmulSelfAclnnTensor();
    Status CreateMatmulMat2AclnnTensor();
    Status CreateMatmulOutAclnnTensor();
    Status CreateAddmmSelfAclnnTensor();
    Status CreateAddmmMat1AclnnTensor();
    Status CreateAddmmMat2AclnnTensor();
    Status CreateAddmmOutAclnnTensor();
    aclnnStatus SetAclnnMatmulWorkspaceExecutor();
    aclnnStatus SetAclnnAddmmWorkspaceExecutor();
    aclnnStatus SetAclnnMatmulWeightNzWorkspaceExecutor();
    aclnnStatus SetAclnnAddmmWeightNzWorkspaceExecutor();
    aclnnStatus SetAclnnBatchMatMulWeightNzWorkspaceExecutor();
    std::shared_ptr<AclNNTensor> CreateXAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateWeightAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateWeightNzAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateBiasAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> CreateOutputAclnnTensor(int aclnnTensorIndex);
    std::shared_ptr<AclNNTensor> InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex);

private:
    infer::LinearParam param_;

    bool isWeightNz_ = false;
    bool isBatch_ = false;

    size_t aclInTensorNum_ = 0;
    size_t aclOutTensorNum_ = 0;
    size_t aclInTensorListNum_ = 0;
    size_t aclOutTensorListNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;
    size_t aclInTensorListIndex_ = 0;
    size_t atbOutTensorIndex_ = 0;
    size_t aclOutTensorIndex_ = 0;
    size_t aclOutTensorListIndex_ = 0;

    size_t matmulSelfAclTensorIndex_ = 0;
    size_t matmulMat2AclTensorIndex_ = 0;
    size_t matmulOutAclTensorIndex_ = 0;
    size_t addmmSelfAclTensorIndex_ = 0;
    size_t addmmMat1AclTensorIndex_ = 0;
    size_t addmmMat2AclTensorIndex_ = 0;
    size_t addmmOutAclTensorIndex_ = 0;

    aclScalar *alpha_ = nullptr;
    aclScalar *beta_ = nullptr;

    static AclnnMatmulGetWorkspaceSizeFunc aclnnMatmulGetWorkspaceSizeFunc_;
    static AclnnMatmulExecuteFunc aclnnMatmulExecuteFunc_;
    static AclnnAddmmGetWorkspaceSizeFunc aclnnAddmmGetWorkspaceSizeFunc_;
    static AclnnAddmmExecuteFunc aclnnAddmmExecuteFunc_;
    static AclnnMatmulWeightNzGetWorkspaceSizeFunc aclnnMatmulWeightNzGetWorkspaceSizeFunc_;
    static AclnnMatmulWeightNzExecuteFunc aclnnMatmulWeightNzExecuteFunc_;
    static AclnnAddmmWeightNzGetWorkspaceSizeFunc aclnnAddmmWeightNzGetWorkspaceSizeFunc_;
    static AclnnAddmmWeightNzExecuteFunc aclnnAddmmWeightNzExecuteFunc_;
    static AclnnBatchMatMulWeightNzGetWorkspaceSizeFunc aclnnBatchMatMulWeightNzGetWorkspaceSizeFunc_;
    static AclnnBatchMatMulWeightNzExecuteFunc aclnnBatchMatMulWeightNzExecuteFunc_;
};
} // namespace atb
#endif // ATB_LINEAR_ACLNN_RUNNER_H