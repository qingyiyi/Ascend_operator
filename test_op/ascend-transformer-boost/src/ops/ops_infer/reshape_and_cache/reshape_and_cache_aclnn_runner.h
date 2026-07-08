/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RESHAPE_AND_CACHE_ACLNN_RUNNER_H
#define ATB_RESHAPE_AND_CACHE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnScatterPaKvCacheGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *key, const aclTensor *keyCacheRef,
    const aclTensor *slotMapping, const aclTensor *value, const aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional, const aclTensor *compressSeqOffsetOptional, const aclTensor *seqLensOptional,
    char *cacheModeOptional, char *scatterModeOptional, const aclIntArray *stridesOptional,
    const aclIntArray *offsetsOptional, uint64_t *workspaceSize, aclOpExecutor **executor);
using AclnnScatterPaKvCacheFunc = aclnnStatus (*)(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

namespace atb {
class ReshapeAndCacheAclnnRunner : public AclnnRunner {
public:
    explicit ReshapeAndCacheAclnnRunner(const infer::ReshapeAndCacheParam &param);
    ~ReshapeAndCacheAclnnRunner() override;
    static Status LoadAclnnFuncs();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    Status CreateKeyAclnnTensor();
    Status CreateValueAclnnTensor();
    Status CreateKeyCacheAclnnTensor();
    Status CreateValueCacheAclnnTensor();
    Status CreateSlotMappingAclnnTensor();

private:
    infer::ReshapeAndCacheParam param_;

    size_t aclInTensorNum_ = 0;

    size_t atbInTensorIndex_ = 0;
    size_t aclInTensorIndex_ = 0;

    size_t keyAclTensorIndex_ = 0;
    size_t valueAclTensorIndex_ = 0;
    size_t keyCacheRefAclTensorIndex_ = 0;
    size_t valueCacheRefAclTensorIndex_ = 0;
    size_t slotMappingAclTensorIndex_ = 0;

    static AclnnScatterPaKvCacheGetWorkspaceSizeFunc aclnnScatterPaKvCacheGetWorkspaceSizeFunc_;
    static AclnnScatterPaKvCacheFunc aclnnScatterPaKvCacheFunc_;
};
}  // namespace atb
#endif