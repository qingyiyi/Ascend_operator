/*
 * Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_SELF_ATTENTION_ACLNN_RUNNER_H
#define ATB_SELF_ATTENTION_ACLNN_RUNNER_H

#include <string>
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc = aclnnStatus (*)(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional, int64_t numHeads, double scaleValue,
    int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode,
    int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode,
    int64_t valueAntiquantMode, int64_t queryQuantMode, int64_t pseType, const aclTensor *attentionOut,
    const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);
using AclnnFusedInferAttentionScoreV5Func = aclnnStatus (*)(void *workspace, uint64_t workspaceSize,
                                                            aclOpExecutor *executor, const aclrtStream stream);

namespace atb {
struct AclnnFusedInferAttentionScoreV5Param {
    int64_t numHeads = 0;
    double scaleValue = 0.0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    std::string inputLayoutStr = "TND";
    int64_t numKeyValueHeads = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;
    int64_t blockSize = 0;
    int64_t antiquantMode = 0;
    bool softmaxLseFlag = false;
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    int64_t queryQuantMode = 0;
    int64_t pseType = 0;
};

class SelfAttentionAclnnRunner : public AclnnRunner {
public:
    explicit SelfAttentionAclnnRunner(const atb::infer::SelfAttentionParam &opParam);
    ~SelfAttentionAclnnRunner() override;
    static Status LoadMethod();

protected:
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    Status LaunchAclnnKernel() override;

private:
    void GetTensorNum();
    void InitTensorIndex();
    void InitAclnnParam();
    Status CreateQueryAclnnTensor();
    Status CreateKeyAclnnTensorList();
    Status CreateValueAclnnTensorList();
    Status CreatePseShiftAclnnTensor();
    Status CreateAttenMaskAclnnTensor();
    Status CreateActualSeqLengthsAclIntArray();
    Status CreateAttentionOutAclnnTensor();
    std::shared_ptr<AclNNTensor> InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex);

private:
    infer::SelfAttentionParam param_;
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

    size_t queryAclTensorIndex_ = 0;
    size_t keyAclTensorListIndex_ = 0;
    size_t valueAclTensorListIndex_ = 0;
    size_t pseShiftAclTensorIndex_ = 0;
    size_t attenMaskAclTensorIndex_ = 0;
    size_t attentionOutAclTensorIndex_ = 0;

    int64_t batch_ = 0;
    // Decoder: for all batch, qSeqlen is 1; update kvCache in FA
    int64_t isDecoder_ = false;

    aclIntArray *actualSeqLengths_ = nullptr;

    AclnnFusedInferAttentionScoreV5Param aclnnParam_;

    static AclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc aclnnFusedInferAttentionScoreV5GetWorkspaceSizeFunc_;
    static AclnnFusedInferAttentionScoreV5Func aclnnFusedInferAttentionScoreV5Func_;
};
} // namespace atb
#endif // ATB_SELF_ATTENTION_ACLNN_RUNNER_H
