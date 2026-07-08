/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ELEWISE_ACLNN_RUNNER_H
#define ATB_ELEWISE_ACLNN_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/aclnn_runner.h"

using AclnnCastGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclDataType, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnCastExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnMulsGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclScalar *, const aclTensor *,uint64_t *, aclOpExecutor **);
using AclnnMulsExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnCosGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, uint64_t *,aclOpExecutor **);
using AclnnCosExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnSinGetWorkspaceSizeFunc =  aclnnStatus (*)(const aclTensor *, const aclTensor *, uint64_t *,aclOpExecutor **);
using AclnnSinExecuteFunc =  aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnLogicalNotGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, uint64_t *,aclOpExecutor **);
using AclnnLogicalNotExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnAddGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclScalar *, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnAddExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnSubGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclScalar *, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnSubExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnMulGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnMulExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnDivGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnDivExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnLtTensorGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnLtTensorExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnGtTensorGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensor *, const aclTensor *, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnGtTensorExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

using AclnnAscendQuantGetWorkspaceSizeFunc = aclnnStatus(*)(const aclTensor *, const aclTensor *, const aclTensor *, bool, const char *, int32_t, int32_t, const aclTensor *, uint64_t *, aclOpExecutor **);
using AclnnAscendQuantExecuteFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

namespace atb {
class ElewiseAclnnRunner : public AclnnRunner {
public:
    explicit ElewiseAclnnRunner(const infer::ElewiseParam &param);
    ~ElewiseAclnnRunner() override;

    static Status LoadMethod();

protected:
    template <typename T>
    aclnnStatus CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                                void** deviceAddr, aclDataType dataType, aclTensor** tensor,
                                uint64_t size);
    Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) override;
    Status ProcessNormalTensors(const RunnerVariantPack &runnerVariantPack,
                                atb::SVector<std::shared_ptr<AclNNTensor>> &tensors,
                                bool isOutput);
    Status ProcessQuantTensors(const RunnerVariantPack &runnerVariantPack);
    Status CreateAclNNTensorByAtbTensor(atb::Tensor atbTensor, int index, std::shared_ptr<AclNNTensor>& tensorPtr);
    aclnnStatus CreateQuantParamTensor(atb::Tensor baseTensor,
                                       void* deviceAddr,
                                       aclTensor *paramTensor,
                                       float paramValue,
                                       std::shared_ptr<AclNNTensor>& tensorPtr);
    Status LaunchAclnnKernel() override;
    aclnnStatus SetAclNNWorkspaceExecutor() override;
    aclnnStatus HandleSub(aclOpExecutor** executor);
    aclnnStatus HandleCast(aclOpExecutor** executor);
    aclnnStatus HandleMuls(aclOpExecutor** executor);
    aclnnStatus HandleCos(aclOpExecutor** executor);
    aclnnStatus HandleSin(aclOpExecutor** executor);
    aclnnStatus HandleLogicalNot(aclOpExecutor** executor);
    aclnnStatus HandleAdd(aclOpExecutor** executor);
    aclnnStatus HandleMul(aclOpExecutor** executor);
    aclnnStatus HandleRealDiv(aclOpExecutor** executor);
    aclnnStatus HandleLess(aclOpExecutor** executor);
    aclnnStatus HandleGreater(aclOpExecutor** executor);
    aclnnStatus HandleQuant(aclOpExecutor** executor);

private:
    infer::ElewiseParam param_;
    uint32_t inTensorNum_ = 0;
    aclScalar *alpha_ = nullptr;
    void* scaleDeviceAddr_ = nullptr;
    void* offsetDeviceAddr_ = nullptr;
    aclTensor* scaleTensor_ = nullptr;
    aclTensor* offsetTensor_ = nullptr;

    static AclnnCastGetWorkspaceSizeFunc aclnnCastGetWorkspaceSizeFunc_;
    static AclnnCastExecuteFunc aclnnCastExecuteFunc_;

    static AclnnMulsGetWorkspaceSizeFunc aclnnMulsGetWorkspaceSizeFunc_;
    static AclnnMulsExecuteFunc aclnnMulsExecuteFunc_;

    static AclnnCosGetWorkspaceSizeFunc aclnnCosGetWorkspaceSizeFunc_;
    static AclnnCosExecuteFunc aclnnCosExecuteFunc_;

    static AclnnSinGetWorkspaceSizeFunc aclnnSinGetWorkspaceSizeFunc_;
    static AclnnSinExecuteFunc aclnnSinExecuteFunc_;

    static AclnnLogicalNotGetWorkspaceSizeFunc aclnnLogicalNotGetWorkspaceSizeFunc_;
    static AclnnLogicalNotExecuteFunc aclnnLogicalNotExecuteFunc_;

    static AclnnAddGetWorkspaceSizeFunc aclnnAddGetWorkspaceSizeFunc_;
    static AclnnAddExecuteFunc aclnnAddExecuteFunc_;

    static AclnnSubGetWorkspaceSizeFunc aclnnSubGetWorkspaceSizeFunc_;
    static AclnnSubExecuteFunc aclnnSubExecuteFunc_;

    static AclnnMulGetWorkspaceSizeFunc aclnnMulGetWorkspaceSizeFunc_;
    static AclnnMulExecuteFunc aclnnMulExecuteFunc_;

    static AclnnDivGetWorkspaceSizeFunc aclnnDivGetWorkspaceSizeFunc_;
    static AclnnDivExecuteFunc aclnnDivExecuteFunc_;

    static AclnnLtTensorGetWorkspaceSizeFunc aclnnLtTensorGetWorkspaceSizeFunc_;
    static AclnnLtTensorExecuteFunc aclnnLtTensorExecuteFunc_;

    static AclnnGtTensorGetWorkspaceSizeFunc aclnnGtTensorGetWorkspaceSizeFunc_;
    static AclnnGtTensorExecuteFunc aclnnGtTensorExecuteFunc_;

    static AclnnAscendQuantGetWorkspaceSizeFunc aclnnAscendQuantGetWorkspaceSizeFunc_;
    static AclnnAscendQuantExecuteFunc aclnnAscendQuantExecuteFunc_;
};
} // namespace atb
#endif
