/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/utils_internal.h"
#include "atbops/params/params.h"
#include "atb/utils/operation_register.h"

namespace {
static const int MATMUL_SELF_ACLNN_TENSOR_IDX = 0;
static const int MATMUL_MAT2_ACLNN_TENSOR_IDX = 1;
static const int MATMUL_OUT_ACLNN_TENSOR_IDX = 0;
static const int ADDMM_SELF_ACLNN_TENSOR_IDX = 0;
static const int ADDMM_MAT1_ACLNN_TENSOR_IDX = 1;
static const int ADDMM_MAT2_ACLNN_TENSOR_IDX = 2;
static const int ADDMM_OUT_ACLNN_TENSOR_IDX = 0;
static const int DEFAULT_ALIGN = 16;
} // namespace

namespace atb {
AclnnMatmulGetWorkspaceSizeFunc LinearAclnnRunner::aclnnMatmulGetWorkspaceSizeFunc_ = nullptr;
AclnnMatmulExecuteFunc LinearAclnnRunner::aclnnMatmulExecuteFunc_ = nullptr;
AclnnAddmmGetWorkspaceSizeFunc LinearAclnnRunner::aclnnAddmmGetWorkspaceSizeFunc_ = nullptr;
AclnnAddmmExecuteFunc LinearAclnnRunner::aclnnAddmmExecuteFunc_ = nullptr;
AclnnMatmulWeightNzGetWorkspaceSizeFunc LinearAclnnRunner::aclnnMatmulWeightNzGetWorkspaceSizeFunc_ = nullptr;
AclnnMatmulWeightNzExecuteFunc LinearAclnnRunner::aclnnMatmulWeightNzExecuteFunc_ = nullptr;
AclnnAddmmWeightNzGetWorkspaceSizeFunc LinearAclnnRunner::aclnnAddmmWeightNzGetWorkspaceSizeFunc_ = nullptr;
AclnnAddmmWeightNzExecuteFunc LinearAclnnRunner::aclnnAddmmWeightNzExecuteFunc_ = nullptr;
AclnnBatchMatMulWeightNzGetWorkspaceSizeFunc LinearAclnnRunner::aclnnBatchMatMulWeightNzGetWorkspaceSizeFunc_ = nullptr;
AclnnBatchMatMulWeightNzExecuteFunc LinearAclnnRunner::aclnnBatchMatMulWeightNzExecuteFunc_ = nullptr;

LinearAclnnRunner::LinearAclnnRunner(const infer::LinearParam &param) : AclnnRunner("LinearAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::LinearAclnnRunner";

    GetTensorNum();
}

LinearAclnnRunner::~LinearAclnnRunner()
{
    if (alpha_) {
        if (aclDestroyScalar(alpha_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "alpha aclDestroyScalar error";
        }
        alpha_ = nullptr;
    }
    if (beta_) {
        if (aclDestroyScalar(beta_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "beta aclDestroyScalar error";
        }
        beta_ = nullptr;
    }
}

Status LinearAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::LoadAclnnFuncs";
    if ((aclnnMatmulGetWorkspaceSizeFunc_ && aclnnMatmulExecuteFunc_) &&
        (aclnnAddmmGetWorkspaceSizeFunc_ && aclnnAddmmExecuteFunc_) &&
        (aclnnMatmulWeightNzGetWorkspaceSizeFunc_ && aclnnMatmulWeightNzExecuteFunc_) &&
        (aclnnAddmmWeightNzGetWorkspaceSizeFunc_ && aclnnAddmmWeightNzExecuteFunc_) &&
        (aclnnBatchMatMulWeightNzGetWorkspaceSizeFunc_ && aclnnBatchMatMulWeightNzExecuteFunc_)) {
        return NO_ERROR;
    }
    Status st = NO_ERROR;
    if (!aclnnMatmulGetWorkspaceSizeFunc_ || !aclnnMatmulExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnMatmulGetWorkspaceSize", "aclnnMatmul", aclnnMatmulGetWorkspaceSizeFunc_,
                                      aclnnMatmulExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnAddmmGetWorkspaceSizeFunc_ || !aclnnAddmmExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnAddmmGetWorkspaceSize", "aclnnAddmm", aclnnAddmmGetWorkspaceSizeFunc_,
                                      aclnnAddmmExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnMatmulWeightNzGetWorkspaceSizeFunc_ || !aclnnMatmulWeightNzExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnMatmulWeightNzGetWorkspaceSize", "aclnnMatmulWeightNz",
                                      aclnnMatmulWeightNzGetWorkspaceSizeFunc_, aclnnMatmulWeightNzExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnAddmmWeightNzGetWorkspaceSizeFunc_ || !aclnnAddmmWeightNzExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnAddmmWeightNzGetWorkspaceSize", "aclnnAddmmWeightNz",
                                      aclnnAddmmWeightNzGetWorkspaceSizeFunc_, aclnnAddmmWeightNzExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    if (!aclnnBatchMatMulWeightNzGetWorkspaceSizeFunc_ || !aclnnBatchMatMulWeightNzExecuteFunc_) {
        st = LoadFromSharedObjectFile("aclnnBatchMatMulWeightNzGetWorkspaceSize", "aclnnBatchMatMulWeightNz",
                                      aclnnBatchMatMulWeightNzGetWorkspaceSizeFunc_,
                                      aclnnBatchMatMulWeightNzExecuteFunc_);
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

Status LinearAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix()
                  << "LinearAclnnRunner::BuildAclnnVariantPack, runnerVariantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    isWeightNz_ = runnerVariantPack.inTensors[1].desc.format == ACL_FORMAT_FRACTAL_NZ;
    isBatch_ = runnerVariantPack.inTensors[1].desc.shape.dimNum == 3 ||
               (runnerVariantPack.inTensors[1].desc.shape.dimNum == 4 &&
                runnerVariantPack.inTensors[1].desc.shape.dims[0] != 1);
    InitTensorIndex();
    aclnnVariantPack_.aclInTensors.reserve(aclInTensorNum_);
    aclnnVariantPack_.aclInTensors.resize(aclInTensorNum_);
    aclnnVariantPack_.aclOutTensors.reserve(aclOutTensorNum_);
    aclnnVariantPack_.aclOutTensors.resize(aclOutTensorNum_);
    Status st = NO_ERROR;
    if (param_.hasBias) {
        st = CreateAddmmMat1AclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateAddmmMat2AclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateAddmmSelfAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateAddmmOutAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    } else {
        st = CreateMatmulSelfAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateMatmulMat2AclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
        st = CreateMatmulOutAclnnTensor();
        if (st != NO_ERROR) {
            return st;
        }
    }
    return NO_ERROR;
}

aclnnStatus LinearAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::SetAclNNWorkspaceExecutor";
    if (isWeightNz_) {
        if (param_.hasBias) {
            return SetAclnnAddmmWeightNzWorkspaceExecutor();
        } else {
            if (isBatch_) {
                return SetAclnnBatchMatMulWeightNzWorkspaceExecutor();
            } else {
                return SetAclnnMatmulWeightNzWorkspaceExecutor();
            }
        }
    } else {
        if (param_.hasBias) {
            return SetAclnnAddmmWorkspaceExecutor();
        } else {
            return SetAclnnMatmulWorkspaceExecutor();
        }
    }
}

Status LinearAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LinearAclnnRunner::LaunchAclnnKernel";

    aclrtStream executeStream = GetExecuteStream(atbVariantPack_.context);

    aclnnStatus ret = ACL_SUCCESS;
    if (isWeightNz_) {
        if (param_.hasBias) {
            ret = aclnnAddmmWeightNzExecuteFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                                 aclnnExecutor_.get(), executeStream);
        } else {
            if (isBatch_) {
                ret = aclnnBatchMatMulWeightNzExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                                           atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(),
                                                           executeStream);
            } else {
                ret = aclnnMatmulWeightNzExecuteFunc_(atbVariantPack_.workspaceBuffer,
                                                      atbVariantPack_.workspaceBufferSize, aclnnExecutor_.get(),
                                                      executeStream);
            }
        }
    } else {
        if (param_.hasBias) {
            ret = aclnnAddmmExecuteFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                         aclnnExecutor_.get(), executeStream);
        } else {
            ret = aclnnMatmulExecuteFunc_(atbVariantPack_.workspaceBuffer, atbVariantPack_.workspaceBufferSize,
                                          aclnnExecutor_.get(), executeStream);
        }
    }
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    return NO_ERROR;
}

void LinearAclnnRunner::GetTensorNum()
{
    if (param_.hasBias) {
        aclInTensorNum_ = 3; // 3: self, mat1, mat2
    } else {
        aclInTensorNum_ = 2; // 2: self, mat2
    }
    aclOutTensorNum_ = 1;
}

void LinearAclnnRunner::InitTensorIndex()
{
    atbInTensorIndex_ = 0;
    aclInTensorIndex_ = 0;
    atbOutTensorIndex_ = 0;
    aclOutTensorIndex_ = 0;
    matmulSelfAclTensorIndex_ = 0;
    matmulMat2AclTensorIndex_ = 0;
    matmulOutAclTensorIndex_ = 0;
    addmmSelfAclTensorIndex_ = 0;
    addmmMat1AclTensorIndex_ = 0;
    addmmMat2AclTensorIndex_ = 0;
    addmmOutAclTensorIndex_ = 0;
}

Status LinearAclnnRunner::CreateMatmulSelfAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateMatmulSelfAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateXAclnnTensor(MATMUL_SELF_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul self aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    matmulSelfAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateMatmulMat2AclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateMatmulMat2AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = isWeightNz_ ?
                                                      CreateWeightNzAclnnTensor(MATMUL_MAT2_ACLNN_TENSOR_IDX) :
                                                      CreateWeightAclnnTensor(MATMUL_MAT2_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul mat2 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    matmulMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateMatmulOutAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateMatmulOutAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateOutputAclnnTensor(MATMUL_OUT_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "matmul out aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    matmulOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmSelfAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmSelfAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateBiasAclnnTensor(ADDMM_SELF_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm self aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    addmmSelfAclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmMat1AclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmMat1AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateXAclnnTensor(ADDMM_MAT1_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm mat1 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    addmmMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmMat2AclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmMat2AclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = isWeightNz_ ? CreateWeightNzAclnnTensor(ADDMM_MAT2_ACLNN_TENSOR_IDX) :
                                                                CreateWeightAclnnTensor(ADDMM_MAT2_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm mat2 aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclInTensors.at(aclInTensorIndex_) = aclnnTensorPtr;
    addmmMat2AclTensorIndex_ = aclInTensorIndex_++;
    return NO_ERROR;
}

Status LinearAclnnRunner::CreateAddmmOutAclnnTensor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::CreateAddmmOutAclnnTensor";

    std::shared_ptr<AclNNTensor> aclnnTensorPtr = CreateOutputAclnnTensor(ADDMM_OUT_ACLNN_TENSOR_IDX);
    if (!aclnnTensorPtr->tensor) {
        ATB_LOG(ERROR) << GetLogPrefix() << "addmm out aclCreateTensor failed";
        return ERROR_INTERNAL_ERROR;
    }
    aclnnVariantPack_.aclOutTensors.at(aclOutTensorIndex_) = aclnnTensorPtr;
    addmmOutAclTensorIndex_ = aclOutTensorIndex_++;
    return NO_ERROR;
}

aclnnStatus LinearAclnnRunner::SetAclnnMatmulWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnMatmulWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(matmulSelfAclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(matmulMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(matmulOutAclTensorIndex_)->tensor;

    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnMatmulGetWorkspaceSizeFunc_(self, mat2, out, cubeMathType,
                                                       &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearAclnnRunner::SetAclnnAddmmWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnAddmmWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(addmmSelfAclTensorIndex_)->tensor;
    aclTensor *mat1 = aclnnVariantPack_.aclInTensors.at(addmmMat1AclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(addmmMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(addmmOutAclTensorIndex_)->tensor;

    if (alpha_) {
        if (aclDestroyScalar(alpha_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "alpha aclDestroyScalar error";
            return ERROR_INTERNAL_ERROR;
        }
        alpha_ = nullptr;
    }
    if (beta_) {
        if (aclDestroyScalar(beta_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "beta aclDestroyScalar error";
            return ERROR_INTERNAL_ERROR;
        }
        beta_ = nullptr;
    }

    float alphaValue = 1.0f;
    alpha_ = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    float betaValue = 1.0f;
    beta_ = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnAddmmGetWorkspaceSizeFunc_(self, mat1, mat2, beta_, alpha_, out, cubeMathType,
                                                      &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearAclnnRunner::SetAclnnMatmulWeightNzWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnMatmulWeightNzWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(matmulSelfAclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(matmulMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(matmulOutAclTensorIndex_)->tensor;

    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnMatmulWeightNzGetWorkspaceSizeFunc_(self, mat2, out, cubeMathType,
                                                               &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearAclnnRunner::SetAclnnAddmmWeightNzWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnAddmmWeightNzWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(addmmSelfAclTensorIndex_)->tensor;
    aclTensor *mat1 = aclnnVariantPack_.aclInTensors.at(addmmMat1AclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(addmmMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(addmmOutAclTensorIndex_)->tensor;

    if (alpha_) {
        if (aclDestroyScalar(alpha_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "alpha aclDestroyScalar error";
            return ERROR_INTERNAL_ERROR;
        }
        alpha_ = nullptr;
    }
    if (beta_) {
        if (aclDestroyScalar(beta_) != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "beta aclDestroyScalar error";
            return ERROR_INTERNAL_ERROR;
        }
        beta_ = nullptr;
    }

    float alphaValue = 1.0f;
    alpha_ = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
    float betaValue = 1.0f;
    beta_ = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnAddmmWeightNzGetWorkspaceSizeFunc_(self, mat1, mat2, beta_, alpha_, out, cubeMathType,
                                                              &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

aclnnStatus LinearAclnnRunner::SetAclnnBatchMatMulWeightNzWorkspaceExecutor()
{
    ATB_LOG(INFO) << "LinearAclnnRunner::SetAclnnBatchMatMulWeightNzWorkspaceExecutor";

    aclTensor *self = aclnnVariantPack_.aclInTensors.at(matmulSelfAclTensorIndex_)->tensor;
    aclTensor *mat2 = aclnnVariantPack_.aclInTensors.at(matmulMat2AclTensorIndex_)->tensor;
    aclTensor *out = aclnnVariantPack_.aclOutTensors.at(matmulOutAclTensorIndex_)->tensor;

    int8_t cubeMathType = 1;
    aclOpExecutor *rawExecutePtr = aclnnExecutor_.get();

    aclnnStatus ret = aclnnBatchMatMulWeightNzGetWorkspaceSizeFunc_(
        self, mat2, out, cubeMathType, &(atbVariantPack_.workspaceBufferSize), &rawExecutePtr);
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutePtr, [this](aclOpExecutor *ptr) {
        if (ptr && executorRepeatable_) {
            aclDestroyAclOpExecutor(ptr);
        }
    });
    return ret;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateXAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    if (viewShape.dimNum == 3 && !isBatch_) {
        viewShape.dims[0] = viewShape.dims[0] * viewShape.dims[1];
        viewShape.dims[1] = viewShape.dims[2];
        viewShape.dimNum = 2;
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    if (param_.transposeA) {
        int64_t m = viewShape.dims[viewShape.dimNum - 1];
        int64_t k = viewShape.dims[viewShape.dimNum - 2];
        viewShape.dims[viewShape.dimNum - 2] = m;
        viewShape.dims[viewShape.dimNum - 1] = k;
        int64_t lastStride = aclnnTensorPtr->strides[viewShape.dimNum - 1];
        aclnnTensorPtr->strides[viewShape.dimNum - 1] = aclnnTensorPtr->strides[viewShape.dimNum - 2];
        aclnnTensorPtr->strides[viewShape.dimNum - 2] = lastStride;
    }
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateWeightAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    if (param_.transposeB) {
        int64_t k = viewShape.dims[viewShape.dimNum - 1];
        int64_t n = viewShape.dims[viewShape.dimNum - 2];
        viewShape.dims[viewShape.dimNum - 2] = k;
        viewShape.dims[viewShape.dimNum - 1] = n;
        int64_t lastStride = aclnnTensorPtr->strides[viewShape.dimNum - 1];
        aclnnTensorPtr->strides[viewShape.dimNum - 1] = aclnnTensorPtr->strides[viewShape.dimNum - 2];
        aclnnTensorPtr->strides[viewShape.dimNum - 2] = lastStride;
    }
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateWeightNzAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    Dims storageShape = atbTensor.desc.shape;
    if (viewShape.dimNum == 4) {
        Dims oldShape = viewShape;
        if (isBatch_) {
            viewShape.dims[0] = oldShape.dims[0];
            viewShape.dims[1] = oldShape.dims[2];
            viewShape.dims[2] = oldShape.dims[1] * oldShape.dims[3];
            viewShape.dimNum = 3;
            ATB_LOG(INFO) << GetLogPrefix() << "viewShape: [" << viewShape.dims[0] << ", " << viewShape.dims[1] << ", "
                          << viewShape.dims[2] << "]";
        } else {
            viewShape.dims[0] = oldShape.dims[2];
            viewShape.dims[1] = oldShape.dims[1] * oldShape.dims[3];
            viewShape.dimNum = 2;
            ATB_LOG(INFO) << GetLogPrefix() << "viewShape: [" << viewShape.dims[0] << ", " << viewShape.dims[1] << "]";
        }
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    if (param_.transposeB) {
        int64_t k = viewShape.dims[viewShape.dimNum - 1];
        int64_t n = viewShape.dims[viewShape.dimNum - 2];
        viewShape.dims[viewShape.dimNum - 2] = k;
        viewShape.dims[viewShape.dimNum - 1] = n;
        int64_t lastStride = aclnnTensorPtr->strides[viewShape.dimNum - 1];
        aclnnTensorPtr->strides[viewShape.dimNum - 1] = aclnnTensorPtr->strides[viewShape.dimNum - 2];
        aclnnTensorPtr->strides[viewShape.dimNum - 2] = lastStride;
    }
    if (storageShape.dimNum == 2) {
        Dims oldShape = storageShape;
        storageShape.dims[0] = 1;
        storageShape.dims[1] = UtilsInternal::AlignUp(oldShape.dims[1], DEFAULT_ALIGN) / DEFAULT_ALIGN;
        storageShape.dims[2] = UtilsInternal::AlignUp(oldShape.dims[0], DEFAULT_ALIGN);
        storageShape.dims[3] = DEFAULT_ALIGN;
        storageShape.dimNum = 4;
        ATB_LOG(INFO) << GetLogPrefix() << "storageShape: [" << storageShape.dims[0] << ", " << storageShape.dims[1]
                      << ", " << storageShape.dims[2] << ", " << storageShape.dims[3] << "]";
    } else if (storageShape.dimNum == 3) {
        Dims oldShape = storageShape;
        storageShape.dims[0] = oldShape.dims[0];
        storageShape.dims[1] = 1;
        storageShape.dims[2] = UtilsInternal::AlignUp(oldShape.dims[2], DEFAULT_ALIGN) / DEFAULT_ALIGN;
        storageShape.dims[3] = UtilsInternal::AlignUp(oldShape.dims[1], DEFAULT_ALIGN);
        storageShape.dims[4] = DEFAULT_ALIGN;
        storageShape.dimNum = 5;
        ATB_LOG(INFO) << GetLogPrefix() << "storageShape: [" << storageShape.dims[0] << ", " << storageShape.dims[1]
                      << ", " << storageShape.dims[2] << ", " << storageShape.dims[3] << ", " << storageShape.dims[4]
                      << "]";
    } else if (storageShape.dimNum == 4) {
        if (isBatch_) {
            Dims oldShape = storageShape;
            storageShape.dims[0] = oldShape.dims[0];
            storageShape.dims[1] = 1;
            storageShape.dims[2] = oldShape.dims[1];
            storageShape.dims[3] = oldShape.dims[2];
            storageShape.dims[4] = oldShape.dims[3];
            storageShape.dimNum = 5;
            ATB_LOG(INFO) << GetLogPrefix() << "storageShape: [" << storageShape.dims[0] << ", " << storageShape.dims[1]
                          << ", " << storageShape.dims[2] << ", " << storageShape.dims[3] << ", "
                          << storageShape.dims[4] << "]";
        }
    }
    aclnnTensorPtr->tensor =
        aclCreateTensor(viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
                        atbTensor.desc.format, storageShape.dims, storageShape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateBiasAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.inTensors.at(atbInTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::CreateOutputAclnnTensor(int aclnnTensorIndex)
{
    Tensor atbTensor = atbVariantPack_.outTensors.at(atbOutTensorIndex_++);
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = InitAclnnTensor(atbTensor, aclnnTensorIndex);
    Dims viewShape = atbTensor.desc.shape;
    if (viewShape.dimNum == 3 && !isBatch_) {
        viewShape.dims[0] = viewShape.dims[0] * viewShape.dims[1];
        viewShape.dims[1] = viewShape.dims[2];
        viewShape.dimNum = 2;
    }
    aclnnTensorPtr->strides = GetCopyTensorStride(viewShape);
    aclnnTensorPtr->tensor = aclCreateTensor(
        viewShape.dims, viewShape.dimNum, atbTensor.desc.dtype, aclnnTensorPtr->strides.data(), 0,
        atbTensor.desc.format, atbTensor.desc.shape.dims, atbTensor.desc.shape.dimNum, atbTensor.deviceData);
    return aclnnTensorPtr;
}

std::shared_ptr<AclNNTensor> LinearAclnnRunner::InitAclnnTensor(Tensor atbTensor, int aclnnTensorIndex)
{
    std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
    aclnnTensorPtr->atbTensor = atbTensor;
    aclnnTensorPtr->tensorIdx = aclnnTensorIndex;
    aclnnTensorPtr->needUpdateTensorDataPtr = true;
    return aclnnTensorPtr;
}

REG_RUNNER_TYPE(LinearAclnnRunner);
} // namespace atb
