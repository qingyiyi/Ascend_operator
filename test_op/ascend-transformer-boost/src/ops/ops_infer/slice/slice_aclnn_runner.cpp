/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "slice_aclnn_runner.h"
#include <aclnn/opdev/op_errno.h>
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/operation_register.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t INDEX_ZERO = 0;
static const uint32_t SIZE_TWO = 2;
} // namespace

namespace atb {
AclnnSliceV2GetWorkspaceSizeFunc SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_ = nullptr;
AclnnSliceV2Func SliceAclnnRunner::aclnnExecuteFunc_ = nullptr;
AclnnCastGetWorkspaceSizeFunc SliceAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ = nullptr;
AclnnCastExecuteFunc SliceAclnnRunner::aclnnCastExecuteFunc_ = nullptr;

SliceAclnnRunner::SliceAclnnRunner(const infer::SliceParam &param) : AclnnRunner("SliceAclnnRunner"), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "SliceAclnnRunner::SliceAclnnRunner created";
}

void SliceAclnnRunner::CleanUp()
{
    aclnnStatus ret = 0;
    if (self_ != nullptr) {
        ret = aclDestroyTensor(self_);
        if (ret != ACL_SUCCESS)
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy self_->tensor failed with return value: " << ret;
        self_ = nullptr;
    }
    if (out_ != nullptr) {
        ret = aclDestroyTensor(out_);
        if (ret != ACL_SUCCESS)
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy out_->tensor failed with return value: " << ret;
        out_ = nullptr;
    }
}

SliceAclnnRunner::~SliceAclnnRunner()
{
    CleanUp();
}

Status SliceAclnnRunner::LoadAclnnFuncs()
{
    ATB_LOG(INFO) << "SliceAclnnRunner LoadAclnnFuncs";
    if (SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_ != nullptr && SliceAclnnRunner::aclnnExecuteFunc_ != nullptr &&
        SliceAclnnRunner::aclnnCastGetWorkspaceSizeFunc_ != nullptr &&
        SliceAclnnRunner::aclnnCastExecuteFunc_ != nullptr) {
        return NO_ERROR;
    }
    Status st = NO_ERROR;
    st = LoadFromSharedObjectFile("aclnnSliceV2GetWorkspaceSize", "aclnnSliceV2",
                                  SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_, SliceAclnnRunner::aclnnExecuteFunc_);
    if (st != NO_ERROR) {
        return st;
    }
    st = LoadFromSharedObjectFile("aclnnCastGetWorkspaceSize", "aclnnCast",
                                  SliceAclnnRunner::aclnnCastGetWorkspaceSizeFunc_,
                                  SliceAclnnRunner::aclnnCastExecuteFunc_);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status SliceAclnnRunner::BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack";
    ATB_LOG(INFO) << GetLogPrefix() << "variantPack: " << runnerVariantPack.ToString();
    atbVariantPack_ = runnerVariantPack;
    Status ret = NO_ERROR;
    // self
    aclnnVariantPack_.aclInTensors.reserve(IN_TENSOR_NUM);
    aclnnVariantPack_.aclInTensors.resize(IN_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        atb::Tensor atbTensor = runnerVariantPack.inTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        aclnnVariantPack_.aclInTensors[i] = aclnnTensorPtr;
    }

    // output
    aclnnVariantPack_.aclOutTensors.reserve(OUT_TENSOR_NUM);
    aclnnVariantPack_.aclOutTensors.resize(OUT_TENSOR_NUM);
    for (size_t i = 0; i < aclnnVariantPack_.aclOutTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensorPtr = std::make_shared<AclNNTensor>();
        ATB_LOG(INFO) << GetLogPrefix() << "BuildAclnnVariantPack outTensor index: " << i;
        atb::Tensor atbTensor = runnerVariantPack.outTensors.at(i);
        aclnnTensorPtr->atbTensor = atbTensor;
        aclnnTensorPtr->strides = GetCopyTensorStride(atbTensor.desc.shape);
        ret = CallAclCreateTensor(atbTensor.desc.shape, atbTensor.desc.shape, atbTensor, aclnnTensorPtr,
                                  atbTensor.desc.dtype);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create aclTensor by aclCreateTensor failed!";
            return ret;
        }
        aclnnTensorPtr->tensorIdx = static_cast<int>(i);
        aclnnTensorPtr->needUpdateTensorDataPtr = true;
        aclnnVariantPack_.aclOutTensors[i] = aclnnTensorPtr;
    }
    aclnnStatus status = ACL_SUCCESS;
    selfBufferSize_ = 0;
    // temp self
    if (this->atbVariantPack_.inTensors.at(INDEX_ZERO).desc.dtype == ACL_UINT32) {
        selfBufferSize_ = this->atbVariantPack_.inTensors.at(INDEX_ZERO).dataSize * SIZE_TWO;
        atb::SVector<int64_t> strides = GetCopyTensorStride(this->atbVariantPack_.inTensors.at(INDEX_ZERO).desc.shape);
        Dims viewDims;

        viewDims.dimNum = this->atbVariantPack_.inTensors.at(INDEX_ZERO).desc.shape.dimNum;
        for (size_t i = 0; i < viewDims.dimNum; i++) {
            viewDims.dims[i] = this->atbVariantPack_.inTensors.at(INDEX_ZERO).desc.shape.dims[i];
        }
        if (self_ != nullptr) {
            status = aclDestroyTensor(self_);
            if (status != ACL_SUCCESS) {
                ATB_LOG(ERROR) << GetLogPrefix() << "destroy self_->tensor failed with statusurn value: " << status;
                return ERROR_CANN_ERROR;
            }
            self_ = nullptr;
        }
        self_ = aclCreateTensor(viewDims.dims, viewDims.dimNum, ACL_INT64, strides.data(), 0,
                                this->atbVariantPack_.inTensors.at(INDEX_ZERO).desc.format, viewDims.dims,
                                viewDims.dimNum, nullptr);
        if (self_ == nullptr) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create int64 indices by aclCreateTensor failed!";
            return ERROR_CANN_ERROR;
        }
    }
    outBufferSize_ = 0;
    // temp out
    if (this->atbVariantPack_.outTensors.at(INDEX_ZERO).desc.dtype == ACL_UINT32) {
        outBufferSize_ = this->atbVariantPack_.outTensors.at(INDEX_ZERO).dataSize * SIZE_TWO;
        atb::SVector<int64_t> strides = GetCopyTensorStride(this->atbVariantPack_.outTensors.at(INDEX_ZERO).desc.shape);
        Dims viewDims;

        viewDims.dimNum = this->atbVariantPack_.outTensors.at(INDEX_ZERO).desc.shape.dimNum;
        for (size_t i = 0; i < viewDims.dimNum; i++) {
            viewDims.dims[i] = this->atbVariantPack_.outTensors.at(INDEX_ZERO).desc.shape.dims[i];
        }
        if (out_ != nullptr) {
            status = aclDestroyTensor(out_);
            if (status != ACL_SUCCESS) {
                ATB_LOG(ERROR) << GetLogPrefix() << "destroy out->tensor failed with return value: " << status;
                return ERROR_CANN_ERROR;
            }
            out_ = nullptr;
        }
        out_ = aclCreateTensor(viewDims.dims, viewDims.dimNum, ACL_INT64, strides.data(), 0,
                               this->atbVariantPack_.outTensors.at(INDEX_ZERO).desc.format, viewDims.dims,
                               viewDims.dimNum, nullptr);
        if (out_ == nullptr) {
            ATB_LOG(ERROR) << GetLogPrefix() << "create int64 indices by aclCreateTensor failed!";
            return ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

aclnnStatus SliceAclnnRunner::SetAclNNWorkspaceExecutor()
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Slice setup start.";
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn Slice, aclInTensors size: " << aclnnVariantPack_.aclInTensors.size()
                  << ", aclOutTensors size: " << aclnnVariantPack_.aclOutTensors.size();
    std::shared_ptr<AclNNTensor> xPtr = aclnnVariantPack_.aclInTensors.at(0);
    aclTensor *x = xPtr->tensor;                                       // self
    aclTensor *output = aclnnVariantPack_.aclOutTensors.at(0)->tensor; // out
    Dims xDims = xPtr->atbTensor.desc.shape;

    SVector<int64_t> offsets = param_.offsets;
    SVector<int64_t> size = param_.size;
    int64_t dimNum = offsets.size();
    int64_t steps[dimNum];
    int64_t axes[dimNum];
    int64_t starts[dimNum];
    int64_t ends[dimNum];
    for (int64_t i = 0; i < dimNum; ++i) {
        steps[i] = 1;
        axes[i] = i;
        starts[i] = offsets[i];
        if (offsets[i] < 0) {
            starts[i] = xDims.dims[i] + offsets[i];
        } else {
            starts[i] = offsets[i];
        }
        if (size[i] == -1) {
            ends[i] = xDims.dims[i];
        } else {
            ends[i] = starts[i] + size[i];
        }
    }

    aclnnStatus ret = ACL_SUCCESS;
    if (stepsArray_) {
        ret = aclDestroyIntArray(stepsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray steps failed!";
            return ret;
        }
        stepsArray_ = nullptr;
    }
    if (axesArray_) {
        ret = aclDestroyIntArray(axesArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray axes failed!";
            return ret;
        }
        axesArray_ = nullptr;
    }
    if (startsArray_) {
        ret = aclDestroyIntArray(startsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray starts failed!";
            return ret;
        }
        startsArray_ = nullptr;
    }
    if (endsArray_) {
        ret = aclDestroyIntArray(endsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray ends failed!";
            return ret;
        }
        startsArray_ = nullptr;
    }
    stepsArray_ = aclCreateIntArray(steps, dimNum);
    axesArray_ = aclCreateIntArray(axes, dimNum);
    startsArray_ = aclCreateIntArray(starts, dimNum);
    endsArray_ = aclCreateIntArray(ends, dimNum);

    // aclnnSliceV2 does not support uint32, convert uint32 to int64
    if (selfBufferSize_ != 0) {
        aclOpExecutor *rawCastExecutorPtr = this->aclnnCastExecutor1st_.get();
        ret = SliceAclnnRunner::aclnnCastGetWorkspaceSizeFunc_(x, ACL_INT64, self_, &(this->cast1stWorkspaceSize_),
                                                               &rawCastExecutorPtr);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclnnCastGetWorkspaceSize failed!";
            return ret;
        }
        ret = aclSetAclOpExecutorRepeatable(rawCastExecutorPtr);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Set Cast AclOpExecutorRepeatable failed!";
            return ret;
        }
        this->aclnnCastExecutor1st_ = std::shared_ptr<aclOpExecutor>(rawCastExecutorPtr, [this](aclOpExecutor *ptr) {
            if (ptr) { // 可复用时才手动销毁aclOpExecutor
                aclDestroyAclOpExecutor(ptr);
            }
        });
    }

    aclOpExecutor *rawExecutorPtr = aclnnExecutor_.get();
    ret = SliceAclnnRunner::aclnnGetWorkspaceSizeFunc_(selfBufferSize_ == 0 ? x : self_, startsArray_, endsArray_,
                                                       axesArray_, stepsArray_, outBufferSize_ == 0 ? output : out_,
                                                       &(this->sliceWorkspaceSize_), &rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "aclnnGetWorkspaceSize failed!";
        return ret;
    }
    ret = aclSetAclOpExecutorRepeatable(rawExecutorPtr);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Set Slice AclOpExecutorRepeatable failed!";
        return ret;
    }
    aclnnExecutor_ = std::shared_ptr<aclOpExecutor>(rawExecutorPtr, [this](aclOpExecutor *ptr) {
        if (ptr) { // 可复用时才手动销毁aclOpExecutor
            aclDestroyAclOpExecutor(ptr);
        }
    });

    // convert int64 to uint32
    if (outBufferSize_ != 0) {
        aclOpExecutor *rawCastExecutorPtr = this->aclnnCastExecutor2nd_.get();
        ret = SliceAclnnRunner::aclnnCastGetWorkspaceSizeFunc_(out_, ACL_UINT32, output, &(this->cast2ndWorkspaceSize_),
                                                               &rawCastExecutorPtr);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclnnCastGetWorkspaceSize failed!";
            return ret;
        }
        ret = aclSetAclOpExecutorRepeatable(rawCastExecutorPtr);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Set Cast AclOpExecutorRepeatable failed!";
            return ret;
        }
        this->aclnnCastExecutor2nd_ = std::shared_ptr<aclOpExecutor>(rawCastExecutorPtr, [this](aclOpExecutor *ptr) {
            if (ptr) { // 可复用时才手动销毁aclOpExecutor
                aclDestroyAclOpExecutor(ptr);
            }
        });
    }
    this->atbVariantPack_.workspaceBufferSize = this->sliceWorkspaceSize_ + this->cast1stWorkspaceSize_ +
                                                this->cast2ndWorkspaceSize_ + this->selfBufferSize_ +
                                                this->outBufferSize_;
    ATB_LOG(INFO) << GetLogPrefix() << "workspaceSize: " << atbVariantPack_.workspaceBufferSize;
    return ret;
}

bool SliceAclnnRunner::useCache()
{
    return false;
}

Status SliceAclnnRunner::LaunchAclnnKernel()
{
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute start.";
    void *executeStream = GetExecuteStream(atbVariantPack_.context);
    aclnnStatus ret = ACL_SUCCESS;
    if (selfBufferSize_ != 0) {
        ret = aclSetOutputTensorAddr(this->aclnnCastExecutor1st_.get(), INDEX_ZERO, this->self_,
                                     this->atbVariantPack_.workspaceBuffer + this->sliceWorkspaceSize_ +
                                         this->cast1stWorkspaceSize_ + this->cast2ndWorkspaceSize_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclSetOutputTensorAddr failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
        ret = aclSetInputTensorAddr(this->aclnnExecutor_.get(), INDEX_ZERO, this->self_,
                                    this->atbVariantPack_.workspaceBuffer + this->sliceWorkspaceSize_ +
                                        this->cast1stWorkspaceSize_ + this->cast2ndWorkspaceSize_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclSetInputTensorAddr failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    if (outBufferSize_ != 0) {
        ret = aclSetOutputTensorAddr(this->aclnnExecutor_.get(), INDEX_ZERO, this->out_,
                                     this->atbVariantPack_.workspaceBuffer + this->sliceWorkspaceSize_ +
                                         this->cast1stWorkspaceSize_ + this->cast2ndWorkspaceSize_ + selfBufferSize_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclSetOutputTensorAddr failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
        ret = aclSetInputTensorAddr(this->aclnnCastExecutor2nd_.get(), INDEX_ZERO, this->out_,
                                    this->atbVariantPack_.workspaceBuffer + this->sliceWorkspaceSize_ +
                                        this->cast1stWorkspaceSize_ + this->cast2ndWorkspaceSize_ + selfBufferSize_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "aclSetInputTensorAddr failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    if (selfBufferSize_ != 0) {
        ret = SliceAclnnRunner::aclnnCastExecuteFunc_(atbVariantPack_.workspaceBuffer + this->sliceWorkspaceSize_,
                                                      this->cast1stWorkspaceSize_, aclnnExecutor_.get(), executeStream);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    ret = SliceAclnnRunner::aclnnExecuteFunc_(atbVariantPack_.workspaceBuffer, this->sliceWorkspaceSize_,
                                              aclnnExecutor_.get(), executeStream);
    if (ret != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
        return ERROR_CANN_ERROR;
    }
    if (outBufferSize_ != 0) {
        ret = SliceAclnnRunner::aclnnCastExecuteFunc_(
            atbVariantPack_.workspaceBuffer + this->sliceWorkspaceSize_ + this->cast1stWorkspaceSize_,
            this->cast2ndWorkspaceSize_, this->aclnnCastExecutor2nd_.get(), executeStream);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op kernel launch failed with return value: " << ret;
            return ERROR_CANN_ERROR;
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "LaunchAclnnKernel execute success.";
    if (stepsArray_) {
        ret = aclDestroyIntArray(stepsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray steps failed!";
            return ERROR_CANN_ERROR;
        }
        stepsArray_ = nullptr;
    }
    if (axesArray_) {
        ret = aclDestroyIntArray(axesArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray axes failed!";
            return ERROR_CANN_ERROR;
        }
        axesArray_ = nullptr;
    }
    if (startsArray_) {
        ret = aclDestroyIntArray(startsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray starts failed!";
            return ERROR_CANN_ERROR;
        }
        startsArray_ = nullptr;
    }
    if (endsArray_) {
        ret = aclDestroyIntArray(endsArray_);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << GetLogPrefix() << "destroy aclIntArray ends failed!";
            return ERROR_CANN_ERROR;
        }
        endsArray_ = nullptr;
    }
    return NO_ERROR;
}

REG_RUNNER_TYPE(SliceAclnnRunner);
} // namespace atb
