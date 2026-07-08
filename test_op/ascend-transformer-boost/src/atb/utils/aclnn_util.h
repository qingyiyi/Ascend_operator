/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATB_ACLNN_UTILS_H
#define ATB_ACLNN_UTILS_H

#include <mki/utils/dl/dl.h>
#include <mki/utils/env/env.h>

#include "atb/types.h"
#include "atb/utils/runner_variant_pack.h"

namespace atb {

/// Calculate stride along each dimension when copying a tensor.
///
/// \param tensorDims The size of each axis in a tensor.
/// \return The number of steps needed in memory to move to the next element
///     along each dimension when copying a tensor.
atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims);

/// Calculate stride along each dimension when transposing a tensor.
///
/// \param tensorDims The size of each axis in a tensor.
/// \return The number of steps needed in memory to move to the next element
///     along each dimension when transposing a tensor.
atb::SVector<int64_t> GetTransposeTensorStride(atb::Dims &tensorDims);

/// Call `aclCreateTensor` API to create `aclTensor`.
///
/// \param viewDims The dimension of tensor's view shape.
/// \param storageDims The dimension of tensor's storage shape.
/// \param atbTensor The tensor passed through ATB framework.
/// \param aclnnTensor A pointer to an `AclNNTensor` object whose `tensor` attribute is updated
///     using the return value of `aclCreateTensor`.
/// \param dataType the dataType of the tensor, used for specifying empty tensors.
/// \return A status code that indicates whether `aclTensor` has been created.
atb::Status CallAclCreateTensor(atb::Dims &viewDims, atb::Dims &storageDims, atb::Tensor &atbTensor,
                                std::shared_ptr<AclNNTensor> aclnnTensor, aclDataType dataType = ACL_DT_UNDEFINED, int64_t offset = 0);

/// Reshape a tensor by squeezing batch size axis and seq len axis if the tensor's shape has two dimensions.
///
/// \param atbTensor An `atb::Tensor` object whose tensor shape requires reshaping.
/// \return The `atb::Tensor` after reshaping.
atb::Tensor SqueezeBatchSeq(atb::Tensor atbTensor);

/// Check whether `aclnnVariantPack` and `atbVariantPack` are the same, except for tensors' device data.
///
/// Two variant packs, `aclnnVariantPack` and `atbVariantPack`, are considered the same if they have the same
/// number of tensors, and each corresponding tensor in the variant packs
/// has identical data type，format，shape and host data.
/// \param aclnnVariantPack An `AclNNVariantPack` object containing tensor info of existing AclNN operation.
/// \param atbVariantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
/// \return A boolean value that indicates whether `aclnnVariantPack` and `atbVariantPack` are the same,
/// except for tensors' device data.
bool IsAclnnAtbVariankPackEqual(const AclNNVariantPack &aclnnVariantPack, const VariantPack &atbVariantPack);

/// Check whether `aclnnVariantPack` and `atbVariantPack` are the same, except for tensors' device data.
///
/// Two variant packs, `aclnnVariantPack` and `atbVariantPack`, are considered the same if they have the same
/// number of tensors, and each corresponding tensor in the variant packs
/// has identical data type，format，shape and host data.
/// \param aclnnVariantPack An `AclNNVariantPack` object containing tensor info of existing AclNN operation.
/// \param atbVariantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
/// \return A boolean value that indicates whether `aclnnVariantPack` and `atbVariantPack` are the same,
/// except for tensors' device data.
bool IsAclnnRunnerVariankPackEqual(const AclNNVariantPack &aclnnVariantPack,
                                   const RunnerVariantPack &runnerVariantPack);

/// Create a pointer to `AclNNTensor` by configuring it with tensor information extracted from `atbTensor`.
///
/// `needUpdateTensorDataPtr` is set to true, `atbTensor` and `tensorIdx` are updated with input parameters.
/// `strides` is updated by `GetCopyTensorStride`.
/// \param atbTensor Tensor passed through the ATB framework.
/// \param tensorIdx The index of the tensor in `aclOpExecutor`'s parameter list.
/// \return A pointer to an `AclNNTensor` object whose attributes are updated.
std::shared_ptr<AclNNTensor> CreateTensor(atb::Tensor atbTensor, int tensorIdx);

int ConvertTensorToSeqLengths(atb::Tensor &tensor, aclIntArray *&actualSeqLengths);

std::shared_ptr<AclNNTensor> CreateAclnnTensor(Tensor atbTensor, int aclnnTensorIndex, Dims viewShape,
                                               SVector<int64_t> strides);

/// Load GetWorkSpaceSize and Execute func from aclnn .so file.
///
/// \param workSpaceSizeFuncName The name of first symbol to load.
/// \param executeFuncName The name of second symbol to load.
/// \param workSpaceSizefunc Reference to a pointer that receives the address of the first symbol.
/// \param executeFunc Reference to a pointer that receives the address of the second symbol.
/// \return Status
///         - `NO_ERROR` Successfully loaded and resolved both symbols.
///         - `ERROR_INVALID_PARAM` loaded failed.
template <typename GetWorkspaceSizeFunc, typename ExecuteFunc>
Status LoadFromSharedObjectFile(const char *workSpaceSizeFuncName, const char *executeFuncName,
                                GetWorkspaceSizeFunc *&workSpaceSizeFunc, ExecuteFunc *&executeFunc)
{
    ATB_LOG(INFO) << "Loading func " << workSpaceSizeFuncName << " and " << executeFuncName;
    static std::once_flag onceFlag;
    static std::unique_ptr<Mki::Dl> mkiDl;
    static std::atomic<Status> initFlag{ERROR_INVALID_PARAM};

    std::call_once(onceFlag, []() {
        std::string soPath;
        const char *ascendHomePath = Mki::GetEnv("ASCEND_HOME_PATH");
        if (ascendHomePath == nullptr || *ascendHomePath == '\0') {
            ATB_LOG(ERROR) << "ASCEND_HOME_PATH is null.";
            initFlag.store(ERROR_INVALID_PARAM, std::memory_order_release);
            return;
        }
        soPath = std::string(ascendHomePath) + "/lib64/libopapi.so";
        ATB_LOG(INFO) << "Loading path: " << soPath;

        auto dl = std::make_unique<Mki::Dl>(soPath, false);
        if (!dl || !dl->IsValid()) {
            ATB_LOG(ERROR) << "Load so file failed. Please check soPath: " << soPath;
            initFlag.store(ERROR_INVALID_PARAM, std::memory_order_release);
            return;
        }
        mkiDl = std::move(dl);
        ATB_LOG(INFO) << "Dlopen " << soPath << " successfully.";
        initFlag.store(NO_ERROR, std::memory_order_release);
    });

    Status st = initFlag.load(std::memory_order_acquire);
    if (st != NO_ERROR)
        return st;
    if (!mkiDl) {
        ATB_LOG(ERROR) << "mkiDl is null, check if it was reset.";
        return ERROR_INVALID_PARAM;
    }
    void *sym1 = mkiDl->GetSymbol(workSpaceSizeFuncName);
    if (!sym1) {
        ATB_LOG(ERROR) << "Symbol " << workSpaceSizeFuncName << " not found.";
        return ERROR_INVALID_PARAM;
    }
    void *sym2 = mkiDl->GetSymbol(executeFuncName);
    if (!sym2) {
        ATB_LOG(ERROR) << "Symbol " << executeFuncName << " not found.";
        return ERROR_INVALID_PARAM;
    }
    workSpaceSizeFunc = reinterpret_cast<GetWorkspaceSizeFunc *>(sym1);
    executeFunc = reinterpret_cast<ExecuteFunc *>(sym2);
    return st;
}
} // namespace atb
#endif