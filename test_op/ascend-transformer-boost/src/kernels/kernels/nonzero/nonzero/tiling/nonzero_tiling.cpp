/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/math/math.h>
#include <mki/utils/SVector/SVector.h>
#include "tiling_data.h"
#include "nonzero_tiling.h"

namespace AsdOps {
Status NonzeroTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    NonzeroTilingData *tilingDataPtr =
        reinterpret_cast<AsdOps::NonzeroTilingData *>(kernelInfo.GetTilingHostAddr());
    MKI_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    int64_t xNumel = launchParam.GetInTensor(0).Numel();
    MKI_CHECK(xNumel <= std::numeric_limits<uint32_t>::max() && xNumel > 0, "xNumel invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    tilingDataPtr->xNumel = static_cast<uint32_t>(xNumel);
    Mki::SVector<int64_t> xShape = launchParam.GetInTensor(0).desc.dims;
    MKI_CHECK(xShape.size() <= MAX_DIM_NUM && xShape.size() > 0, "xshape invalid",
                 return Status::FailStatus(ERROR_INVALID_VALUE));
    for (uint32_t i = 0; i < xShape.size(); i++) {
        MKI_CHECK(xShape[i] <= std::numeric_limits<uint32_t>::max(), "xshape[i] invalid",
                  return Status::FailStatus(ERROR_INVALID_VALUE));
        tilingDataPtr->xDims[i] = static_cast<uint32_t>(xShape[i]);
    }
    tilingDataPtr->xdimLength = xShape.size();
    MKI_CHECK(xShape.size() < std::numeric_limits<uint64_t>::max() / sizeof(int64_t) / tilingDataPtr->xNumel,
              "yshape invalid", return Status::FailStatus(ERROR_INVALID_VALUE));
    uint64_t sysWorkspaceSize = 16;
    kernelInfo.GetScratchSizes() = {sysWorkspaceSize};
    kernelInfo.SetBlockDim(1);

    return Status::OkStatus();
}
} // namespace AsdOps