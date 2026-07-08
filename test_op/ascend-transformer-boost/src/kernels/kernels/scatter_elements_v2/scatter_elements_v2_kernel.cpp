/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "asdops/params/scatter_elements_v2.h"

#include <mki/base/kernel_base.h>
#include <mki/utils/log/log.h>
#include <mki/utils/math/tensor_utils.h>
#include <mki/utils/platform/platform_info.h>
#include <mki_loader/op_register.h>

#include "kernels/scatter_elements_v2/tiling/scatter_elements_v2_tiling.h"

namespace AsdOps {
using namespace Mki;

static constexpr char NONE[] = "none";
static constexpr char ADD[] = "add";
template <TensorDType INPUT_TYPE, TensorDType INDICE_TYPE, const char *REDUCTION>
class ScatterElementsV2Kernel : public KernelBase {
public:
    explicit ScatterElementsV2Kernel(const std::string &kernelName, const BinHandle *handle) noexcept
        : KernelBase(kernelName, handle)
    {
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        // 检查参数类型是否为 ScatterElemens
        MKI_CHECK(launchParam.GetParam().Type() == typeid(OpParam::ScatterElementsV2),
                  "ScatterElements: param type invalid", return false);
        MKI_CHECK(PlatformInfo::Instance().GetPlatformType() == PlatformType::ASCEND_910B,
                  "ScatterElements operator only supported ASCEND_910B platform", return false);
        // 检查输入张量数量是否为 3（input, indices, updates）
        MKI_CHECK(launchParam.GetInTensorCount() == 3, "input num invalid", return false);
        // 检查输出张量数量是否为 1（output）
        MKI_CHECK(launchParam.GetOutTensorCount() == 1, "output num invalid", return false);
        return true;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        // 调用 ScatterElementsV2 的通用分片函数
        return ScatterElementsV2CommonTiling(GetName(), launchParam, kernelInfo_, *GetBinHandle());
    }
};

// ScatterElementsV2Kernel
// Note: The 'attr' parameter is deprecated and has no effect. Any value passed will be trated the same.
using ScatterElementsV2Int32Int32Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Int32Int64Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT32, TENSOR_DTYPE_INT64, NONE>;

using ScatterElementsV2Bfloat16Int32Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Bfloat16Int64Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_BF16, TENSOR_DTYPE_INT64, NONE>;

using ScatterElementsV2Uint8Int32Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_UINT8, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Uint8Int64Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_UINT8, TENSOR_DTYPE_INT64, NONE>;

using ScatterElementsV2Int8Int32Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Int8Int64Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_INT8, TENSOR_DTYPE_INT64, NONE>;

using ScatterElementsV2Float32Int32Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Float32Int64Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT, TENSOR_DTYPE_INT64, NONE>;

using ScatterElementsV2Float16Int32Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT32, NONE>;
using ScatterElementsV2Float16Int64Kernel = ScatterElementsV2Kernel<TENSOR_DTYPE_FLOAT16, TENSOR_DTYPE_INT64, NONE>;

REG_KERNEL_BASE(ScatterElementsV2Int32Int32Kernel);
REG_KERNEL_BASE(ScatterElementsV2Int32Int64Kernel);
REG_KERNEL_BASE(ScatterElementsV2Bfloat16Int32Kernel);
REG_KERNEL_BASE(ScatterElementsV2Bfloat16Int64Kernel);
REG_KERNEL_BASE(ScatterElementsV2Uint8Int32Kernel);
REG_KERNEL_BASE(ScatterElementsV2Uint8Int64Kernel);
REG_KERNEL_BASE(ScatterElementsV2Int8Int32Kernel);
REG_KERNEL_BASE(ScatterElementsV2Int8Int64Kernel);
REG_KERNEL_BASE(ScatterElementsV2Float32Int32Kernel);
REG_KERNEL_BASE(ScatterElementsV2Float32Int64Kernel);
REG_KERNEL_BASE(ScatterElementsV2Float16Int32Kernel);
REG_KERNEL_BASE(ScatterElementsV2Float16Int64Kernel);
} // namespace AsdOps