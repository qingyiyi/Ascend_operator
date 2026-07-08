/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_COHERE_LAYER_NORM_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_COHERE_LAYER_NORM_H_

#include <mki/kernel_info.h>
#include <mki/launch_param.h>
#include <mki/utils/status/status.h>
#include "kernels/norm/coherelayernorm/tiling/tiling_data.h"

namespace AsdOps {
using namespace Mki;
Status CohereLayerNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo, const
                             KernelBufferInfoCohereLayerNorm &kernelBufferInfo);
} // namespace AsdOps
#endif