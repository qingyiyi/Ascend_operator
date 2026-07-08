/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/param_compare.h"
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/infer_op_params.h"
#include "atb/train_op_params.h"
#include "atb/utils/tensor_util.h"

namespace atb {
bool IsLaunchParamEqual(const Mki::LaunchParam &launchParam1, const Mki::LaunchParam &launchParam2)
{
    if (launchParam1.GetInTensorCount() != launchParam2.GetInTensorCount()) {
        return false;
    }

    for (size_t i = 0; i < launchParam1.GetInTensorCount(); ++i) {
        if (!TensorUtil::AsdOpsTensorDescEqual(launchParam1.GetInTensor(i).desc, launchParam2.GetInTensor(i).desc)) {
            return false;
        }
    }

    const Mki::Any &specificParam1 = launchParam1.GetParam();
    const Mki::Any &specificParam2 = launchParam2.GetParam();
    auto &opParamCompareMap = OpParamRegister::GetOpParamCompareMap();
    auto it = opParamCompareMap.find(specificParam1.Type().hash_code());
    if (it != opParamCompareMap.end()) {
        return it->second(specificParam1, specificParam2);
    } else {
        ATB_LOG(WARN) << "Can not compare param of " << specificParam1.Type().name();
        return false;
    }
}

// 注册OpParam的Map
OpParamRegister::OpParamRegister(size_t typeHashCode, ParamCompareFunc func) noexcept
{
    auto &opParamCompareMap = GetOpParamCompareMap();
    auto res = opParamCompareMap.emplace(typeHashCode, std::move(func));
    if (!res.second) {
        ATB_LOG(ERROR) << "Op param hash code: " << typeHashCode << " has been registered";
    }
}

std::map<std::size_t, ParamCompareFunc> &OpParamRegister::GetOpParamCompareMap()
{
    static std::map<std::size_t, ParamCompareFunc> opParamCompareMap;
    return opParamCompareMap;
}
} // namespace atb