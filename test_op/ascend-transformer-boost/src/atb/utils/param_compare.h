/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_PARAM_COMPARE_H
#define ATB_PARAM_COMPARE_H
#include <functional>
#include <map>
#include <mki/launch_param.h>
#include "atb/utils/log.h"

namespace atb {
bool IsLaunchParamEqual(const Mki::LaunchParam &launchParam1, const Mki::LaunchParam &launchParam2);
using ParamCompareFunc = std::function<bool(const Mki::Any &, const Mki::Any &)>;

template <typename T> bool ParamCompareFuncImpl(const Mki::Any &any1, const Mki::Any &any2)
{
    if (any1.Type() != any2.Type()) {
        ATB_LOG(WARN) << "param1: " << any1.Type().name() << " and param2: " << any2.Type().name()
                      << " can not be compared";
        return false;
    }
    const auto &content1 = Mki::AnyCast<T>(any1);
    const auto &content2 = Mki::AnyCast<T>(any2);
    return content1 == content2;
}

class OpParamRegister {
public:
    OpParamRegister(size_t typeHashCode, ParamCompareFunc func) noexcept;
    static std::map<std::size_t, ParamCompareFunc> &GetOpParamCompareMap();
};
}

#define CONCAT(a, b) a##b
#define CONCAT2(a, b) CONCAT (a, b)
#define UNIQUE_NAME(base) CONCAT2(base, __COUNTER__)
#define REG_OP_PARAM(typeName) \
    static atb::OpParamRegister UNIQUE_NAME(opParamRegister)(typeid(typeName).hash_code(), atb::ParamCompareFuncImpl<typeName>)
#endif