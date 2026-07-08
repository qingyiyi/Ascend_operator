/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/plugin_runner.h"
#include "atb/utils/log.h"

namespace atb {
PluginRunner::PluginRunner(Operation *operation) : Runner("PluginRunner"), operation_(operation) {}

PluginRunner::~PluginRunner() {}

Status PluginRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    if (operation_) {
        variantPack_.inTensors = runnerVariantPack.inTensors;
        variantPack_.outTensors = runnerVariantPack.outTensors;
        return operation_->Setup(variantPack_, workspaceSize_, runnerVariantPack.context);
    }

    return ERROR_INVALID_PARAM;
}

uint64_t PluginRunner::GetWorkspaceBufferSizeImpl()
{
    return workspaceSize_;
}

Status PluginRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (operation_) {
        variantPack_.inTensors = runnerVariantPack.inTensors;
        variantPack_.outTensors = runnerVariantPack.outTensors;
        return operation_->Execute(variantPack_, runnerVariantPack.workspaceBuffer,
                                   runnerVariantPack.workspaceBufferSize, runnerVariantPack.context);
    }

    return ERROR_INVALID_PARAM;
}
} // namespace atb