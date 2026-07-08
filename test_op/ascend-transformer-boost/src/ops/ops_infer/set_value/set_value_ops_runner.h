/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SET_VALUE_OPS_RUNNER_H
#define ATB_SET_VALUE_OPS_RUNNER_H
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class SetValueOpsRunner : public OpsRunner {
public:
    explicit SetValueOpsRunner(const infer::SetValueParam &param);
    ~SetValueOpsRunner() override;

private:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::SetValueParam param_;
};
} // namespace atb
#endif