/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ALL_GATHER_HCCL_RUNNER_H
#define ATB_ALL_GATHER_HCCL_RUNNER_H
#include "atb/runner/hccl_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class AllGatherHcclRunner : public HcclRunner {
public:
    explicit AllGatherHcclRunner(const infer::AllGatherParam &param);
    AllGatherHcclRunner(const infer::AllGatherParam &param, bool useRankTableFile);
    AllGatherHcclRunner(const infer::AllGatherParam &param, HcclComm hcclComm);
    ~AllGatherHcclRunner() override;

protected:
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

private:
    infer::AllGatherParam param_;
};
} // namespace atb

#endif // ATB_ALL_GATHER_HCCL_RUNNER_H
