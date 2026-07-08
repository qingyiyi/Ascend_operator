/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ALL_TO_ALLVV2_HCCL_RUNNER_H
#define ATB_ALL_TO_ALLVV2_HCCL_RUNNER_H
#include "atb/runner/hccl_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class AllToAllVV2HcclRunner : public HcclRunner {
public:
    explicit AllToAllVV2HcclRunner(const infer::AllToAllVV2Param &param);
    AllToAllVV2HcclRunner(const infer::AllToAllVV2Param &param, bool useRankTableFile);
    AllToAllVV2HcclRunner(const infer::AllToAllVV2Param &param, HcclComm hcclComm);
    ~AllToAllVV2HcclRunner() override;

protected:
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

private:
    infer::AllToAllVV2Param param_;
};
} // namespace atb
#endif // ATB_NONZERO_OPS_RUNNER_H
