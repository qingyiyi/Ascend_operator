/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_RESHAPE_AND_CACHE_OPS_RUNNER_SISO_H
#define ATB_RESHAPE_AND_CACHE_OPS_RUNNER_SISO_H

#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class ReshapeAndCacheOpsRunnerSISO : public OpsRunner {
public:
    explicit ReshapeAndCacheOpsRunnerSISO(const infer::ReshapeAndCacheParam &param);
    ~ReshapeAndCacheOpsRunnerSISO() override;

private:
    infer::ReshapeAndCacheParam param_;
};
} // namespace atb
#endif