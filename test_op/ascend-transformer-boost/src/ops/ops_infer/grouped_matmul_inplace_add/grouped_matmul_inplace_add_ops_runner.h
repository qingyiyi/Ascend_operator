/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GROUPED_MATMUL_INPLACE_ADD_OPS_RUNNER_H
#define ATB_GROUPED_MATMUL_INPLACE_ADD_OPS_RUNNER_H
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class GroupedMatmulInplaceAddOpsRunner : public OpsRunner {
public:
    explicit GroupedMatmulInplaceAddOpsRunner(const infer::GroupedMatmulInplaceAddParam &param);
    ~GroupedMatmulInplaceAddOpsRunner() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    infer::GroupedMatmulInplaceAddParam param_;
};

namespace infer {
inline bool operator==(const GroupedMatmulInplaceAddParam &left, const GroupedMatmulInplaceAddParam &right)
{
    return left.transposeA == right.transposeA && left.transposeB == right.transposeB;
}
} // namespace infer
} // namespace atb
#endif // ATB_GROUPED_MATMUL_INPLACE_ADD_OPS_RUNNER_H