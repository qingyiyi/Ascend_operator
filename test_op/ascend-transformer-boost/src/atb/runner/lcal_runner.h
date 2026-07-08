/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LCAL_RUNNER_H
#define ATB_LCAL_RUNNER_H

#include <memory>
#include <utility>
#include "lccl.h"
#include "lcoc.h"
#include "atb/runner/runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class LcalRunner : public Runner {
public:
    explicit LcalRunner(const std::string &name, int32_t rank, int32_t rankSize,
                        const infer::CommMode commMode, const std::string &commDomain, Context &context);
    ~LcalRunner() override;

protected:
    Lcal::LcalComm *GetLcalComm();
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;

protected:
    int64_t runnerTypeIdx_ = -1;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    infer::CommMode commMode_ = infer::CommMode::COMM_MULTI_PROCESS;
    std::string commDomain_;

private:
    void InitLcalComm();
    std::pair<int32_t, int32_t> ParseCommDomain(const std::string &commDomain) const;
    std::shared_ptr<Lcal::LcalComm> CreateLcalComm();

private:
    std::shared_ptr<Lcal::LcalComm> lcalComm_;
    int32_t lcalErrorCode_ = 0;
    bool magicNumberDisabled_ = false;
};
} // namespace atb
#endif
