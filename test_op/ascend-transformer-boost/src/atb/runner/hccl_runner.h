/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_HCCL_RUNNER_H
#define ATB_HCCL_RUNNER_H
#include <hccl/hccl_types.h>
#include <mki/utils/share_memory/share_memory.h>
#include "atb/runner/runner.h"
#include "atb/infer_op_params.h"

namespace atb {
struct CommInitInfo {
    int signal = 0;
    HcclRootInfo hcclRootInfo = {};
    bool barrier[1]; // Flexible array member
};

using HcclCommSharedPtr = std::shared_ptr<void>;

class HcclRunner : public Runner {
public:
    explicit HcclRunner(const std::string &name, int rank = 0,
                        int rankSize = 0, int rankRoot = 0, const std::string &commDomain = "");
    explicit HcclRunner(const std::string &name, int rank = 0,
                        const std::string &rankTableFile = "", const std::string &commDomain = "");
    HcclRunner(const std::string &name, HcclComm hcclComm);
    ~HcclRunner() override;
    HcclCommSharedPtr GetHcclCommSharedPtr() const;

protected:
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

protected:
    int64_t runnerTypeIdx_ = -1;
    int rank_ = 0;
    int rankSize_ = 0;
    int rankRoot_ = 0;
    std::string allReduceType_ = "sum";
    HcclCommSharedPtr hcclComm_;
    HcclRootInfo hcclRootInfo_ = {};
    std::string rankTableFile_;
    bool useRankTableFile_ = false;
    std::string commDomain_;

private:
    void Init();
    HcclCommSharedPtr CreateHcclComm();
    HcclCommSharedPtr CreateHcclCommInMulitProcess();
    HcclCommSharedPtr CreateHcclCommInMulitProcessByRootInfo();
    HcclCommSharedPtr CreateHcclCommInMulitProcessByRankFile() const;
    bool CreateHcclRootInfo();
    void ShmGetHcclRootInfo(Mki::ShareMemory &shm, const CommInitInfo &shmInfo);
    void ShmSetHcclRootInfo(Mki::ShareMemory &shm, CommInitInfo &shmInfo);
    bool ShmBarrier(Mki::ShareMemory &shm, CommInitInfo &shmInfo);
    void ShmSetReady(Mki::ShareMemory &shm, CommInitInfo &shmInfo) const;
    HcclResult HcclExecute(RunnerVariantPack &runnerVariantPack);
};
} // namespace atb
#endif
