/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_hccl_runner.h"
#include <hccl/hccl.h>
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include "atb/utils.h"
#include "atb/utils/common_utils.h"
#include "atb/utils/operation_register.h"

namespace atb {
AllGatherHcclRunner::AllGatherHcclRunner(const infer::AllGatherParam &param, bool useRankTableFile)
    : HcclRunner(!useRankTableFile ? HcclRunner("AllGatherHcclRunner", param.rank,
                                                param.rankSize, param.rankRoot, param.commDomain) :
                                     HcclRunner("AllGatherHcclRunner", param.rank,
                                                param.rankTableFile, param.commDomain)),
      param_(param)
{
    ATB_LOG(INFO) << "AllGatherHcclRunner::AllGatherHcclRunner called";
}

AllGatherHcclRunner::AllGatherHcclRunner(const infer::AllGatherParam &param, HcclComm hcclComm)
    : HcclRunner("AllGatherHcclRunner", hcclComm), param_(param)
{
    ATB_LOG(INFO) << "AllGatherHcclRunner::AllGatherHcclRunner ext called";
}

Status AllGatherHcclRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    if (!hcclComm_) {
        ATB_LOG(ERROR) << "hcclComm is null, rank: " << param_.rank;
        return ERROR_COMM_EMPTY;
    }

    if (!runnerVariantPack.inTensors[0].deviceData || !runnerVariantPack.outTensors[0].deviceData) {
        ATB_LOG(ERROR) << " device tensor is null";
        return ERROR_INVALID_PARAM;
    }
    HcclResult ret = HcclAllGather(
        runnerVariantPack.inTensors[0].deviceData, runnerVariantPack.outTensors[0].deviceData,
        Utils::GetTensorNumel(runnerVariantPack.inTensors[0]), GetHcclDtype(runnerVariantPack.inTensors[0].desc.dtype),
        hcclComm_.get(), GetExecuteStream(runnerVariantPack.context));
    if (ret != HCCL_SUCCESS) {
        ATB_LOG(ERROR) << "hccl Execute failed, HcclResult:" << ret;
        return ConvertHcclResultToStatus(ret);
    }
    return NO_ERROR;
}

AllGatherHcclRunner::~AllGatherHcclRunner() {}
REG_RUNNER_TYPE(AllGatherHcclRunner);
} // namespace atb