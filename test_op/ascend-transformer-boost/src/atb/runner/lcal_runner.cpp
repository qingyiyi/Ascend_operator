/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/lcal_runner.h"
#include <algorithm>
#include <cctype>
#include <atb/utils/log.h>
#include <acl/acl.h>
#include "atb/utils/config.h"
#include "atb/utils/comm_pool.h"
#include "atb/utils/common_utils.h"
#include "atb/utils/singleton.h"
#include "atb/utils/operation_register.h"

namespace atb {
LcalRunner::LcalRunner(const std::string &name, int32_t rank, int32_t rankSize,
                       const infer::CommMode commMode, const std::string &commDomain, Context &context)
    : Runner(name), rank_(rank), rankSize_(rankSize), commMode_(commMode),
      commDomain_(commDomain)
{
    magicNumberDisabled_ = context.GetLaunchMode() == GRAPH_LAUNCH_MODE;
    runnerTypeIdx_ = RunnerTypeRegister::GetRunnerTypeIdx(name);
    ATB_LOG(INFO) << GetLogPrefix() << "LcalRunner::LcalRunner " << runnerTypeIdx_ << " called, rank : "
                  << rank << "/" << rankSize << " commMode: " << commMode_
                  << " commDomain: " << commDomain_ << " magicNumberDisabled: " << magicNumberDisabled_;
    InitLcalComm();
}

LcalRunner::~LcalRunner() {}

Lcal::LcalComm *LcalRunner::GetLcalComm()
{
    return lcalComm_.get();
}

void LcalRunner::InitLcalComm()
{
    lcalComm_ = GetSingleton<CommPool<Lcal::LcalComm>>().GetComm(std::to_string(rank_) + "_" + commDomain_,
                                                                 std::bind(&LcalRunner::CreateLcalComm, this));
    if (lcalComm_) {
        ATB_LOG(INFO) << GetLogPrefix() << "get lcal comm from comm pool success, rank : " << rank_;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "get lcal comm from comm pool failed, rank : " << rank_;
    }
}

std::pair<int32_t, int32_t> LcalRunner::ParseCommDomain(const std::string &commDomain) const
{
    constexpr int32_t defaultDomainSize = 200;
    constexpr int32_t maxDomainId = 65535;
    if (commDomain.empty()) {
        return {0, defaultDomainSize};
    }
    auto safeStoi = [](const std::string &str, int32_t defaultValue = -1) -> int32_t {
        try {
            return std::stoi(str);
        } catch (const std::invalid_argument &) {
            return defaultValue;
        } catch (const std::out_of_range &) {
            return defaultValue;
        }
    };
    size_t colonPos = commDomain.find(':');
    if (colonPos != std::string::npos) {
        std::string idStr = commDomain.substr(0, colonPos);
        std::string sizeStr = commDomain.substr(colonPos + 1);
        if (idStr.empty() || sizeStr.empty() ||
            !std::all_of(idStr.begin(), idStr.end(), ::isdigit) ||
            !std::all_of(sizeStr.begin(), sizeStr.end(), ::isdigit)) {
            ATB_LOG(ERROR) << "commDomain must contain numeric id and size";
            return {-1, -1};
        }
        int32_t id = safeStoi(idStr);
        int32_t size = safeStoi(sizeStr);
        if (id < 0 || id > maxDomainId || size <= 0) {
            ATB_LOG(ERROR) << "Invalid range: id should be 0-65535 and size > 0";
            return {-1, -1};
        }
        return {id, size};
    }

    // 兼容旧格式：只传编号
    if (!std::all_of(commDomain.begin(), commDomain.end(), ::isdigit)) {
        ATB_LOG(ERROR) << "commDomain must be a number or in id:size format, got: " << commDomain;
        return {-1, -1};
    }

    int32_t id = safeStoi(commDomain);
    if (id < 0 || id > maxDomainId) {
        ATB_LOG(ERROR) << "commDomain id is not in 0-65535, commDomain: " << commDomain;
        return {-1, -1};
    }
    return {id, defaultDomainSize};
}

std::shared_ptr<Lcal::LcalComm> LcalRunner::CreateLcalComm()
{
    LcalCommPtr comm = nullptr;
    if (commMode_ == infer::CommMode::COMM_MULTI_PROCESS) {
        auto commonId = ParseCommDomain(commDomain_);
        ATB_LOG(INFO) << GetLogPrefix() << "Lccl COMM_MULTI_PROCESS commDomain is : "
            << commonId.first << ":" << commonId.second;
        if (commonId.first == -1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Invalid commDomain: " << commonId.first;
            return std::shared_ptr<Lcal::LcalComm>();
        }
        lcalErrorCode_ = LcalCommInitRankWithCustDomainSize(commonId.first, commonId.second, rankSize_,
                                                            rank_, &comm, magicNumberDisabled_);
    } else if (commMode_ == infer::CommMode::COMM_MULTI_THREAD) {
        lcalErrorCode_ = LcalCommInitThread(rank_, rankSize_, commDomain_.c_str(), &comm);
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "Invalid commMode: " << commMode_;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    if (lcalErrorCode_ != Lcal::LCAL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "init LcalComm failed, lcalErrorCode : " << lcalErrorCode_;
        return std::shared_ptr<Lcal::LcalComm>();
    }
    return std::shared_ptr<Lcal::LcalComm>(static_cast<Lcal::LcalComm *>(comm));
}

Status LcalRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    (void)runnerVariantPack;
    if (lcalErrorCode_ == Lcal::LCAL_SUCCESS) {
        return NO_ERROR;
    }
    if (lcalErrorCode_ == Lcal::LCAL_ERROR_OUT_OF_MEMORY) {
        ATB_LOG(ERROR) << "error code:" << ERROR_OUT_OF_DEVICE_MEMORY
                       << ", out of NPU memory! Please check if the memory is enough.";
        return ConvertLcclResultToStatus(lcalErrorCode_);
    }
    return ERROR_RT_FAIL;
}

} // namespace atb
