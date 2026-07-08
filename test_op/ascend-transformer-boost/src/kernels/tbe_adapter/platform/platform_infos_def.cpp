/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform/platform_infos_def.h"
#include <mutex>
#include "mki/utils/dl/dl.h"
#include "mki/utils/env/env.h"
#include "mki/utils/log/log.h"
#include "mki/types.h"
#include "platform_infos_impl.h"

namespace {
using AclrtGetResInCurrentThreadFunc = int(*)(int, uint32_t*);

int GetResInCurrentThread(int type, uint32_t &resource)
{
    static std::once_flag onceFlag;
    static std::atomic<int> initFlag{Mki::ERROR_FUNC_NOT_INITIALIZED};  
    static std::unique_ptr<Mki::Dl> mkiDl; // 持久保存，避免库被卸载
    static AclrtGetResInCurrentThreadFunc aclFn = nullptr;

    std::call_once(onceFlag, []() {
        std::string p;
        const char *c = Mki::GetEnv("ASCEND_HOME_PATH");
        if (c) {
            p = std::string(c) + "/lib64/libascendcl.so";
        } else {
            p = "libascendcl.so";
        }
        auto dl = std::make_unique<Mki::Dl>(p, false);
        if (!dl->IsValid()) {
            MKI_LOG(ERROR) << "Try load libascendcl.so failed: " << p;
            initFlag.store(Mki::ERROR_FUNC_NOT_FOUND, std::memory_order_release);
            return;
        }
        auto sym = dl->GetSymbol("aclrtGetResInCurrentThread");
        if (sym == nullptr) {
            MKI_LOG(WARN) << "Symbol aclrtGetResInCurrentThread not found in: " << p;
            initFlag.store(Mki::ERROR_FUNC_NOT_FOUND, std::memory_order_release);
            return;
        }
        mkiDl = std::move(dl); // 保留句柄，防止卸载
        aclFn = reinterpret_cast<AclrtGetResInCurrentThreadFunc>(sym);
        initFlag.store(Mki::NO_ERROR, std::memory_order_release);
        MKI_LOG(INFO) << "Loaded libascendcl.so and resolved aclrtGetResInCurrentThread from: " << p;
    });

    // 初始化结果判定
    int rc = initFlag.load(std::memory_order_acquire);
    if (rc != Mki::NO_ERROR) {
        return rc;
    }

    if (type != 0 && type != 1) {
        MKI_LOG(ERROR) << "aclrtGetResInCurrentThread not support resource type: " << type;
        return Mki::ERROR_INVALID_VALUE;
    }

    // 调用前检查函数指针有效性
    if (aclFn == nullptr) {
        MKI_LOG(ERROR) << "aclrtGetResInCurrentThread function pointer is null.";
        return Mki::ERROR_FUNC_NOT_FOUND;
    }

    // 调用底层函数
    const int ret = aclFn(type, &resource);
    if (ret != 0) {
        MKI_LOG(ERROR) << "aclrtGetResInCurrentThread failed. type: " << type << " err: " << ret;
        return Mki::ERROR_RUN_TIME_ERROR;
    }

    MKI_LOG(INFO) << "Got resource in current thread. type: " << type << " resource: " << resource;
    return Mki::NO_ERROR;
}
}

namespace fe {
constexpr uint32_t MAX_CORE_NUM = 128;
std::mutex g_asdopsFePlatMutex;

bool PlatFormInfos::Init()
{
    platform_infos_impl_ = std::make_shared<PlatFormInfosImpl>();
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return true;
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val)
{
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, key, val);
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, std::map<std::string, std::string> &res)
{
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, res);
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetAICoreIntrinsicDtype()
{
    if (platform_infos_impl_ == nullptr) {
        return {};
    }
    return platform_infos_impl_->GetAICoreIntrinsicDtype();
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetVectorCoreIntrinsicDtype()
{
    if (platform_infos_impl_ == nullptr) {
        return {};
    }
    return platform_infos_impl_->GetVectorCoreIntrinsicDtype();
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, key, val);
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return platform_infos_impl_->GetPlatformRes(label, res);
}

void PlatFormInfos::SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetAICoreIntrinsicDtype(intrinsicDtypes);
}

void PlatFormInfos::SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetVectorCoreIntrinsicDtype(intrinsicDtypes);
}

void PlatFormInfos::SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetFixPipeDtypeMap(fixpipeDtypeMap);
}

void PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type)
{
    uint32_t coreNum = 0;
    int8_t resType = core_type == "VectorCore" ? 1 : 0;
    int getResRet = GetResInCurrentThread(resType, coreNum);

    if (getResRet == Mki::NO_ERROR) {
        core_num_ = coreNum;
        if (core_num_ == 0 || core_num_ > MAX_CORE_NUM) {
            MKI_LOG(ERROR) << "core_num is out of range : " << core_num_;
            core_num_ = 1;
        }
        return;
    }

    std::string coreNumStr = "";
    std::string coreTypeStr = "";
    if (core_type == "VectorCore") {
        coreTypeStr = "vector_core_cnt";
    } else {
        coreTypeStr = "ai_core_cnt";
    }
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    (void)GetPlatformRes("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Set PlatFormInfos::core_num_ to " << coreTypeStr << ": " << coreNumStr;
    if (coreNumStr.empty()) {
        core_num_ = 1;
        MKI_LOG(ERROR) << "CoreNumStr is empty!";
    } else {
        core_num_ = std::strtoul(coreNumStr.c_str(), nullptr, 10); // 10 进制
    }
    if (core_num_ == 0 || core_num_ > MAX_CORE_NUM) {
        MKI_LOG(ERROR) << "core_num is out of range : " << core_num_;
        core_num_ = 1;
    }
}

uint32_t PlatFormInfos::GetCoreNumByType(const std::string &core_type)
{
    uint32_t coreNum = 0;
    int8_t resType = core_type == "VectorCore" ? 1 : 0;
    int getResRet = GetResInCurrentThread(resType, coreNum);
    
    if (getResRet == Mki::NO_ERROR) {
        if (coreNum > MAX_CORE_NUM) {
            MKI_LOG(ERROR) << "core_num is out of range : " << coreNum;
            return 1;
        }
        return coreNum;
    }

    std::string coreNumStr = "";
    std::string coreTypeStr = core_type == "VectorCore" ? "vector_core_cnt" : "ai_core_cnt";
    std::lock_guard<std::mutex> lockGuard(g_asdopsFePlatMutex);
    (void)GetPlatformRes("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Get PlatFormInfos::core_num_ to " << coreTypeStr << ": " << coreNumStr;
    if (coreNumStr.empty()) {
        MKI_LOG(ERROR) << "CoreNumStr is empty!";
        return 1;
    } else {
        coreNum = std::strtoul(coreNumStr.c_str(), nullptr, 10); // 10 进制
    }
    if (coreNum > MAX_CORE_NUM) {
        MKI_LOG(ERROR) << "core_num is out of range : " << coreNum;
        return 1;
    }
    return coreNum;
}

void PlatFormInfos::SetCoreNum(const uint32_t &coreNum)
{
    MKI_LOG(DEBUG) << "Set PlatFormInfos::core_num_: " << coreNum;
    core_num_ = coreNum;
}

uint32_t PlatFormInfos::GetCoreNum() const
{
    MKI_LOG(DEBUG) << "Get PlatFormInfos::core_num_: " << core_num_;
    return core_num_;
}

void PlatFormInfos::GetLocalMemSize(const LocalMemType &memType, uint64_t &size)
{
    std::string sizeStr;
    switch (memType) {
        case LocalMemType::L0_A: {
            (void)GetPlatformRes("AICoreSpec", "l0_a_size", sizeStr);
            break;
        }
        case LocalMemType::L0_B: {
            (void)GetPlatformRes("AICoreSpec", "l0_b_size", sizeStr);
            break;
        }
        case LocalMemType::L0_C: {
            (void)GetPlatformRes("AICoreSpec", "l0_c_size", sizeStr);
            break;
        }
        case LocalMemType::L1: {
            (void)GetPlatformRes("AICoreSpec", "l1_size", sizeStr);
            break;
        }
        case LocalMemType::L2: {
            (void)GetPlatformRes("SoCInfo", "l2_size", sizeStr);
            break;
        }
        case LocalMemType::UB: {
            (void)GetPlatformRes("AICoreSpec", "ub_size", sizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformRes("SoCInfo", "memory_size", sizeStr);
            break;
        }
        default: {
            break;
        }
    }

    if (sizeStr.empty()) {
        size = 0;
    } else {
        try {
            size = static_cast<uint64_t>(std::stoll(sizeStr.c_str()));
        } catch (const std::invalid_argument &e) {
            size = 0;
        } catch (const std::out_of_range &e) {
            size = 0;
        }
    }
}

void PlatFormInfos::GetLocalMemBw(const LocalMemType &memType, uint64_t &bwSize)
{
    std::string bwSizeStr;
    switch (memType) {
        case LocalMemType::L2: {
            (void)GetPlatformRes("AICoreMemoryRates", "l2_rate", bwSizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformRes("AICoreMemoryRates", "ddr_rate", bwSizeStr);
            break;
        }
        default: {
            break;
        }
    }

    if (bwSizeStr.empty()) {
        bwSize = 0;
    } else {
        try {
            bwSize = static_cast<uint64_t>(std::stoll(bwSizeStr.c_str()));
        } catch (const std::invalid_argument &e) {
            bwSize = 0;
        } catch (const std::out_of_range &e) {
            bwSize = 0;
        }
    }
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetFixPipeDtypeMap()
{
    if (platform_infos_impl_ == nullptr) {
        return {};
    }
    return platform_infos_impl_->GetFixPipeDtypeMap();
}

void PlatFormInfos::SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    if (platform_infos_impl_ == nullptr) {
        return;
    }
    platform_infos_impl_->SetPlatformRes(label, res);
}
} // namespace fe
