/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_CONFIG_H
#define ATB_CONFIG_H
#include <string>
#include <mutex>

namespace atb {
// Environment variable of KernelCacheType. The default value is KERNEL_CACHE_BOTH.
enum KernelCacheType : uint32_t {
    KERNEL_NOT_CACHE = 0,
    KERNEL_CACHE_LOCAL,
    KERNEL_CACHE_GLOBAL,
    KERNEL_CACHE_BOTH,
};

class Config {
public:
    Config();
    ~Config();
    std::string GetAtbHomePath() const;
    bool IsStreamSyncEveryKernelEnable() const;
    bool IsStreamSyncEveryRunnerEnable() const;
    bool IsStreamSyncEveryOperationEnable() const;
    bool Is910B() const;
    bool Is310P() const;
    bool Is910A() const;
    bool Is310B() const;
    uint32_t GetWorkspaceMemAllocAlgType() const;
    uint32_t GetLocalKernelCacheCount() const;
    uint32_t GetGlobalKernelCacheCount() const;
    bool IsCompareTilingEveryKernelEnable() const;
    std::string GetShareMemoryNameSuffix() const;
    bool IsMatmulShuffleKEnable() const;

private:
    static bool IsEnable(const char *env, bool enable = false);
    void InitAtbHomePath();
    void InitWorkspaceSize();
    void InitSocVersion();
    void InitKernelCache();
    void InitVariable(const char *envName, uint32_t min, uint32_t max, uint32_t &value) const;
    void InitShareMemoryNameSuffix();

private:
    std::string atbHomePath_;
    bool isStreamSyncEveryKernelEnable_ = false;
    bool isStreamSyncEveryRunnerEnable_ = false;
    bool isStreamSyncEveryOperationEnable_ = false;
    bool is910B_ = false;
    bool is310P_ = false;
    bool is910A_ = false;
    bool is310B_ = false;
    uint32_t workspaceMemAllocAlgType_ = 1;
    uint32_t localKernelCacheCount_ = 1;
    uint32_t globalKernelCacheCount_ = 1;
    bool isCompareTilingEveryKernelEnable_ = false;
    std::string shareMemoryNameSuffix_;
    bool isMatmulShuffleKEnable_ = false;
};
} // namespace atb
#endif