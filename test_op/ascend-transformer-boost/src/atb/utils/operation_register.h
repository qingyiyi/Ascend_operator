/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #ifndef ATB_OPERATION_REGISTER_H
 #define ATB_OPERATION_REGISTER_H
 #include <map>
 #include <string>

namespace atb {
class RunnerTypeRegister {
public:
    RunnerTypeRegister(const char* runnerType) noexcept
    {
        ATB_CHECK(runnerType != nullptr && runnerType[0] != '\0', "Invalid runnerType provided", return);
        auto &runnerTypeMap = GetRunnerTypeMap();
        auto res = runnerTypeMap.emplace(runnerType, GetRunnerIdx()++);
        if (!res.second) {
            ATB_LOG(ERROR) << "RunnerType: " << runnerType << " has been registered";
            GetRunnerIdx()--;
        }
    }

    static std::map<std::string, int64_t> &GetRunnerTypeMap()
    {
        static std::map<std::string, int64_t> runnerTypeMap;
        return runnerTypeMap;
    }

    static int64_t &GetRunnerIdx()
    {
        static int64_t runnerIdx = 0;
        return runnerIdx;
    }

    static int64_t GetRunnerTypeIdx(const std::string &runnerType)
    {
        if (runnerType.empty()) {
            ATB_LOG(ERROR) << "Invalid runnerType provided";
            return -1;
        }
        auto &runnerTYpeMap = GetRunnerTypeMap();
        auto it = runnerTYpeMap.find(runnerType);
        if (it == runnerTYpeMap.end()) {
            ATB_LOG(ERROR) << "Can not find the runnerTypeIdx by runner name: " << runnerType;
            return -1;
        }
        return it->second;
    }

    static size_t GetRunnerTypeMapSize()
    {
        auto &runnerTypeMap = GetRunnerTypeMap();
        return runnerTypeMap.size();
    }
};
} // namespace atb
#define REG_RUNNER_TYPE(runnerType) \
    static atb::RunnerTypeRegister runnerType##RunnerTypeRegister(#runnerType)
#endif