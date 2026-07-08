/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_COMM_POOL_H
#define ATB_COMM_POOL_H
#include <map>
#include <string>
#include <memory>
#include <mutex>
#include <functional>
#include "atb/utils/log.h"

namespace atb {
// 通信域池子
template <class Comm> class CommPool {
public:
    using CommSharedPtr = std::shared_ptr<Comm>;
    using CommCreateFunc = std::function<CommSharedPtr()>;

    CommPool() = default;
    ~CommPool() {}

    CommSharedPtr GetComm(const std::string &key, CommCreateFunc commCreateFunc)
    {
        ATB_LOG(INFO) << "GetComm Key: " << key;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = commMap_.find(key);
            if (it != commMap_.end()) {
                return it->second;
            }
        }

        CommSharedPtr newComm = commCreateFunc();
        if (!newComm) {
            ATB_LOG(ERROR) << "CommPool commCreateFunc fail";
            return newComm;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            commMap_[key] = newComm;
        }
        return newComm;
    }

private:
    mutable std::mutex mutex_;
    std::map<std::string, CommSharedPtr> commMap_;
};
} // namespace atb
#endif