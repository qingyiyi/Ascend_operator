/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_DEFAULT_DEVICE_ALLOCATOR_H
#define ATB_DEFAULT_DEVICE_ALLOCATOR_H
#include <cstdint>
#include <map>
#include "atb/context/allocator/allocator.h"

namespace atb {
class DefaultDeviceAllocator : public Allocator {
public:
    DefaultDeviceAllocator();
    ~DefaultDeviceAllocator() override;
    void *Allocate(size_t bufferSize) override;
    Status Deallocate(void *addr) override;
private:
    std::map<void*, size_t> memMap;
    size_t currentAllocateSize_ = 0;
};
} // namespace atb
#endif