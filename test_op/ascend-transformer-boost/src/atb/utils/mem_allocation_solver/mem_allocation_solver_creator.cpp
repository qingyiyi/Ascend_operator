/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/mem_allocation_solver/mem_allocation_solver_creator.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"

namespace atb {

thread_local std::shared_ptr<MemAllocationSolver> g_memAllocationSolver;

std::shared_ptr<MemAllocationSolver> GetGlobalMemAllocationSolver()
{
    if (g_memAllocationSolver) {
        return g_memAllocationSolver;
    } else {
        g_memAllocationSolver = CreateMemAllocationSolver();
        return g_memAllocationSolver;
    }
}

std::shared_ptr<MemAllocationSolver> CreateMemAllocationSolver()
{
    uint32_t allocAlgType = GetSingleton<Config>().GetWorkspaceMemAllocAlgType();
    if (allocAlgType == 0) {
        return std::make_shared<BruteforceMemAllocationSolver>();
    } else if (allocAlgType == 1) {
        return std::make_shared<BlockMemAllocationSolver>();
    } else if (allocAlgType == 2) { // 2: 有序bestFit block算法
        return std::make_shared<HeapMemAllocationSolver>();
    } else {
        return std::make_shared<NoblockMemAllocationSolver>();
    }
}

} // namespace atb