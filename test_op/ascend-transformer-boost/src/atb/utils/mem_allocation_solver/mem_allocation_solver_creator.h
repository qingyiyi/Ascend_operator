/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDTRANSFORM_MEM_ALLOCATION_SOLVER_CREATOR_H
#define ASDTRANSFORM_MEM_ALLOCATION_SOLVER_CREATOR_H
#include <memory>
#include "bruteforce_mem_allocation_solver.h"
#include "block_mem_allocation_solver.h"
#include "heap_mem_allocation_solver.h"
#include "noblock_mem_allocation_solver.h"

namespace atb {
std::shared_ptr<MemAllocationSolver> GetGlobalMemAllocationSolver();

std::shared_ptr<MemAllocationSolver> CreateMemAllocationSolver();
} // namespace atb
#endif