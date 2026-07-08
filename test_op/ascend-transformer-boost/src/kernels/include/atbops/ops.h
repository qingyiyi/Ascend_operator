/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_LOADER_OPS_H
#define ATBOPS_LOADER_OPS_H
#include <vector>
#include <string>
#include <memory>
#include "mki/kernel.h"
#include "mki/operation.h"
#include "mki_loader/op_schedule_base.h"

namespace AtbOps {
class OpScheduleBase;

class Ops {
public:
    /**
     * @brief Return the singleton object
     *
     * @return Ops&
     */
    static Ops &Instance();
    /**
     * @brief Get the All Operations object
     *
     * @return std::vector<Mki::Operation *>
     */
    std::vector<Mki::Operation *> GetAllOperations() const;
    /**
     * @brief Get the Operation By Name object
     *
     * @param[const std::string&] opName
     * @return Mki::Operation*
     */
    Mki::Operation *GetOperationByName(const std::string &opName) const;
    /**
     * @brief Get the Kernel Instance By Name
     *
     * @param[const std::string&] opName
     * @return Mki::Kernel*
     */
    Mki::Kernel *GetKernelInstance(const std::string &kernelName) const;
    /**
     * @brief update schedule for dynamic BishengIR bin
     *
     */
    void UpdateSchedule();

private:
    Ops();
    ~Ops();

private:
    std::unique_ptr<Mki::OpScheduleBase> opSchedule_;
};
} // namespace AtbOps

#endif
