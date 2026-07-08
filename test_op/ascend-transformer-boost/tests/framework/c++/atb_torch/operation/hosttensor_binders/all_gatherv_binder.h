/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ALLGATHERVBINDER_H
#define ALLGATHERVBINDER_H
#include "hosttensor_binder.h"
#include <iostream>
#include <vector>

class AllGatherVBinder : public HostTensorBinder {
public:
    AllGatherVBinder();
    virtual ~AllGatherVBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    int64_t sendCount;
    std::vector<int64_t> recvCounts;
    std::vector<int64_t> rdispls;
};

#endif