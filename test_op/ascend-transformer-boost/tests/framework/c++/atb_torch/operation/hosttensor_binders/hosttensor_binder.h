/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATB_HOSTTENSOR_BINDER_H
#define ATB_HOSTTENSOR_BINDER_H
#include <nlohmann/json.hpp>
#include <atb/atb_infer.h>

class HostTensorBinder {
public:
    HostTensorBinder() = default;
    virtual ~HostTensorBinder() = default;
    virtual void ParseParam(const nlohmann::json &paramJson) = 0;
    virtual void BindTensor(atb::VariantPack &variantPack) = 0;
    virtual void SetTensorList(const std::vector<atb::Tensor> &tensorList, const std::string &tensorListName);
};

#endif