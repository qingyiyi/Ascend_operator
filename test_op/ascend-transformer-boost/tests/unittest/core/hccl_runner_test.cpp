/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "atb/operation.h"
#include "atb/utils.h"
#include "atb/runner/runner.h"
#include "atb/runner/hccl_runner.h"
#include "atb/utils/runner_variant_pack.h"
#include "test_utils/test_utils.h"

using namespace atb;

TEST(TestHcclRunner, ExecuteFailRunnerTypeInvalid)
{
    Runner *runner = new HcclRunner("TestHcclRunner", 0, 1, 0);
    RunnerVariantPack variantPack;
    variantPack.inTensors.resize(1);
    variantPack.inTensors.at(0).desc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 1, 2 }, 2 } };
    variantPack.inTensors.at(0).dataSize = atb::Utils::GetTensorSize(variantPack.inTensors.at(0));
    variantPack.inTensors.at(0).deviceData = malloc(variantPack.inTensors.at(0).dataSize);
    atb::Status st = runner->PreExecute(variantPack);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    delete runner;
}

TEST(TestHcclRunner, ExecuteFailDeviceDataIsNull)
{
    Runner *runner = new HcclRunner("TestHcclRunner", 0, 1, 0);
    RunnerVariantPack variantPack;
    variantPack.inTensors.resize(1);
    variantPack.inTensors.at(0).desc = { ACL_FLOAT16, ACL_FORMAT_ND, { { 1, 2 }, 2 } };
    variantPack.inTensors.at(0).dataSize = atb::Utils::GetTensorSize(variantPack.inTensors.at(0));
    variantPack.inTensors.at(0).deviceData = nullptr;
    atb::Status st = runner->PreExecute(variantPack);
    EXPECT_EQ(st, atb::ERROR_INVALID_PARAM);
    delete runner;
}