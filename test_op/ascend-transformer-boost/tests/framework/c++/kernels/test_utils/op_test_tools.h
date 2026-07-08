/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MKIOPTESTS_OPTEST_TOOLS_H
#define MKIOPTESTS_OPTEST_TOOLS_H

#include <ATen/ATen.h>
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <mki/utils/log/log.h>
#include <mki/utils/status/status.h>
#include <mki/utils/SVector/SVector.h>
#include "float_util.h"
#include "op_test.h"

namespace Mki {
namespace Test {
float GetMare(float *actual, float *golden, uint32_t dataLen);

float GetMere(float *actual, float *golden, uint32_t dataLen);

float GetRmse(float *actual, float *golden, uint32_t dataLen);

float GetEb(float *actual, float *golden, uint32_t dataLen);

Status MatchAtTensor(at::Tensor &out, at::Tensor &gt, at::Tensor &gt_high, std::string &dtype);

void PrepareScalarParams(std::vector<uint32_t> &scalars, bool &dynamicBatch, 
                         MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir);

void PrepareScalarParams(std::vector<uint32_t> &scalars, MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir);

void PrepareVectors(std::vector<int32_t> &q_seqlen, std::vector<int32_t> &kv_seqlen, 
                    std::vector<int32_t> &batch_state, MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir);

void PrepareVectors(std::vector<int32_t> &q_seqlen, std::vector<int32_t> &kv_seqlen, 
                    MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir);

void PrepareExpect(std::vector<uint16_t> &expect, std::vector<uint32_t> &expect_high, 
                   MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir);

void PrepareKvFiles(uint32_t batch, std::vector<std::string> &kvFiles, 
                    MkiOpTest &opTest, const std::string &atbopsHomeDir, const std::string &subDir, bool isRelay = false, uint32_t share_len = 0);

SVector<std::string> GetInDataFiles(const std::string &atbopsHomeDir, const std::string &subDir, bool isRelay = false);
} // namespace Test
} // namespace Mki

#endif