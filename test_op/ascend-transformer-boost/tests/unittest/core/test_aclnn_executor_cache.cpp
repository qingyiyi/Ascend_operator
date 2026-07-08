/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <acl/acl.h>
#include "atb/kernel_cache/aclnn_executor_cache.h"
#include "atb/utils.h"
#include "atb/utils/runner_variant_pack.h"
#include "atb/types.h"


// 设置各个inTensor的属性
static void CreateTensorDescs(atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::Dims> dimsVec)
{
    inTensorDescs.resize(dimsVec.size());
    for (size_t i = 0; i < inTensorDescs.size(); ++i) {
        inTensorDescs.at(i).dtype = ACL_FLOAT16;
        inTensorDescs.at(i).format = ACL_FORMAT_ND;
        inTensorDescs.at(i).shape.dimNum = dimsVec[i].dimNum;
        for (size_t j = 0; j < dimsVec[i].dimNum; ++j) {
            inTensorDescs.at(i).shape.dims[j] = dimsVec[i].dims[j];
        }
    }
}

// 设置各个inTensor并且为各个inTensor分配内存空间，此处的inTensor为手动设置，工程实现上可以使用torchTensor转换或者其他简单数据结构转换的方式
static void CreateInTensors(atb::SVector<atb::Tensor> &inTensors, atb::SVector<atb::TensorDesc> &inTensorDescs,
                            uint32_t value)
{
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensors.at(i).desc = inTensorDescs.at(i);
        inTensors.at(i).dataSize = atb::Utils::GetTensorSize(inTensors.at(i));
        std::vector<uint16_t> hostData(atb::Utils::GetTensorNumel(inTensors.at(i)), value); // 一段全value的hostBuffer
        int ret = aclrtMalloc(&inTensors.at(i).deviceData, inTensors.at(i).dataSize,
                              ACL_MEM_MALLOC_HUGE_FIRST); // 分配NPU内存
        ASSERT_EQ(ret, 0);
        ret = aclrtMemcpy(inTensors.at(i).deviceData, inTensors.at(i).dataSize, hostData.data(),
                          hostData.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE); // 拷贝CPU内存到NPU侧
        ASSERT_EQ(ret, 0);
    }
}

// 设置各个outTensor并且为outTensor分配内存空间，同inTensor设置
static void CreateOutTensors(atb::SVector<atb::Tensor> &outTensors, atb::SVector<atb::TensorDesc> &outTensorDescs)
{
    for (size_t i = 0; i < outTensors.size(); i++) {
        outTensors.at(i).desc = outTensorDescs.at(i);
        outTensors.at(i).dataSize = atb::Utils::GetTensorSize(outTensors.at(i));
        int ret = aclrtMalloc(&outTensors.at(i).deviceData, outTensors.at(i).dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, 0);
    }
}

static void TestInit(atb::Context *&context, aclrtStream &stream)
{
    // 前置操作
    uint32_t deviceId = 0;
    ASSERT_EQ(aclrtSetDevice(deviceId), 0);
    ASSERT_EQ(aclrtCreateStream(&stream), 0);
    ASSERT_EQ(atb::CreateContext(&context), 0);
    context->SetExecuteStream(stream);
}

static void TestFinalize(atb::Context *&context, aclrtStream &stream)
{
    // 释放stream和context
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtDestroyStream(stream), 0);
    ASSERT_EQ(DestroyContext(context), 0);
}

static void CreateRunnerVariantPack(atb::RunnerVariantPack &pack, atb::SVector<atb::Dims> inDimsVec,
                                    atb::SVector<atb::Dims> outDimsVec)
{
    // 创建一个仅包含in/out Tensor的runnerVariantPack
    atb::SVector<atb::TensorDesc> inTensorDescs;
    CreateTensorDescs(inTensorDescs, inDimsVec);
    pack.inTensors.resize(inDimsVec.size());
    CreateInTensors(pack.inTensors, inTensorDescs, 0);
    atb::SVector<atb::TensorDesc> outTensorDescs;
    CreateTensorDescs(outTensorDescs, outDimsVec);
    pack.inTensors.resize(outDimsVec.size());
    CreateInTensors(pack.inTensors, outTensorDescs, 0);
}

TEST(TestAclnnExecutorCache, TestFetchAdd)
{
    aclrtStream stream;
    atb::Context *context = nullptr;
    TestInit(context, stream);

    atb::AclnnExecutorCache cache = atb::AclnnExecutorCache();
    std::string opName = "ElewiseOperation";

    aclOpExecutor *raw_ptr = reinterpret_cast<aclOpExecutor *>(0x123);
    atb::AclnnCacheSlot cacheSlot = {10, std::shared_ptr<aclOpExecutor>(raw_ptr, [](aclOpExecutor *) {})};
    atb::RunnerVariantPack cacheKey;
    atb::SVector<atb::Dims> dimsVec = {{{2, 3}, 2}};
    CreateRunnerVariantPack(cacheKey, dimsVec, dimsVec);
    
    atb::Status stat = cache.FetchCacheSlot(opName, cacheKey, cacheSlot);
    ASSERT_EQ(stat, atb::ERROR_INVALID_PARAM) << "Expect ERROR_INVALID_PARAM fetching on empty cache, but got " << stat;
    stat = cache.AddCacheSlot(opName, cacheKey, cacheSlot);
    ASSERT_EQ(stat, atb::NO_ERROR) << "Expect NO_ERROR adding a key, but got " << stat;
    stat = cache.FetchCacheSlot(opName, cacheKey, cacheSlot);
    ASSERT_EQ(stat, atb::NO_ERROR) << "Expect NO_ERROR fetching after adding, but got " << stat;
    TestFinalize(context, stream);
}
