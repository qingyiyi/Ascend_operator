#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import random
import sys
import unittest
 
import numpy as np
import torch
import torch_npu
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
 
MAX_SEQ_LEN = 1024
 
 
class ReshapeAndCacheDataGenerator():

    def set_reshape_and_cache_param(self, num_heads, head_size, block_size, num_blocks, dtype, batch):
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_blocks = num_blocks*num_heads
        self.dtype = dtype
        self.batch = batch
        self.seqLen = np.random.randint(1, 20, size=(self.batch)).astype(np.int32)
        self.num_tokens = np.sum(self.seqLen)

        if self.dtype == "bfloat16":
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
        else:
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
            # self.key = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
            # self.value = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
        self.wins = 2*np.ones((self.num_heads * self.batch)).astype(np.int32)
        value = []
        sumX = 0
        for i in range(self.batch*self.num_heads):
            x = (self.wins[i] + self.block_size - 1) // self.block_size
            value.append(sumX * self.block_size)
            sumX += x

        self.slot_mapping = np.array(value).astype(np.int32)

        
        self.key_cache = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)
        self.value_cache = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)

    def generate_test_data(self):
        key_expect = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)
        value_expect = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)
        self.new_seq = self.seqLen.copy()
        self.new_seq[0] = self.seqLen[0]

        for n in range(1, len(self.seqLen)):
            self.new_seq[n] = self.seqLen[n] + self.new_seq[n-1]

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            curSlot = slot
            win = self.wins[i]
            for j in range(win):
                block_index = curSlot // self.block_size
                block_offset = curSlot % self.block_size

                curBatch = i // self.num_heads

                bsID = self.new_seq[curBatch] - win + j
                headID = i % self.num_heads

                token_key = self.key[bsID][headID]
                token_v = self.value[bsID][headID]
                key_expect[block_index][block_offset] = token_key
                value_expect[block_index][block_offset] = token_v
                curSlot += 1

        ret_data = self.key, self.value, self.key_cache, self.value_cache, self.slot_mapping, self.wins, self.seqLen,\
            key_expect, value_expect
        return ret_data

    def golden_compare(self, out_tensor, golden_out_tensor):
        expect = golden_out_tensor
        actual = out_tensor
        abs_diff = torch.abs(expect-actual)
        print((abs_diff > 0).nonzero().size(0))
        return (not (abs_diff > 0).nonzero().size(0) > 0)

    def test_reshape_and_cache_compress_head(self):
        batch = 3
        seq_len = 10
        num_heads = 2
        head_size = 128
        block_size = 128
        num_blocks = 8
        dtype = "int8"
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch) 

 
OP_NAME = "ReshapeAndCacheOperation"
PARAM = json.dumps({"compressType": 1})
 
data_generator = ReshapeAndCacheDataGenerator()
data_generator.test_reshape_and_cache_compress_head()
data = data_generator.generate_test_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

 
class TestReshapeAndCacheOperationCompress(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors[7], in_tensors[8]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return data_generator.golden_compare(out_tensor, golden_out_tensor)
        # return True
 
    def test(self):
        if not TestReshapeAndCacheOperationCompress.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2],\
                in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6]],
                [in_tensors[2], in_tensors[3]])
 
 
if __name__ == '__main__':
    unittest.main()