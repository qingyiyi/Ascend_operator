#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import logging
import json
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "RopeOperation"

class TestRopeOperation2(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        seqlen = in_tensors[0].size()[1]
        batch = in_tensors[0].size()[0]
        q_head_num = in_tensors[0].size()[2]
        k_head_num = in_tensors[1].size()[2]
        rot_dim = in_tensors[2].size()[1]

        q = in_tensors[0]
        k = in_tensors[1]
        qshaped = q.reshape(batch, -1, q_head_num, rot_dim // 2, 2)
        kshaped = k.reshape(batch, -1, k_head_num, rot_dim // 2, 2)
        cos = in_tensors[2].view(-1, 2)[:, 0].view(batch, -1, 1, qshaped.size(3))
        sin = in_tensors[3].view(-1, 2)[:, 0].view(batch, -1, 1, qshaped.size(3))

        q_out2 = torch.stack(
            [
                qshaped[..., 0] * cos - qshaped[..., 1] * sin,
                qshaped[..., 1] * cos + qshaped[..., 0] * sin,
            ],
            -1,
        )

        q_out2 = q_out2.flatten(3)
        k_out2 = torch.stack(
            [
                kshaped[..., 0] * cos - kshaped[..., 1] * sin,
                kshaped[..., 1] * cos + kshaped[..., 0] * sin,
            ],
            -1,
        )
        k_out2 = k_out2.flatten(3)

        return [q_out2, k_out2]

    def test_2d_half(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        ntoken = 512    
        seqlen = 256
        batch = 2
        q_head_num = 32
        k_head_num = 2
        head_size = 128
        # head_dim / 2时，前半部分旋转，后半部分不做
        intensor0 = torch.rand(batch, seqlen, q_head_num, head_size // 2).npu().half()
        intensor1 = torch.rand(batch, seqlen, k_head_num, head_size // 2).npu().half()
        # op需要cos/sin重复一次
        intensor2 = torch.rand(ntoken, head_size // 4, 1).repeat(1, 1, 2).view(ntoken, head_size // 2).npu().half()
        intensor3 = torch.rand(ntoken, head_size // 4, 1).repeat(1, 1, 2).view(ntoken, head_size // 2).npu().half()
        intensor4 = torch.tensor([seqlen, seqlen], dtype=torch.int32).npu()
        self.execute(OP_NAME, {"rotaryCoeff": 64},
                     [intensor0, intensor1, intensor2, intensor3, intensor4])

def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)

class TestRopeOperation3(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        hidden_size = in_tensors[0].size()[1]
        hidden_size1 = in_tensors[1].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        head_num1 = hidden_size1 // head_size
        q = in_tensors[0].view(batch, seqlen, head_num, head_size).float()
        k = in_tensors[1].view(batch, seqlen, head_num1, head_size).float()
        cos = in_tensors[2].view(batch, seqlen, head_size).unsqueeze(2)
        sin = in_tensors[3].view(batch, seqlen, head_size).unsqueeze(2)
        q_embed = ((q * cos) + (rotate_half(q) * sin)).view(ntoken, hidden_size)
        k_embed = ((k * cos) + (rotate_half(k) * sin)).view(ntoken, hidden_size1)
        return [q_embed.half(), k_embed.half()]

    def test(self):
        return
        ntoken = 128
        seqlen = 128
        hidden_sizeq = 1024
        hidden_sizek = 128
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_sizeq).npu().half()
        intensor1 = torch.rand(ntoken, hidden_sizek).npu().half()
        intensor2 = torch.rand(ntoken, head_size).npu()
        intensor3 = torch.rand(ntoken, head_size).npu()
        intensor4 = torch.tensor([seqlen], dtype=torch.int32).npu()
        # llama highPrecision精度验证
        self.execute(OP_NAME, {"rotaryCoeff": 2, "cosFormat": 0},
                     [intensor0, intensor1, intensor2, intensor3, intensor4])

class TestRopeOperation4(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        params = self.op_params
        params = json.loads(json.dumps(params))
        if in_tensors[0].dtype == torch.bfloat16:
            in_tensors[0] = in_tensors[0].to(torch.float32)
            in_tensors[1] = in_tensors[1].to(torch.float32)
            in_tensors[2] = in_tensors[2].to(torch.float32)
            in_tensors[3] = in_tensors[3].to(torch.float32)
            dtype = np.float32
        else:
            dtype = np.float16
        q = np.array(in_tensors[0].cpu()).astype(dtype)
        kk = np.array(in_tensors[1].cpu()).astype(dtype)
        cos = np.array(in_tensors[2].cpu()).astype(dtype)
        sin = np.array(in_tensors[3].cpu()).astype(dtype)
        seqlen = np.array(in_tensors[4].cpu()).astype(np.int32)
        batch = seqlen.shape[0]
        rotaryCoeff = params['rotaryCoeff']
        headDim = cos.shape[-1]
        hiddensizeQ = 0
        hiddensizeK = 0
        headNumQ = 0
        headNumK = 0
        realHeadNumQ = 0
        realHeadNumK = 0
        realHeadDim = 0
        realBatch = batch
        realSeqLen = seqlen[0]
        isFour = False
        if len(q.shape) == 4:
            isFour = True
            hiddensizeQ = q.shape[-1] * q.shape[-2]
            hiddensizeK = kk.shape[-1] * kk.shape[-2]
            realHeadNumQ = q.shape[-2]
            realHeadNumK = kk.shape[-2]
            realHeadDim = q.shape[-1]
            realBatch = q.shape[0]
            realSeqLen = q.shape[1]
        else:
            hiddensizeQ = q.shape[-1]
            hiddensizeK = kk.shape[-1]
            realHeadNumQ = hiddensizeQ // headDim
            realHeadNumK = hiddensizeK // headDim
            realHeadDim = cos.shape[-1]
        headNumQ = hiddensizeQ // headDim
        headNumK = hiddensizeK // headDim
        hiddensize = max(hiddensizeQ, hiddensizeK)
        headNum = max(headNumQ, headNumK)
        ntokens = np.sum(seqlen)
        if len(q.shape) != len(cos.shape):
            q = q.reshape((ntokens, hiddensizeQ))
            kk = kk.reshape((ntokens, hiddensizeK))
        rope_q = np.zeros(shape=(ntokens, hiddensizeQ)).astype(dtype)
        rope_k = np.zeros(shape=(ntokens, hiddensizeK)).astype(dtype)
        prefix_Ntokens = 0
        cosTable = np.zeros(shape=(ntokens, hiddensize)).astype(dtype)
        for i in range(ntokens):
            for j in range(headNum):
                cosTable[i][j*headDim:(j+1)*headDim] = cos[i][:]
        for i in range(batch):
            curr_seqLen = seqlen[i]
            q1 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(dtype)
            k1 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(dtype)

            for i in range(prefix_Ntokens, prefix_Ntokens + curr_seqLen):
                q1[i-prefix_Ntokens] = q[i] * cosTable[i][:hiddensizeQ]
                k1[i-prefix_Ntokens] = kk[i] * cosTable[i][:hiddensizeK] 
            q2 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(dtype)
            k2 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(dtype)        
            for k in range(headNum):
                src_ = k * headDim
                dst_ = (k + 1) * headDim
                strdie = headDim // 2
                rotaryStrdie = headDim // rotaryCoeff
                rotaryTimesPerHead = rotaryCoeff / 2
                for cycle in range(int(rotaryTimesPerHead)):
                    src =  src_ + cycle * rotaryStrdie * 2
                    dst = src + rotaryStrdie * 2
                    for curr_seqLeni in range(curr_seqLen):
                        if k < headNumQ:
                            q2[curr_seqLeni][src:src + rotaryStrdie] = q[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                            q2[curr_seqLeni][src + rotaryStrdie:dst] = q[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                            q2[curr_seqLeni][src:dst] = q2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
                        if k < headNumK:
                            k2[curr_seqLeni][src:src + rotaryStrdie] = kk[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                            k2[curr_seqLeni][src + rotaryStrdie:dst] = kk[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                            k2[curr_seqLeni][src:dst] = k2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
            rope_q[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += q1 + q2
            rope_k[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += k1 + k2      
            
            prefix_Ntokens += curr_seqLen

        if isFour:
            rope_q = rope_q.reshape((realBatch, realSeqLen, realHeadNumQ, realHeadDim))
            rope_k = rope_k.reshape((realBatch, realSeqLen, realHeadNumK, realHeadDim))
        if dtype == np.float32:
            return [torch.tensor(rope_q).bfloat16(), torch.tensor(rope_k).bfloat16()]
        else:
            return [torch.tensor(rope_q), torch.tensor(rope_k)]
    
    def test_rope_cos_format0(self):
        if operation_test.get_soc_version() not in ('Ascend910B', 'Ascend950'):
            print("this testcase only supports Ascend910B adn Ascend950")
            return

        for head_size in [32, 64, 128, 256]:
            ntoken = random.randint(1,1024)
            seqlen = ntoken
            hidden_sizek = head_size * random.randint(1,6)
            hidden_sizeq = hidden_sizek * random.randint(1,6)
            for dtype in [torch.bfloat16, torch.float16]:
                intensor0 = torch.rand(ntoken, hidden_sizeq).npu().to(dtype)
                intensor1 = torch.rand(ntoken, hidden_sizek).npu().to(dtype)
                intensor4 = torch.tensor([seqlen], dtype=torch.int32).npu()
                for coeff in [2, 4, head_size//2, head_size]:
                    rotary_coeff = coeff
                    cos_dim = head_size if rotary_coeff in [2, 4, head_size] else head_size // 2
                    sin_dim = cos_dim
                    
                    intensor2 = torch.rand(ntoken, cos_dim).npu().to(dtype)
                    intensor3 = torch.rand(ntoken, sin_dim).npu().to(dtype)
                    
                    self.op_params = {"rotaryCoeff": rotary_coeff}
                    self.execute(OP_NAME, self.op_params, [intensor0, intensor1, intensor2, intensor3, intensor4])

if __name__ == '__main__':
    unittest.main()
