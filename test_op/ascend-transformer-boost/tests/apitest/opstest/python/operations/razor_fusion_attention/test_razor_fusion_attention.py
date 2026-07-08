#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

from enum import Enum
import torch
import logging
import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from precision_calcu import compare_cv
import operation_test
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
sys.path.append("./tests/pythontest")
save_path = "./"


class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2


np.random.seed(123)
MASK_TYPE_NO_MASK = 0
MASK_TYPE_RAZOR_FUSION = 11

logging.getLogger().setLevel(logging.DEBUG)


class TestRazorFusionAttention(operation_test.OperationTest):

    def close_pack(self, in_data, seq_len):
        kv = in_data.numpy()
        dim1len = np.size(kv, -2)
        if max(seq_len) > dim1len:
            return None
        kv = kv.reshape(np.prod(kv.shape[0:-1]), kv.shape[-1])
        c_offset = 0
        s_offset = 0
        for i, len in enumerate(seq_len):
            kv[c_offset:c_offset + seq_len[i]
            ][:] = kv[s_offset:s_offset + seq_len[i]][:]
            c_offset += seq_len[i]
            s_offset += dim1len
        return torch.from_numpy(kv[0:sum(seq_len)][:])

    def set_data_params(self, dynamic_batch=False, batch_state=None, window_size=0,
                        is_mask=False, is_decoder=False, is_razor_fusion = True,
                        batch = 1, kv_head = 1, heads = 1, embeddim = 128, embeddimv = 0, max_seq = 2048,
                        kv_seqLen = [], is_clamp = 0, clamp_min = 0, preTokens = 0, nextTokens = 0,
                        tileQ = 0, tileKv = 0, razorLen = 0, baseM = 0, textQLen = 0, textKvLen = 0,
                        clamp_max = 0, data_type = torch.float16, mask_type = 0,
                        no_cache = True, long_seq = False,
                        is_sqrt = False, left_align = False, scaleType = ScaleType.SCALE_TOR.value, fav3 = False,
                        tor = 1.0, bnsd = False, is_compress = False, q_seqlens=None, dim_num=2):
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.preTokens = preTokens
        self.nextTokens = nextTokens
        self.tileQ = tileQ
        self.tileKv = tileKv
        self.razorLen = razorLen
        self.baseM = baseM
        self.textQLen = textQLen
        self.textKvLen = textKvLen
        self.is_razor_fusion = is_razor_fusion
        self.batch = batch
        self.kv_head = kv_head
        self.heads = heads
        self.embeddim = embeddim
        self.embeddimv = embeddimv
        self.max_seq = max_seq
        self.kv_seqLen = kv_seqLen
        self.dynamic_batch = dynamic_batch
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = tor
        self.is_int8_flag = False
        self.bnsd = bnsd
        self.window_size = window_size
        self.is_compress = is_compress
        self.q_seqlens = q_seqlens if q_seqlens is not None else kv_seqLen
        self.dim_num = dim_num

        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, self.q_seqlens)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)

        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim)))
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)

        self.gen_mask(batch, data_type, mask_type)
        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

    def gen_mask(self, batch, data_type, mask_type):
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            MASK_TYPE_RAZOR_FUSION : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:q_s, :kv_s]))),
            MASK_TYPE_NO_MASK : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: 0))
        }
        self.mask_info = mask_type_dict[mask_type]
        mask = self.gen_razor_fusion_mask(self.razorLen, self.tileQ, self.tileKv, self.textQLen, self.textKvLen,
                                          self.preTokens, self.nextTokens, self.baseM)
        post_mask_coff = 1
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff

    def gen_razor_fusion_mask(self, razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM):
        np.set_printoptions(threshold=np.inf)

        mask_sizeQ = razorLen * tileQ + textQLen
        mask_sizeK = razorLen * tileKv + textKvLen
        print("generate razor mask:", razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM)
        mask = np.zeros((mask_sizeQ, mask_sizeK), dtype=int)
        preTokensBlock = preTokens // baseM
        nextTokensBlock = nextTokens // baseM
        idx = razorLen // baseM * baseM
        mask[:, int(idx): int(razorLen)] = 0
        mask[int(idx): int(razorLen), :] = 0
        for i in range((razorLen + baseM - 1) // baseM):
            start = i - preTokensBlock + 1 if i >= preTokensBlock else 0
            end = i + nextTokensBlock if i < preTokensBlock else start + preTokensBlock + nextTokensBlock - 1
            end = (razorLen + baseM - 1) // baseM if end > (razorLen + baseM - 1) // baseM else end
            for j in range(start, end):
                mask[i * baseM: (i + 1) * baseM, j * baseM: (j + 1) * baseM] = 1
        mask[razorLen:, :] = 0
        mask[:, razorLen:] = 0
        for i in range(tileQ):
            for j in range(tileKv):
                mask[i * razorLen: (i + 1) * razorLen, j * razorLen: (j + 1) * razorLen] = mask[0: razorLen,
                                                                                           0: razorLen]

        mask[razorLen * tileQ:, :] = 1
        mask[:, razorLen * tileKv:] = 1
        mask = mask[None, None, :]
        mask = 1 - mask
        return mask * -10000

    def gen_out_tensor(self):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        is_razor_fusion = self.is_razor_fusion
        q = self.q
        k = self.k
        v = self.v
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        ans_concat = None
        ans_concat_true = None
        out_true = None
        self.encoder_logN = torch.tensor([2.0] * self.max_seq).to(torch.float32)
        self.encoder_logN.uniform_(1, 2)
        self.decoder_logN = torch.tensor([2.0] * batch).to(torch.float32)
        self.decoder_logN.uniform_(1, 2)
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                    out_true = output
                else:
                    out = torch.cat((out, output), 0)
                    out_true = torch.cat((out_true, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            if self.scaleType == ScaleType.SCALE_LOGN_FP32.value:
                if is_decoder:
                    score *= self.decoder_logN[idx]
                else:
                    score *= self.encoder_logN[None, :q_s, None]
            score *= self.tor
            temp_mask = self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
            if is_mask or is_razor_fusion:
                score = score + temp_mask

            s_qk = score
            s_qk_true = score.to(torch.float32)
            score = score.numpy().astype(np.float32)
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)           
            p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
            p_true = torch.from_numpy(p_true)
            p = p_true.to(torch.bfloat16)
            o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)

            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            o_true = o_true.view(heads, q_s, embedv)
            o_true = torch.permute(o_true, (1, 0, 2)).contiguous()
            o = o.view(heads, q_s, embedv)
            o = torch.permute(o, (1, 0, 2)).contiguous()
            if out is None:
                out = o
                out_true = o_true
            else:
                out = torch.cat((out, o), 0)
                out_true = torch.cat((out_true, o_true), 0)
            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        # golden data
        out = out.view(q_ntokens, heads * embedv)
        self.golden_out = out.to(self.data_type)
        out_true = out_true.view(q_ntokens, heads * embedv)
        self.golden_out_true = out_true.to(torch.float32)
        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)
    
    def group_mm_torch(self, heads, group_num, A, B, dtype=torch.float32):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(dtype), B[i:(i + 1), :, :].to(dtype))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def qkMM(self, query,key):
        result = None
        qk_k = key.shape[1]
        for qk_k_split in range(0, qk_k, 128):
            sub_k = 128
            if qk_k_split == 512:
                sub_k = 64
            query_k = query[:, :, qk_k_split : qk_k_split + sub_k]
            key_k = key[:, qk_k_split : qk_k_split + sub_k, :]
            result_split = torch.matmul(query_k.to(torch.float32), key_k.to(torch.float32))
            if result is None:
                result = result_split
            else:
                result = result + result_split
        return result

    def softmax(self,
        qk_result,
        is_first,
        gm
    ):
        sim = qk_result.numpy()
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm #全局的减去最新的
        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub.astype(np.float32))
        if qk_result.dtype == torch.float16:
            sim_sub = sim_sub.astype(np.float16)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        return torch.from_numpy(sim_sub), row_sum, dm, gm #返回P~ 、P~的rowsum和rowmax的差值 都是局部的结果


    def golden_calc(self, in_tensors):
        golden_out = self.golden_out.clone().detach().requires_grad_(True).half().npu()
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        print("golden_compare")
        print(out_tensors.shape)
        result_double = compare_cv(self.golden_out_true.flatten(), golden_tensors.cpu().flatten(), out_tensors.cpu().flatten())
        self.rfa_compare_result = result_double
        return self.rfa_compare_result

    def test_razor_fusion_attention_case_fa_razor_fusion_norm_bfloat16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        razorLen = 512
        tileQ = 1
        tileKv = 2
        textQLen = 128
        textKvLen = 256
        preTokens = 128   #128对齐
        nextTokens = 256  #128对齐

        max_seq = max(tileQ * razorLen + textQLen, tileKv * razorLen + textKvLen)
        q_seqLen = [tileQ * razorLen + textQLen] * batch
        kv_seqLen = [tileKv * razorLen + textKvLen] * batch
        baseM = kv_seqLen[0] if kv_seqLen[0] <= 128 else 128
        OP_NAME = "RazorFusionAttentionOperation"
        tor = 0.088
        OP_PARAM = {"headNum":1, "kvHeadNum":1, "qkScale": tor, "tileQ": tileQ,  "tileKv": tileKv,
                    "razorLen": razorLen, "textQLen": textQLen,  "textKvLen": textKvLen, "preTokens": preTokens,
                    "nextTokens": nextTokens}
        data_type = torch.bfloat16

        self.set_data_params(batch = batch, max_seq = max_seq, kv_seqLen = kv_seqLen, baseM = baseM, tileQ = tileQ,
                             razorLen = razorLen, textQLen = textQLen, textKvLen = textKvLen, tileKv = tileKv,
                             preTokens = preTokens, nextTokens = nextTokens, data_type = data_type,
                             q_seqlens=q_seqLen, mask_type = MASK_TYPE_RAZOR_FUSION, tor = tor)
        self.gen_out_tensor()
        self.execute(OP_NAME, OP_PARAM, [self.q.npu(), self.k.npu(), self.v.npu()])

    def test_razor_fusion_attention_case_fa_razor_fusion_norm_float16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        razorLen = 500
        tileQ = 2
        tileKv = 3
        textQLen = 128
        textKvLen = 256
        preTokens = 128   #128对齐
        nextTokens = 256  #128对齐

        max_seq = max(tileQ * razorLen + textQLen, tileKv * razorLen + textKvLen)
        q_seqLen = [tileQ * razorLen + textQLen] * batch
        kv_seqLen = [tileKv * razorLen + textKvLen] * batch
        baseM = kv_seqLen[0] if kv_seqLen[0] <= 128 else 128
        OP_NAME = "RazorFusionAttentionOperation"
        tor = 0.088
        OP_PARAM = {"headNum":1, "kvHeadNum":1, "qkScale": tor, "tileQ": tileQ,  "tileKv": tileKv,
                    "razorLen": razorLen, "textQLen": textQLen,  "textKvLen": textKvLen, "preTokens": preTokens,
                    "nextTokens": nextTokens}
        data_type = torch.float16

        self.set_data_params(batch = batch, max_seq = max_seq, kv_seqLen = kv_seqLen, baseM = baseM, tileQ = tileQ,
                             razorLen = razorLen, textQLen = textQLen, textKvLen = textKvLen, tileKv = tileKv,
                             preTokens = preTokens, nextTokens = nextTokens, data_type = data_type,
                             q_seqlens=q_seqLen, mask_type = MASK_TYPE_RAZOR_FUSION, tor = tor)
        self.gen_out_tensor()
        self.execute(OP_NAME, OP_PARAM, [self.q.npu(), self.k.npu(), self.v.npu()])

    def test_razor_fusion_attention_case_fa_razor_fusion_norm_bfloat16_dim3(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 3
        razorLen = 128
        tileQ = 3
        tileKv = 2
        textQLen = 10
        textKvLen = 10
        preTokens = 256   #128对齐
        nextTokens = 128  #128对齐

        max_seq = max(tileQ * razorLen + textQLen, tileKv * razorLen + textKvLen)
        q_seqLen = [tileQ * razorLen + textQLen] * batch
        kv_seqLen = [tileKv * razorLen + textKvLen] * batch
        baseM = kv_seqLen[0] if kv_seqLen[0] <= 128 else 128
        OP_NAME = "RazorFusionAttentionOperation"
        tor = 0.088
        OP_PARAM = {"headNum":1, "kvHeadNum":1, "qkScale": tor, "tileQ": tileQ,  "tileKv": tileKv,
                    "razorLen": razorLen, "textQLen": textQLen,  "textKvLen": textKvLen, "preTokens": preTokens,
                    "nextTokens": nextTokens}
        data_type = torch.bfloat16

        self.set_data_params(batch = batch, max_seq = max_seq, kv_seqLen = kv_seqLen, baseM = baseM, tileQ = tileQ,
                             razorLen = razorLen, textQLen = textQLen, textKvLen = textKvLen, tileKv = tileKv,
                             preTokens = preTokens, nextTokens = nextTokens, data_type = data_type,
                             q_seqlens=q_seqLen, mask_type = MASK_TYPE_RAZOR_FUSION, tor = tor, dim_num =3)
        self.gen_out_tensor()
        # [M, N]转成[M, 1, N]
        self.q = self.q.unsqueeze(1)
        self.k = self.k.unsqueeze(1)
        self.v = self.v.unsqueeze(1)
        self.execute(OP_NAME, OP_PARAM, [self.q.npu(), self.k.npu(), self.v.npu()])

    def test_razor_fusion_attention_case_fa_razor_fusion_norm_float16_dim3(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 1
        razorLen = 311
        tileQ = 2
        tileKv = 3
        textQLen = 128
        textKvLen = 256
        preTokens = 128   #128对齐
        nextTokens = 256  #128对齐

        max_seq = max(tileQ * razorLen + textQLen, tileKv * razorLen + textKvLen)
        q_seqLen = [tileQ * razorLen + textQLen] * batch
        kv_seqLen = [tileKv * razorLen + textKvLen] * batch
        baseM = kv_seqLen[0] if kv_seqLen[0] <= 128 else 128
        OP_NAME = "RazorFusionAttentionOperation"
        tor = 0.088
        OP_PARAM = {"headNum":1, "kvHeadNum":1, "qkScale": tor, "tileQ": tileQ,  "tileKv": tileKv,
                    "razorLen": razorLen, "textQLen": textQLen,  "textKvLen": textKvLen, "preTokens": preTokens,
                    "nextTokens": nextTokens}
        data_type = torch.float16

        self.set_data_params(batch = batch, max_seq = max_seq, kv_seqLen = kv_seqLen, baseM = baseM, tileQ = tileQ,
                             razorLen = razorLen, textQLen = textQLen, textKvLen = textKvLen, tileKv = tileKv,
                             preTokens = preTokens, nextTokens = nextTokens, data_type = data_type,
                             q_seqlens=q_seqLen, mask_type = MASK_TYPE_RAZOR_FUSION, tor = tor, dim_num =3)
        self.gen_out_tensor()
        # [M, N]转成[M, 1, N]
        self.q = self.q.unsqueeze(1)
        self.k = self.k.unsqueeze(1)
        self.v = self.v.unsqueeze(1)
        self.execute(OP_NAME, OP_PARAM, [self.q.npu(), self.k.npu(), self.v.npu()])


if __name__ == '__main__':
    unittest.main()
