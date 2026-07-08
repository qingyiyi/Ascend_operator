#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import builtins
import torch
import numpy as np
import torch_npu
import json
import random
import os
from ctypes import CDLL
import math
import sys
import shutil
import logging
import re
from enum import Enum
from typing import Any, Dict, List
import copy


class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2


np.random.seed(123)

MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5
MASK_TYPE_SWA = 6
MASK_TYPE_SWA_DECODER = 7
MASK_TYPE_ALIBI_WITH_PREFIX_BATCH = 8
MASK_TYPE_NO_BATCH_WITH_PREFIX = 9
MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX = 10

dtype_map = {
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int32": torch.int32,
    "float": torch.float,
}
format_dict = {
    "undefined": -1,
    "nchw": 0,
    "nhwc": 1,
    "nd": 2,
    "nc1hwc0": 3,
    "fractal_z": 4,
    "nc1hwc0_c04": 12,
    "hwcn": 16,
    "ndhwc": 27,
    "fractal_nz": 29,
    "ncdhw": 30,
    "ndc1hwc0": 32,
    "fractal_z_3d": 33,
}


# 将5维nz转4维nd
def convert_nz_to_nd(x):
    aux_dims = [0, 0, 0, 0]
    aux_dims[0] = x.size(0)
    aux_dims[1] = x.size(1)
    aux_dims[2] = x.size(3)
    aux_dims[3] = x.size(2) * x.size(4)
    return x.transpose(2, 3).reshape(aux_dims)


# 将4维nz转3维nd
def convert_nz_to_nd_mask(x):
    aux_dims = [0, 0, 0]
    aux_dims[0] = x.size(0)
    aux_dims[1] = x.size(2)
    aux_dims[2] = x.size(1) * x.size(3)
    return x.transpose(1, 2).reshape(aux_dims)


# 用于将4维nd转为nz
def shape_nd_to_nz(shape, dtype="float16"):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3])
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)

# 根据csv配置生成tensor
def gen_tensor(i, shapes, datatype, format, data_gen_ranges):
    return torch.from_numpy(np.random.uniform(0, 1, size=shapes[i])).to(datatype)


# 获取q的ntokens和headsize
def get_q_info(shape, head_num):
    # BSND:
    if len(shape) == 4:
        return shape[0] * shape[1], shape[3]
    elif len(shape) == 2:
        return shape[0], shape[1] // head_num
    else:
        return shape[0], shape[2]
    # BNSD:


def get_q_dims_BNSD(shape, head_num):
    return shape[0] * shape[2], shape[3]


# 获取batch, kv的headsize
def get_cache_info(shape, head_num):
    # BSND:
    if len(shape) == 4:  # [layerNum, batch, maxSeqLen, khiddenSize]
        return shape[1], shape[2], shape[3] // head_num
    elif len(shape) == 3:  # [batch, maxSeqLen, hiddenSize]
        return shape[0], shape[1], shape[2] // head_num
    elif len(shape) == 5:  # [layer, batch, hiddenSize/16, maxSeqLen, 16]
        return shape[1], shape[3], (shape[2] * 16) // head_num


# pa encoder:
def get_cache_info_pa_encoder(shape, head_num):
    # BSND:
    if len(shape) == 4:  # [layerNum, batch, maxSeqLen, khiddenSize]
        return shape[1], shape[2], shape[3] // head_num
    elif len(shape) == 5:  # [batch, head_num, embedim / 16, kv_max_seq, 16]
        return shape[0], shape[3], (shape[2] * 16) // head_num


# BNSD:
def get_cache_info_BNSD(shape, head_num):
    # BSND:
    if len(shape) == 4:  # [layerNum, batch, maxSeqLen, khiddenSize]
        return shape[1], shape[2], shape[3] // head_num
    elif len(shape) == 5:  # [batch, head_num, embedim / 16, kv_max_seq, 16]
        return shape[0], shape[3], (shape[2] * 16) // head_num

class SelfAttentionMaskGen():
    def __init__(self, batch=1, heads=1, data_type=torch.float16, mask_type=MASK_TYPE_NO_HEAD, max_seq=2048,
                    window_size=0, long_seq=False, kv_seqLen=None, dynamic_batch=False, is_decoder=False):
        self.batch = batch
        self.heads = heads
        self.data_type = data_type
        self.mask_type = mask_type
        self.q_max_seq = max_seq
        self.kv_max_seq = max_seq
        self.max_seq = max_seq
        self.is_triu_mask = False
        self.left_align = False
        self.mask_compress = False
        self.is_alibi = False
        self.compress_type = 0
        self.shape = None
        self.is_910b = True
        self.format = 'nd'
        self.mask_info = self.get_mask_info(mask_type, self.batch, self.heads)
        self.window_size = window_size
        self.long_seq = long_seq
        self.kv_seqLen = kv_seqLen
        self.dynamic_batch = dynamic_batch
        self.is_decoder = is_decoder


    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        # slopes = torch.ones(n_heads)
        return slopes

    def get_alibi_bias(self, n_heads, max_seqlen):
        if not self.left_align:
            self.bias = torch.arange(max_seqlen)
            self.bias = self.bias[None, :] - self.bias[:, None]
            if (self.is_sqrt):
                self.bias = torch.sqrt(
                    torch.abs(self.bias)) * torch.sign(self.bias)
            bias = torch.empty(
                n_heads,
                max_seqlen,
                max_seqlen
            )[:, :max_seqlen, :max_seqlen].copy_(self.bias)
            self.alibi_slopes = self.get_alibi_slopes(n_heads)
        else:
            self.bias = torch.arange(max_seqlen, dtype=torch.float32).unsqueeze(
                0).unsqueeze(0).expand(n_heads, max_seqlen, -1)
            self.alibi_slopes = torch.Tensor(self.get_interleave(n_heads))
            bias = self.bias
        bias = bias * self.alibi_slopes[:, None, None]
        return bias

    def get_interleave(self, n, alibi_bias_max=8.0):
        def get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self.get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def gen_swa_cmp(self, max_seq, window_size):
        swa_mask = np.ones(shape=(1, 512, 512)) * self.pre_mask_coff
        pp_n = 128 if self.embeddim <= 128 else 64
        if window_size <= pp_n * 3:
            true_size = window_size
        else:
            if window_size % pp_n == 0:
                true_size = pp_n * 3
            else:
                true_size = pp_n * 2 + window_size % pp_n
        triu_mask = np.triu(swa_mask, 1)
        tril_mask = np.tril(swa_mask, -true_size)
        swa_mask = triu_mask + tril_mask
        swa_mask = torch.from_numpy(swa_mask).to(torch.float32)
        return swa_mask

    def gen_swa_mask(self, max_seq, window_size, pre_mask_coff, cache_type=0):
        swa_mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        if window_size < max_seq and self.is_decoder:
            if cache_type == 1:
                for idx, kvseqlen in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, :window_size] = 0
            else:
                for idx, kvseqlen in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, kvseqlen - window_size: kvseqlen] = 0
        elif window_size < max_seq or self.is_compress:
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -window_size)
            swa_mask = triu_mask + tril_mask
        else:
            swa_mask = np.triu(swa_mask, 1)
        return swa_mask
    
    def get_mask_info(self, mask_type, batch, heads):
        q_max_seq = self.q_max_seq
        kv_max_seq = self.kv_max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s,:kv_s]))),
            MASK_TYPE_ALIBI_WITH_PREFIX_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :,kv_s-q_s:kv_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH: ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:,kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_NO_HEAD: ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH_WITH_PREFIX : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s,:kv_s]))),
            MASK_TYPE_SWA: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_SWA_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_NO_MASK: ((1, q_max_seq, kv_max_seq),
                                (lambda mask, idx, q_s, kv_s: 0))
        }
        # kernel中mask的系数
        if self.data_type == torch.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        elif self.data_type == torch.bfloat16 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = -float("inf")
        elif self.data_type == torch.float32 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        self.mask_info = mask_type_dict[mask_type]
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.kv_max_seq)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff
        return mask_type_dict[mask_type]

    def gen_mask(self, batch, heads, data_type, mask_type, window_size, is_compress, cache_type=0):
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_WITH_PREFIX_BATCH: ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, kv_s-q_s:kv_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH: ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_NO_HEAD: ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH_WITH_PREFIX : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_SWA: ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_SWA_DECODER: ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_NO_MASK: ((1, q_max_seq, kv_max_seq),
                                (lambda mask, idx, q_s, kv_s: 0))
        }
        # kernel中mask的系数
        if data_type == torch.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        elif data_type == torch.bfloat16 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = -float("inf")
        elif data_type == torch.float32 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        if data_type == torch.float16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi:
                select_zero = False
            elif self.dynamic_batch or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        else:
            if self.is_alibi or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        if self.is_triu_mask:
            select_zero = False
        self.mask_info = mask_type_dict[mask_type]
        mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(self.max_seq), k=300)
        if self.window_size > 0:
            mask = self.gen_swa_mask(
                self.max_seq, window_size, pre_mask_coff, cache_type)
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
            mask += self.alibi_bias.numpy()
        if select_zero:
            mask.flat[zero_indice] = 0
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff
        return self.mask

    def gen_mask_for_golden(self, mask_param):
        # 根据opparam以及csv shape 选择mask_type，然后生成golden计算压缩前的mask
        self.gen_mask(**mask_param)

    def gen_mask_for_asdops(self):
        # 用于算子计算的mask
        # 非压缩场景直接传golden计算用的mask
        # 压缩场景通过csv的shape推测mask_type，然后修改max_seq，生成压缩后的mask
        mask = None
        max_seq = max(self.q_max_seq, self.kv_max_seq)
        if not self.mask_compress:
            mask = self.mask
            # 310p场景需要reshape
            if not self.is_910b:
                mask = convert_nd_to_nz(mask)
                if len(mask.shape) == 2:
                    mask = mask.reshape(1, max_seq // 16, max_seq, 16)
                else:
                    mask = mask.reshape(-1, max_seq // 16, max_seq, 16)
        # 压缩mask场景，算子和golden需要的mask shape不同
        else:
            if self.is_alibi:
                if len(self.shape) == 2 and self.shape[0] == 256 and self.shape[1] == 256: # ALIBI_COMPRESS [256, 256]
                    if self.left_align:
                        mask = np.ones((256,256)) * -(float("inf"))
                        mask = np.triu(mask, 1)
                        mask = self.bias[0, :256, :256] + mask
                    else:
                        self.alibi_slopes *= -1
                        mask = np.ones((256,256)) * 60000
                        mask = np.triu(mask, 1)
                        mask = self.bias[:256, :256] * -1 + mask
                    mask = mask.to(self.data_type)
                    if not self.is_910b:
                        mask = convert_nd_to_nz(mask)
                        mask = mask.reshape(1, 16, 256, 16)
                elif len(self.shape) == 3 and self.shape[2] == 128: # ALIBI_COMPRESS [head_num, seqlen, 128]
                    if self.left_align:
                        mask = np.ones(shape=self.mask_info[0]) * -(float("inf"))
                        mask = np.triu(mask, 1)
                        mask += self.alibi_bias.numpy()
                        mask = mask[0, :, :, :128]
                        mask = torch.from_numpy(mask)
                    else:
                        mask = self.mask[0, :, :, :128]
                    mask = mask.to(self.data_type)
                    if not self.is_910b:
                        mask = convert_nd_to_nz(mask)
                        mask = mask.reshape(self.heads, 128 // 16, self.q_max_seq, 16)
            else:
                if self.data_type == torch.float16:
                    pre_mask_coff = -10000.0
                else:
                    pre_mask_coff = 1
                mask = np.ones(self.shape) * pre_mask_coff
                mask = np.triu(mask, 1)
                mask = torch.from_numpy(mask).to(self.data_type)
        return mask


class SelfAttentionGenOutTensor:
    def __init__(self,fa_golden):
        for key, value in fa_golden.__dict__.items():
            if not callable(value):  # 过滤掉方法
                setattr(self, key, value)
        if self.pa_encoder:
            query_shape = self.in_tensors[self.query_id].shape
            self.ntokens,self.head_num,self.head_size = query_shape
            # self.head_num = query_shape[1]
            # self.head_size = query_shape[2]

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

    def set_data_params(self, dynamic_batch=False, batch_state=None, window_size=0, cache_type=0,
                        is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4,
                        batch=1, kv_head=1, heads=1, embeddim=128, embeddimv=0, max_seq=2048,
                        kv_seqLen=[], is_clamp=0, clamp_min=0,
                        clamp_max=0, data_type=torch.float16, op_type=0, mask_type=0,
                        no_cache=False, long_seq=False, is_triu_mask=False, is_multi_layer=False,
                        is_sqrt=False, left_align=False, scaleType=ScaleType.SCALE_TOR.value, fav3=False,
                        tor=1, bnsd=False, is_compress=False, q_seqlens=None, num_blocks=None,
                        block_size=None):
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.alibi_dim = alibi_dim
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
        self.is_triu_mask = is_triu_mask
        self.is_multi_layer = is_multi_layer
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = tor
        self.is_int8_flag = False
        self.online = False
        self.bnsd = bnsd
        self.window_size = window_size
        self.is_compress = is_compress
        self.cache_type = cache_type
        self.q_seqLens = q_seqlens if q_seqlens is not None else kv_seqLen

        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqLen, self.q_ntokens = __class__.gen_seq_len(batch, [
                                                             1] * batch)
        else:
            self.q_seqLen, self.q_ntokens = __class__.gen_seq_len(
                batch, self.q_seqLens)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        if is_multi_layer:
            self.layer_id = torch.from_numpy(
                np.array([1], dtype=np.int32)).to(torch.int32)
        else:
            self.layer_id = torch.from_numpy(
                np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqLen)
        self.kv_max_seq = np.max(self.kv_seqLen)
        q = torch.from_numpy(
            np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)))

        self.q = q.to(data_type)

        if num_blocks is None:
            self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
            self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
        else:
            # kv cache shape: (num_blocks, block_size, num_heads, head_size)
            self.k_cache = torch.from_numpy(
                np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            self.v_cache = torch.from_numpy(
                np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)

            batch = len(kv_seqLen)
            max_context_len = max(kv_seqLen)
            print("max_context_len: ", max_context_len)
            print("block_size: ", block_size)
            max_num_blocks_per_seq = (
                max_context_len + block_size - 1) // block_size
            block_tables = []   # (batch, max_num_blocks_per_seq)
            offset = 0
            for i in range(batch):
                num_blocks_cur_seq = (kv_seqLen[i] + block_size - 1) // block_size
                # padding block table with 0
                block_table = [
                    random.randint(0, num_blocks-1) if j < num_blocks_cur_seq else 0 for j in range(max_num_blocks_per_seq)
                ]
                offset += num_blocks_cur_seq
                block_tables.append(block_table)
            self.block_tables = torch.from_numpy(np.array(block_tables)).to(torch.int32)
            print("block_tables : ", block_tables)
            self.block_tables = torch.from_numpy(
                np.array(block_tables)).to(torch.int32)
            self.k = torch.stack([self.k_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(
                torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.v = torch.stack([self.v_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(
                torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.k = self.k.reshape(
                1, batch, max_context_len, kv_head * self.embeddim)
            self.v = self.v.reshape(
                1, batch, max_context_len, kv_head * self.embeddim)

        if self.fav3:
            self.is_int8_flag = True
            self.q_scale, self.q_offset, self.q_int8 = self.quant_per_head(
                self.q, heads, embeddim, (self.q_ntokens, heads * self.embeddim))
            self.k_scale, self.k_offset, self.k_int8 = self.quant_per_head(
                self.k, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.v_scale, self.v_offset, self.v_int8 = self.quant_per_head(
                self.v, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.k_scale = (self.k_scale.view(kv_head, 1) *
                            torch.ones([kv_head, heads // kv_head])).view(-1)
            self.k_offset = (self.k_offset.view(kv_head, 1) *
                             torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_scale = (self.v_scale.view(kv_head, 1) *
                            torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_offset = (self.v_offset.view(kv_head, 1) *
                             torch.ones([kv_head, heads // kv_head])).view(-1)
            self.offline_scale = torch.from_numpy(np.random.uniform(
                1 / 127, 3 / 127, size=(heads))).to(torch.float32)

            self.q_int8 = torch.from_numpy(
                np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim))).to(torch.int8)
            self.k_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(torch.int8)
            self.v_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(
                self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(torch.int8)
        self.mask_generator.gen_mask({batch, heads, data_type, mask_type, window_size, is_compress, cache_type})
        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

    def quant_per_head(self, data, heads, embeddim, shape):
        temp = data.view(-1, heads, self.embeddim)
        scale = torch.stack([self.fav3_quant(
            temp[:, i, :], data_min=-1, data_max=1, symmetric=True)[0] for i in range(heads)])
        offset = torch.stack([self.fav3_quant(
            temp[:, i, :], data_min=-1, data_max=1, symmetric=True)[1] for i in range(heads)])
        int8_data = torch.zeros_like(temp)
        for i in range(heads):
            int8_data[:, i, :] = (
                (temp[:, i, :] / scale[i]).round_() + offset[i])
        int8_data = int8_data.view(shape).to(torch.int8)
        return scale, offset, int8_data

    def fav3_quant(self, data, data_min=0, data_max=0, symmetric=False, bit=8):
        n = 2 ** (bit - 1)
        if symmetric:
            quant_min, quant_max = -(n - 1), (n - 1)
        else:
            quant_min, quant_max = -n, (n - 1)
        span = quant_max - quant_min
        if data_min == data_max:
            data_max = data.max().item()
            data_min = data.min().item()
        if symmetric:
            scale = max(data_max, -data_min) / (float(span) / 2)
            offset = 0
        else:
            scale = (data_max - data_min) / float(span)
            offset = (data_min * quant_min + data_max *
                      quant_max) / (data_min - data_max)
        # 量化公式：x / scale + offset
        return torch.tensor(float(scale), dtype=torch.float), torch.tensor(int(offset), dtype=torch.float)

    def quantize_tensor_symmetric(self, x, prev_max_abs_vals=None, num_bits=8):
        if x.dtype != torch.float:
            x = x.to(torch.float)

        quant_min = -2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1

        current_max_abs_vals = x.abs().max(dim=1).values
        if prev_max_abs_vals is not None:
            max_abs_vals = torch.max(prev_max_abs_vals, current_max_abs_vals)
        else:
            max_abs_vals = current_max_abs_vals
        scales = max_abs_vals / (quant_max)
        x_q = torch.clamp(torch.round(
            x / scales.unsqueeze(1)), quant_min, quant_max)
        x_q = torch.round(x_q)
        x_q = x_q.to(torch.int8)
        return x_q, scales, max_abs_vals

    def dequantize_tensor(self, x_q, scales, value):
        x_deq = x_q.to(torch.float32)
        scales = scales.unsqueeze(1)
        x_deq = x_deq * value
        x_deq = x_deq * scales
        return x_deq

    def online_softmax(self, s_qk, q_s, v_slice, heads, kv_head, embed, online, dtype):
        ans = None
        group_num = heads // kv_head
        for head_idx in range(heads):
            s_head_idx = s_qk[head_idx]
            O = torch.zeros((q_s, embed)).to(dtype)
            Br = q_s
            Bc = 128
            self.row_block_size = Br
            self.col_block_size = Bc
            d = embed
            V_mat = v_slice[head_idx // group_num]
            Tr = q_s // Br
            Tc = q_s // Bc

            d = embed
            Tr = q_s // Br
            Tc = q_s // Bc

            start_row_idx = 0
            start_col_idx = 0

            for i in range(Tr):

                Oi = torch.zeros((Br, d)).to(dtype)  # shape Br x d
                li = torch.zeros((Br, 1)).to(dtype)  # shape Br x 1
                mi = torch.full((Br, 1), -torch.inf).to(dtype)  # shape Br x 1
                pp_max_num = None

                for j in range(Tc):

                    Sij = s_head_idx[i * Br: (i + 1) * Br, start_col_idx +
                                     j * Bc: start_col_idx + (j + 1) * Bc].to(dtype)

                    Vj = V_mat[start_col_idx + j *
                               Bc: start_col_idx + (j + 1) * Bc, :]

                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * \
                        li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(
                                Pij_hat, pp_max_num)
                            if pp_max_num == None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(
                                dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * \
                                self.offline_scale[head_idx]
                            Oi = Oi * \
                                torch.exp((mi - mi_new).to(torch.float32)
                                          ).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * \
                            torch.exp((mi - mi_new).to(torch.float32)
                                      ).to(dtype) + Pij_hat @ Vj.to(dtype)

                    mi = mi_new

                if (q_s % Bc != 0):
                    Bc = q_s % Bc
                    start_row_idx = (
                        q_s // self.row_block_size) * self.row_block_size
                    start_col_idx = (
                        q_s // self.col_block_size) * self.col_block_size

                    Sij = s_head_idx[i * Br: (i + 1) * Br,
                                     start_col_idx: start_col_idx + Bc].to(dtype)
                    Vj = V_mat[start_col_idx: start_col_idx + Bc, :]
                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * \
                        li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(
                                Pij_hat, pp_max_num)
                            if pp_max_num == None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(
                                dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * \
                                self.offline_scale[head_idx]
                            Oi = Oi * \
                                torch.exp((mi - mi_new).to(torch.float32)
                                          ).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * \
                            torch.exp((mi - mi_new).to(torch.float32)
                                      ).to(dtype) + Pij_hat @ Vj.to(dtype)
                Oi = Oi / li

                O[i * Br: (i + 1) * Br, :] = Oi

            if ans is None:
                ans = O
            else:
                ans = torch.cat((ans, O), 1)
        return ans

    def gen_out_tensor(self, online=False):
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
        q_seqlen = self.q_seqLen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        online = self.online
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        if self.fav3:
            q = self.q_int8
            k = self.k_int8
            v = self.v_int8
        q_ntokens = self.q_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        ans_concat = None
        ans_concat_true = None
        out_true = None

        self.encoder_logN = torch.tensor(
            [2.0] * self.max_seq).to(torch.float32)
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
                    if not self.fav3:
                        out_true = output
                else:
                    out = torch.cat((out, output), 0)
                    if not self.fav3:
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

            if self.fav3:
                score = __class__.group_mm_torch(
                    heads, kv_head, q_slice, k_slice_t, torch.int32)
            else:
                score = __class__.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if self.fav3:
                # score:[heads,m,n]
                score = score.to(torch.float32)
                score = score * self.q_scale.view(heads, 1, 1)
                score = score.to(torch.float16)

            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            if self.scaleType == ScaleType.SCALE_LOGN_FP32.value:
                if is_decoder:
                    score *= self.decoder_logN[idx]
                else:
                    score *= self.encoder_logN[None, :q_s, None]

            if self.fav3:
                score = score * torch.tensor(self.tor, dtype=torch.float16)
            else:
                score *= self.tor
            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(
                    np.minimum(score, clamp_max_brc)))
            if is_mask:
                if self.mask_type in (1, 3) and q_s > mask.shape[1]:
                    # 压缩norm mask, 使用当前最大seqlen生成mask
                    no_compress_mask = np.ones(shape=(1, self.max_seq, self.max_seq)).astype(np.float16)
                    no_compress_mask = np.triu(no_compress_mask, 1)
                    no_compress_mask *= -10000.0
                    score = score + no_compress_mask[:, :q_s, :kv_s]
                else:
                    score = score + mask[:, :q_s, :kv_s]

            s_qk = score
            s_qk_true = score.to(torch.float32)
            score = score.numpy().astype(np.float32)
            if self.is_int8_flag:
                ans = self.online_softmax(
                    s_qk, q_s, v_slice, heads, kv_head, embed, online, torch.float16)
                if ans_concat is None:
                    ans_concat = ans
                else:
                    ans_concat = torch.cat((ans_concat, ans), 0)

                ans_true = self.online_softmax(
                    s_qk_true, q_s, v_slice, heads, kv_head, embed, online, torch.float32)
                if ans_concat_true is None:
                    ans_concat_true = ans_true
                else:
                    ans_concat_true = torch.cat((ans_concat_true, ans_true), 0)

            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)

            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)
            if self.fav3:
                p = score_exp
                p = p * 127
                p = torch.from_numpy(p).to(torch.int8)
            else:
                p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
                p_true = torch.from_numpy(p_true)
                p = p_true.to(torch.bfloat16)
                o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)

            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            if self.fav3:
                o = o.to(torch.float)
                v_scale = self.v_scale
                v_scale = v_scale.view(heads, 1, 1)
                o = o * v_scale
                o = o / 127
                o = o / score_sum.reshape((heads, q_s, 1))
            else:
                o_true = o_true.view(heads, q_s, embedv)
                o_true = torch.permute(o_true, (1, 0, 2)).contiguous()
            o = o.view(heads, q_s, embedv)
            o = torch.permute(o, (1, 0, 2)).contiguous()
            if out is None:
                out = o
                if not self.fav3:
                    out_true = o_true
            else:
                out = torch.cat((out, o), 0)
                if not self.fav3:
                    out_true = torch.cat((out_true, o_true), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        # golden data

        if self.is_int8_flag:
            ans_concat = ans_concat.view(q_ntokens, heads * embedv)
            ans_concat_true = ans_concat_true.view(q_ntokens, heads * embedv)
            self.golden_out = ans_concat
            self.golden_out_true = ans_concat_true
        else:
            out = out.view(q_ntokens, heads * embedv)
            self.golden_out = out.to(self.data_type)
            out_true = out_true.view(q_ntokens, heads * embedv)
            self.golden_out_true = out_true.to(torch.float32)

        if self.long_seq:
            self.max_seq = 128
            self.mask = self.mask_generator.gen_mask(self.batch, self.heads, self.data_type, self.mask_type, 0, False, 0)
            self.in_tensors[self.mask_id] = self.mask

    @staticmethod
    def gen_seq_len(batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)
        out = out.flatten().to(torch.float32)
        out_len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info(f"maxDiff {max_diff}")
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / out_len)
        logging.info("5/1000 Accuracy is %f",  1 - float(strict_error_count) / out_len)
        if self.data_type == torch.bfloat16:
            logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[2])
        else:
            logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[0])
        calc_times = self.heads * self.max_seq + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else :
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        elif self.data_type == torch.float16:
            if calc_times < 2048:
                error = 2**(-8)
            else :
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        else :
            if calc_times < 2048:
                error = 2**(-11)
            elif calc_times >= 2048 and calc_times < 16384:
                error = 2**(-10)
            else:
                error = 2**(-14)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res

    def kvcache(self, newkv, cache_in):
        cache_out = cache_in.clone()
        token_offset = self.kv_seqLen
        seqlen = self.q_seqLen
        #kvcache golden
        prefix_ntokens = 0
        for i in range(self.batch):
            for j in range(seqlen[i]):
                cache_out[int(self.layer_id[0])][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]
        return cache_out
    # 合轴成2维
    def process_qkv(self, q):
        out = q.clone()
        # BSND:          
        if len(q.shape) == 3:
            dim0, dim1, dim2 = q.shape
            out = out.contiguous().view(dim0, dim1*dim2)
        elif len(q.shape) == 4:
            dim0, dim1, dim2, dim3 = q.shape
            out = out.contiguous().view(dim0*dim1, dim2*dim3)
        return out
        
    def process_kvcache(self, k):
        # pa encoder 将2维的kvcache转换成3维的
        if k.dim() == 2:
            k = k.view(k.shape[0], self.kv_head, k.shape[1] // self.kv_head)
        return k

    def process_before_golden(self):
        # 处理mask
        
        # 为了使golden函数算子golden一致，提供它需要的变量
        self.q = self.in_tensors[self.query_id]
        if not self.bnsd:
            self.q = self.process_qkv(self.q) # q在golen计算过程中是2维的
        
        if self.kcache_id == -1:
            self.kcache_id = self.key_id
        if self.vcache_id == -1:
            self.vcache_id = self.value_id

        # qscale
        if not self.pa_encoder:
            self.q *= self.qscale
        if self.pa_encoder: # [nTokens, head_num, head_size]
            kcache_nd = self.in_tensors[self.kcache_id]
            if self.mlaVHeadSize > 0:
                vcache_nd = kcache_nd
            else:
                vcache_nd = self.in_tensors[self.vcache_id]
        elif self.is_910b:
            kcache_nd = self.in_tensors[self.kcache_id]
            if len(kcache_nd.shape) == 3: #  动态batch[batch, maxSeqLen, hiddenSize]
                kcache_nd = kcache_nd.unsqueeze(0)
            vcache_nd = self.in_tensors[self.vcache_id]
            if len(vcache_nd.shape) == 3: #  动态batch[batch, maxSeqLen, hiddenSize]
                vcache_nd = vcache_nd.unsqueeze(0)
        else: # 310p场景需要调整kvcache为nd
            kcache_nd = convert_nz_to_nd(self.in_tensors[self.kcache_id])
            vcache_nd = convert_nz_to_nd(self.in_tensors[self.vcache_id])
        # 非bypass，且非pa encoder 情况需要kvcache
        if not self.by_pass and not self.pa_encoder:
            key = self.in_tensors[self.key_id] # k在kvcache golen计算过程中是2维的
            self.k = self.kvcache(self.process_qkv(key), kcache_nd) # golden里的self.k是kcache
            value = self.in_tensors[self.value_id]# v在kvcache golen计算过程中是2维的
            self.v = self.kvcache(self.process_qkv(value), vcache_nd) # golden里的self.v是vcache                       
        else: # pa encoder 将2维的kvcache转换成4维的
            self.q = self.process_qkv(self.q)
            kcache_nd = self.process_qkv(kcache_nd)
            self.k = kcache_nd.view(1, 1, kcache_nd.shape[-2], kcache_nd.shape[-1])
            vcache_nd = self.process_qkv(vcache_nd)
            self.v = vcache_nd.view(1, 1, vcache_nd.shape[-2], vcache_nd.shape[-1])
            self.in_tensors[self.query_id] = self.q
            self.in_tensors[self.key_id] = self.k
            self.in_tensors[self.value_id] = self.v
        if self.bnsd:
            # 将bnsd格式的qkv转换成bsnd的
            self.q = torch.permute(self.q, (0, 2, 1, 3))
            # self.q = self.process_qkv(self.q) # q在golen计算过程中是2维的
            self.k = torch.permute(self.k, (0, 1, 3, 2, 4)).contiguous()
            self.k = self.k.view(self.k.shape[0], self.k.shape[1], self.k.shape[2], self.k.shape[3]*self.k.shape[4])
            self.v = torch.permute(self.v, (0, 1, 3, 2, 4)).contiguous()
            self.v = self.v.view(self.v.shape[0], self.v.shape[1], self.v.shape[2], self.v.shape[3]*self.v.shape[4])

        # 动态batch
        if self.dynamic_batch:
            self.batch_state = self.in_tensors[self.batch_status_id]

    @staticmethod
    def group_mm_torch(heads, group_num, A, B, dtype=torch.float32):
        need_cast = False
        if isinstance(A, np.ndarray):
            A = torch.tensor(A)
            need_cast = True
        if isinstance(B, np.ndarray):
            B = torch.tensor(B)
            need_cast = True
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(
                A[i * group_head: (i + 1) * group_head, :, :].to(dtype), B[i:(i + 1), :, :].to(dtype))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        if need_cast:
            score = score.numpy()
        return score

    def golden_calc(self):
        golden_out = self.golden_out.clone().detach().requires_grad_(True)
        if self.pa_encoder:
            golden_out = golden_out.view(self.q_ntokens, self.head_num, self.head_size)
        return [golden_out.detach()]

    def golden_compare(self, out_tensors, golden_tensors):
        result_single = self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.005, 0.005])
        if self.is_int8_flag:
            result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
            return (result_double or result_single)
        else:
            result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
            return (result_double or result_single)

class SelfAttentionGolden:
    def __init__(
        self,
        dynamic_batch=False,
        batch_state=None,
        is_mask=False,
        is_decoder=False,
        is_alibi=False,
        alibi_dim=4,
        batch=0,
        kv_head=1,
        heads=1,
        embeddim=128,
        embeddimv=0,
        max_seq=2048,
        kv_seqLen=[],
        is_clamp=0,
        clamp_min=0,
        q_seqLen=[],
        clamp_max=0,
        data_type=torch.float16,
        op_type=0,
        mask_type=0,
        no_cache=False,
        long_seq=False,
        is_triu_mask=False,
        is_multi_layer=False,
        is_sqrt=False,
        left_align=False,
        scaleType=ScaleType.SCALE_TOR.value,
        fav3=False,
        is_bypass=False,
        is_pa_encoder=False,
        is_bnsd=False,
        is_mask_compress=False,
    ):
        # def __init__(self, op_params: dict[str, Any]):
        # 每个tensor的id表示它对应第几个输入，-1代表这个tensor没有被传入
        self.query_id = -1
        self.key_id = -1
        self.value_id = -1
        self.kcache_id = -1
        self.vcache_id = -1
        self.mask_id = -1
        self.token_offset_id = -1
        self.seqlen_id = -1
        self.layerid_id = -1
        self.batch_status_id = -1
        self.slopes_id = -1
        self.qk_descale_id = -1
        self.qk_offset_id = -1
        self.vpv_descale_id = -1
        self.vpv_offset_id = -1
        self.p_scale_id = -1
        self.logn_id = -1
        self.tensor_ids = [] # [self.query_id, self.key_id, ... , self.logn_id] 方便遍历
        self.in_tensors = []  # 需要两套intensor
        # self.intensors_for_golden = []
        # golden计算需要的参数，命名和算子golden同步
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state  # 需要通过tensor shape推理
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.alibi_dim = alibi_dim
        self.batch = batch  # 需要通过tensor shape推理
        self.kv_head = kv_head
        self.heads = heads
        self.q_ntokens = 0
        self.embeddim = embeddim  # 需要通过tensor shape推理
        self.embeddimv = embeddimv  # 需要通过tensor shape推理
        self.max_seq = max_seq
        self.q_seqLen = q_seqLen  # 在gen_seqlen_and_token_offset函数中获取
        
        self.kv_seqLen = kv_seqLen  # 在gen_seqlen_and_token_offset函数中获取
        
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.previous_type = None
        self.isbf16 = False
        if self.data_type == torch.bfloat16:
            self.isbf16 = True
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_triu_mask = is_triu_mask
        self.is_multi_layer = is_multi_layer
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = 1
        self.window_size = 0
        self.cache_type = 0
        self.online = False
        self.is_int8_flag = False
        self.layer = 0
        # 一些加速库特有的参数
        self.mlaVHeadSize = 0
        self.by_pass = is_bypass
        self.pa_encoder = is_pa_encoder
        self.bnsd = is_bnsd
        self.mask_compress = is_mask_compress
        self.compress_type = 0  # 和fa MaskType对应
        self.is_910b = True
        self.qscale = 1
        self.layer_id = [0]
        self.is_tensor_customize = []
        self.high_precision = False
        # 用SelfAttentionMaskGen处理mask
        self.mask_generator = None

    def load_from_op_params(self, op_params: Dict[str, Any]):
        for op_k, op_v in op_params.items():
            if hasattr(self, op_k):
                self.op_k = op_v

        if "batchRunStatusEnable" in op_params and op_params["batchRunStatusEnable"] is True:
            self.dynamic_batch = op_params["batchRunStatusEnable"]
        if "inputLayout" in op_params and op_params["inputLayout"] == 1:
            self.bnsd = True
        if "maskType" in op_params:
            if op_params["maskType"] != 0:
                self.is_mask = True
                if op_params["maskType"] in [2, 4, 5, 6]:
                    self.is_alibi = True
                if op_params["maskType"] in [3, 4, 5, 6]:
                    self.mask_compress = True
                    self.compress_type = op_params["maskType"]
                if op_params["maskType"] == 6:
                    self.left_align = True
                if op_params["maskType"] == 5:
                    self.is_sqrt = True
        if "isTriuMask" in op_params and op_params["isTriuMask"] == 1:
            self.mask_compress = True
            self.is_triu_mask = True
            if "headNum" in op_params:
                self.heads = op_params["headNum"]
                if "kvHeadNum" in op_params.keys() and op_params["kvHeadNum"] > 0:
                    self.kv_head = op_params["kvHeadNum"]
                else:
                    self.kv_head = self.heads
            else:
                logging.error("error! No headNum")

        if "headNum" in op_params.keys():
            self.heads = op_params["headNum"]
            if "kvHeadNum" in op_params.keys() and op_params["kvHeadNum"] > 0:
                self.kv_head = op_params["kvHeadNum"]
            else:
                self.kv_head = self.heads
        else:
            logging.error("error! No headNum")
        if "clampType" in op_params.keys() and op_params["clampType"] == 1:
            self.is_clamp = True
            self.clamp_min = op_params["clampMin"]
            self.clamp_max = op_params["clampMax"]
        if "qScale" in op_params.keys():
            self.qscale = op_params["qScale"]
        if "qkScale" in op_params.keys():
            self.tor = op_params["qkScale"]
        if "kernelType" in op_params.keys() and op_params["kernelType"] == 1:
            self.high_precision = True
        if "scaleType" in op_params.keys() and op_params["scaleType"] == 1:
            self.scaleType = op_params["scaleType"]
        if "mlaVHeadSize" in op_params.keys() and op_params["mlaVHeadSize"] > 0:
            self.mlaVHeadSize = op_params["mlaVHeadSize"]
        if "windowSize" in op_params.keys() and op_params["windowSize"] > 0:
            self.window_size = op_params["windowSize"]
        if "cacheType" in op_params.keys() and op_params["cacheType"] > 0:
            self.cache_type = op_params["cacheType"]
        if "inputLayout" in op_params.keys() and op_params["inputLayout"] > 0:
            self.bnsd = True
        if "quantType" in op_params.keys() and op_params["quantType"] > 0:
            self.fav3 = True
            if op_params["quantType"] == 3:
                self.online = True
            if "outDataType" in op_params.keys():
                if op_params["outDataType"] == 1:
                    self.data_type = torch.float16
                if op_params["outDataType"] == 27:
                    self.data_type = torch.bfloat16
        if "calcType" in op_params.keys():
            if op_params["calcType"] == 3:
                self.pa_encoder = True
            if op_params["calcType"] == 2:
                self.is_decoder = True
        if "maskType"  in op_params.keys():
            self.mask_type = op_params["maskType"]
        if "batch" in op_params.keys():
                self.batch = op_params["batch"]
        index = 0
        if self.pa_encoder:
            self.query_id, self.key_id, self.value_id = range(index, index+3)
            index += 3
            if self.is_mask:
                self.mask_id = index
                index += 1
            self.seqlen_id = index
            index += 1
            if self.is_alibi:
                self.slopes_id = index
                index += 1
            if self.scaleType == 1:
                # logN缩放
                self.logn_id = index
                index += 1


    def __str__(self):
        intensors_shapes = [t.shape for t in self.in_tensors]
        no_tensor_dict = self.__dict__.copy()
        no_tensor_dict.pop("in_tensors")
        return "intensors_shapes: " + str(intensors_shapes) + "\n" + str(no_tensor_dict)

    def get_cache_info(self, shape, head_num):
        if self.bnsd:
            if self.is_910b:  # [layer, batch, head_num, seq_len, head_size]
                self.batch, self.max_seq, self.embeddimv = shape[1], shape[3], shape[4]
            elif  len(shape) == 5:  # [layer, batch*head_num, head_size / 16, kv_max_seq, 16]
                self.batch, self.max_seq, self.embeddimv = shape[0], shape[3], (shape[2] * 16) // head_num
            else:
                logging.error("wrong cache shape!")
        # BSND:
        elif len(shape) == 4:  # [layerNum, batch, maxSeqLen, khiddenSize]
            self.batch, self.max_seq, self.embeddimv =  shape[1], shape[2], shape[3] // head_num
        elif len(shape) == 3:
            if self.pa_encoder:  # [nTokens, head_num, head_size]
                self.embeddimv = shape[2]
            else:  # [batch, maxSeqLen, hiddenSize]
                self.batch, self.max_seq, self.embeddimv =shape[0],shape[1],shape[2] // head_num
        elif len(shape) == 5:  # [layer, batch, hiddenSize/16, maxSeqLen, 16]
            self.batch, self.max_seq, self.embeddimv = shape[1], shape[3], (shape[2] * 16) // head_num
        elif len(shape) == 2:
            self.embeddimv = shape[1] // head_num

    def prepare_in_tensors(self, in_tensors: List[torch.tensor]):
        if self.seqlen_id != -1:
            self.q_seqLen = in_tensors[self.seqlen_id]
            self.batch = in_tensors[self.seqlen_id].shape[0]
        if self.pa_encoder:
            self.kv_seqLen = self.q_seqLen
        if self.bnsd:
            self.q_ntokens, self.embeddim = get_q_dims_BNSD(in_tensors[self.query_id].shape, self.heads)
        else:
            self.q_ntokens, self.embeddim = get_q_info(in_tensors[self.query_id].shape, self.heads)
        if self.mlaVHeadSize > 0:
            self.embeddimv = self.mlaVHeadSize
        else:
            self.get_cache_info(in_tensors[self.vcache_id].shape, self.kv_head)
        self.in_tensors = in_tensors
        self.max_seq = max(max(self.q_seqLen), max(self.kv_seqLen))
        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if self.mask_generator == None:
            self.mask_generator = SelfAttentionMaskGen(batch=self.batch, heads=self.heads,data_type=self.data_type,
                mask_type=self.mask_type, max_seq=self.max_seq, window_size=self.window_size,
                long_seq=self.long_seq, kv_seqLen=self.kv_seqLen, dynamic_batch=self.dynamic_batch, 
                is_decoder=self.is_decoder)

        if self.is_mask:
            # golden mask
            self.mask = self.in_tensors[self.mask_id]
            if list(self.mask.shape) == [128, 128] and self.mask_type == 1: # 128 compress mask
                self.long_seq = True
                self.mask_compress = True
                self.mask = self.mask_generator.gen_mask(self.batch, self.heads, self.data_type, self.mask_type, self.window_size, self.mask_compress, self.cache_type)
                self.in_tensors[self.mask_id] = self.mask

    def gen_out_tensor(self):
        out_generator = SelfAttentionGenOutTensor(self)
        out_generator.process_before_golden()
        out_generator.gen_out_tensor()
        return out_generator.golden_calc()

    @staticmethod
    def group_matmul(heads, group_num, A, B):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                    B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        logging.debug(score.shape)
        return score

    def calc_expect_func(self, batch, seqlen, heads, embed, group_num=32):
        is_mask = False
        variate_seq = False
        is_decoder = False
        max_seq = 2048
        src_type = 'float16'
        fp32 = True
        logging.debug(f"group_num: {group_num}")
        logging.debug("q_seq is:")
        if is_decoder:
            q_seqlen, q_ntokens = SelfAttentionGenOutTensor.gen_seq_len(batch, [1])
            kv_seqlen, kv_ntokens = SelfAttentionGenOutTensor.gen_seq_len(batch, seqlen)
        else:
            q_seqlen, q_ntokens = SelfAttentionGenOutTensor.gen_seq_len(batch, seqlen)
            kv_seqlen, kv_ntokens = q_seqlen, q_ntokens   # crossattention时，q_seqlen != k_seqlen

        max_s = np.max(q_seqlen)
        ntokens2 = (q_seqlen * kv_seqlen).sum()

        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
        mask = np.ones(shape=(1, max_s, max_s)).astype(np.float16)  # 使用当前最大seqlen生成mask
        mask = np.triu(mask, 1)
        mask *= -10000.0
        logging.debug(mask)

        q_offset = 0
        k_offset = 0
        v_offset = 0

        s = None
        _p = None
        out = None

        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.reshape(kv_s, group_num, embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.reshape(kv_s, group_num, embed)
            v_slice = np.transpose(v_slice, (1, 0, 2))
            score = __class__.group_matmul(heads, group_num, q_slice, k_slice_t)
            if s is None:
                s = score.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)

            tor = np.float16(1.0 / math.sqrt(1.0 * embed))
            score = score * tor
            if is_mask:
                score = score + mask[:, :q_s, :kv_s]
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score.astype(np.float32))
            if not fp32:
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                out_sub = __class__.group_matmul(heads, group_num, p, v_slice)
            else:
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)
                out_sub = __class__.group_matmul(heads, group_num, p, v_slice)
                out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

            out_sub = out_sub.reshape(heads, q_s, embed)
            out_sub = np.transpose(out_sub, (1, 0, 2))
            out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = np.concatenate((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        logging.debug("==> data generate finished!")

        q = q.astype(src_type).reshape(-1, heads, 128)
        k = k.astype(src_type).reshape(-1, group_num, 128)
        v = v.astype(src_type).reshape(-1, group_num, 128)
        mask = mask.astype(src_type).reshape(max_s, max_s)
        q_len = q_seqlen.astype(np.int32)
        out = out.astype(src_type).reshape(-1, heads, 128)

        ret_data = q, k, v, q_len, out
        return ret_data
    
    def tensor_customize(self, shapes, datatype, format, data_gen_ranges):
        low, high = map(float, data_gen_ranges.split(','))
        if low < -50 or high > 50:
            low = float(-1)
            high = float(1)
            logging.info(f"data gen range is too big, use [{low}, {high}) instead.")
        tensor = None
        if datatype in [torch.float32, torch.float16,  torch.bfloat16]:
            tensor = (high - low) * torch.rand(shapes, dtype=datatype) + low
        elif datatype in [torch.int32, torch.int16, torch.int8]:
            tensor = torch.randint(low=int(low), high=int(high), size=shapes, dtype=datatype)
        elif datatype == torch.bool:
            tensor = torch.randint(0, 2, size=shapes, dtype=datatype)
        return tensor.npu()
