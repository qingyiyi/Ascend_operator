#
# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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
import torch.nn.functional as F
import os
from ctypes import CDLL
import math
import sys
import shutil
import logging
import re
from enum import Enum, IntEnum
from functools import reduce
from en_dtypes import hifloat8

from self_attention_golden import SelfAttentionGolden, SelfAttentionGenOutTensor

dtype_dict = {"float": torch.float32, "float16": torch.float16, "int8": torch.int8, "int32": torch.int32, "uint8": torch.uint8,
              "int16": torch.int16, "uint16": torch.int16, "uint32": torch.int32, "int64": torch.int64, "uint64": torch.int64,
              "double": torch.double, "bool": torch.bool, "complex64": torch.complex64, "complex128": torch.complex128, "bf16": torch.bfloat16,
              "hifloat8": torch.bits8, "float8_e5m2": torch.float8_e5m2, "float8_e4m3fn": torch.float8_e4m3fn}

format_dict = {"undefined": -1, "nchw": 0, "nhwc": 1, "nd": 2, "nc1hwc0": 3,
                "fractal_z": 4, "nc1hwc0_c04": 12, "hwcn": 16, "ndhwc": 27,
                "fractal_nz": 29, "ncdhw": 30, "ndc1hwc0": 32, "fractal_z_3d": 33}

class OpTypes(Enum):
    NA = 0 # new standard is not available
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9

def get_precision_and_eb_threshold(op_type, dtype, compute_num):
    precision_threshold = 0
    eb_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT]:
        if dtype in [torch.int8]:
            precision_threshold = 1
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.float16]:
            if compute_num != -1 and compute_num >= 2048:
                precision_threshold = 2**(-7)
            else:
                precision_threshold = 2**(-8)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            if compute_num != -1 and compute_num >= 2048:
                precision_threshold = 2**(-6)
                eb_threshold = 2**(-6)
            else:
                precision_threshold = 2**(-7)
                eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-11)
            eb_threshold = 2**(-14)
            if compute_num != -1:
                if compute_num < 2048:
                    precision_threshold = 2**(-11)
                elif compute_num < 16384:
                    precision_threshold = 2**(-10)
                else:
                    precision_threshold = 2**(-9)
    if op_type in [OpTypes.COMPUTE_FLOAT_HIGH_PRECISION]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-11)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-8)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-14)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.VECTOR_FUSION]:
        if dtype in [torch.float16]:
            precision_threshold = 2**(-9)
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2**(-7)
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            precision_threshold = 2**(-12)
            eb_threshold = 2**(-14)
    if op_type in [OpTypes.CV_FUSION]:
        precision_threshold = 1022 #最大相对误差10/平均相对误差2/均方根误差2
        if dtype in [torch.float16]:
            eb_threshold = 2**(-10)
        if dtype in [torch.bfloat16]:
            eb_threshold = 2**(-7)
        if dtype in [torch.float32]:
            eb_threshold = 2**(-14)
    logging.debug(f"op_type: {op_type}, dtype: {dtype}, precision_threshold: {precision_threshold}, eb_threshold: {eb_threshold}")
    return precision_threshold, eb_threshold

def get_soc_version():
    device_name = torch.npu.get_device_name()
    if (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name, re.I):
        soc_version = "Ascend910B"
    elif re.search("Ascend950", device_name, re.I):
        soc_version = "Ascend950"
    elif re.search("Ascend310P", device_name, re.I):
        soc_version = "Ascend310P"
    elif re.search("Ascend950", device_name, re.I):
        soc_version = "Ascend950"
    elif (re.search("Ascend910ProB", device_name, re.I) or re.search("Ascend910B", device_name, re.I) or
    re.search("Ascend910PremiumA", device_name, re.I) or re.search("Ascend910ProA", device_name, re.I) or
    re.search("Ascend910A", device_name, re.I)):
        soc_version = "Ascend910A"
    elif (re.search("Ascend310B", device_name, re.I)):
        soc_version = "Ascend310B"
    elif (re.search("Ascend950", device_name, re.I)):
        soc_version = "Ascend950"
    else:
        logging.error("device_name {} is not supported".format(device_name))
        quit(1)
    return soc_version

def numpy_bfloat16():
    try:
        import bfloat16ext
        return bfloat16ext.bfloat16
    except ImportError:
        import tensorflow
        return tensorflow.bfloat16.as_numpy_dtype

class TensorBinFile:
    ATTR_VERSION = "$Version"
    ATTR_END = "$End"
    ATTR_OBJECT_LENGTH = "$Object.Length"
    ATTR_OBJECT_COUNT = "$Object.Count"
    ATTR_OBJECT_PREFIX = "$Object."

    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []
        self.__parse_bin_file()

    def get_tensor(self):
        if self.dtype == 0:
            dtype = np.float32
        elif self.dtype == 1:
            dtype = np.float16
        elif self.dtype == 2:  # int8
            dtype = np.int8
        elif self.dtype == 3:  # int32
            dtype = np.int32
        elif self.dtype == 9:  # int64
            dtype = np.int64
        elif self.dtype == 12:
            dtype = np.bool8
        elif self.dtype == 27:  # bfloat16
            dtype = torch.bfloat16
        else:
            logging.error("error, unsupport dtype:", self.dtype)
            pass
        if dtype == torch.bfloat16:
            tensor = torch.frombuffer(bytearray(self.obj_buffer), dtype=dtype)
        else:
            tensor = torch.tensor(np.frombuffer(self.obj_buffer, dtype=dtype))
        tensor = tensor.view(self.dims).npu()
        return torch_npu.npu_format_cast(tensor, self.format)

    def __parse_bin_file(self):
        end_str = f"{TensorBinFile.ATTR_END}=1"
        with open(self.file_path, "rb") as fd:
            file_data = fd.read()
            begin_offset = 0
            for i in range(len(file_data)):
                if file_data[i] == ord("\n"):
                    line = file_data[begin_offset: i].decode("utf-8")
                    begin_offset = i + 1
                    fields = line.split("=")
                    attr_name = fields[0]
                    attr_value = fields[1]
                    if attr_name == TensorBinFile.ATTR_END:
                        self.obj_buffer = file_data[i + 1:]
                        break
                    elif attr_name.startswith("$"):
                        self.__parse_system_atrr(attr_name, attr_value)
                    else:
                        self.__parse_user_attr(attr_name, attr_value)
                        pass

    def __parse_system_atrr(self, attr_name, attr_value):
        if attr_name == TensorBinFile.ATTR_OBJECT_LENGTH:
            self.obj_len = int(attr_value)
        elif attr_name == TensorBinFile.ATTR_OBJECT_PREFIX:
            pass

    def __parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            for i in range(len(self.dims)):
                self.dims[i] = int(self.dims[i])
        logging.debug(f"parse user attr finished, dtype: {self.dtype}, format: {self.format}, dims: {self.dims}")

class DataGen():
    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        pass

    @staticmethod
    def zero(shape, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor.npu()
        '''
        if datatype == "float8_e4m3fn":
            data = torch.zeros(shape, dtype=torch.float16).to(dtype=torch.float8_e4m3fn).npu()
        elif datatype == "float8_e5m2":
            data = torch.zeros(shape, dtype=torch.float16).to(dtype=torch.float8_e5m2).npu()
        elif datatype == "hifloat8":
            data = torch.zeros(shape, dtype=torch.int8).view(torch.bits8).npu()
        else:
            data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
        return torch_npu.npu_format_cast(data, format_dict[format])

    @staticmethod
    def one(shape, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor.npu()
        '''
        data = torch.ones(shape, dtype=dtype_dict[datatype]).npu()
        return torch_npu.npu_format_cast(data, format_dict[format])

    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor.npu()
        '''
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        data = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype]).npu()
        return torch_npu.npu_format_cast(data, format_dict[format])

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        '''
        Returns:
            torch.Tensor.npu()
        '''
        # default: generate random values
        return DataGen.random(shapes[i], datatype, format, data_gen_ranges, op_params)

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        return bin.get_tensor()

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        pass

    @staticmethod
    def golden(in_tensors, op_params) -> [torch.Tensor]:
        '''
        Returns:
            list of torch.Tensor.cpu()
        '''
        pass

    @staticmethod
    def performance_threshold(op_params):
        return {"SetupTime(us)": 1000000, "ExecuteTime(us)": 1000000, "SyncTime(us)": 1000000}

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.NA

# Note:
# Every class should use opname as class name and be inherited from class DataGen. Currently you only need to override function customize and golden.
# 1. Generally, function golden must be overrided.
# 2. Overriding function customize is optional; when function zero/one/random can not satisfy your need, override it to generate your own defined data.
# 3. Function customize and golden must be staticmethods.
# 4. Golden data should not be generated from npu; other function can use npu to generate input data and output data.

class CumsumOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if in_tensors[0].dtype == torch.bfloat16:
            golden_result = np.cumsum(in_tensors[0].float(), axis = json_data['axes'][0])
        else:
            golden_result = np.cumsum(in_tensors[0], axis = json_data['axes'][0])
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class FillOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if json_data["withMask"]:
            golden_result = in_tensors[0].masked_fill_(in_tensors[1].bool(),json_data["value"][0]);
        else:
            golden_result = torch.full(json_data["outDim"],json_data["value"][0],dtype=torch.float16)
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class ConcatOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        axis_num = json_data["concatDim"] if json_data["concatDim"] >= 0\
            else json_data["concatDim"] + len(in_tensors[0].size())
        golden_result = torch.cat(in_tensors, axis=axis_num)
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class DynamicNTKOperation(DataGen):
    OUTPUT_DATA_TYPE_FP16 = 1
    OUTPUT_DATA_TYPE_BF16 = 27
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            logging.debug(DynamicNTKOperation.in_tensors[i].shape)
            return DynamicNTKOperation.in_tensors[i]
        ntokens = shapes[0][0]
        batch = shapes[1][0]
        head_size_half = shapes[1][1]
        if batch > 1:
            seqlens = np.random.rand(batch-1)
            ratio = sum(seqlens)/ntokens
            seqlens = seqlens / ratio
            seqlens = seqlens.astype(int)
            seqlens = np.append(seqlens, ntokens-sum(seqlens))
        else:
            seqlens = np.array([ntokens])
        max_seq_len = max(seqlens)
        positionIds = np.random.randint(0, max_seq_len, size=ntokens)
        inv_freq = np.random.rand(batch, head_size_half)
        positionIds = torch.from_numpy(positionIds).npu().to(torch.int32)
        seqlens = torch.from_numpy(seqlens).npu().to(torch.int32)
        inv_freq = torch.from_numpy(inv_freq).npu().to(torch.float32)
        ret_data = positionIds, inv_freq, seqlens
        DynamicNTKOperation.in_tensors = ret_data
        return DynamicNTKOperation.in_tensors[0]
    @staticmethod
    def golden(in_tensors, op_params):
        json_param = json.loads(op_params)
        position_ids = in_tensors[0].cpu().to(torch.int32)
        num_tokens = position_ids.shape[0]
        seqlens = in_tensors[2].cpu().to(torch.int32)
        batch_num = seqlens.shape[0]
        inv_freq = in_tensors[1].cpu().to(torch.float32)
        dim = inv_freq.shape[1] * 2
        sin_out = torch.zeros([num_tokens, dim], dtype=torch.float32)
        cos_out = torch.zeros([num_tokens, dim], dtype=torch.float32)
        off = 0
        for batch_id in range(batch_num):
            pos_len = seqlens[batch_id]
            freqs = torch.einsum('i,j->ij', position_ids[off:off + pos_len].to(torch.float32), inv_freq[batch_id])
            emb = torch.cat((freqs, freqs), dim = -1)
            cos_out[off:off + pos_len, :] = emb.cos()
            sin_out[off:off + pos_len, :] = emb.sin()
            off += pos_len
        if json_param['outputType'] == DynamicNTKOperation.OUTPUT_DATA_TYPE_FP16:
            # print([sin_out.to(torch.float16), cos_out.to(torch.float16)])
            return [sin_out.to(torch.float16), cos_out.to(torch.float16)]
        else:
            # print([sin_out.to(torch.bfloat16), cos_out.to(torch.bfloat16)])
            return [sin_out.to(torch.bfloat16), cos_out.to(torch.bfloat16)]
    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class SplitOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if "splitSizes" in json_data and len(json_data["splitSizes"]) != 0:
            if json_data['splitNum'] == 3:
                x = in_tensors[0]
                y = torch.split(x, json_data["splitSizes"], dim = json_data['splitDim'])
                return [y[0], y[1], y[2]]
            elif json_data['splitNum'] == 2:
                x = in_tensors[0]
                y = torch.split(x, json_data["splitSizes"], dim = json_data['splitDim'])
                return [y[0], y[1]]
            else:
                return [torch.zeros_like(x) for x in in_tensors]
        else:
            golden_result = torch.chunk(in_tensors[0], chunks=json_data['splitNum'], dim=json_data['splitDim'])
        return golden_result

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class SetValueOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        golden_result = [in_tensors[0].clone(), in_tensors[1].clone()]
        if len(json_data["starts"]) == 3:
            golden_result[0][json_data['starts'][0]:json_data['ends'][0],json_data['starts'][1]:json_data['ends'][1],json_data['starts'][2]:json_data['ends'][2]].copy_(in_tensors[1])
        return golden_result

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class LinearOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i == 0:
            MatmulCommon.reset()
            transpose_a = MatmulCommon.get_param_value(op_params, "transposeA", False)
            return MatmulCommon.gen_input(shapes, i, datatype, data_gen_ranges, transpose_a)
        if i == 1:
            transpose_b = MatmulCommon.get_param_value(op_params, "transposeB", True)
            return MatmulCommon.gen_weight(shapes, i, datatype, format, data_gen_ranges, transpose_b)
        if i == 2:
            has_bias = MatmulCommon.get_param_value(op_params, "hasBias", True)
            en_accum = MatmulCommon.get_param_value(op_params, "enAccum", False)
            if has_bias:
                return torch_npu.npu_format_cast(MatmulCommon.gen_bias(shapes, i, datatype, data_gen_ranges), format_dict[format])
            if en_accum:
                return torch_npu.npu_format_cast(MatmulCommon.gen_accum(shapes, i, datatype, data_gen_ranges), format_dict[format])
            out_data_type = MatmulCommon.get_param_value(op_params, "outDataType", -1)
            quantMode = MatmulCommon.get_param_value(op_params, "quantMode", 0)
            return torch_npu.npu_format_cast(MatmulCommon.gen_deq(shapes, i, data_gen_ranges, out_data_type, quantMode), format_dict[format])
        if i == 3:
            out_data_type = MatmulCommon.get_param_value(op_params, "outDataType", -1)
            quantMode = MatmulCommon.get_param_value(op_params, "quantMode", 0)
            if quantMode == 2:
                return torch_npu.npu_format_cast(MatmulCommon.gen_pertoken_scale(shapes, i, data_gen_ranges, out_data_type), format_dict[format])
            return torch_npu.npu_format_cast(MatmulCommon.gen_deq(shapes, i, data_gen_ranges, out_data_type, quantMode), format_dict[format])

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        en_accum = MatmulCommon.get_param_value(op_params, "enAccum", False)
        if en_accum:
            output_tensor_list[0] = input_tensor_list[2]

    @staticmethod
    def golden(in_tensors, op_params):
        out_data_type = MatmulCommon.get_param_value(op_params, "outDataType", -1)
        matmultype = MatmulCommon.get_param_value(op_params, "matmulType", 0)
        quantMode = MatmulCommon.get_param_value(op_params, "quantMode", 0)

        x = MatmulCommon.input_golden.cpu() if MatmulCommon.input_golden is not None else None
        weight = MatmulCommon.weight_golden.cpu() if MatmulCommon.weight_golden is not None else None
        bias = MatmulCommon.bias_golden.cpu() if MatmulCommon.bias_golden is not None else None
        deq_scale = MatmulCommon.deq_golden.cpu() if MatmulCommon.deq_golden is not None else None
        accum = MatmulCommon.accum_golden.cpu() if MatmulCommon.accum_golden is not None else None
        pertoken_scale = MatmulCommon.pertoken_scale_golden.cpu() if MatmulCommon.pertoken_scale_golden is not None else None
        
        if out_data_type == -1:
            if bias is not None and bias.dtype == torch.bfloat16:
                x = x.to(torch.float64)
                weight = weight.to(torch.float64)
            else:
                x = x.to(torch.float)
                weight = weight.to(torch.float)
            if bias is not None and x.dtype != torch.float16:
                bias = bias.to(torch.float)
        else:
            x = x.to(torch.int32)
            weight = weight.to(torch.int32)
        if matmultype == 1:
            if len(weight.shape) == 2:
                golden_result = torch.einsum('mbk,kn->mbn', x, weight)
            else:
                golden_result = torch.einsum('mbk,bkn->mbn', x, weight)
            if bias is not None:
                bias_expanded = bias.unsqueeze(0)
                golden_result = golden_result + bias_expanded
        elif len(x.shape) == 3 and len(weight.shape) == 3:
            output_list = []
            if quantMode == 2:
                pertoken_scale = pertoken_scale.unsqueeze(-1)
            for i in range(x.shape[0]):
                x_i = x[i:i + 1, :].squeeze(0)
                weight_i = weight[i:i + 1, :].squeeze(0)
                output_i = torch.matmul(x_i, weight_i)
                if bias is not None and bias.dtype == torch.bfloat16:
                    output_i = output_i.to(torch.bfloat16)
                if bias is not None:
                    output_i = output_i.to(bias.dtype) + bias[i:i + 1, :]
                if deq_scale is not None:
                    if quantMode == 2:
                        output_i = output_i * deq_scale
                    else:
                        output_i = output_i * deq_scale[i:i + 1, :]
                if pertoken_scale is not None and quantMode == 2:
                    output_i = output_i * pertoken_scale
                if accum is not None:
                    output_i = output_i + accum[i:i + 1, :].squeeze(0)
                output_list.append(output_i)
            golden_result = torch.stack(output_list, dim=0)
        else:
            golden_result = torch.matmul(x, weight)
            if bias is not None:
                if bias.dtype == torch.bfloat16 and get_soc_version() != "Ascend950":
                    golden_result = golden_result.to(torch.bfloat16)
                golden_result = golden_result.to(bias.dtype) + bias
            if deq_scale is not None:
                golden_result = golden_result * deq_scale
            if pertoken_scale is not None and quantMode == 2:
                pertoken_scale = pertoken_scale.unsqueeze(-1)
                golden_result = golden_result * pertoken_scale
            if accum is not None:
                golden_result = golden_result + accum
        if accum is None:
            if out_data_type == -1:
                if bias is not None and bias.dtype == torch.bfloat16:
                    golden_result = golden_result.to(torch.bfloat16)
                else:
                    golden_result = golden_result.to(torch.float32)
            elif out_data_type == 27:
                golden_result = golden_result.to(torch.bfloat16)
            else:
                golden_result = golden_result.to(torch.float16)
        MatmulCommon.reset()
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params):
        bin = TensorBinFile(intensor_file)
        input = bin.get_tensor()
        input_cpu = input.cpu()
        shape = input_cpu.shape
        has_bias = MatmulCommon.get_param_value(op_params, "hasBias", True)
        en_accum = MatmulCommon.get_param_value(op_params, "enAccum", False)
        out_data_type = MatmulCommon.get_param_value(op_params, "outDataType", -1)
        quantMode = MatmulCommon.get_param_value(op_params, "quantMode", 0)
        if i == 0:
            input_golden = input_cpu
            transpose_a = MatmulCommon.get_param_value(op_params, "transposeA", False)
            if transpose_a:
                input_golden = torch.transpose(input_cpu, 0, 1) if len(shape) == 2 else torch.transpose(input_cpu, 1, 2)
            MatmulCommon.input_golden = input_golden
        if i == 1:
            weight_golden = input_cpu
            if len(shape) == 4:
                weight_golden = torch.transpose(input_cpu, 1, 2)
                if shape[0] == 1:
                    weight_golden = weight_golden.reshape(weight_golden.size()[1],
                                                          weight_golden.size()[2] * weight_golden.size()[3])
                else:
                    weight_golden = weight_golden.reshape(weight_golden.size()[0], weight_golden.size()[1],
                                                          weight_golden.size()[2] * weight_golden.size()[3])
            transpose_b = MatmulCommon.get_param_value(op_params, "transposeB", True)
            if transpose_b:
                dim_num = len(weight_golden.size())
                weight_golden = torch.transpose(weight_golden, 0, 1) if dim_num == 2 else torch.transpose(weight_golden, 1, 2)
            MatmulCommon.weight_golden = weight_golden
        if i == 2:
            if has_bias:
                MatmulCommon.bias_golden = input_cpu
            elif en_accum:
                MatmulCommon.accum_golden = input_cpu
            elif out_data_type == 1 and get_soc_version() != "Ascend910A":
                deq_np = np.frombuffer(input_cpu.numpy().astype(np.uint32).tobytes(), dtype=np.float32).copy()
                deq_tensor = torch.from_numpy(deq_np)
                MatmulCommon.deq_golden = deq_tensor
            else:
                MatmulCommon.deq_golden = input_cpu

        if i == 3:
            if quantMode == 2:
                MatmulCommon.pertoken_scale_golden = input_cpu
            elif out_data_type == 1 and get_soc_version() != "Ascend910A":
                deq_np = np.frombuffer(input_cpu.numpy().astype(np.uint32).tobytes(), dtype=np.float32).copy()
                deq_tensor = torch.from_numpy(deq_np)
                MatmulCommon.deq_golden = deq_tensor
            else:
                MatmulCommon.deq_golden = input_cpu
        return input_cpu.npu()

class GroupedMatmulInplaceAddOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i == 0:
            MatmulCommon.reset()
            transpose_a = MatmulCommon.get_param_value(op_params, "transposeA", False)
            return MatmulCommon.gen_input(shapes, i, datatype, data_gen_ranges, transpose_a)
        if i == 1:
            transpose_b = MatmulCommon.get_param_value(op_params, "transposeB", False)
            return MatmulCommon.gen_weight(shapes, i, datatype, format, data_gen_ranges, transpose_b)
        if i == 2:
            return MatmulCommon.gen_groupList(shapes, i, data_gen_ranges)
        if i == 3:
            return MatmulCommon.gen_accum(shapes, i, datatype, data_gen_ranges)

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        output_tensor_list[0] = input_tensor_list[3]

    @staticmethod
    def golden(in_tensors, op_params):
        x = MatmulCommon.input_golden
        weight = MatmulCommon.weight_golden
        groupList = MatmulCommon.groupList_golden
        accum = MatmulCommon.accum_golden
        x = x.to(torch.float32)
        weight = weight.to(torch.float32)
        mout = accum.shape[0]
        nout = accum.shape[1]
        m = x.shape[0]
        n = weight.shape[1]
        num = len(groupList)
        golden_result = torch.zeros(m * num, n)
        flag = (mout % num == 0 and mout / num == m)
        for i in range (num) :
            startIdx = 0 if i == 0 else groupList[i - 1]
            endIdx = groupList[i]
            startNum = i * x.shape[0]
            endNum = (i + 1) * x.shape[0]
            golden_result[startNum:endNum, :] = torch.mm(x[:, startIdx:endIdx], weight[startIdx:endIdx, :])
        if not flag:
            golden_result = golden_result.reshape(accum.shape)
        golden_result = torch.add(golden_result, accum)
        MatmulCommon.reset()
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.VECTOR_FUSION

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params):
        bin = TensorBinFile(intensor_file)
        input = bin.get_tensor()
        input_cpu = input.cpu()
        shape = input_cpu.shape
        if i == 0:
            input_golden = input_cpu
            transpose_a = MatmulCommon.get_param_value(op_params, "transposeA", False)
            if transpose_a:
                input_golden = torch.transpose(input_cpu, 0, 1)
            MatmulCommon.input_golden = input_golden
        if i == 1:
            MatmulCommon.weight_golden = input_cpu
        if i == 2:
            MatmulCommon.groupList_golden = input_cpu
        if i == 3:
            MatmulCommon.accum_golden = input_cpu
        return input_cpu.npu()

class LinearParallelOperation(DataGen):

    residual_golden = None
    scale_golden = None
    def __init__(self):
        pass

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params):
        bin = TensorBinFile(intensor_file)
        input = bin.get_tensor()
        input_cpu = input.cpu()
        shape = input_cpu.shape
        #i表示intensor的下标，表示第几个intensor
        if i == 0:
            input_golden = input_cpu
            MatmulCommon.input_golden = input_golden
        if i == 1:
            weight_golden = input_cpu
            if len(shape) == 4:
                weight_golden = torch.transpose(input_cpu, 1, 2)
                if shape[0] == 1:
                    weight_golden = weight_golden.reshape(weight_golden.size()[1],
                                                          weight_golden.size()[2] * weight_golden.size()[3])
                else:
                    weight_golden = weight_golden.reshape(weight_golden.size()[0], weight_golden.size()[1],
                                                          weight_golden.size()[2] * weight_golden.size()[3])
            transpose_b = MatmulCommon.get_param_value(op_params, "transWeight", True)
            if transpose_b:
                dim_num = len(weight_golden.size())
                weight_golden = torch.transpose(weight_golden, 0, 1) if dim_num == 2 else torch.transpose(weight_golden, 1, 2)
            MatmulCommon.weight_golden = weight_golden
        return input_cpu.npu()

    @staticmethod
    def gen_residual(shape, datatype, data_gen_ranges):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        LinearParallelOperation.residual_golden = (high - low) * torch.rand(shape, dtype=dtype_dict[datatype]) + low
        return LinearParallelOperation.residual_golden.npu()

    @staticmethod
    def gen_scale(shape, datatype, data_gen_ranges):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        torch_data_type = dtype_dict[datatype]
        if torch_data_type == torch.int64:
            scale_origin = (high - low) * torch.rand(shape, dtype=torch.float32) + low
            scale_origin = ((scale_origin.view(torch.int32) >> 13) << 13).view(torch.float32)
            LinearParallelOperation.scale_golden = torch.stack((scale_origin, torch.zeros_like(scale_origin)), dim=-1) \
                .flatten().view(torch.int64).contiguous()
        else:
            LinearParallelOperation.scale_golden = (high - low) * torch.rand(shape, dtype=dtype_dict[datatype]) + low
        return LinearParallelOperation.scale_golden.npu()

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        json_data = json.loads(op_params)
        torch.manual_seed(random_seed)
        torch.npu.config.allow_internal_format = True
        if i == 0:
            return MatmulCommon.gen_input(shapes, i, datatype, data_gen_ranges)
        if i == 1:
            transpose_b = MatmulCommon.get_param_value(op_params, "transWeight", True)
            return MatmulCommon.gen_weight(shapes, i, datatype, format, data_gen_ranges, transpose_b)
        if i == 2:
            quant_type = -1
            if json_data.get('quantType') is not None:
                quant_type = json_data['quantType']
            if quant_type == -1 and json_data['hasResidual']:
                return LinearParallelOperation.gen_residual(shapes[i], datatype, data_gen_ranges)
            else:
                return MatmulCommon.gen_bias(shapes, i, datatype, data_gen_ranges)
        if i == 3:
            return LinearParallelOperation.gen_scale(shapes[i], datatype, data_gen_ranges)
        if i == 4:
            return LinearParallelOperation.gen_residual(shapes[i], datatype, data_gen_ranges)

    def golden_pure_linear(in_tensors, quant_type = -1, group_size = 0, out_data_type = -1):
        weight = MatmulCommon.weight_golden
        bias = MatmulCommon.bias_golden

        if quant_type is not None and quant_type >= 0:
            is_quant_after = in_tensors[0].dtype == torch.int8 and in_tensors[1].dtype == torch.int8
            quant_tensor = MatmulCommon.golden(True) if is_quant_after else weight.to(torch.float32)

            if bias is not None and bias.nelement() != 0:
                quant_tensor = quant_tensor.to(torch.float) + bias.to(torch.float)
            if LinearParallelOperation.scale_golden.dtype is torch.int64:
                scale = LinearParallelOperation.scale_golden.view(torch.float32)[::2]
            else:
                scale = LinearParallelOperation.scale_golden.to(torch.float)
            if quant_type == 2:
                dequantized_groups = [group * scale[i] for i, group in enumerate(quant_tensor.split(group_size, dim=0))]
                quant_tensor = torch.cat(dequantized_groups, dim=0)
            else:
                quant_tensor = quant_tensor.clone().to(torch.float) * scale
            if is_quant_after:
                result_dtype = torch.float16 if out_data_type == 1 else torch.bfloat16
                result = quant_tensor.to(result_dtype)
            else:
                MatmulCommon.weight_golden = quant_tensor.to(in_tensors[0].dtype)
                result = MatmulCommon.golden(True)
        else:
            result = MatmulCommon.golden(True)

        MatmulCommon.reset()
        LinearParallelOperation.scale_golden = None
        return [result]

    def golden_linear_all_reduce(in_tensors, rank_size, quant_type = -1, group_size = 0, out_data_type = -1):
        linear_result = LinearParallelOperation.golden_pure_linear(in_tensors, quant_type, group_size, out_data_type)[0]
        # allReduce
        golden_result = linear_result.clone()
        for i in range(rank_size - 1):
            golden_result += linear_result
        residual = LinearParallelOperation.residual_golden
        if residual is not None:
            golden_result = golden_result + residual
        LinearParallelOperation.residual_golden = None
        return [golden_result]

    def golden_linear_reduce_scatter(in_tensors, rank, rank_size, quant_type = -1, group_size = 0, out_data_type = -1):
        matmul_result = LinearParallelOperation.golden_pure_linear(in_tensors, quant_type, group_size, out_data_type)[0]
        sum_tensor = matmul_result.clone()
        for i in range(rank_size - 1):
            sum_tensor += matmul_result
        if len(in_tensors[0].shape) == 3 and in_tensors[0].shape[0] == 1:
            chunks_size = int(in_tensors[0].shape[1] / rank_size)
            chunks = torch.split(sum_tensor, chunks_size, dim = 1)
        else:
            chunks_size = int(in_tensors[0].shape[0] / rank_size)
            chunks = torch.split(sum_tensor, chunks_size)
        golden_result = chunks[rank]
        if LinearParallelOperation.residual_golden is not None:
            golden_result = golden_result + LinearParallelOperation.residual_golden
        LinearParallelOperation.residual_golden = None
        MatmulCommon.reset()
        return [golden_result]

    def golden_all_gather_linear(in_tensors, rank_size, quant_type = -1, group_size = 0, out_data_type = -1):
        golden_mid_tensor = in_tensors[0].clone()
        for i in range(rank_size - 1):
            golden_mid_tensor = torch.cat((golden_mid_tensor, in_tensors[0]), dim=0)
        MatmulCommon.input_golden = golden_mid_tensor
        golden_result = LinearParallelOperation.golden_pure_linear(golden_mid_tensor, quant_type, group_size, out_data_type)[0]
        if LinearParallelOperation.residual_golden is not None:
            golden_result = golden_result + LinearParallelOperation.residual_golden
        LinearParallelOperation.residual_golden = None
        MatmulCommon.reset()
        return [golden_result]

    def golden_all_gather_linear_reduce_scatter(in_tensors, rank, ag_dim, rs_dim, inner_dim_is_ag):
        golden_mid_tensor = in_tensors[0].clone()
        for i in range(ag_dim - 1):
            golden_mid_tensor = torch.cat((golden_mid_tensor, in_tensors[0]), dim=0)
        MatmulCommon.input_golden = golden_mid_tensor
        matmul_result = MatmulCommon.golden(True)
        sum_tensor = matmul_result.clone()
        for i in range(rs_dim - 1):
            sum_tensor += matmul_result
        chunks = torch.split(sum_tensor, int(sum_tensor.shape[0]/rs_dim))
        if inner_dim_is_ag:
            rs_rank = rank // ag_dim
        else:
            rs_rank = rank % rs_dim
        golden_result = chunks[rs_rank]
        if LinearParallelOperation.residual_golden is not None:
            golden_result = golden_result + LinearParallelOperation.residual_golden
        MatmulCommon.reset()
        return [golden_result]

    def golden_all_gather_linear_v2(in_tensors, rank_size):
        golden_mid_tensor = in_tensors[0].clone()
        for i in range(rank_size - 1):
            golden_mid_tensor = torch.cat((golden_mid_tensor, in_tensors[0]), dim=0)
        MatmulCommon.input_golden = golden_mid_tensor
        golden_result = MatmulCommon.golden(True)
        if LinearParallelOperation.residual_golden is not None:
            golden_result = golden_result + LinearParallelOperation.residual_golden
        MatmulCommon.reset()
        return [golden_result, golden_mid_tensor]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        backend = json_data['backend']
        rank_size = json_data['rankSize']
        rank = json_data['rank']
        if json_data.get('twoDimTPInfo') is not None:
            ag_dim = json_data['twoDimTPInfo']['agDim']
            rs_dim = json_data['twoDimTPInfo']['rsDim']
            inner_dim_is_ag = json_data['twoDimTPInfo']['innerDimIsAg']
        if backend != 'lcoc' and backend != 'mc2':
            return LinearParallelOperation.golden_linear_all_reduce(in_tensors, rank_size)
        type = 0
        quant_type = -1
        group_size = 0
        out_data_type = -1
        if json_data.get('type') is not None:
            type = json_data['type']
        if json_data.get('quantType') is not None:
            quant_type = json_data['quantType']
        if json_data.get('quantGroupSize') is not None:
            group_size = json_data['quantGroupSize']
        if json_data.get('outDataType') is not None:
            out_data_type = json_data['outDataType']
        if type == 0:
            return LinearParallelOperation.golden_linear_all_reduce(in_tensors, rank_size, quant_type, group_size, out_data_type)
        elif type == 1:
            return LinearParallelOperation.golden_linear_reduce_scatter(in_tensors, rank, rank_size,quant_type, group_size, out_data_type)
        elif type == 2:
            if json_data.get('keepIntermediate') and json_data['keepIntermediate'] == True:
                return LinearParallelOperation.golden_all_gather_linear_v2(in_tensors, rank_size)
            return LinearParallelOperation.golden_all_gather_linear(in_tensors, rank_size, quant_type, group_size, out_data_type)
        elif type == 3:
            return LinearParallelOperation.golden_pure_linear(in_tensors, quant_type, group_size, out_data_type)
        elif type == 4:
            return LinearParallelOperation.golden_all_gather_linear_reduce_scatter(in_tensors, rank, ag_dim, rs_dim, inner_dim_is_ag)

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class LinearSparseOperation(DataGen):
    case_num = 1
    compress_index = None
    # bash compress_weight tool
    ascend_home_path = os.getenv("ASCEND_HOME_PATH")
    compress_graph_path = f"{ascend_home_path}/tools/msmodelslim/pytorch/weight_compression/compress_graph"
    compile_result_path = f"{compress_graph_path}/build/compress_excutor"

    def __init__(self):
        pass

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        if len(input_tensor_list[1].size()) != 2:
            return
        if not os.path.exists(LinearSparseOperation.compile_result_path):
            os.chdir(LinearSparseOperation.compress_graph_path)
            command = f'bash build.sh {LinearSparseOperation.ascend_home_path}'
            if os.system(command) != 0:
                logging.error("compile compress_weight tool failed")
                exit(1)
        def input_yes(prompt=""):
            if prompt == "Please enter 'yes' or 'no'\n":
                return "yes"
            return builtins.input(prompt)
        builtins.input = input_yes
        sys.path.append(f"{LinearSparseOperation.ascend_home_path}/tools/msmodelslim/pytorch/")
        from weight_compression import CompressConfig, Compressor
        work_dir = os.path.join(os.getcwd(), "atb_temp/linear_sparse_weight")
        if LinearSparseOperation.case_num == 1:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)
        weight_path = os.path.join(work_dir, f"case{LinearSparseOperation.case_num}.npy")
        LinearSparseOperation.case_num = LinearSparseOperation.case_num + 1
        tensor_dict = {"key": input_tensor_list[1].cpu()}
        np.save(weight_path, tensor_dict)
        compress_config = CompressConfig()
        compressor = Compressor(compress_config, weight_path)
        compress_weight, LinearSparseOperation.compress_index, compress_info = compressor.run()
        weight_npu = torch.tensor(compress_weight["key"].astype(np.int8)).npu()
        input_tensor_list[1] = weight_npu
        input_tensor_list[4] = torch.tensor(LinearSparseOperation.compress_index["key"].astype(np.int8)).npu()

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        if i == 0:
            transpose_a = MatmulCommon.get_param_value(op_params, "transposeA", False)
            return MatmulCommon.gen_input(shapes, i, datatype, data_gen_ranges, transpose_a)
        if i == 1:
            transpose_b = MatmulCommon.get_param_value(op_params, "transposeB", True)
            return MatmulCommon.gen_weight(shapes, i, datatype, format, data_gen_ranges, transpose_b)
        if i == 2:
            return MatmulCommon.gen_bias(shapes, i, datatype, data_gen_ranges)
        if i == 3:
            return MatmulCommon.gen_deq(shapes, i, data_gen_ranges, 1, 0)
        if i == 4:
            return LinearSparseOperation.random(shapes[4], datatype, format, data_gen_ranges, op_params)

    @staticmethod
    def golden(in_tensors, op_params):
        MatmulCommon.linear_type = LinearType.int8int8_int32_fp16
        golden_result = MatmulCommon.golden()
        LinearSparseOperation.compress_index = None
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_QUANT


class LinearType:
    fp16fp16_fp32_fp16 = 0
    bf16bf16_fp32_bf16 = 1
    int8int8_int32_fp16 = 2
    int8int8_int32_bf16 = 3


class MatmulCommon:
    datatype_fp16 = "float16"
    datatype_bf16 = "bf16"
    datatype_int8 = "int8"
    datatype_int32 = "int32"
    datatype_float = "float"

    input_golden = None
    weight_golden = None
    bias_golden = None
    accum_golden = None
    deq_golden = None
    matmultype_golden = None
    pertoken_scale_golden = None
    linear_type = -1
    groupList_golden = None

    def __init__(self):
        pass

    @staticmethod
    def get_param_value(op_params, param_name: str, default_value):
        json_data = json.loads(op_params)
        return json_data[param_name] if param_name in json_data else default_value

    @staticmethod
    def __get_data_range(data_gen_ranges):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        return low, high

    @staticmethod
    def gen_input(shapes, i, datatype, data_gen_ranges, transpose_a: bool = False):
        shape = shapes[i]
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        if datatype == MatmulCommon.datatype_fp16:
            input_cpu = (high - low) * torch.rand(shape, dtype=torch.float16) + low
            MatmulCommon.linear_type = LinearType.fp16fp16_fp32_fp16
        elif datatype == MatmulCommon.datatype_bf16:
            input_cpu = (high - low) * torch.rand(shape, dtype=torch.bfloat16) + low
            MatmulCommon.linear_type = LinearType.bf16bf16_fp32_bf16
        elif datatype == MatmulCommon.datatype_float:
            input_cpu = (high - low) * torch.rand(shape, dtype=torch.float32) + low
        elif datatype == MatmulCommon.datatype_int8:
            low_int = int(low)
            high_int = int(high)
            input_cpu = torch.randint(low=low_int, high=high_int, size=tuple(shape), dtype=torch.int8)
        else:
            logging.error("inTensor0 datatype error!")
            return None
        input_golden = input_cpu
        if transpose_a:
            input_golden = torch.transpose(input_cpu, 0, 1) if len(shape) == 2 else torch.transpose(input_cpu, 1, 2)
        input_npu = input_cpu.npu()
        MatmulCommon.input_golden = input_golden
        return input_npu

    @staticmethod
    def gen_weight(shapes, i, datatype, format, data_gen_ranges, transpose_b: bool):
        shape = shapes[i]
        fixed_shape = ([shape[2], shape[1] * shape[3]] if shape[0] == 1 else [
                       shape[0], shape[2], shape[1] * shape[3]]) if len(shape) == 4 else shape
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        if datatype == MatmulCommon.datatype_fp16:
            weight_cpu = (high - low) * torch.rand(fixed_shape, dtype=torch.float16) + low
        elif datatype == MatmulCommon.datatype_bf16:
            weight_cpu = (high - low) * torch.rand(fixed_shape, dtype=torch.bfloat16) + low
        elif datatype == MatmulCommon.datatype_float:
            weight_cpu = (high - low) * torch.rand(fixed_shape, dtype=torch.float32) + low
        elif datatype == MatmulCommon.datatype_int8:
            low_int = int(low)
            high_int = int(high)
            weight_cpu = torch.randint(low=low_int, high=high_int, size=tuple(fixed_shape), dtype=torch.int8)
        else:
            logging.error("inTensor1 datatype error!")
            return None
        weight_golden = weight_cpu
        if transpose_b:
            dim_num = len(weight_golden.size())
            weight_golden = torch.transpose(weight_golden, 0, 1) if dim_num == 2 else torch.transpose(weight_golden, 1, 2)
        weight_npu = weight_cpu.npu()
        if format == "fractal_nz":
            weight_npu = torch_npu.npu_format_cast(weight_npu, 29)
            if len(shape) == 4:
                weight_npu = weight_npu.reshape(shape[0],shape[1],shape[2],shape[3])
        MatmulCommon.weight_golden = weight_golden
        return weight_npu

    @staticmethod
    def gen_bias(shapes, i, datatype, data_gen_ranges):
        shape = shapes[i]
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        if datatype == MatmulCommon.datatype_fp16:
            bias_cpu = (high - low) * torch.rand(shape, dtype=torch.float16) + low
        elif datatype == MatmulCommon.datatype_bf16:
            bias_cpu = (high - low) * torch.rand(shape, dtype=torch.bfloat16) + low
        elif datatype == MatmulCommon.datatype_float:
            bias_cpu = (high - low) * torch.rand(shape, dtype=torch.float) + low
        elif datatype == MatmulCommon.datatype_int32:
            low_int = int(low)
            high_int = int(high)
            bias_cpu = torch.randint(low=low_int, high=high_int, size=tuple(shape), dtype=torch.int)
        else:
            logging.error("bias datatype error!")
            return None
        bias_golden = bias_cpu
        bias_npu = bias_cpu.npu()
        MatmulCommon.bias_golden = bias_golden
        return bias_npu

    @staticmethod
    def gen_accum(shapes, i, datatype, data_gen_ranges):
        shape = shapes[i]
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        accum_cpu = (high - low) * torch.rand(shape, dtype=torch.float) + low
        accum_golden = accum_cpu
        accum_npu = accum_cpu.npu()
        MatmulCommon.accum_golden = accum_golden
        return accum_npu

    @staticmethod
    def gen_deq(shapes, i, data_gen_ranges, out_data_type, quant_mode):
        shape = shapes[i]
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        if quant_mode == 2:
            MatmulCommon.deq_golden = ((high - low) * torch.rand(shape) + low).type(dtype_dict["float"])
            return MatmulCommon.deq_golden.npu()
        if out_data_type == 1:
            low_int = int(low)
            high_int = int(high)
            deq_golden = torch.randint(low=low_int, high=high_int, size=tuple(shape), dtype=torch.int).float()
            deq_np = np.frombuffer(deq_golden.numpy().tobytes(), dtype=np.uint32).astype(np.int64)
            if get_soc_version() == "Ascend910A":
                deq_npu = deq_golden.npu()
            else:
                deq_npu = torch.tensor(deq_np, dtype=torch.int64).npu()
            MatmulCommon.deq_golden = deq_golden
            return deq_npu.view(shape)
        elif out_data_type == 27:
            MatmulCommon.deq_golden = ((high - low) * torch.rand(shape) + low).type(dtype_dict["float"])
            return MatmulCommon.deq_golden.npu()
        return None

    @staticmethod
    def gen_pertoken_scale(shapes, i, data_gen_ranges, out_data_type):
        shape = shapes[i]
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        if out_data_type == 1 or out_data_type == 27:
            MatmulCommon.pertoken_scale_golden = ((high - low) * torch.rand(shape) + low).type(dtype_dict["float"])
            return MatmulCommon.pertoken_scale_golden.npu()
        return None

    @staticmethod
    def gen_groupList(shapes, i, data_gen_ranges):
        num = shapes[i][0]
        k = shapes[1][0]
        low, high = MatmulCommon.__get_data_range(data_gen_ranges)
        low_int = 1
        high_int = k - 1
        if num == 0 :
            groupList = torch.tensor([])
        else :
            groupList = torch.randint(1, k - 1, (num - 1,), dtype=torch.int64)
            element = torch.tensor([k])
            groupList = torch.cat((groupList, element))
        groupList_golden = torch.sort(groupList).values
        groupList_npu = groupList_golden.npu()
        MatmulCommon.groupList_golden = groupList_golden
        return groupList_npu

    @staticmethod
    def golden(is_parallel: bool = False) -> torch.Tensor:
        input = MatmulCommon.input_golden
        weight = MatmulCommon.weight_golden
        bias = MatmulCommon.bias_golden
        deq = MatmulCommon.deq_golden
        accum = MatmulCommon.accum_golden
        groupList = MatmulCommon.groupList_golden
        pertoken_deq = MatmulCommon.pertoken_scale_golden
        if MatmulCommon.linear_type == LinearType.fp16fp16_fp32_fp16 or input.dtype == torch.float16:
            golden_result = torch.matmul(input.to(torch.float32), weight.to(torch.float32))
        elif MatmulCommon.linear_type == LinearType.bf16bf16_fp32_bf16 or input.dtype == torch.bfloat16:
            golden_result = torch.matmul(input.to(torch.float32), weight.to(torch.float32))
        elif MatmulCommon.linear_type == LinearType.int8int8_int32_fp16 or input.dtype == torch.int8:
            golden_result = torch.matmul(input.to(torch.int32), weight.to(torch.int32))
        else:
            return None
        if is_parallel:
            return golden_result
        if bias is not None:
            golden_result = golden_result + bias
        if deq is not None:
            golden_result = golden_result * deq
        if accum is not None:
            golden_result = golden_result + accum
        if pertoken_deq is not None:
            golden_result = golden_result * pertoken_deq

        MatmulCommon.reset()
        return golden_result

    @staticmethod
    def reset():
        MatmulCommon.input_golden = None
        MatmulCommon.weight_golden = None
        MatmulCommon.bias_golden = None
        MatmulCommon.accum_golden = None
        MatmulCommon.deq_golden = None
        MatmulCommon.linear_type = -1
        MatmulCommon.groupList_golden = None
        MatmulCommon.pertoken_scale_golden = None

class GatherOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        # 获取 x_shape 和 indexs_shape
        x_shape = shapes[0]
        indexs_shape = shapes[1]

        # 从 op_params 中获取 axis
        json_data = json.loads(op_params)
        axis = json_data["axis"]

        # 确保 axis 在 x_shape 的有效范围内
        if axis < 0 or axis >= len(x_shape):
            raise ValueError(f"Axis {axis} is out of bounds for x with shape {x_shape}")

        if i == 0:
            # 生成 x 的数据
            if datatype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
                low, high = -100, 100  # 默认整数上下限
                x_data = ((high - low) * torch.rand(x_shape) + low).type(dtype_dict[datatype])
            else:
                low, high = -100.0, 100.0  # 默认浮动点上下限
                x_data = ((high - low) * torch.rand(x_shape) + low).type(dtype_dict[datatype])
            return x_data.npu()
        else:
            # 生成 indexs 的数据
            x_size_at_axis = x_shape[axis]
            # 检查 x_size_at_axis 是否大于 0
            if x_size_at_axis <= 0:
                raise ValueError(f"x.shape[{axis}] must be greater than 0, got {x_size_at_axis}")
            # 生成在 [0, x_size_at_axis) 范围内的索引
            low, high = 0, x_size_at_axis
            indexs_data = ((high - low) * torch.rand(indexs_shape) + low).type(dtype_dict[datatype])

        return indexs_data.npu()
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        axis = json_data["axis"]
        batchDims = json_data["batchDims"]
        if in_tensors[1].ndim == 1:
            golden_result = torch.index_select(in_tensors[0], axis, in_tensors[1].to(torch.int64))
        else:
            if batchDims == 0:
                if axis == 0:
                    if in_tensors[0].ndim == 2 and in_tensors[1].ndim == 2:
                        embedding = torch.nn.Embedding(in_tensors[0].shape[0],in_tensors[0].shape[1])
                        embedding.weight.data.copy_(in_tensors[0])
                        embedding.weight.requires_grad = False
                        golden_result = embedding(in_tensors[1]).detach()
                        return [golden_result]
                outputSize = []
                dim0 = 1
                for i in range(0,axis):
                    outputSize.append(in_tensors[0].shape[i])
                    dim0 *= in_tensors[0].shape[i]
                dim1 = in_tensors[0].shape[axis]
                for i in range(0,in_tensors[1].ndim):
                    outputSize.append(in_tensors[1].shape[i])
                dim2 = 1
                for i in range(axis + 1,in_tensors[0].ndim):
                    outputSize.append(in_tensors[0].shape[i])
                    dim2 *= in_tensors[0].shape[i]
                # inputFlatten = in_tensors[0].clone().reshape(-1)
                indicesFlatten = in_tensors[1].clone().reshape(-1)
                logging.debug("outputSize",outputSize)

                golden_result = torch.zeros(outputSize, dtype=in_tensors[0].dtype, device=in_tensors[0].device).reshape(-1)
                idx = 0
                for i in range(0, dim0):
                    inputIdx = i * dim1 * dim2
                    for indice in indicesFlatten:
                        for k in range(0, dim2):
                            golden_result[idx] = in_tensors[0].flatten()[inputIdx + indice * dim2 + k]
                            idx += 1
                golden_result = golden_result.reshape(outputSize)
            else:
                params = in_tensors[0]
                indices = in_tensors[1]
                
                # 计算实际要索引的维度
                slice_axis = axis - batchDims
                
                # 获取 batch 维度的大小
                batch_shape = params.shape[:batchDims]
                
                # 使用 torch 的高级索引展开所有 batch 维度
                # 将前 batchDims 维度展平为一个维度，方便迭代
                flat_batch_size = 1
                for dim in batch_shape:
                    flat_batch_size *= dim

                # 展平 batch 维度，得到 [N, ...] 的形式，N 是总 batch 数
                # 注意：view(-1, ...) 不能直接用，因为每个 batch 的处理是独立的，在这使用 reshape + 迭代
                params_flat = params.reshape((flat_batch_size,) + params.shape[batchDims:])
                indices_flat = indices.reshape((flat_batch_size,) + indices.shape[batchDims:])
                
                results = []
                for i in range(flat_batch_size):
                    param_slice = params_flat[i]  # 形状: (D0, D1, ...)
                    index_slice = indices_flat[i]  # 形状: (I1, I2, ..., Ik)
                    
                    # 展平索引为 1D
                    flat_indices = index_slice.reshape(-1)
                    
                    # 在 slice_axis 上选择
                    selected = torch.index_select(param_slice, dim=slice_axis, index=flat_indices)
                    
                    # 计算目标形状
                    selection_shape = index_slice.shape  # 原始 indices 的形状
                    pre_axis_shape = selected.shape[:slice_axis]  # slice_axis 前的形状
                    post_axis_shape = selected.shape[slice_axis + 1:]  # slice_axis 后的形状（注意 index_select 会替换该维度）
                    
                    # 新形状 = pre_axis + selection_shape + post_axis
                    new_shape = list(pre_axis_shape) + list(selection_shape) + list(post_axis_shape)
                    
                    try:
                        selected = selected.view(new_shape)
                    except Exception as e:
                        raise RuntimeError(f"Failed to view selected tensor. "
                                        f"Current shape: {selected.shape}, "
                                        f"Target shape: {new_shape}, "
                                        f"Total elements: {selected.numel()}") from e
                    
                    results.append(selected)
                
                # 将结果堆叠回原始 batch 结构
                golden_result = torch.stack(results, dim=0)
                # 重新 reshape 到原始 batch shape + 输出空间形状
                output_shape = batch_shape + golden_result.shape[1:]
                golden_result = golden_result.reshape(output_shape)
        # 返回结果
        return [golden_result.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class AllReduceOperation(DataGen):
    scale_golden = None
    offset_golden = None
    @staticmethod
    def gen_input(shape, datatype, format, data_gen_ranges, op_params):
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllReduceOperation.intensors = []
        intensor_cpu = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
        for _ in range(rankSize):
            AllReduceOperation.intensors.append(intensor_cpu)
        intensor_npu = AllReduceOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor_npu, format_dict[format])

    @staticmethod
    def gen_scale(shape, datatype, format, data_gen_ranges):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllReduceOperation.scale_golden = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
        return torch_npu.npu_format_cast(AllReduceOperation.scale_golden.npu(), format_dict[format])

    @staticmethod
    def gen_offset(shape, datatype, format, data_gen_ranges):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllReduceOperation.offset_golden = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
        return torch_npu.npu_format_cast(AllReduceOperation.offset_golden.npu(), format_dict[format])

    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllReduceOperation.intensors = []
        intensor_cpu = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
        for _ in range(rankSize):
            AllReduceOperation.intensors.append(intensor_cpu)
        intensor_npu = AllReduceOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor_npu, format_dict[format])

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        json_data = json.loads(op_params)
        torch.manual_seed(random_seed)
        if "quantType" in json_data and json_data["quantType"] != 0:
            if i == 0:
                return AllReduceOperation.gen_input(shapes[i], datatype, format, data_gen_ranges, op_params)
            if i == 1:
                return AllReduceOperation.gen_scale(shapes[i], datatype, format, data_gen_ranges)
            if i == 2:
                return AllReduceOperation.gen_offset(shapes[i], datatype, format, data_gen_ranges)
        else:
            if i==0:
                return AllReduceOperation.gen_input(shapes[i], datatype, format, data_gen_ranges, op_params)

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        intensor_npu = bin.get_tensor()
        AllReduceOperation.intensors = []
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        for i in range(rankSize):
            AllReduceOperation.intensors.append(intensor_npu.cpu())
        return intensor_npu

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        os.environ["LCCL_DETERMINISTIC"]="1"
        os.environ["HCCL_DETERMINISTIC"]="true"

    def sum_cal(inTensors, op_params):
        json_data = json.loads(op_params)
        if ("quantType" in json_data and json_data["quantType"] != 0) or inTensors[0].dtype == torch.bfloat16:
            result = inTensors[0].clone().to(torch.float)
        else:
            result = inTensors[0].clone()
        for i in range(1, len(inTensors)):
            result += inTensors[i]
        return [result.to(torch.bfloat16) if inTensors[0].dtype == torch.bfloat16 else result ]

    def sum_cal_quant(inTensors, op_params):
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        result = AllReduceOperation.sum_cal(inTensors,op_params)[0]
        scale = AllReduceOperation.scale_golden.to(torch.float)
        offset = AllReduceOperation.offset_golden.to(torch.float)
        result = (result.clone().to(torch.float) + offset * rankSize) * scale
        AllReduceOperation.scale_golden = None
        AllReduceOperation.offset_golden = None
        return [result]

    def max_cal(inTensors):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.max(result,inTensors[i])
        return [result]

    def min_cal(inTensors):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.min(result,inTensors[i])
        return [result]

    def prod_cal(inTensors):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.mul(result,inTensors[i])
        return [result]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        allreduceType = json_data['allReduceType']
        backend = json_data['backend']
        logging.debug("backend: %s, allreduceType: %s", backend, allreduceType)
        logging.debug("env: %s", os.getenv("LCCL_DETERMINISTIC"))
        logging.debug("env: %s", os.getenv("HCCL_DETERMINISTIC"))

        res = []
        if "quantType" in json_data and json_data["quantType"] != 0:
            if not hasattr(AllReduceOperation, 'scale_golden'):
                AllReduceOperation.scale_golden = in_tensors[1]
            if not hasattr(AllReduceOperation, 'offset_golden'):
                AllReduceOperation.offset_golden = in_tensors[2]
            res = AllReduceOperation.sum_cal_quant(AllReduceOperation.intensors, op_params)
            delattr(AllReduceOperation, 'scale_golden')
            delattr(AllReduceOperation, 'offset_golden')
        else:
           if allreduceType == "sum":
              res = AllReduceOperation.sum_cal(AllReduceOperation.intensors, op_params)
           elif allreduceType == "max":
              res = AllReduceOperation.max_cal(AllReduceOperation.intensors)
           elif allreduceType == "min":
              res = AllReduceOperation.min_cal(AllReduceOperation.intensors)
           elif allreduceType == "prod":
              res = AllReduceOperation.prod_cal(AllReduceOperation.intensors)
        res = [torch.nan_to_num(tensor, nan=0.0, posinf=None, neginf=None) for tensor in res]
        return res

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if "quantType" in json_data and json_data["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        return OpTypes.COMPUTE_FLOAT

class ReduceScatterVOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = 0
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = 1
        high = 17
        ReduceScatterVOperation.intensors = []
        intensor_cpu = torch.randint(low, high, shape, dtype=dtype_dict[datatype])
        for _ in range(rankSize):
            ReduceScatterVOperation.intensors.append(intensor_cpu)
        intensor_npu = ReduceScatterVOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor_npu, format_dict[format])
    
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        if i == 1:
            return torch.full(shapes[i], torch.numel(ReduceScatterVOperation.intensors[0]) // rankSize, dtype=torch.int64).npu()
        elif i == 2:
            if len(shapes[i]) != 1 or shapes[i][0] != rankSize:
                return torch.full(shapes[i], torch.numel(ReduceScatterVOperation.intensors[0]), dtype=torch.int64).npu()
            return torch.numel(ReduceScatterVOperation.intensors[0]) // rankSize * torch.arange(rankSize, dtype=torch.int64).npu()
        elif i == 3:
            return torch.full(shapes[i], torch.numel(ReduceScatterVOperation.intensors[0]) // rankSize, dtype=torch.int64).npu()
        elif i == 4:
            return torch.full(shapes[i], 0, dtype=torch.float16).npu()
        else:
            raise ValueError("Index Error")
        
    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        host_tensor_dict = {}
        host_tensor_dict["sendCounts"] = input_tensor_list[1].tolist()
        host_tensor_dict["sdispls"] = input_tensor_list[2].tolist()
        host_tensor_dict["recvCount"] = input_tensor_list[3].tolist()[0]
        
        run_param = json.dumps(host_tensor_dict)
        operation.set_varaintpack_param(run_param)
    
    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        intensor_npu = bin.get_tensor()
        ReduceScatterVOperation.intensors = []
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        for i in range(rankSize):
            ReduceScatterVOperation.intensors.append(intensor_npu.cpu())
        return intensor_npu

    def sum_cal(inTensors,rank,rankSize):
        result = inTensors[0].clone()
        for i in range(1, len(inTensors)):
            result += inTensors[i]
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]

    def max_cal(inTensors,rank,rankSize):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.max(result,inTensors[i])
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]

    def min_cal(inTensors,rank,rankSize):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.min(result,inTensors[i])
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        reduceType = json_data['reduceType']
        backend = json_data['backend']
        logging.debug("backend: %s, reduceType: %s", backend, reduceType)

        if reduceType == "sum":
           return ReduceScatterVOperation.sum_cal(ReduceScatterVOperation.intensors,rank,rankSize)
        elif reduceType == "max":
            return ReduceScatterVOperation.max_cal(ReduceScatterVOperation.intensors,rank,rankSize)
        elif reduceType == "min":
            return ReduceScatterVOperation.min_cal(ReduceScatterVOperation.intensors,rank,rankSize)

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        reduceType = json_data['reduceType']
        if reduceType == "sum":
            return OpTypes.CV_FUSION
        return OpTypes.MOVE

class ReduceScatterOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = 1
        high = 17
        ReduceScatterOperation.intensors = []
        intensor_cpu = torch.randint(low, high, shape, dtype=dtype_dict[datatype])
        for _ in range(rankSize):
            ReduceScatterOperation.intensors.append(intensor_cpu)
        intensor_npu = ReduceScatterOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor_npu, format_dict[format])

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        intensor_npu = bin.get_tensor()
        ReduceScatterOperation.intensors = []
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        for i in range(rankSize):
            ReduceScatterOperation.intensors.append(intensor_npu.cpu())
        return intensor_npu

    def sum_cal(inTensors,rank,rankSize):
        result = inTensors[0].clone()
        for i in range(1, len(inTensors)):
            result += inTensors[i]
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]

    def max_cal(inTensors,rank,rankSize):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.max(result,inTensors[i])
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]

    def min_cal(inTensors,rank,rankSize):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.min(result,inTensors[i])
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]

    def prod_cal(inTensors,rank,rankSize):
        result = inTensors[0]
        for i in range(1,len(inTensors)):
            result = torch.mul(result,inTensors[i])
        chunks = torch.split(result,int(inTensors[0].shape[0]/rankSize))
        return [chunks[rank]]


    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        reduceType = json_data['reduceType']
        backend = json_data['backend']
        logging.debug("backend: %s, reduceType: %s", backend, reduceType)

        if reduceType == "sum":
           return ReduceScatterOperation.sum_cal(ReduceScatterOperation.intensors,rank,rankSize)
        elif reduceType == "max":
            return ReduceScatterOperation.max_cal(ReduceScatterOperation.intensors,rank,rankSize)
        elif reduceType == "min":
            return ReduceScatterOperation.min_cal(ReduceScatterOperation.intensors,rank,rankSize)
        elif reduceType == "prod":
            return ReduceScatterOperation.prod_cal(ReduceScatterOperation.intensors,rank,rankSize)

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        reduceType = json_data['reduceType']
        if reduceType == "sum" or reduceType == "prod":
            return OpTypes.CV_FUSION
        return OpTypes.MOVE

class AllToAllVOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllToAllVOperation.intensors = []
        AllToAllVOperation.type = dtype_dict[datatype]

        for i in range(rankSize):
            intensor = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
            AllToAllVOperation.intensors.append(intensor)
        intensor = AllToAllVOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        sendCounts = json_data['sendCounts']
        sdispls = json_data['sdispls']
        rank = json_data['rank']
        recvCounts = json_data['recvCounts']
        golden_result = []
        for j in range(rankSize):
            tensor_list = AllToAllVOperation.intensors[j].reshape(-1).tolist()
            golden_result += tensor_list[sdispls[rank]:sdispls[rank]+sendCounts[rank]]
        size = sum(recvCounts)
        golden_out_tensor = torch.tensor(golden_result,dtype=AllToAllVOperation.type).reshape(1,size)
        return [golden_out_tensor.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class AllToAllVV2Operation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllToAllVV2Operation.intensors = []
        AllToAllVV2Operation.type = dtype_dict[datatype]

        for i in range(rankSize):
            intensor = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
            AllToAllVV2Operation.intensors.append(intensor)
        intensor = AllToAllVV2Operation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        sendCounts = json_data['sendCounts']
        sdispls = json_data['sdispls']
        rank = json_data['rank']
        recvCounts = json_data['recvCounts']
        golden_result = []
        for j in range(rankSize):
            tensor_list = AllToAllVV2Operation.intensors[j].reshape(-1).tolist()
            golden_result += tensor_list[sdispls[rank]:sdispls[rank]+sendCounts[rank]]
        size = sum(recvCounts)
        golden_out_tensor = torch.tensor(golden_result,dtype=AllToAllVV2Operation.type).reshape(1,size)
        return [golden_out_tensor.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class AllToAllOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllToAllOperation.intensors = []
        AllToAllOperation.type = dtype_dict[datatype]
        AllToAllOperation.shape = shape

        for i in range(rankSize):
            intensor = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
            AllToAllOperation.intensors.append(intensor)
        intensor = AllToAllOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        rank = json_data['rank']
        trans = False
        if json_data.get('transpose') is not None:
             trans = json_data['transpose']
        goldenTensors = []
        height, width = AllToAllOperation.shape if len(AllToAllOperation.shape) == 2 else (0, 0)
        for i in range(rankSize):
            golden_out = []
            for j in range(rankSize):
                intensorj = AllToAllOperation.intensors[j]
                if trans:
                    intensorj = intensorj.reshape(height, rankSize, -1).permute(1, 0, 2)
                golden_out_list = intensorj.reshape(-1).tolist()
                split = golden_out_list[i*len(golden_out_list) // rankSize:(i+1)*len(golden_out_list) // rankSize]
                golden_out += split
            golden_out_tensor = torch.tensor(golden_out,dtype=AllToAllOperation.type).reshape(AllToAllOperation.shape)
            if trans:
                golden_out_tensor = golden_out_tensor.reshape(height * rankSize, width // rankSize)
            goldenTensors.append(golden_out_tensor)
        return [goldenTensors[rank].cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class AllGatherVOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllGatherVOperation.intensors = []
        AllGatherVOperation.type = dtype_dict[datatype]

        for i in range(rankSize):
            intensor = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
            AllGatherVOperation.intensors.append(intensor)
        intensor = AllGatherVOperation.intensors[rank].clone().npu()
        return torch_npu.npu_format_cast(intensor, format_dict[format])
    
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        if i == 1:
            return torch.full(shapes[i], torch.numel(AllGatherVOperation.intensors[0]), dtype=torch.int64).npu()
        elif i == 2:
            return torch.full(shapes[i], torch.numel(AllGatherVOperation.intensors[0]), dtype=torch.int64).npu()
        elif i == 3:
            if len(shapes[i]) != 1 or shapes[i][0] != rankSize:
                return torch.full(shapes[i], torch.numel(AllGatherVOperation.intensors[0]), dtype=torch.int64).npu()
            return torch.numel(AllGatherVOperation.intensors[0]) * torch.arange(rankSize, dtype=torch.int64).npu()
        elif i == 4:
            return torch.full(shapes[i], 0, dtype=torch.float16).npu()
        else:
            raise ValueError("Index Error")

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        os.environ["HCCL_DETERMINISTIC"]="true"

        host_tensor_dict = {}
        host_tensor_dict["sendCount"] = input_tensor_list[1].tolist()[0]
        host_tensor_dict["recvCounts"] = input_tensor_list[2].tolist()
        host_tensor_dict["rdispls"] = input_tensor_list[3].tolist()

        run_param = json.dumps(host_tensor_dict)
        operation.set_varaintpack_param(run_param)

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        intensor_npu = bin.get_tensor()
        AllGatherVOperation.intensors = []
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        for i in range(rankSize):
            AllGatherVOperation.intensors.append(intensor_npu.cpu())
        return intensor_npu

    @staticmethod
    def golden(in_tensors, op_params):
        golden_result = torch.cat(AllGatherVOperation.intensors, dim=0)
        return [golden_result.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class AllGatherOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        AllGatherOperation.intensors = []
        for i in range(rankSize):
            intensor = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
            AllGatherOperation.intensors.append(intensor)
        intensor = AllGatherOperation.intensors[rank].clone().npu()
        torch.npu.config.allow_internal_format = True
        return torch_npu.npu_format_cast(intensor, format_dict[format])

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        os.environ["LCCL_DETERMINISTIC"]="1"
        os.environ["HCCL_DETERMINISTIC"]="true"

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        intensor_npu = bin.get_tensor()
        AllGatherOperation.intensors = []
        json_data = json.loads(op_params)
        rankSize = json_data['rankSize']
        for i in range(rankSize):
            AllGatherOperation.intensors.append(intensor_npu.cpu())
        return intensor_npu

    @staticmethod
    def golden(in_tensors, op_params):
        golden_result = torch.stack(AllGatherOperation.intensors, dim=0)
        return [golden_result.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class BlockCopyOperation(DataGen):
    srcBlks = None
    @staticmethod
    def shape_nd_to_nz(shape, dtype='float16'):
        assert len(shape) >= 2
        batch = shape[:-2] # 最后两维nd->nz
        a, b = shape[-2], shape[-1]
        a0, b0 = 16, 16
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    @staticmethod
    def gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    @staticmethod
    def convert_nd_to_nz(x):
        array_trans = BlockCopyOperation.gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
        x_shape = BlockCopyOperation.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans) # x原始需要对齐，才能reshape
    @staticmethod
    def generate_dstBlks(srcBlks, blockCount, dstCount):
        # 创建包含所有可能整数的集合
        all_possible_integers = set(range(blockCount))
        # 从集合中移除数组 a 中的所有元素
        integers_not_in_src = all_possible_integers - set(srcBlks)
        # 确保剩余的整数数量足够
        if len(integers_not_in_src) < dstCount:
            raise ValueError("Not enough unique integers available")
        # 从剩余的集合中随机选择 x 个整数
        selected_integers = np.random.choice(list(integers_not_in_src), size=dstCount, replace=False)
        return selected_integers

    @staticmethod
    def generate_cumSum(srcCount, dstCount):
        if srcCount > dstCount:
            raise ValueError("srcCount cannot be greater than dstCount")
        # 生成包含1到count的所有整数的列表
        full_range = list(range(1, dstCount))
        # 从full_range中随机选择length个不重复的整数
        random_sequence = random.sample(full_range, srcCount - 1)
        cumSum = np.append(random_sequence, dstCount)
        cumSum.sort()
        return cumSum

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        blockCount = shapes[0][0]
        shape = shapes[i]

        if i < 2:
            tensor = random(shape, datatype, format, data_gen_ranges, op_params)
            if format == "fractal_nz":
                return BlockCopyOperation.convert_nd_to_nz(tensor)
            return tensor
        elif i == 2:
            BlockCopyOperation.srcBlks = np.random.randint(blockCount, size=shapes[i][0]).astype(np.int32)
            return torch.from_numpy(BlockCopyOperation.srcBlks).npu()
        elif i == 3:
            return torch.from_numpy(BlockCopyOperation.generate_dstBlks(BlockCopyOperation.srcBlks, blockCount, shapes[i][0]).astype(np.int32)).int().npu()
        elif i == 4:
            return torch.from_numpy(BlockCopyOperation.generate_cumSum(shapes[2][0], shapes[3][0]).astype(np.int32)).int().npu()
        else:
            raise ValueError("Index Error")

    @staticmethod
    def golden(in_tensors, op_params):
        kCacheGolden = in_tensors[0].cpu().clone()
        vCacheGolden = in_tensors[1].cpu().clone()
        srcBlks = in_tensors[2].cpu()
        dstBlks = in_tensors[3].cpu()
        cumSum = in_tensors[4].cpu()
        startIdx = 0
        endIdx = 0
        for i in range(len(srcBlks)):
            srcBlk = srcBlks[i]
            endIdx = cumSum[i]
            for j in range(startIdx, endIdx):
                dstBlk = dstBlks[j]
                kCacheGolden[dstBlk] = kCacheGolden[srcBlk].clone()
                vCacheGolden[dstBlk] = vCacheGolden[srcBlk].clone()
            startIdx = endIdx
        return [kCacheGolden, vCacheGolden, in_tensors[2], in_tensors[3], in_tensors[4]]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class BroadcastOperation(DataGen):
    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        random_seed = int(os.environ["random_seed"])
        torch.manual_seed(random_seed)
        json_data = json.loads(op_params)
        rank = json_data['rank']
        rankSize = json_data['rankSize']
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        BroadcastOperation.intensors = []
        for i in range(rankSize):
            intensor = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype])
            BroadcastOperation.intensors.append(intensor)
        intensor = BroadcastOperation.intensors[rank].clone().npu()
        torch.npu.config.allow_internal_format = True
        return torch_npu.npu_format_cast(intensor, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        rankRoot = json_data['rankRoot']
        golden_result = BroadcastOperation.intensors[rankRoot]
        return [golden_result.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class UnpadOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i == 0:
            batch = shapes[0][0]
            max_seq_len = shapes[0][1]
            input_ids = np.zeros(shapes[0])
            seq_len = np.random.randint(1, max_seq_len, size=batch)
            low = float(data_gen_ranges.split(',')[0])
            high = float(data_gen_ranges.split(',')[1])
            for i in range(batch):
                input_ids[i][0: seq_len[i]] = np.random.randint(low, high, size=seq_len[i])
            temp_arr = batch * [max_seq_len]
            zeros_num = np.array(temp_arr) - np.array(seq_len)
            cum_offsets_now = zeros_num
            cum_offsets_now = np.cumsum(zeros_num)
            token_num = np.sum(seq_len)

            input_ids = torch.from_numpy(np.array(input_ids).astype(np.int64))
            cum_offsets_now = torch.from_numpy(np.array(cum_offsets_now).reshape(batch, 1).astype(np.int32))
            token_num = torch.from_numpy(np.array(token_num).reshape(1, 1).astype(np.int64))
            seq_len = torch.from_numpy(np.array(seq_len).reshape(batch, 1).astype(np.int32))
            UnpadOperation.intensors = [input_ids.npu(),cum_offsets_now.npu(),token_num.npu(),seq_len.npu()]
            return UnpadOperation.intensors[0]
        else:
            return UnpadOperation.intensors[i]

    @staticmethod
    def golden(in_tensors, op_params):
        input_ids = in_tensors[0]
        cum_offsets_now = in_tensors[1].reshape(-1)
        token_num = in_tensors[2]
        seq_len = in_tensors[3]
        batch = in_tensors[0].shape[0]
        total_length_imm = in_tensors[0].shape[1]

        x_remove_padding = input_ids[0][0:seq_len[0]]
        for i in range(1, batch):
            x_remove_padding = np.concatenate((x_remove_padding, input_ids[i][0:seq_len[i]]))
        x_remove_padding = np.pad(x_remove_padding, (0, (batch * total_length_imm - token_num[0][0])))
        cum_offsets_out = np.zeros(batch)
        for i in range(1, batch):
            cum_offsets_out[i] = cum_offsets_now[i - 1]
        padding_offset =seq_len[0] * [0]
        for i in range(1, batch):
            temp_pad_out =seq_len[i] * [cum_offsets_now[i - 1]]
            padding_offset = np.concatenate((padding_offset, temp_pad_out))
        zero_offset = np.zeros((1,batch * total_length_imm - token_num[0][0]))
        padding_offset = np.append(padding_offset, zero_offset)
        x_remove_padding = torch.from_numpy(x_remove_padding.reshape(1, batch * total_length_imm)).long()
        cum_offsets_out = torch.from_numpy(cum_offsets_out.reshape(batch, 1)).int()
        padding_offset = torch.from_numpy(padding_offset.reshape(1, batch * total_length_imm).astype(np.int32))
        return [x_remove_padding, cum_offsets_out, padding_offset]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class PadOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i == 0:
            batch = shapes[3][0]
            max_seq_len = shapes[3][1]
            hidden_dim = shapes[0][1]
            seq_len = np.random.randint(1, max_seq_len, size=shapes[2][0]) #几个真实数据
            input_ids = np.zeros((batch,max_seq_len),dtype = np.int64)
            token_num = np.sum(seq_len)
            temp_arr = batch * [max_seq_len]
            zeros_num = np.array(temp_arr) - np.array(seq_len)
            cum_offsets_now = zeros_num
            cum_offsets_now = np.cumsum(zeros_num)
            tmp_out = torch.from_numpy(np.random.uniform(-1,1,(token_num, hidden_dim)).astype(np.float16))
            low = float(data_gen_ranges.split(',')[0])
            high = float(data_gen_ranges.split(',')[1])
            for i in range(batch):
                input_ids[i][0: seq_len[i]] = np.random.randint(low, high, size=seq_len[i])
            padding_offset = seq_len[0] * [0]
            for i in range(1, batch):
                temp_pad_out = seq_len[i] * [cum_offsets_now[i - 1]]
                padding_offset = np.concatenate((padding_offset, temp_pad_out))
            input_ids = torch.from_numpy(input_ids).long()
            padding_offset = torch.from_numpy(np.array(padding_offset).reshape(1, token_num)).int()
            seq_len = torch.from_numpy(seq_len.reshape(batch, 1)).int()
            PadOperation.intensors = [tmp_out,padding_offset,seq_len,input_ids]
            pad0 =  PadOperation.intensors[0].type(dtype_dict[datatype]).npu()
            return torch_npu.npu_format_cast(pad0, format_dict[format])
        else:
            pad =  PadOperation.intensors[i].type(dtype_dict[datatype]).npu()
            return torch_npu.npu_format_cast(pad, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        tmp_out = in_tensors[0]
        padding_offset = in_tensors[1]
        seq_len = in_tensors[2]
        input_ids = in_tensors[3]
        batch = input_ids.shape[0]
        hidden_dim = tmp_out.shape[1]
        max_seq_len = input_ids.shape[1]

        golden_result = np.zeros((batch,hidden_dim)).astype(np.float16)
        tempVal = 0
        for i in range(batch):
            tempVal = tempVal + seq_len[i][0]
            golden_result[i] = tmp_out[tempVal - 1]
        golden_result = torch.from_numpy(golden_result)
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class RepeatOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        golden_result = in_tensors[0].repeat(json_data["multiples"])
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class ActivationGolden:
    @staticmethod
    def relu_golden(in_tensors):
        return torch.nn.functional.relu(in_tensors)

    @staticmethod
    def gelu_golden(in_tensors, geluMode):
        float_in_tensors = in_tensors.float()
        approx = "tanh" if geluMode == 0 else "none"
        float_result = torch.nn.functional.gelu(float_in_tensors, approximate=approx)
        return float_result

    @staticmethod
    def fast_gelu_golden(in_tensors):
        float_in_tensors = in_tensors.float()
        float_result = float_in_tensors * torch.exp(0.851 * (float_in_tensors - torch.abs(float_in_tensors))) / (1 + torch.exp(-1.702 * torch.abs(float_in_tensors)))
        return float_result.half()

    @staticmethod
    def sigmoid_golden(in_tensors):
        float_in_tensors = in_tensors.float()
        float_result = 1 / (1 + np.exp(-float_in_tensors))
        return float_result

    @staticmethod
    def swish_golden(in_tensors, scale):
        float_in_tensors = in_tensors.float()
        float_result = float_in_tensors / (1 + np.exp(-float_in_tensors * scale))
        return float_result

    @staticmethod
    def log_golden(in_tensors):
        return torch.log(in_tensors)

    @staticmethod
    def swigluforward_golden(in_tensors, dim):
        dtype = in_tensors.dtype
        float_in_tensors = in_tensors.float()
        a, b = float_in_tensors.chunk(2, dim = dim)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        float_result = F.silu(a) * b
        return float_result.to(dtype)

    @staticmethod
    def faster_gelu_golden(in_tensor) -> torch.Tensor:
        dtype = in_tensor.dtype
        in_tensor = in_tensor.float()
        abs_value = torch.abs(in_tensor)
        float_res = in_tensor * torch.sigmoid(1.702 * abs_value) * torch.exp(0.851 * (in_tensor - abs_value))
        return float_res.to(dtype)


    def swish(x):
        return x * torch.sigmoid(x)

    def swish_grad(x):
        return torch.sigmoid(x) + x * (1 - torch.sigmoid(x)) * torch.sigmoid(x)

    @staticmethod
    def swiglubackward_golden(in_tensors, dim):
        tensor_y_grad = in_tensors[0].float()
        x = in_tensors[1].float()
        a, b = x.chunk(2, dim = dim)
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        y1 = b * tensor_y_grad * ActivationGolden.swish_grad(a)
        y2 = tensor_y_grad * ActivationGolden.swish(a)
        return torch.cat((y1,y2), dim = dim)


class ActivationOperation(DataGen):
    ACTIVATION_RELU = 1
    ACTIVATION_GELU = 2
    ACTIVATION_FAST_GELU = 3
    ACTIVATION_SWISH = 4
    ACTIVATION_LOG = 5
    ACTIVATION_SWIGLU_FORWARD = 6
    ACTIVATION_SWIGLU_BACKWARD = 7
    ACTIVATION_SIGMOID = 8
    ACTIVATION_FASTER_GELU_FORWARD = 9

    golden_func = {
        ACTIVATION_RELU: ActivationGolden.relu_golden,
        ACTIVATION_GELU: ActivationGolden.gelu_golden,
        ACTIVATION_FAST_GELU: ActivationGolden.fast_gelu_golden,
        ACTIVATION_SWISH: ActivationGolden.swish_golden,
        ACTIVATION_LOG: ActivationGolden.log_golden,
        ACTIVATION_SWIGLU_FORWARD: ActivationGolden.swigluforward_golden,
        ACTIVATION_SWIGLU_BACKWARD: ActivationGolden.swiglubackward_golden,
        ACTIVATION_SIGMOID: ActivationGolden.sigmoid_golden,
        ACTIVATION_FASTER_GELU_FORWARD: ActivationGolden.faster_gelu_golden
    }
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if json_data["activationType"] == ActivationOperation.ACTIVATION_SWIGLU_FORWARD:
            golden_result = ActivationOperation.golden_func[json_data["activationType"]](in_tensors[0], json_data["dim"])
        elif json_data["activationType"] == ActivationOperation.ACTIVATION_SWIGLU_BACKWARD:
            golden_result = ActivationOperation.golden_func[json_data["activationType"]](in_tensors, json_data["dim"])
        elif json_data["activationType"] == ActivationOperation.ACTIVATION_GELU:
            geluMode = json_data["geluMode"] if "geluMode" in json_data else 0
            golden_result = ActivationOperation.golden_func[json_data["activationType"]](in_tensors[0], geluMode)
        elif json_data["activationType"] == ActivationOperation.ACTIVATION_SWISH:
            golden_result = ActivationOperation.golden_func[json_data["activationType"]](in_tensors[0], json_data["scale"])
        else :
            golden_result = ActivationOperation.golden_func[json_data["activationType"]](in_tensors[0])
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["activationType"] in [
            ActivationOperation.ACTIVATION_FAST_GELU,
            ActivationOperation.ACTIVATION_FASTER_GELU_FORWARD,
            ActivationOperation.ACTIVATION_GELU,
            ActivationOperation.ACTIVATION_LOG,
            ActivationOperation.ACTIVATION_RELU
        ]:
            return OpTypes.COMPUTE_FLOAT
        else:
            return OpTypes.VECTOR_FUSION


class WhereOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        golden_result = torch.where(in_tensors[0].bool(), in_tensors[1], in_tensors[2])
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class TransdataOperation(DataGen):
    ALIGN_INT8 = 32
    DEFAULT_ALIGN = 16

    @staticmethod
    def round_up(x, align):
        if align == 0:
            return -1
        return (x + align - 1) // align * align

    @staticmethod
    def custom_pad(x, pad_dims):
        return torch.nn.functional.pad(x, pad_dims)

    @staticmethod
    def custom_reshape(x, target_shape):
        return x.reshape(target_shape)

    @staticmethod
    def custom_transpose(x, dim1, dim2):
        return x.transpose(dim1, dim2)

    @staticmethod
    def golden_nd_to_nz_3d(in_tensors):
        aux_dims = [0, 0, 0, 0]
        aux_dims[0] = in_tensors[0].size(0)
        aux_dims[1] = TransdataOperation.round_up(in_tensors[0].size(1), TransdataOperation.DEFAULT_ALIGN)

        pad_dims = [0, 0, 0, 0]
        pad_dims[3] = TransdataOperation.round_up(in_tensors[0].size(1), TransdataOperation.DEFAULT_ALIGN) - in_tensors[0].size(1)

        if in_tensors[0].dtype == torch.int8:
            aux_dims[2] = TransdataOperation.round_up(in_tensors[0].size(2), TransdataOperation.ALIGN_INT8) // TransdataOperation.ALIGN_INT8
            aux_dims[3] = TransdataOperation.ALIGN_INT8
            pad_dims[1] = TransdataOperation.round_up(in_tensors[0].size(2), TransdataOperation.ALIGN_INT8) - in_tensors[0].size(2)
        else:
            aux_dims[2] = TransdataOperation.round_up(in_tensors[0].size(2), TransdataOperation.DEFAULT_ALIGN) // TransdataOperation.DEFAULT_ALIGN
            aux_dims[3] = TransdataOperation.DEFAULT_ALIGN
            pad_dims[1] = TransdataOperation.round_up(in_tensors[0].size(2), TransdataOperation.DEFAULT_ALIGN) - in_tensors[0].size(2)

        return TransdataOperation.custom_transpose(
                    TransdataOperation.custom_reshape(
                        TransdataOperation.custom_pad(in_tensors[0], pad_dims),
                        aux_dims
                    ),
                    1, 2
                ).contiguous()

    @staticmethod
    def golden_nd_to_nz_2d(in_tensors):
        aux_dims = [0, 0, 0, 0]
        aux_dims[0] = 1
        aux_dims[1] = TransdataOperation.round_up(in_tensors[0].size(0), TransdataOperation.DEFAULT_ALIGN)

        pad_dims = [0, 0, 0, 0]
        pad_dims[3] = TransdataOperation.round_up(in_tensors[0].size(0), TransdataOperation.DEFAULT_ALIGN) - in_tensors[0].size(0)

        if in_tensors[0].dtype == torch.int8:
            aux_dims[2] = TransdataOperation.round_up(in_tensors[0].size(1), TransdataOperation.ALIGN_INT8) // TransdataOperation.ALIGN_INT8
            aux_dims[3] = TransdataOperation.ALIGN_INT8
            pad_dims[1] = TransdataOperation.round_up(in_tensors[0].size(1), TransdataOperation.ALIGN_INT8) - in_tensors[0].size(1)
        else:
            aux_dims[2] = TransdataOperation.round_up(in_tensors[0].size(1), TransdataOperation.DEFAULT_ALIGN) // TransdataOperation.DEFAULT_ALIGN
            aux_dims[3] = TransdataOperation.DEFAULT_ALIGN
            pad_dims[1] = TransdataOperation.round_up(in_tensors[0].size(1), TransdataOperation.DEFAULT_ALIGN) - in_tensors[0].size(1)

        return TransdataOperation.custom_transpose(
                    TransdataOperation.custom_reshape(
                        TransdataOperation.custom_pad(in_tensors[0], pad_dims),
                        aux_dims
                    ),
                    1, 2
                ).contiguous()

    @staticmethod
    def golden_nz_to_nd(in_tensors, outCrops):
        aux_dims = [0, 0, 0]
        aux_dims[0] = in_tensors[0].size(0)
        aux_dims[1] = in_tensors[0].size(2)
        aux_dims[2] = in_tensors[0].size(1) * in_tensors[0].size(3)
        return TransdataOperation.custom_reshape(
                    TransdataOperation.custom_transpose(in_tensors[0], 1, 2),
                    aux_dims
                )[:, :outCrops[0], :outCrops[1]]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if json_data["transdataType"] == 2:
            if len(in_tensors[0].size()) == 3:
                golden_result = TransdataOperation.golden_nd_to_nz_3d(in_tensors)
            else:
                golden_result = TransdataOperation.golden_nd_to_nz_2d(in_tensors)
        else:
            golden_result = TransdataOperation.golden_nz_to_nd(in_tensors, json_data["outCrops"])

        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CAST


class TopkToppSamplingOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        json_data = json.loads(op_params)
        rand_seed = json_data["randSeed"] if "randSeed" in json_data else 0
        torch.manual_seed(rand_seed)
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        probs = ((high - low) * torch.rand(shapes[i]) + low).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).type(dtype_dict[datatype]).npu()
        if i == 0:
            return torch_npu.npu_format_cast(probs, format_dict[format])
        else:
            exp_div = torch.empty_like(probs.cpu()).exponential_(1)
            exp_div = exp_div.npu()
            return torch_npu.npu_format_cast(exp_div, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):

        def logprobs_golden(in_tensors, logprobs_size):
            sorted_probs = in_tensors[0].to(torch.float32).cpu().numpy()
            cumsumed_probs = in_tensors[1]
            select_index = in_tensors[2]
            select_prob = np.take_along_axis(cumsumed_probs, select_index, axis=-1).astype(np.float32)

            logprobs_output = sorted_probs[:, :logprobs_size]
            logprobs_output = np.log(np.divide(logprobs_output, select_prob))
            batch = sorted_probs.shape[0]
            increasing_index = np.tile(np.arange(logprobs_size), batch).reshape(batch, -1)
            mask = ~torch.tensor(increasing_index <= select_index)
            logprobs_output = torch.tensor(logprobs_output).masked_fill(mask=mask, value=-9999.0)

            return logprobs_output
        
        def toppsample_golden_firstpick(probs_cumsumed, topp, rand, temp_ub_ele_aligned=16384):
            batch_size = probs_cumsumed.shape[0]
            vocab_size = probs_cumsumed.shape[-1]
            per_core_run_num = (vocab_size + temp_ub_ele_aligned - 1) // temp_ub_ele_aligned
            for b in range(batch_size):
                remove_val = topp[b] if topp.shape[0] > 1 else topp[0]
                idx_return = None
                # 模拟分块遍历（对应FirstPick中的循环）
                for j in range(per_core_run_num):
                    start_idx = j * temp_ub_ele_aligned
                    end_idx = min(start_idx + temp_ub_ele_aligned, vocab_size)
                    # 获取当前块的最后一个值
                    block_last_val = probs_cumsumed[b, end_idx - 1]
                    # 找到第一个大于temp_rand_val的块
                    if block_last_val >= remove_val * rand[b] and idx_return is None:
                        idx_return = j
                if idx_return is None:
                    idx_return = per_core_run_num - 1
            return idx_return
        
        json_data = json.loads(op_params)
        topktopp_sampling_type = json_data["topkToppSamplingType"]
        libc = CDLL("libc.so.6")
        if topktopp_sampling_type == 0:
            topk = json_data["topk"]
            rand_seed = json_data["randSeed"]
            libc.srand(rand_seed)
            rand_list = [libc.rand() / 0x7fffffff for i in range(512)]
            probs = in_tensors[0].npu()
            topp = in_tensors[1].npu()
            probs_sorted, indices_sorted = torch.topk(probs, k=topk, dim=-1, sorted=True)
            probs_cumsumed = torch.cumsum(probs_sorted, dim=-1)
            # 转npu计算以提高精度
            probs_sorted_sumed = probs_cumsumed.cpu().to(torch.float32).numpy()
            topp_cpu = topp.cpu().to(torch.float32).numpy()
            rand = np.array(rand_list).reshape(-1, 1)[0:probs.shape[0]]
            idx_return = toppsample_golden_firstpick(probs_sorted_sumed, topp_cpu, rand, 16384)
            bool_judge = (probs_sorted_sumed < topp_cpu)
            sum_val = np.sum(bool_judge, axis=-1, keepdims=True) - 1
            sum_val[sum_val < 0] = 0
            topp_v = np.take_along_axis(probs_sorted_sumed, np.minimum((idx_return + 1) * 16384 - 1, sum_val), axis=-1)
            topp_v *= rand
            bool_judge_one = probs_sorted_sumed <= topp_v
            res = np.sum(bool_judge_one, axis=-1, keepdims=True)
            res[res < 0] = 0
            res_torch = torch.from_numpy(res).npu()
            indices_sampled = torch.gather(indices_sorted, dim=-1, index=res_torch.long())
            probs_sampled = torch.gather(probs_sorted, dim=-1, index=res_torch.long())
            return [indices_sampled.cpu(), probs_sampled.cpu()]
        else:
            probs = in_tensors[0].npu()
            topk = in_tensors[1].npu()
            topp = in_tensors[2].npu()
            probs_sorted, idx_sorted = torch.sort(probs, descending=True, stable=True)
            gather_topk = torch.gather(probs_sorted, dim=1, index=topk)
            topk_mask = torch.lt(probs_sorted, gather_topk).to(torch.bool)
            probs_masked_topk = probs_sorted.masked_fill(topk_mask, 0)
            probs_cumsumed = torch.cumsum(probs_masked_topk, dim=-1)
            if topktopp_sampling_type == 2 or topktopp_sampling_type == 4:
                exp = in_tensors[3].npu()
                topp_mask = torch.gt(probs_cumsumed, topp).to(torch.bool)
                probs_masked_topp = probs_masked_topk.masked_fill(topp_mask, 0)
                divided_probs = torch.div(probs_masked_topp.to(exp.dtype), exp)
                argmax_probs, argmax_idx = torch.sort(divided_probs, descending=True, stable=True)
                argmax_probs = argmax_probs[..., :1]
                argmax_idx = argmax_idx[..., :1]
                outtensor_probs = torch.gather(probs_sorted, dim=1, index=argmax_idx)
                outtensor_idx = torch.gather(idx_sorted, dim=1, index=argmax_idx)
                if topktopp_sampling_type == 4:
                    logProbsSize = json_data["logProbsSize"]
                    mask_tensor = torch.logical_or(topp_mask, topk_mask).cpu()
                    mask_tensor[:, 0] = 0
                    sum_val = np.sum(~mask_tensor.numpy(), axis=-1, keepdims=True) - 1
                    sum_val[sum_val < 0] = 0
                    probs_cumsumed = probs_cumsumed.to(torch.float32).cpu().numpy()
                    logprobs_output = logprobs_golden([probs_sorted, probs_cumsumed, sum_val], logProbsSize)
                    return [outtensor_idx.to(torch.int32).cpu(), outtensor_probs.to(torch.float16).cpu(),
                            logprobs_output.to(torch.float32).cpu()]

                return [outtensor_idx.to(torch.int32).cpu(), outtensor_probs.to(torch.float16).cpu()]
            if topktopp_sampling_type == 1 or topktopp_sampling_type == 3:
                topp = topp.to(torch.float32).cpu().numpy()
                probs_cumsumed = probs_cumsumed.to(torch.float32).cpu().numpy()
                rand = None
                randnp_new = np.zeros(probs_cumsumed.shape[0])
                for i in range(probs_cumsumed.shape[0]):
                    if topktopp_sampling_type == 1:
                        randSeeds = json_data["randSeeds"]
                        libc.srand(randSeeds[i])
                        rand_num = libc.rand() / 0x7fffffff
                        randnp_new[i] = rand_num
                        randnp_new = randnp_new.reshape(-1, 1)
                        rand = randnp_new[0:probs_cumsumed.shape[0]]
                    else:
                        rand = in_tensors[3].cpu().numpy().astype(np.float32)
                idx_return = toppsample_golden_firstpick(probs_cumsumed, topp, rand, 16384)
                bool_judge = (probs_cumsumed < topp)
                sum_val = np.sum(bool_judge, axis=-1, keepdims=True) - 1
                sum_val[sum_val < 0] = 0
                topp_v = np.take_along_axis(probs_cumsumed,  np.minimum((idx_return + 1) * 16384 - 1,sum_val), axis=-1)
                topp_v_new = rand * topp_v
                bool_judge_one = (probs_cumsumed < topp_v_new)
                res = np.sum(bool_judge_one, axis=-1, keepdims=True)
                res[res < 0] = 0
                res_idx = torch.from_numpy(res).to(torch.int64).npu()
                outtensor_probs = torch.gather(probs_sorted, dim=1, index=res_idx)
                outtensor_idx = torch.gather(idx_sorted, dim=1, index=res_idx)
                if topktopp_sampling_type == 3:
                    logProbsSize = json_data["logProbsSize"]
                    logprobs_output = logprobs_golden([probs_sorted, probs_cumsumed, sum_val], logProbsSize)
                    return [outtensor_idx.to(torch.int32).cpu(), outtensor_probs.to(torch.float16).cpu(),
                            logprobs_output.to(torch.float32).cpu()]
                else:
                    return [outtensor_idx.to(torch.int32).cpu(), outtensor_probs.to(torch.float16).cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class NonzeroOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        in_tensors = torch.randint(low=0, high=2, size=shapes[0], dtype=dtype_dict[datatype]).npu()
        in_tensors = torch_npu.npu_format_cast(in_tensors, format_dict[format])
        NonzeroOperation.ret_data = in_tensors
        return in_tensors

    @staticmethod
    def golden(in_tensors, op_params):
        num_non_negative = torch.count_nonzero(in_tensors[0])
        paddingNum = in_tensors[0].numel() - num_non_negative
        padding = torch.zeros((len(in_tensors[0].shape), paddingNum))
        result = torch.stack(list(torch.nonzero(in_tensors[0], as_tuple=True)))
        result = torch.cat((result, padding), dim=-1).long()
        return [result, torch.tensor(num_non_negative).long()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class SwigluQuantOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        in_tensors = torch.empty(shapes[0]).uniform_(0, 1).to(dtype_dict[datatype]).npu()
        in_tensors = torch_npu.npu_format_cast(in_tensors, format_dict[format])
        return in_tensors

    @staticmethod
    def golden(in_tensors, op_params):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def do_swiglu(a, b):
            sigmoid_mul_a = sigmoid(a) * a
            swiglu_y = sigmoid_mul_a * b
            return swiglu_y

        def do_quant(swiglu_y):
            y_tmp = swiglu_y
            y_tmp = np.array(y_tmp)
            y_max = np.amax(np.abs(y_tmp), axis=1)  # 动态量化依赖于每一行的最大值来调整缩放因子
            dynamic_scale_tmp = 127 / y_max
            dynamic_scale = dynamic_scale_tmp.reshape(-1, 1)  # 将一维行向量转换为二维，如(256)转为(256,1)
            y_tmp = y_tmp * dynamic_scale
            quant_y_tmp = np.round(y_tmp)  # 使用 numpy 进行乘法和四舍五入
            quant_y = quant_y_tmp.astype(np.int8)  # 转换为 int8 类型
            dynamic_scale_output = np.array([])
            dynamic_scale = 1 / dynamic_scale
            dynamic_scale_output = np.append(dynamic_scale_output, dynamic_scale)
            return quant_y,dynamic_scale_output

        def SwiGluQuantGolden(x_golden):
            x_golden = np.array(x_golden.cpu().float()).astype(np.float32)
            a, b = np.split(x_golden, 2, axis=1)
            wiglu_y = do_swiglu(a, b)
            quant_y, dynamic_scale = do_quant(wiglu_y)
            return torch.from_numpy(quant_y).npu(), torch.from_numpy(dynamic_scale).npu()

        if in_tensors[0].dtype == torch.bfloat16:
            in_tensors[0] = in_tensors[0].to(torch.float32)
        golden_res = SwiGluQuantGolden(in_tensors[0].cpu())
        return [golden_res[0].to('cpu'), golden_res[1].to('cpu')]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_QUANT

class IndexAddOperation(DataGen):
    in_tensors = []

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        json_data = json.loads(op_params)
        index_type = json_data["indexType"] if "indexType" in json_data else 0
        if i == 1:
            if index_type == 1:
                axis = json_data["axis"] if "axis" in json_data else 0
                d_axis = shapes[0][axis]
                d_x = shapes[1][0]
                if d_axis >= d_x:
                    in_tensor_indices = torch.randperm(d_x, dtype=dtype_dict[datatype])
                else:
                    half0 = torch.randperm(d_axis, dtype=dtype_dict[datatype])
                    half1 = torch.randint(0, d_axis, (d_x - d_axis,), dtype=dtype_dict[datatype])
                    in_tensor_indices = torch.cat([half0, half1])
            elif index_type == 2:
                in_tensor_indices = torch.randint(0, shapes[0][0], tuple(shapes[1]), dtype=dtype_dict[datatype])
            return torch_npu.npu_format_cast(in_tensor_indices.npu(), format_dict[format])
        if i == 3:
            if index_type == 2:
                valid_indices_num = random.randint(0, shapes[1][0])
                in_tensor_valid_indices_num = torch.tensor([valid_indices_num], dtype=dtype_dict[datatype])
            return torch_npu.npu_format_cast(in_tensor_valid_indices_num.npu(), format_dict[format])

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        output_tensor_list[0] = input_tensor_list[0]
        IndexAddOperation.in_tensors = [tensor.cpu() for tensor in input_tensor_list]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        index_type = json_data["indexType"] if "indexType" in json_data else 0
        axis = json_data["axis"] if "axis" in json_data else 0
        in_tensor_var = IndexAddOperation.in_tensors[0]
        in_tensor_indices = IndexAddOperation.in_tensors[1]
        in_tensor_updates = IndexAddOperation.in_tensors[2]

        if index_type == 1:
            in_tensor_alpha = IndexAddOperation.in_tensors[3]
            if IndexAddOperation.in_tensors[0].dtype == torch.bfloat16:
                in_tensor_var = in_tensor_var.float()
                in_tensor_updates = in_tensor_updates.float()
                in_tensor_alpha = in_tensor_alpha.float()
            in_tensor_var.index_add_(axis, in_tensor_indices, in_tensor_updates, alpha=in_tensor_alpha.item())
            return [in_tensor_var]
        if index_type == 2:
            in_tensor_valid_indices_num = IndexAddOperation.in_tensors[3]
            valid_indices_num = in_tensor_valid_indices_num.item()
            if valid_indices_num == 0:
                return [in_tensor_var]
            in_tensor_var = in_tensor_var.float()
            in_tensor_updates = in_tensor_updates.float()
            in_tensor_indices = torch.split(in_tensor_indices, in_tensor_valid_indices_num.item(), dim=0)[0]
            in_tensor_updates = torch.split(in_tensor_updates, in_tensor_valid_indices_num.item(), dim=0)[0]
            in_tensor_var.index_add_(axis, in_tensor_indices, in_tensor_updates, alpha=1)
            return [in_tensor_var]
        return []

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class LayerNormOperation(DataGen):
    LAYER_NORM_NORM = 1
    LAYER_NORM_PRENORM = 2
    LAYER_NORM_POSTNORM = 3

    @staticmethod
    def golden(in_tensors, op_params):
        json_param = json.loads(op_params)
        json_data = []
        if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM:
            json_data = json_param["normParam"]
        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_PRENORM:
            json_data = json_param["preNormParam"]
        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_POSTNORM:
            json_data = json_param["postNormParam"]
        eps = json_data['epsilon'] if 'epsilon' in json_data.keys() else 1e-5
        is_quant = json_data['quantType'] != 0
        quant_scale=1
        quant_offset=0
        quant_alpha=1
        if is_quant and ('dynamicQuantType' not in json_data.keys() or json_data['dynamicQuantType'] == 0):
            quant_scale = in_tensors[3] if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM else in_tensors[4]
            quant_offset = in_tensors[4] if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM else in_tensors[5]

        def layer_norm_quant(layer_norm_res):
            golden_result_quant = (layer_norm_res * quant_scale + quant_offset).float()
            golden_result_quant = torch.round(golden_result_quant)
            golden_result_quant = torch.clamp(golden_result_quant, -128, 127)
            return golden_result_quant.type(torch.int8)

        if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM:
            input = in_tensors[0].float()
            weight = in_tensors[1].float()
            bias = in_tensors[2].float()
            if not is_quant:
                axis = json_data['beginNormAxis'] if 'beginNormAxis' in json_data.keys() else 0
                normalized_shape = in_tensors[0].shape[axis:]
                golden_result = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
            if is_quant:
                if 'dynamicQuantType' in json_data.keys() and json_data['dynamicQuantType'] != 0:
                    layer_norm_result = torch.nn.functional.layer_norm(input, weight.shape, weight, bias, eps).half()
                    dynamic_quant_x = layer_norm_result.cpu().numpy()
                    if json_data['dynamicQuantType'] == 1:
                        input_abs = np.abs(dynamic_quant_x)
                        scale = np.max(input_abs, axis=-1, keepdims=True)
                        scale = scale.astype(np.float32)
                        dynamic_quant_scale = scale / 127
                        dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                        dynamic_quant_x = dynamic_quant_x * 127
                        dynamic_quant_x = dynamic_quant_x / scale
                        dynamic_quant_y = np.round(dynamic_quant_x)
                        return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                                torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32)]
                    if json_data['dynamicQuantType'] == 2:
                        row_max = np.max(dynamic_quant_x, axis=-1, keepdims=True)
                        row_min = np.min(dynamic_quant_x, axis=-1, keepdims=True)
                        row_max = row_max.astype(np.float32)
                        row_min = row_min.astype(np.float32)
                        dynamic_quant_scale = (row_max - row_min) / 255
                        dynamic_quant_offset = - (row_max + row_min) / (2 * dynamic_quant_scale)

                        dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                        dynamic_quant_x = dynamic_quant_x / dynamic_quant_scale
                        dynamic_quant_x = dynamic_quant_x + dynamic_quant_offset
                        dynamic_quant_x = np.clip(dynamic_quant_x, -128, 127)
                        dynamic_quant_y = np.round(dynamic_quant_x)
                        return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                                torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32),
                                torch.from_numpy(dynamic_quant_offset.squeeze(axis=-1)).to(torch.float32)]
                weight = weight.reshape(in_tensors[0].shape[-1])
                bias = bias.reshape(in_tensors[0].shape[-1])
                normalized_shape = (in_tensors[0].shape[-1],)
                layer_norm_res = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps).to(torch.float16)
                golden_result = layer_norm_res.to(torch.float16)
                golden_result_quant = layer_norm_quant(layer_norm_res)

        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_PRENORM:
            weight = in_tensors[2].float()
            bias = in_tensors[3].float()
            weight = weight.reshape(in_tensors[0].shape[-1])
            bias = bias.reshape(in_tensors[0].shape[-1])
            normalized_shape = (in_tensors[0].shape[-1],)
            zoom_scale_value = json_data['zoomScaleValue']
            add_result = torch.add(in_tensors[0].float(), zoom_scale_value * in_tensors[1].float())
            golden_result = torch.nn.functional.layer_norm(add_result, normalized_shape, weight, bias, eps)

        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_POSTNORM:
            weight = in_tensors[2].float()
            bias = in_tensors[3].float()
            weight = weight.reshape(in_tensors[0].shape[-1])
            bias = bias.reshape(in_tensors[0].shape[-1])
            normalized_shape = (in_tensors[0].shape[-1],)
            if not is_quant:
                zoom_scale_value = json_data['zoomScaleValue']
                input = torch.add(in_tensors[0].float(), zoom_scale_value * in_tensors[1].float())
                golden_result = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
            if is_quant:
                input = torch.add(in_tensors[0].float(), in_tensors[1].float())
                layer_norm_res = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps).to(torch.float16)
                golden_result = (layer_norm_res * quant_alpha).to(torch.float16)
                golden_result_quant = layer_norm_quant(layer_norm_res)
        else:
            return -1

        if not is_quant:
            if json_param['layerType'] == LayerNormOperation.LAYER_NORM_PRENORM:
                if in_tensors[0].dtype == torch.float16:
                    return [golden_result.half(), add_result.half()]
                else:
                    return [golden_result, add_result]
            else:
                return [golden_result.half()] if in_tensors[0].dtype == torch.float16 else [golden_result]
        else:
            if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM:
                return [golden_result_quant]
            return [golden_result, golden_result_quant]

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        pass

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["layerType"] == LayerNormOperation.LAYER_NORM_NORM \
        and json_data["normParam"]["quantType"] == 0:
            return OpTypes.COMPUTE_FLOAT
        elif json_data["layerType"] == LayerNormOperation.LAYER_NORM_NORM \
        and json_data["normParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        elif ((json_data["layerType"] == LayerNormOperation.LAYER_NORM_PRENORM and \
               json_data["preNormParam"]["quantType"] == 0) or \
              (json_data["layerType"] == LayerNormOperation.LAYER_NORM_POSTNORM and \
               json_data["postNormParam"]["quantType"] == 0)):
            return OpTypes.VECTOR_FUSION
        else:
            return OpTypes.COMPUTE_QUANT

class LayerNormWithStrideOperation(DataGen):
    LAYER_NORM_NORM = 1
    LAYER_NORM_PRENORM = 2
    LAYER_NORM_POSTNORM = 3

    @staticmethod
    def golden(in_tensors, op_params):
        json_param = json.loads(op_params)
        json_data = []
        if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM:
            json_data = json_param["normParam"]
        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_PRENORM:
            json_data = json_param["preNormParam"]
        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_POSTNORM:
            json_data = json_param["postNormParam"]
        eps = json_data['epsilon'] if 'epsilon' in json_data.keys() else 1e-5
        is_quant = json_data['quantType'] != 0
        quant_scale=1
        quant_offset=0
        quant_alpha=1
        if is_quant and ('dynamicQuantType' not in json_data.keys() or json_data['dynamicQuantType'] == 0):
            quant_scale = in_tensors[3] if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM else in_tensors[4]
            quant_offset = in_tensors[4] if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM else in_tensors[5]

        def layer_norm_quant(layer_norm_res):
            golden_result_quant = (layer_norm_res * quant_scale + quant_offset).float()
            golden_result_quant = torch.round(golden_result_quant)
            golden_result_quant = torch.clamp(golden_result_quant, -128, 127)
            return golden_result_quant.type(torch.int8)

        if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM:
            input = in_tensors[0].float()
            weight = in_tensors[1].float()
            bias = in_tensors[2].float()
            if not is_quant:
                xshape = in_tensors[0].shape
                stride = in_tensors[3].tolist()
                golden_result = torch.nn.functional.layer_norm(input.as_strided(xshape, stride, 0), in_tensors[1].shape, weight, bias, eps)
                golden_result = golden_result.as_strided(xshape, stride, 0)
            if is_quant:
                if 'dynamicQuantType' in json_data.keys() and json_data['dynamicQuantType'] != 0:
                    layer_norm_result = torch.nn.functional.layer_norm(input, weight.shape, weight, bias, eps).half()
                    dynamic_quant_x = layer_norm_result.cpu().numpy()
                    if json_data['dynamicQuantType'] == 1:
                        input_abs = np.abs(dynamic_quant_x)
                        scale = np.max(input_abs, axis=-1, keepdims=True)
                        scale = scale.astype(np.float32)
                        dynamic_quant_scale = scale / 127
                        dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                        dynamic_quant_x = dynamic_quant_x * 127
                        dynamic_quant_x = dynamic_quant_x / scale
                        dynamic_quant_y = np.round(dynamic_quant_x)
                        return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                                torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32)]
                    if json_data['dynamicQuantType'] == 2:
                        row_max = np.max(dynamic_quant_x, axis=-1, keepdims=True)
                        row_min = np.min(dynamic_quant_x, axis=-1, keepdims=True)
                        row_max = row_max.astype(np.float32)
                        row_min = row_min.astype(np.float32)
                        dynamic_quant_scale = (row_max - row_min) / 255
                        dynamic_quant_offset = - (row_max + row_min) / (2 * dynamic_quant_scale)

                        dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                        dynamic_quant_x = dynamic_quant_x / dynamic_quant_scale
                        dynamic_quant_x = dynamic_quant_x + dynamic_quant_offset
                        dynamic_quant_x = np.clip(dynamic_quant_x, -128, 127)
                        dynamic_quant_y = np.round(dynamic_quant_x)
                        return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                                torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32),
                                torch.from_numpy(dynamic_quant_offset.squeeze(axis=-1)).to(torch.float32)]
                normalized_shape = (1, in_tensors[0].shape[-1])
                layer_norm_res = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps).to(torch.float16)
                golden_result = layer_norm_res.to(torch.float16)
                golden_result_quant = layer_norm_quant(layer_norm_res)

        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_PRENORM:
            weight = in_tensors[2].float()
            bias = in_tensors[3].float()
            weight = weight.reshape(in_tensors[0].shape[-1])
            bias = bias.reshape(in_tensors[0].shape[-1])
            normalized_shape = (in_tensors[0].shape[-1],)
            zoom_scale_value = json_data['zoomScaleValue']
            add_result = torch.add(in_tensors[0].float(), zoom_scale_value * in_tensors[1].float())
            golden_result = torch.nn.functional.layer_norm(add_result, normalized_shape, weight, bias, eps)

        elif json_param['layerType'] == LayerNormOperation.LAYER_NORM_POSTNORM:
            weight = in_tensors[2].float()
            bias = in_tensors[3].float()
            weight = weight.reshape(in_tensors[0].shape[-1])
            bias = bias.reshape(in_tensors[0].shape[-1])
            normalized_shape = (in_tensors[0].shape[-1],)
            if not is_quant:
                zoom_scale_value = json_data['zoomScaleValue']
                input = torch.add(in_tensors[0].float(), zoom_scale_value * in_tensors[1].float())
                golden_result = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
            if is_quant:
                input = torch.add(in_tensors[0].float(), in_tensors[1].float())
                layer_norm_res = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps).to(torch.float16)
                golden_result = (layer_norm_res * quant_alpha).to(torch.float16)
                golden_result_quant = layer_norm_quant(layer_norm_res)
        else:
            return -1

        if not is_quant:
            if json_param['layerType'] == LayerNormOperation.LAYER_NORM_PRENORM:
                if in_tensors[0].dtype == torch.float16:
                    return [golden_result.half(), add_result.half()]
                else:
                    return [golden_result, add_result]
            else:
                return [golden_result.half()] if in_tensors[0].dtype == torch.float16 else [golden_result]
        else:
            if json_param['layerType'] == LayerNormOperation.LAYER_NORM_NORM:
                return [golden_result_quant]
            return [golden_result, golden_result_quant]

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["layerType"] == LayerNormOperation.LAYER_NORM_NORM \
        and json_data["normParam"]["quantType"] == 0:
            return OpTypes.COMPUTE_FLOAT
        elif json_data["layerType"] == LayerNormOperation.LAYER_NORM_NORM \
        and json_data["normParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        elif ((json_data["layerType"] == LayerNormOperation.LAYER_NORM_PRENORM and \
               json_data["preNormParam"]["quantType"] == 0) or \
              (json_data["layerType"] == LayerNormOperation.LAYER_NORM_POSTNORM and \
               json_data["postNormParam"]["quantType"] == 0)):
            return OpTypes.VECTOR_FUSION
        else:
            return OpTypes.COMPUTE_QUANT

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        stride = input_tensor_list[3].tolist()
        offset = input_tensor_list[4].tolist()

        run_param = json.dumps({"strides": stride, "offset": offset})
        operation.set_varaintpack_param(run_param)

    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        if i <= 2:
            return torch_npu.npu_format_cast(torch.tensor(np.random.uniform(low, high, size=shapes[i])).to(dtype_dict[datatype]).npu(), format_dict[format])
        if i == 3:
            strides = [1]
            for i in range(1,len(shapes[0])):
                strides.insert(0, int((shapes[0][-1] + 32 - 1) // 32 * 32))
            return torch_npu.npu_format_cast(torch.tensor(strides).to(dtype_dict[datatype]).npu(), format_dict[format])
        if i == 4:
            product = 0
            return torch_npu.npu_format_cast(torch.tensor([product]).to(dtype_dict[datatype]).npu(), format_dict[format])

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        stride = input_tensor_list[3].tolist()
        xshape = input_tensor_list[0].shape
        output_tensor_list[0] = output_tensor_list[0].as_strided(xshape, stride, 0)

class RmsNormOperation(DataGen):
    RMS_NORM_NORM = 1
    RMS_NORM_PRENORM = 2
    RMS_NORM_POSTNORM = 3

    @staticmethod
    def golden(in_tensors, op_params):
        json_param = json.loads(op_params)
        json_data = {}
        if json_param['layerType'] == RmsNormOperation.RMS_NORM_NORM:
            json_data = json_param["normParam"]
        elif json_param['layerType'] == RmsNormOperation.RMS_NORM_PRENORM:
            json_data = json_param["preNormParam"]
        elif json_param['layerType'] == RmsNormOperation.RMS_NORM_POSTNORM:
            json_data = json_param["postNormParam"]
        eps = json_data['epsilon'] if 'epsilon' in json_data.keys() else 0.00001
        x = in_tensors[0].float()
        gamma_origin = in_tensors[1]
        gamma_flatten_fp16 = gamma_origin.view(1, -1)
        gamma = gamma_flatten_fp16.float()
        if json_param['layerType'] == 2 and json_data['quantType'] == 2:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        if json_param['layerType'] == 3 and json_data['quantType'] == 2:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        if (json_param['layerType'] == 3 and json_data['quantType'] == 0) or (json_param['layerType'] == 2 and json_data['quantType'] == 0):
            idx = 1
            if 'hasBias' in json_data.keys() and json_data['hasBias']:
                x = x + in_tensors[idx].float()
                idx += 1
            x = x + in_tensors[idx].float()
            idx += 1
            gamma = in_tensors[idx].float()
        if 'modelType' in json_data.keys() and json_data['modelType'] == 1:
            gamma = 1 + gamma
        gamma_size = float(gamma.size(-1))
        norm =  torch.sum(x / gamma_size * x, dim=-1, keepdim=True) + eps
        if 'precisionMode' in json_data.keys() and json_data['precisionMode'] == 1:
            golden_output = (x / torch.sqrt(norm)).half() * gamma_flatten_fp16
        elif json_param['layerType'] == 1 and 'rstd' in json_data.keys() and json_data['rstd']:
            gamma = gamma_origin.float()
            reduceDims=[]
            edim = x.dim()-gamma.dim()
            for i in range(gamma.dim()):
                reduceDims.append(edim + i)
            rstd = torch.rsqrt(x.pow(2).mean(reduceDims, keepdim=True) + eps)
            return [(x * rstd) * gamma, rstd]
        else:
            golden_output = x * gamma / torch.sqrt(norm)

        def rms_norm_quant(golden_output, beta):
            golden_output = golden_output.float()
            beta = beta.float()
            quant_scale = json_data['quantInputScale'] if 'quantInputScale' in json_data.keys() else 1
            quant_offset = json_data['quantInputOffset'] if 'quantInputOffset' in json_data.keys() else 0
            golden_output = golden_output + beta
            golden_output = golden_output / quant_scale + quant_offset
            golden_output = torch.clamp(golden_output, -128, 127)
            golden_result_quant = torch.round(golden_output)
            return golden_result_quant.type(torch.int8)

        def rms_norm_quant_with_tensor(golden_output, beta, scale, offset):
            golden_output = golden_output.float()
            beta = beta.float()
            scale = scale.float()
            golden_output = golden_output + beta
            golden_output = golden_output / scale + offset
            golden_output = torch.round(golden_output)
            golden_output = golden_output.half()
            golden_result_quant = torch.clamp(golden_output, -128, 127)
            return golden_result_quant.type(torch.int8)

        def rms_postnorm_quant_with_tensor(golden_output, scale, offset):
            golden_output = golden_output.float()
            scale = scale.float()
            golden_output = golden_output / scale + offset
            golden_output = torch.round(golden_output)
            golden_output = golden_output.half()
            golden_result_quant = torch.clamp(golden_output, -128, 127)
            return golden_result_quant.type(torch.int8)

        if json_param['layerType'] == 2 and json_data['quantType'] == 2:
            golden_result = [rms_norm_quant_with_tensor(golden_output, in_tensors[3],in_tensors[4],in_tensors[5]), x]
        elif json_param['layerType'] == 3 and json_data['quantType'] == 2:
            golden_result = [rms_postnorm_quant_with_tensor(golden_output, in_tensors[3],in_tensors[4]), golden_output]
        elif json_param['layerType'] == 1 and json_data['quantType'] == 2:
            if 'dynamicQuantType' not in json_data.keys() or json_data['dynamicQuantType'] == 0:
                golden_result = [rms_norm_quant_with_tensor(golden_output, in_tensors[2],in_tensors[3],in_tensors[4])]
            else:
                golden_output = golden_output + in_tensors[2].float()
                dynamic_quant_x = golden_output.cpu().numpy()
                if json_data['dynamicQuantType'] == 1:
                    input_abs = np.abs(dynamic_quant_x)
                    scale = np.max(input_abs, axis=-1, keepdims=True)
                    scale = scale.astype(np.float32)
                    dynamic_quant_scale = scale / 127
                    dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                    dynamic_quant_x = dynamic_quant_x * 127
                    dynamic_quant_x = dynamic_quant_x / scale
                    dynamic_quant_y = np.round(dynamic_quant_x)
                    return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                            torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32)]
                if json_data['dynamicQuantType'] == 2:
                    row_max = np.max(dynamic_quant_x, axis=-1, keepdims=True)
                    row_min = np.min(dynamic_quant_x, axis=-1, keepdims=True)
                    row_max = row_max.astype(np.float32)
                    row_min = row_min.astype(np.float32)
                    dynamic_quant_scale = (row_max - row_min) / 255
                    dynamic_quant_offset = - (row_max + row_min) / (2 * dynamic_quant_scale)

                    dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                    dynamic_quant_x = dynamic_quant_x / dynamic_quant_scale
                    dynamic_quant_x = dynamic_quant_x + dynamic_quant_offset
                    dynamic_quant_x = np.clip(dynamic_quant_x, -128, 127)
                    dynamic_quant_y = np.round(dynamic_quant_x)
                    return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                            torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32),
                            torch.from_numpy(dynamic_quant_offset.squeeze(axis=-1)).to(torch.float32)]
        elif json_param['layerType'] == 2 and json_data['quantType'] == 0:
            golden_result = [golden_output.half(), x.half()]
        else:
            golden_result = [golden_output.half()]
        return golden_result

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["layerType"] == RmsNormOperation.RMS_NORM_NORM \
        and json_data["normParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        elif json_data["layerType"] == RmsNormOperation.RMS_NORM_PRENORM \
        and json_data["preNormParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        elif json_data["layerType"] == RmsNormOperation.RMS_NORM_POSTNORM \
        and json_data["postNormParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        return OpTypes.VECTOR_FUSION


class RmsNormWithStrideOperation(DataGen):
    RMS_NORM_NORM = 1
    RMS_NORM_PRENORM = 2
    RMS_NORM_POSTNORM = 3

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        stride = input_tensor_list[2].tolist()
        offset = input_tensor_list[3].tolist()

        run_param = json.dumps({"strides": stride, "offset": offset})
        operation.set_varaintpack_param(run_param)

    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        if i <= 1:
            return torch_npu.npu_format_cast(torch.tensor(np.random.uniform(low, high, size=shapes[i])).to(dtype_dict[datatype]).npu(), format_dict[format])
        if i == 2:
            strides = []
            for i in range(1,len(shapes[0])):
                stride = 1
                for j in range(i,len(shapes[0])):
                    stride *= shapes[0][j]
                strides.append(stride)
            strides.append(1)
            return torch_npu.npu_format_cast(torch.tensor(strides).to(dtype_dict[datatype]).npu(), format_dict[format])
        if i == 3:
            product = 0
            return torch_npu.npu_format_cast(torch.tensor([product]).to(dtype_dict[datatype]).npu(), format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        json_param = json.loads(op_params)
        json_data = {}
        if json_param['layerType'] == RmsNormWithStrideOperation.RMS_NORM_NORM:
            json_data = json_param["normParam"]
        elif json_param['layerType'] == RmsNormWithStrideOperation.RMS_NORM_PRENORM:
            json_data = json_param["preNormParam"]
        elif json_param['layerType'] == RmsNormWithStrideOperation.RMS_NORM_POSTNORM:
            json_data = json_param["postNormParam"]
        eps = json_data['epsilon'] if 'epsilon' in json_data.keys() else 0.00001
        x = in_tensors[0].float()
        gamma_origin = in_tensors[1]
        gamma_flatten_fp16 = gamma_origin.view(1, -1)
        gamma = gamma_flatten_fp16.float()
        if json_param['layerType'] == 2 and json_data['quantType'] == 2:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        if json_param['layerType'] == 3 and json_data['quantType'] == 2:
            x = x + in_tensors[1].float()
            gamma = in_tensors[2].float()
        if (json_param['layerType'] == 3 and json_data['quantType'] == 0) or (json_param['layerType'] == 2 and json_data['quantType'] == 0):
            idx = 1
            if 'hasBias' in json_data.keys() and json_data['hasBias']:
                x = x + in_tensors[idx].float()
                idx += 1
            x = x + in_tensors[idx].float()
            idx += 1
            gamma = in_tensors[idx].float()
        if 'modelType' in json_data.keys() and json_data['modelType'] == 1:
            gamma = 1 + gamma
        gamma_size = float(gamma.size(-1))
        norm =  torch.sum(x / gamma_size * x, dim=-1, keepdim=True) + eps
        if 'precisionMode' in json_data.keys() and json_data['precisionMode'] == 1:
            xstride = in_tensors[2]
            xoffset = in_tensors[3]
            max_dims = []
            for i, stride in enumerate(xstride):
                max_dim = (x.numel() - xoffset) // stride
                max_dims.append(max_dim)
            size = tuple(max_dims)
            stride_x = torch.as_strided(x, size=size, stride=xstride, storage_offset=xoffset)
            golden_output = (stride_x / torch.sqrt(norm)).half() * gamma_flatten_fp16
        elif json_param['layerType'] == 1 and 'rstd' in json_data.keys() and json_data['rstd']:
            gamma = gamma_origin.float()
            reduceDims=[]
            edim = x.dim()-gamma.dim()
            for i in range(gamma.dim()):
                reduceDims.append(edim + i)
            rstd = torch.rsqrt(x.pow(2).mean(reduceDims, keepdim=True) + eps)
            return [(x * rstd) * gamma, rstd]
        else:
            golden_output = x * gamma / torch.sqrt(norm)

        def rms_norm_quant(golden_output, beta):
            golden_output = golden_output.float()
            beta = beta.float()
            quant_scale = json_data['quantInputScale'] if 'quantInputScale' in json_data.keys() else 1
            quant_offset = json_data['quantInputOffset'] if 'quantInputOffset' in json_data.keys() else 0
            golden_output = golden_output + beta
            golden_output = golden_output / quant_scale + quant_offset
            golden_output = torch.clamp(golden_output, -128, 127)
            golden_result_quant = torch.round(golden_output)
            return golden_result_quant.type(torch.int8)

        def rms_norm_quant_with_tensor(golden_output, beta, scale, offset):
            golden_output = golden_output.float()
            beta = beta.float()
            scale = scale.float()
            golden_output = golden_output + beta
            golden_output = golden_output / scale + offset
            golden_output = torch.round(golden_output)
            golden_output = golden_output.half()
            golden_result_quant = torch.clamp(golden_output, -128, 127)
            return golden_result_quant.type(torch.int8)

        def rms_postnorm_quant_with_tensor(golden_output, scale, offset):
            golden_output = golden_output.float()
            scale = scale.float()
            golden_output = golden_output / scale + offset
            golden_output = torch.round(golden_output)
            golden_output = golden_output.half()
            golden_result_quant = torch.clamp(golden_output, -128, 127)
            return golden_result_quant.type(torch.int8)

        if json_param['layerType'] == 2 and json_data['quantType'] == 2:
            golden_result = [rms_norm_quant_with_tensor(golden_output, in_tensors[3],in_tensors[4],in_tensors[5]), x]
        elif json_param['layerType'] == 3 and json_data['quantType'] == 2:
            golden_result = [rms_postnorm_quant_with_tensor(golden_output, in_tensors[3],in_tensors[4]), golden_output]
        elif json_param['layerType'] == 1 and json_data['quantType'] == 2:
            if 'dynamicQuantType' not in json_data.keys() or json_data['dynamicQuantType'] == 0:
                golden_result = [rms_norm_quant_with_tensor(golden_output, in_tensors[2],in_tensors[3],in_tensors[4])]
            else:
                golden_output = golden_output + in_tensors[2].float()
                dynamic_quant_x = golden_output.cpu().numpy()
                if json_data['dynamicQuantType'] == 1:
                    input_abs = np.abs(dynamic_quant_x)
                    scale = np.max(input_abs, axis=-1, keepdims=True)
                    scale = scale.astype(np.float32)
                    dynamic_quant_scale = scale / 127
                    dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                    dynamic_quant_x = dynamic_quant_x * 127
                    dynamic_quant_x = dynamic_quant_x / scale
                    dynamic_quant_y = np.round(dynamic_quant_x)
                    return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                            torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32)]
                if json_data['dynamicQuantType'] == 2:
                    row_max = np.max(dynamic_quant_x, axis=-1, keepdims=True)
                    row_min = np.min(dynamic_quant_x, axis=-1, keepdims=True)
                    row_max = row_max.astype(np.float32)
                    row_min = row_min.astype(np.float32)
                    dynamic_quant_scale = (row_max - row_min) / 255
                    dynamic_quant_offset = - (row_max + row_min) / (2 * dynamic_quant_scale)

                    dynamic_quant_x = dynamic_quant_x.astype(np.float32)
                    dynamic_quant_x = dynamic_quant_x / dynamic_quant_scale
                    dynamic_quant_x = dynamic_quant_x + dynamic_quant_offset
                    dynamic_quant_x = np.clip(dynamic_quant_x, -128, 127)
                    dynamic_quant_y = np.round(dynamic_quant_x)
                    return [torch.from_numpy(dynamic_quant_y).to(torch.int8),
                            torch.from_numpy(dynamic_quant_scale.squeeze(axis=-1)).to(torch.float32),
                            torch.from_numpy(dynamic_quant_offset.squeeze(axis=-1)).to(torch.float32)]
        elif json_param['layerType'] == 2 and json_data['quantType'] == 0:
            golden_result = [golden_output.half(), x.half()]
        else:
            golden_result = [golden_output.half()]
        return golden_result

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["layerType"] == RmsNormWithStrideOperation.RMS_NORM_NORM \
        and json_data["normParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        elif json_data["layerType"] == RmsNormWithStrideOperation.RMS_NORM_PRENORM \
        and json_data["preNormParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        elif json_data["layerType"] == RmsNormWithStrideOperation.RMS_NORM_POSTNORM \
        and json_data["postNormParam"]["quantType"] != 0:
            return OpTypes.COMPUTE_QUANT
        return OpTypes.VECTOR_FUSION


class RmsNormBackwardOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        dy_npu = in_tensors[0].npu()
        x_npu = in_tensors[1].npu().float()
        rstd_npu = in_tensors[2].npu()
        gamma_npu = in_tensors[3].npu()
        S = gamma_npu.numel()
        gamma_npu_dtype = gamma_npu.dtype
        dim_l=[]
        edim=x_npu.dim()-gamma_npu.dim()
        for i in range(gamma_npu.dim()):
            dim_l.append(edim+i)
        dgamma = torch.sum(((dy_npu * (x_npu * rstd_npu).to(gamma_npu_dtype)).float()).reshape([-1, S]),
                           dim=0, keepdim=True).reshape(gamma_npu.shape)
        dx = dy_npu * gamma_npu * rstd_npu - \
            torch.mean(dy_npu * gamma_npu * rstd_npu.pow(3) * x_npu, dim=dim_l, keepdim=True) * x_npu
        return [dx.cpu(), dgamma.cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.VECTOR_FUSION

    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i == 2 or i == 1:
            random_seed = int(os.getenv("random_seed",0))
            torch.manual_seed(random_seed)
        if i == 2:
            origin = DataGen.random(shapes[1], datatype, format, data_gen_ranges, op_params)
            return torch_npu.npu_format_cast(torch.rsqrt(origin.to(torch.float).pow(2).mean(-1, keepdim=True) + 1e-6), format_dict[format])
        origin = DataGen.random(shapes[i], datatype, format, data_gen_ranges, op_params)
        return origin

class AsStridedOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        golden_result = torch.as_strided(in_tensors[0], json_data['size'], json_data['stride'], json_data['offset'][0])
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class MultinomialOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        origin = DataGen.random(shapes[i], datatype, format, data_gen_ranges, op_params)
        sm = torch.nn.Softmax(dim=-1)
        return sm(origin)

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        samples = json_data["numSamples"]
        rand_seed = json_data["randSeed"]
        input0 = in_tensors[0].cpu().numpy()
        libc = CDLL("libc.so.6")
        libc.srand(rand_seed)
        rand_list = [libc.rand() / 0x7fffffff for i in range(64)]
        ret = np.zeros(shape=(input0.shape[0], samples))

        sumList = np.cumsum(input0, axis=-1, dtype=np.float16).astype(np.float16)
        for z in range(0, samples):
            for j in range(0, input0.shape[0]):
                for i in range(0, input0.shape[1]):
                    if (sumList[j][i] > rand_list[z]):
                        ret[j][z] = i
                        break
        return [torch.from_numpy(ret.astype(np.int32)).contiguous()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class SliceOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        offsetList = json_data['offsets']
        sizeList = json_data['size']
        for index, offset in enumerate(offsetList):
            offsetList[index] = offset if offset >= 0 else offset + in_tensors[0].shape[index]
        for index, size in enumerate(sizeList):
            sizeList[index] = size if size != -1 else in_tensors[0].shape[index] - offsetList[index]
        # 根据维度数进行切片
        if len(offsetList) == 1:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0]]]
        elif len(offsetList) == 2:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1]]]
        elif len(offsetList) == 3:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1],
                                offsetList[2]:offsetList[2] + sizeList[2]]]
        elif len(offsetList) == 4:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1],
                                offsetList[2]:offsetList[2] + sizeList[2],
                                offsetList[3]:offsetList[3] + sizeList[3]]]
        elif len(offsetList) == 5:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1],
                                offsetList[2]:offsetList[2] + sizeList[2],
                                offsetList[3]:offsetList[3] + sizeList[3],
                                offsetList[4]:offsetList[4] + sizeList[4]]]
        elif len(offsetList) == 6:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1],
                                offsetList[2]:offsetList[2] + sizeList[2],
                                offsetList[3]:offsetList[3] + sizeList[3],
                                offsetList[4]:offsetList[4] + sizeList[4],
                                offsetList[5]:offsetList[5] + sizeList[5]]]
        elif len(offsetList) == 7:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1],
                                offsetList[2]:offsetList[2] + sizeList[2],
                                offsetList[3]:offsetList[3] + sizeList[3],
                                offsetList[4]:offsetList[4] + sizeList[4],
                                offsetList[5]:offsetList[5] + sizeList[5],
                                offsetList[6]:offsetList[6] + sizeList[6]]]
        elif len(offsetList) == 8:
            return [in_tensors[0][offsetList[0]:offsetList[0] + sizeList[0],
                                offsetList[1]:offsetList[1] + sizeList[1],
                                offsetList[2]:offsetList[2] + sizeList[2],
                                offsetList[3]:offsetList[3] + sizeList[3],
                                offsetList[4]:offsetList[4] + sizeList[4],
                                offsetList[5]:offsetList[5] + sizeList[5],
                                offsetList[6]:offsetList[6] + sizeList[6],
                                offsetList[7]:offsetList[7] + sizeList[7]]]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class SoftmaxOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        in_tensor_dim_num = in_tensors[0].dim()
        if in_tensors[0].dtype not in [dtype_dict["float"], dtype_dict["int64"], dtype_dict["bf16"]]:
            in_tensors[0] = in_tensors[0].to(torch.float32)
        json_data["axes"] = [ i % in_tensor_dim_num for i in json_data["axes"] ]
        target_shape = in_tensors[0].shape[json_data["axes"][0]:json_data["axes"][-1] + 1]
        in_tensor_flatten = in_tensors[0].flatten(start_dim=json_data["axes"][0], end_dim=json_data["axes"][-1])
        softmax0 = torch.nn.Softmax(dim=json_data["axes"][0])
        out_tensor = softmax0(in_tensor_flatten)
        out_tensor = out_tensor.unflatten(json_data["axes"][0], target_shape)
        return [out_tensor]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class ReduceOperation(DataGen):
    REDUCE_UNDEFINED = 0
    REDUCE_MAX = 1
    REDUCE_MIN = 2
    REDUCE_SUM = 3
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        opType = json_data['reduceType']
        axis = json_data['axis']
        if opType == 1:
            return [in_tensors[0].amax(axis)]
        elif opType == 2:
            return [in_tensors[0].amin(axis)]
        else:
            return [torch.sum(in_tensors[0], axis)]
    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["reduceType"] in [
            ReduceOperation.REDUCE_MAX,
            ReduceOperation.REDUCE_MIN
        ]:
            return OpTypes.MOVE
        else:
            return OpTypes.COMPUTE_FLOAT


class RopeOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if (i == 2 or i == 3) and json.loads(op_params)["rotaryCoeff"] != 2 and json.loads(op_params)["rotaryCoeff"] != 4:
            ntoken = shapes[i][0]
            head_size = shapes[i][1]
            # op需要cos/sin重复一次
            if datatype == "bf16":
                return torch.rand(ntoken, head_size // 2, 1).repeat(1, 1, 2).view(ntoken, head_size).to(torch.bfloat16).npu()
            else:
                return torch.rand(ntoken, head_size // 2, 1).repeat(1, 1, 2).view(ntoken, head_size).half().npu()
        if i != 0:
            return RopeOperation.unpadRetdata[i]

        min_seqlen = 1
        max_seqlen = 5
        batch = 1
        logging.info(f"{shapes=}")
        if get_soc_version() == "Ascend950":
            batch = shapes[4][0]
            batch = max(batch, 1)
            if len(shapes[0]) == 4: # query shape: 4d, [b,s,n,d]
                min_seqlen = shapes[0][1] // batch
            elif len(shapes[0]) == 2:
                min_seqlen = shapes[0][0] // batch
            max_seqlen = min_seqlen + 1
        seqlen = torch.randint(min_seqlen, max_seqlen, (batch,), dtype=torch.int32)
        json_data = json.loads(op_params)

        logging.info(f"{min_seqlen=} {max_seqlen=} {batch=}")

        logging.info(f"{seqlen=}")

        if "seqlen" in json_data:
            seqlen = torch.tensor(json_data["seqlen"], dtype=torch.int32)
        ntoken = int(seqlen.sum())
        hidden_size = shapes[0][1]
        head_size = shapes[2][1]
        intensor0 = torch.rand(ntoken, hidden_size).npu().half()
        intensor1 = torch.rand(ntoken, hidden_size).npu().half()
        intensor2 = torch.rand(ntoken, head_size).npu().half()
        intensor3 = torch.rand(ntoken, head_size).npu().half()
        intensor4 = seqlen.npu()
        if datatype == "float16":
            RopeOperation.unpadRetdata = [intensor0, intensor1, intensor2, intensor3, intensor4]
        else:
            RopeOperation.unpadRetdata = [intensor0.to(torch.bfloat16), intensor1.to(torch.bfloat16), intensor2.to(torch.bfloat16), intensor3.to(torch.bfloat16), intensor4]
        return RopeOperation.unpadRetdata[0]

    @staticmethod
    def zero(shape, datatype, format, data_gen_ranges, op_params):
        if hasattr(__class__, 'unpadRetdata'):
            realshape = __class__.unpadRetdata[0].size()
        else:
            realshape = shape
        data = torch.zeros(realshape, dtype=dtype_dict[datatype]).npu()
        return torch_npu.npu_format_cast(data, format_dict[format])

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        # 使用完后删除暂存shape
        if hasattr(__class__, 'unpadRetdata'):
            delattr(__class__, 'unpadRetdata')

    @staticmethod
    def rotate_half(x):
        x0, x1 = x.chunk(2, -1)
        return torch.cat((-x1, x0), dim=x0.ndim - 1)

    @staticmethod
    def RopeA5_gloden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if in_tensors[0].dtype in (torch.bfloat16, torch.float16):
            in_tensors[0] = in_tensors[0].to(torch.float32)
            in_tensors[1] = in_tensors[1].to(torch.float32)
            in_tensors[2] = in_tensors[2].to(torch.float32)
            in_tensors[3] = in_tensors[3].to(torch.float32)
            dtype = np.float32
            origin_dtype = in_tensors[0].dtype
        else:
            dtype = np.float16
        q = np.array(in_tensors[0].cpu()).astype(dtype)
        kk = np.array(in_tensors[1].cpu()).astype(dtype)
        cos = np.array(in_tensors[2].cpu()).astype(dtype)
        sin = np.array(in_tensors[3].cpu()).astype(dtype)
        seqlen = np.array(in_tensors[4].cpu()).astype(np.int32)
        batch = seqlen.shape[0]
        rotaryCoeff = json_data['rotaryCoeff']
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
        ret = [torch.tensor(rope_q), torch.tensor(rope_k)]
        if dtype == np.float32:
            ret = [torch.tensor(rope_q).to(origin_dtype), torch.tensor(rope_k).to(origin_dtype)]
        return ret

    @staticmethod
    def golden(in_tensors, op_params):
        if get_soc_version() == "Ascend950":
            return RopeOperation.RopeA5_gloden(in_tensors, op_params)
        json_data = json.loads(op_params)
        dtype = in_tensors[0].dtype
        if dtype == torch.bfloat16:
            in_tensors[0] = in_tensors[0].to(torch.float32)
            in_tensors[1] = in_tensors[1].to(torch.float32)
            in_tensors[2] = in_tensors[2].to(torch.float32)
            in_tensors[3] = in_tensors[3].to(torch.float32)
        if json_data['rotaryCoeff'] == 4:
            if in_tensors[4].size()[0] == 3:
                ntoken = in_tensors[0].size()[0]
                seqlen = in_tensors[4].tolist()
                batch = in_tensors[4].shape[0]
                hidden_size = in_tensors[0].size()[1]
                head_size = in_tensors[2].size()[1]
                head_num = hidden_size // head_size
                q_list = []
                k_list = []
                offset = 0
                for i, _ in enumerate(range(batch)):
                    cur_seqlen = seqlen[i]
                    next_offset = offset + cur_seqlen
                    qlayer = in_tensors[0][offset:next_offset].view(cur_seqlen, head_num, head_size)
                    q0, q1 = qlayer.chunk(2, -1)
                    klayer = in_tensors[1][offset:next_offset].view(cur_seqlen, in_tensors[1].size()[1] // head_size, head_size)
                    k0, k1 = klayer.chunk(2, -1)
                    cos0, cos1 = in_tensors[2][offset:next_offset].unsqueeze(1).chunk(2, -1)
                    sin0, sin1 = in_tensors[3][offset:next_offset].unsqueeze(1).chunk(2, -1)
                    q0 = (q0 * cos0) + (RopeOperation.rotate_half(q0) * sin0)
                    k0 = (k0 * cos0) + (RopeOperation.rotate_half(k0) * sin0)
                    q1 = (q1 * cos1) + (RopeOperation.rotate_half(q1) * sin1)
                    k1 = (k1 * cos1) + (RopeOperation.rotate_half(k1) * sin1)
                    q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(cur_seqlen, hidden_size)
                    q_list.append(q)
                    k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(cur_seqlen, in_tensors[1].size()[1])
                    k_list.append(k)
                    offset = next_offset
                q_sum = torch.cat(tuple(q_list), dim=0)
                k_sum = torch.cat(tuple(k_list), dim=0)
                del RopeOperation.unpadRetdata
                return [q_sum, k_sum]
            else:
                ntoken = in_tensors[0].size()[0]
                seqlen = int(in_tensors[4][0])
                batch = ntoken // seqlen
                hidden_size = in_tensors[0].size()[1]
                head_size = in_tensors[2].size()[1]
                head_num = hidden_size // head_size
                qlayer = in_tensors[0].view(seqlen, batch, head_num, head_size)
                q0, q1 = qlayer.chunk(2, -1)
                klayer = in_tensors[1].view(seqlen, batch, head_num, head_size)
                k0, k1 = klayer.chunk(2, -1)
                cos0, cos1 = in_tensors[2].view(seqlen, batch, 1, head_size).chunk(2, -1)
                sin0, sin1 = in_tensors[3].view(seqlen, batch, 1, head_size).chunk(2, -1)
                q0 = (q0 * cos0) + (RopeOperation.rotate_half(q0) * sin0)
                k0 = (k0 * cos0) + (RopeOperation.rotate_half(k0) * sin0)
                q1 = (q1 * cos1) + (RopeOperation.rotate_half(q1) * sin1)
                k1 = (k1 * cos1) + (RopeOperation.rotate_half(k1) * sin1)
                q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(ntoken, hidden_size)
                k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(ntoken, hidden_size)
                return [q, k]
        elif json_data['rotaryCoeff'] == 2:
            dtype = in_tensors[0].dtype
            if dtype == torch.bfloat16:
                in_tensors[0] = in_tensors[0].to(torch.float32)
                in_tensors[1] = in_tensors[1].to(torch.float32)
                in_tensors[2] = in_tensors[2].to(torch.float32)
                in_tensors[3] = in_tensors[3].to(torch.float32)
            ntoken = in_tensors[0].size()[0]
            seqlen = int(in_tensors[4][0])
            batch = ntoken // seqlen
            if batch == 0:
                batch = 1
                seqlen = ntoken
            hidden_size = in_tensors[0].size()[1]
            hidden_size1 = in_tensors[1].size()[1]
            head_size = in_tensors[2].size()[1]
            head_num = hidden_size // head_size
            head_num1 = hidden_size1 // head_size
            q = in_tensors[0].view(batch, seqlen, head_num, head_size)
            k = in_tensors[1].view(batch, seqlen, head_num1, head_size)
            cos = in_tensors[2].view(batch, seqlen, head_size).unsqueeze(2)
            sin = in_tensors[3].view(batch, seqlen, head_size).unsqueeze(2)
            q_embed = ((q * cos) + (RopeOperation.rotate_half(q) * sin)).view(ntoken, hidden_size)
            k_embed = ((k * cos) + (RopeOperation.rotate_half(k) * sin)).view(ntoken, hidden_size1)
            return [q_embed.to(dtype), k_embed.to(dtype)]
        else:
            if len(in_tensors[0].size()) == 4:
                seqlen = in_tensors[0].size()[1]
                batch = in_tensors[0].size()[0]
                q_head_num = in_tensors[0].size()[2]
                k_head_num = in_tensors[1].size()[2]
            else:
                ntoken = in_tensors[0].size()[0]
                seqlen = int(in_tensors[4][0])
                batch = ntoken // seqlen
                hidden_sizeq = in_tensors[0].size()[1]
                head_size = in_tensors[2].size()[1]
                q_head_num = hidden_sizeq // head_size
                hidden_sizek = in_tensors[1].size()[1]
                k_head_num = hidden_sizek // head_size
            rot_dim = in_tensors[2].size()[1]

            if batch == 0:
                batch = 1
                seqlen = ntoken

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

            if len(in_tensors[0].size()) == 4:
                return [q_out2, k_out2]
            else:
                return [q_out2.view(ntoken, hidden_sizeq), k_out2.view(ntoken, hidden_sizek)]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class RopeQConcatOperation(DataGen):
    @staticmethod
    def rotate_half_for_concat(q, cos):
        head_dim = cos.shape[1]
        # 拆分成 head_num 个 [n,head_dim] 的二维向量
        # print("============== q", q)
        q_splits = torch.split(q, head_dim, dim = 1)
        # 对每个 [n,head_dim] 向量的第二维进行分割，并对第二块乘以 -1再拼回到第一块前面
        processed_q_splits = []
        for q_split in q_splits:
            # 分割第二维
            first_half, second_half = torch.split(q_split, int(head_dim / 2), dim = 1)
            # 拼接回 [n,head_dim] 的二维向量
            processed_q_split = torch.concatenate((-second_half, first_half), axis = 1)
            processed_q_splits.append(processed_q_split)
        # 将所有处理后的 [n,head_dim] 向量拼回 [n,head_num*head_dim] 的二维向量
        return torch.concatenate(processed_q_splits, axis = 1)

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        in_tensors[0] = in_tensors[0].to(torch.float32)
        in_tensors[1] = in_tensors[1].to(torch.float32)
        in_tensors[2] = in_tensors[2].to(torch.float32)
        token_num = in_tensors[1].shape[0]
        head_dim = in_tensors[1].shape[1]
        head_num = in_tensors[3].shape[1]
        pad_cos = torch.tile(in_tensors[1], (1, head_num))
        pad_sin = torch.tile(in_tensors[2], (1, head_num))
        rope_res = in_tensors[0] * pad_cos + RopeQConcatOperation.rotate_half_for_concat(in_tensors[0], in_tensors[1]) * pad_sin
        rope_res = rope_res.reshape(token_num, head_num, head_dim).to(in_tensors[3].dtype)
        return [torch.concatenate((rope_res, in_tensors[3]), axis = 2)]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class ReshapeAndCacheOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            ReshapeAndCacheOperation.in_tensors[i] = torch_npu.npu_dtype_cast(ReshapeAndCacheOperation.in_tensors[i].npu(), dtype_dict[datatype])
            return torch_npu.npu_format_cast(ReshapeAndCacheOperation.in_tensors[i], format_dict[format])
        else:
            if hasattr(ReshapeAndCacheOperation, 'in_tensors'):
                delattr(ReshapeAndCacheOperation, 'in_tensors')

        soc_version = get_soc_version()
        json_data = json.loads(op_params)
        if "kvCacheCfg" in json_data and json_data["kvCacheCfg"] == 1:
            MAX_SEQ_LEN = 1024
            num_tokens = shapes[0][0]
            num_heads = shapes[0][1]
            head_size = shapes[0][2]
            block_size = shapes[1][1]
            num_blocks = shapes[1][0]
            num_heads_cache = shapes[1][2]
            head_size_cache = shapes[1][3]
            hidden_size = 4096
            datatype_dict = {"float16": "float16", "bf16": "bfloat16", "int8": "int8"}
            dtype1 = datatype_dict[datatype]
            dtype = datatype_dict[datatype]
            if dtype1 == "bfloat16":
                dtype = "float32"
            key = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)

            num_slots = block_size * num_blocks
            num_tokens_slots = shapes[2][0]
            slot_list = np.zeros((num_tokens_slots)).astype(np.int32)
            if num_tokens_slots <= num_slots:
                slot_list = random.sample(range(num_slots), num_tokens_slots)
            else:
                range_min = num_slots - num_tokens_slots
                slot_list = random.sample(range(range_min, num_slots), num_tokens_slots)
            slot_mapping = np.array(slot_list).astype(np.int32)

            if soc_version == 'Ascend910B' or soc_version == 'Ascend950':
                key_cache = np.zeros((num_blocks, block_size, num_heads_cache, head_size_cache)).astype(dtype)
                key_expect = np.zeros((num_blocks, block_size, num_heads_cache, head_size_cache)).astype(dtype)

            for j, slot in enumerate(slot_list):
                if slot < 0:
                    continue
                block_index = slot // block_size
                block_offset = slot % block_size

                token_key = key[j]
                if ((soc_version == 'Ascend910B' or soc_version == 'Ascend950') and key_expect[0][0].shape == token_key.shape):
                    key_expect[block_index][block_offset] = token_key

            ret_data = key, key_cache, slot_mapping, key_expect
            in_tensors = ret_data
            if dtype1 == "bfloat16":
                in_tensors = [torch.from_numpy(tensor).bfloat16() for tensor in ret_data]
                in_tensors[2] = torch.from_numpy(ret_data[2])
                in_tensors = [tensor.npu() for tensor in in_tensors]
            else:
                in_tensors = [torch.from_numpy(tensor) for tensor in ret_data]
                in_tensors = [tensor.npu() for tensor in in_tensors]
            ReshapeAndCacheOperation.in_tensors = in_tensors
            return torch_npu.npu_format_cast(ReshapeAndCacheOperation.in_tensors[i], format_dict[format])
        # kvCacheCfg = 0 or 2
        MAX_SEQ_LEN = 1024
        num_tokens = shapes[0][0]
        num_heads = shapes[0][1]
        head_size_k = shapes[0][2]
        num_tokens_v = shapes[1][0]
        num_heads_v  = shapes[1][1]
        head_size_v = shapes[1][2]
        if ((soc_version == 'Ascend910B' or soc_version == 'Ascend950') and ("kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 2)) or soc_version == 'Ascend310P':
            block_size = shapes[2][2]
        else:
            block_size = shapes[2][1]
        num_blocks = shapes[2][0]
        hidden_size = 4096
        datatype_dict = {"float16": "float16", "bf16": "bfloat16", "int8": "int8"}
        dtype1 = datatype_dict[datatype]
        dtype = datatype_dict[datatype]
        if dtype1 == "bfloat16":
            dtype = "float32"
        key = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size_k)).astype(dtype)
        value = np.random.uniform(-1.0, 1.0, size=(num_tokens_v, num_heads_v, head_size_v)).astype(dtype)

        num_tokens_slots = shapes[4][0]
        num_slots = block_size * num_blocks
        slot_list = np.zeros((num_tokens_slots)).astype(np.int32)
        if num_tokens_slots <= num_slots:
            slot_list = random.sample(range(num_slots), num_tokens_slots)
        else:
            range_min = num_slots - num_tokens_slots
            slot_list = random.sample(range(range_min, num_slots), num_tokens_slots)
        slot_mapping = np.array(slot_list).astype(np.int32)

        if soc_version == 'Ascend910B' or soc_version == 'Ascend950':
            key_cache = np.zeros((shapes[2][0], shapes[2][1], shapes[2][2], shapes[2][3])).astype(dtype)
            value_cache = np.zeros((shapes[3][0], shapes[3][1], shapes[3][2], shapes[3][3])).astype(dtype)
            key_expect = np.zeros((shapes[2][0], shapes[2][1], shapes[2][2], shapes[2][3])).astype(dtype)
            value_expect = np.zeros((shapes[3][0], shapes[3][1], shapes[3][2], shapes[3][3])).astype(dtype)
        else:
            key_cache = np.zeros((shapes[2][0],shapes[2][1], shapes[2][2], shapes[2][3])).astype(dtype)
            value_cache = np.zeros((shapes[3][0],shapes[3][1], shapes[3][2], shapes[3][3])).astype(dtype)
            key_expect = np.zeros((shapes[2][0],shapes[2][1], shapes[2][2], shapes[2][3])).astype(dtype)
            value_expect = np.zeros((shapes[3][0],shapes[3][1], shapes[3][2], shapes[3][3])).astype(dtype)

        for j, slot in enumerate(slot_list):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size
            if (j < key.shape[0] and j < value.shape[0]):
                token_key = key[j]
                token_v = value[j]
            else:
                token_key = key[0]
                token_v = value[0]
            if soc_version == 'Ascend910B' and ("kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 2): # 910B NZ
                last_dim_k = 0
                last_dim_v = 16
                if dtype == "int8":
                    last_dim_k = 32
                else:
                    last_dim_k = 16
                token_key = token_key.reshape(num_heads * head_size_k)
                token_v = token_v.reshape(num_heads_v * head_size_v)
                for k in range(num_heads * head_size_k // last_dim_k):
                    if (block_index < key_expect.shape[0] and block_offset < key_expect.shape[2] and key_expect.shape[3] == last_dim_k and k < key_expect.shape[1]):
                        key_expect[block_index][k][block_offset][:] = token_key[k * last_dim_k: k * last_dim_k + last_dim_k]
                for v in range(num_heads_v * head_size_v // last_dim_v):
                    if (block_index < value_expect.shape[0] and block_offset < value_expect.shape[2] and value_expect.shape[3] == last_dim_v and v < value_expect.shape[1]):
                        value_expect[block_index][v][block_offset][:] = token_v[v * last_dim_v: v * last_dim_v + last_dim_v]
            elif soc_version == 'Ascend910B' or soc_version == 'Ascend950': # 910B ND
                if key_expect[0][0].shape == token_key.shape:
                    key_expect[block_index][block_offset] = token_key
                if value_expect[0][0].shape == token_v.shape:
                    value_expect[block_index][block_offset] = token_v
            else: # 310P NZ
                token_key = token_key.reshape(num_heads * head_size_k)
                token_v = token_v.reshape(num_heads_v * head_size_v)
                for k in range(num_heads * head_size_k // 16):
                    if (block_index < key_expect.shape[0] and block_offset < key_expect.shape[2] and key_expect.shape[3] == 16 and k < key_expect.shape[1]):
                        key_expect[block_index][k][block_offset][:] = token_key[k * 16: k * 16 + 16]
                for v in range(num_heads_v * head_size_v // 16):
                    if (block_index < value_expect.shape[0] and block_offset < value_expect.shape[2] and value_expect.shape[3] == 16 and v < value_expect.shape[1]):
                        value_expect[block_index][v][block_offset][:] = token_v[v * 16: v * 16 + 16]

        ret_data = key, value, key_cache, value_cache, slot_mapping, key_expect, value_expect
        in_tensors = ret_data
        if dtype1 == "bfloat16":
            in_tensors = [torch.from_numpy(tensor).bfloat16() for tensor in ret_data]
            in_tensors[4] = torch.from_numpy(ret_data[4])
            in_tensors = [tensor.npu() for tensor in in_tensors]
        else:
            in_tensors = [torch.from_numpy(tensor) for tensor in ret_data]
            in_tensors = [tensor.npu() for tensor in in_tensors]
        ReshapeAndCacheOperation.in_tensors = in_tensors
        return torch_npu.npu_format_cast(ReshapeAndCacheOperation.in_tensors[i], format_dict[format])


    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        json_data = json.loads(op_params)
        if "kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 1:
            output_tensor_list[0] = input_tensor_list[1]
        else:
            output_tensor_list[0] = input_tensor_list[2]
            output_tensor_list[1] = input_tensor_list[3]
        if hasattr(ReshapeAndCacheOperation, 'in_tensors'):
            delattr(ReshapeAndCacheOperation, 'in_tensors')

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if "kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 1:
            num_tokens, num_heads, head_size = in_tensors[0].shape # key
            data_type = in_tensors[0].dtype
            slot_mapping = in_tensors[2]
            soc_version = get_soc_version()
            if soc_version == 'Ascend910B' or soc_version == 'Ascend950':
                num_blocks, block_size, _, _ = in_tensors[1].shape # key_cache
                key_expect = in_tensors[1]
                for i, slot in enumerate(slot_mapping):
                    if slot < 0:
                        continue
                    block_index = slot // block_size
                    block_offset = slot % block_size

                    token_key = in_tensors[0][i]

                    key_expect[block_index][block_offset] = token_key
                return [key_expect]
        # kvCacheCfg = 0 or 2
        num_tokens, num_heads, head_size = in_tensors[0].shape # key
        data_type = in_tensors[0].dtype
        slot_mapping = in_tensors[4]
        soc_version = get_soc_version()
        # 910B nz
        if soc_version == 'Ascend910B' and ("kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 2):
            k_head_size = in_tensors[0].shape[2] # key
            v_head_size = in_tensors[1].shape[2] # value
            last_dim_k = 0
            last_dim_v = 16
            if data_type == torch.int8:
                last_dim_k = 32
            else:
                last_dim_k = 16
            num_blocks, _, block_size, _ = in_tensors[2].shape # key_cache
            key_expect_nz = in_tensors[2]
            value_expect_nz = in_tensors[3]
            for i, slot in enumerate(slot_mapping):
                block_index = slot // block_size
                block_offset = slot % block_size

                token_key = in_tensors[0][i]
                token_v = in_tensors[1][i]
                token_key = token_key.reshape(num_heads * k_head_size)
                token_v = token_v.reshape(num_heads * v_head_size)
                for k in range(num_heads * k_head_size // last_dim_k):
                    key_expect_nz[block_index][k][block_offset][:] = token_key[k * last_dim_k: k * last_dim_k + last_dim_k]
                for v in range(num_heads * v_head_size // last_dim_v):
                    value_expect_nz[block_index][v][block_offset][:] = token_v[v * last_dim_v: v * last_dim_v + last_dim_v]
            return [key_expect_nz, value_expect_nz]
        elif soc_version == 'Ascend910B' or soc_version == 'Ascend950':
            num_blocks, block_size, _, _ = in_tensors[2].shape # key_cache
            key_expect = in_tensors[2]
            value_expect = in_tensors[3]
            if "compressType" in json_data and  json_data["compressType"] == 1:
                wins = in_tensors[5]
                seq_len = in_tensors[6]
                new_seq = seq_len
                new_seq[0] = seq_len[0]
                for n in range(1, len(seq_len)):
                    new_seq[n] = seq_len[n] + seq_len[n-1]

                for i, slot in enumerate(slot_mapping):
                    if slot < 0:
                        continue
                    curSlot = slot
                    win = wins[i]
                    for j in range(win):
                        block_index = curSlot // block_size
                        block_offset = curSlot % block_size
                        curBatch = i // num_heads
                        bsID = new_seq[curBatch] - win + j
                        headID = i % num_heads
                        token_key = in_tensors[0][bsID][headID]
                        token_v = in_tensors[1][bsID][headID]
                        key_expect[block_index][block_offset] = token_key
                        value_expect[block_index][block_offset] = token_v
                        curSlot += 1
                return [key_expect, value_expect]
            else:
                for i, slot in enumerate(slot_mapping):
                    if slot < 0:
                        continue
                    block_index = slot // block_size
                    block_offset = slot % block_size

                    token_key = in_tensors[0][i]
                    token_v = in_tensors[1][i]

                    key_expect[block_index][block_offset] = token_key
                    value_expect[block_index][block_offset] = token_v
                return [key_expect, value_expect]
        # 310P nz
        else:
            num_blocks, _, block_size, _ = in_tensors[2].shape # key_cache
            key_expect_nz = in_tensors[2]
            value_expect_nz = in_tensors[3]
            for i, slot in enumerate(slot_mapping):
                block_index = slot // block_size
                block_offset = slot % block_size

                token_key = in_tensors[0][i]
                token_v = in_tensors[1][i]
                token_key = token_key.reshape(num_heads * head_size)
                token_v = token_v.reshape(num_heads * head_size)
                for k in range(num_heads * head_size // 16):
                    key_expect_nz[block_index][k][block_offset][:] = token_key[k * 16: k * 16 + 16]
                    value_expect_nz[block_index][k][block_offset][:] = token_v[k * 16: k * 16 + 16]
            return [key_expect_nz, value_expect_nz]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class RelayAttentionOperation(DataGen):
    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION

class OnehotOperation(DataGen):

    @staticmethod
    def golden(in_tensors, op_params):	
        data_type = in_tensors[0].dtype	
        json_data = json.loads(op_params)	
        depth = json_data['depth']	


        input0 = in_tensors[0].numpy()	
        res = np.eye(depth)[input0]	
        res = torch.from_numpy(res).to(data_type)	
        return [res]	

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_INTEGER


class ElewiseOperation(DataGen):
    ELEWISE_CAST = 1
    ELEWISE_MULS = 2
    ELEWISE_COS = 3
    ELEWISE_SIN = 4
    ELEWISE_NEG = 5
    ELEWISE_QUANT = 6
    ELEWISE_LOGICAL_NOT = 7
    ELEWISE_ADD = 8
    ELEWISE_MUL = 9
    ELEWISE_REALDIV = 10
    ELEWISE_LOGICAL_AND =11
    ELEWISE_LOGICAL_OR = 12
    ELEWISE_LESS = 13
    ELEWISE_GREATER = 14
    ELEWISE_SUB = 15
    ELEWISE_EQUAL = 16
    ELEWISE_QUANT_PER_CHANNEL = 17
    ELEWISE_DEQUANT_PER_CHANNEL = 18
    ELEWISE_DYNAMIC_QUANT = 19

    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        json_data = json.loads(op_params)
        elewiseType = json_data["elewiseType"]
        if  elewiseType == 12:
            high = float(data_gen_ranges.split(',')[1])
            low = float(data_gen_ranges.split(',')[0])
            if (low <= -1 or high > 1) and datatype == 'int8':
                data = torch.randint(0, 2, shape, dtype=torch.int8).npu()
            else:
                data = ((high - low) * torch.rand(shape) + low).type(dtype_dict[datatype]).npu()
            return torch_npu.npu_format_cast(data, format_dict[format])

        return DataGen.random(shape, datatype, format, data_gen_ranges, op_params)

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        json_data = json.loads(op_params)
        elewiseType = json_data["elewiseType"]
        if elewiseType == 12:
            return ElewiseOperation.random(shapes[i], datatype, format, data_gen_ranges, op_params)

        return DataGen.random(shapes[i], datatype, format, data_gen_ranges, op_params)


    def elewiseCast(in_tensors, op_params):
        json_data = json.loads(op_params)
        outType = json_data['outTensorType']
        golden_result = in_tensors[0]
        if outType == 0:
            golden_result = in_tensors[0].float()
        elif outType == 1:
            golden_result = in_tensors[0].half()
        elif outType == 3:
            golden_result = in_tensors[0].int()
        elif outType == 9:
            golden_result = in_tensors[0].long()
        elif outType == 27:
            golden_result = in_tensors[0].bfloat16()
        return [golden_result]

    def elewiseMuls(in_tensors, op_params):
        json_data = json.loads(op_params)
        if 'mulsParam' in json_data:
            varAttr = json_data['mulsParam']['varAttr']
        else:
            varAttr = json_data['varAttr']
        golden_result = in_tensors[0] * varAttr
        return [golden_result]

    def elewiseCos(in_tensors, op_params):
        golden_result = torch.cos(in_tensors[0].float())
        return [golden_result]

    def elewiseSin(in_tensors, op_params):
        golden_result = torch.sin(in_tensors[0].float())
        return [golden_result]

    def elewiseNeg(in_tensors, op_params):
        golden_result = in_tensors[0] * (-1.0)
        return [golden_result]

    def elewiseQuant(in_tensors, op_params):
        json_data = json.loads(op_params)
        if "outTensorType" in json_data:
            outType = json_data["outTensorType"]
        else:
            outType = 2
        scale = np.float32(json_data["quantParam"]["inputScale"])
        offset = np.int32(json_data["quantParam"]["inputOffset"])
        input_x = in_tensors[0].to(torch.float32).cpu().numpy().astype("float32")

        if (in_tensors[0].dtype == torch.float16):
            scale = scale.astype(np.float16)
            offset = offset.astype(np.float16)
            scale = scale.astype(np.float32)
            offset = offset.astype(np.float32)
        elif (in_tensors[0].dtype == torch.bfloat16):
            scale = scale.astype(numpy_bfloat16())
            offset = offset.astype(numpy_bfloat16())
            scale = scale.astype(np.float32)
            offset = offset.astype(np.float32)

        out = input_x * scale + offset
        out = np.round(out, 8) if outType in (34, 35, 36) else np.round(out, 0)
        if outType == 35:
            out = torch.from_numpy(out).to(torch.float8_e5m2)
        elif outType == 36:
            out = torch.from_numpy(out).to(torch.float8_e4m3fn)
        elif outType == 34:
            out = out.astype(hifloat8, copy=False).view(np.int8)
            out = torch.from_numpy(out)
        else:
            out = torch.from_numpy(out).to(torch.int8)

        return [out]

    def elewiseLogicalNot(in_tensors, op_params):
        golden_result = torch.logical_not(in_tensors[0])
        return [golden_result]

    def elewiseAdd(in_tensors, op_params):
        golden_result = in_tensors[0] + in_tensors[1]
        return [golden_result]

    def elewiseMul(in_tensors, op_params):
        golden_result = in_tensors[0] * in_tensors[1]
        if get_soc_version() != "Ascend910B":
            golden_result = torch.where(torch.isinf(golden_result), torch.tensor(6.5504e+04), golden_result)
        return [golden_result]

    def elewiseRealdiv(in_tensors, op_params):
        golden_result = torch.div(in_tensors[0], in_tensors[1])
        return [golden_result]

    def elewiseLogicalAnd(in_tensors, op_params):
        golden_result = torch.logical_and(in_tensors[0].type(torch.bool), in_tensors[1].type(torch.bool))
        return [golden_result.type(torch.int8)]

    def elewiseLogicalOr(in_tensors, op_params):
        golden_result = torch.logical_or(in_tensors[0].type(torch.bool), in_tensors[1].type(torch.bool))
        return [golden_result.type(torch.int8)]

    def elewiseLess(in_tensors, op_params):
        golden_result = torch.lt(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewiseGreater(in_tensors, op_params):
        golden_result = torch.gt(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewiseSub(in_tensors, op_params):
        golden_result = in_tensors[0] - in_tensors[1]
        return [golden_result]

    def elewiseEqual(in_tensors, op_params):
        golden_result = torch.eq(in_tensors[0], in_tensors[1]).type(torch.int8)
        return [golden_result]

    def elewiseQuantPerChannel(in_tensors, op_params):
        # 获取输入张量
        input_x = in_tensors[0].cpu()
        input_scale = in_tensors[1].cpu()
        input_offset = in_tensors[2].cpu()

        # 对 input_x 和 input_scale 进行广播和除法操作
        result = None
        if get_soc_version() == "Ascend950":
            result = input_x * (1 / input_scale)
        else:
            result = input_x / input_scale

        # 如果有 offset，则加上 offset
        if len(input_offset) > 0:
            result += input_offset

        # 对结果进行四舍五入，并进行 clip 操作，限制范围为 [-128, 127]
        result = result.round()
        result = torch.clamp(result, min=-128, max=127)

        # 转换为 int8 类型
        out = result.to(torch.int8)

        return [out]

    def elewiseDequantPerChannel(in_tensors, op_params):
        input_y = in_tensors[0].cpu().numpy()
        input_scale = in_tensors[1].cpu().numpy()
        input_offset = in_tensors[2].cpu().numpy()
        if len(input_offset) == 0:
            out = np.clip(input_y.astype(np.float16) * input_scale, -65504, 65504)
        else:
            out = np.clip((input_y.astype(np.float16) - input_offset.astype(np.float16)) * input_scale, -65504, 65504)
        return [torch.from_numpy(out).to(torch.float16)]

    def elewiseDynamicQuant(in_tensors, op_params):
        input_x = in_tensors[0].to(torch.float).cpu().numpy()
        shape_input = input_x.shape
        json_data = json.loads(op_params)
        if json_data["quantParam"]["asymmetric"]:
            row_max = np.max(input_x, axis=-1, keepdims=True)
            row_min = np.min(input_x, axis=-1, keepdims=True)
            row_max = row_max.astype(np.float32)
            row_min = row_min.astype(np.float32)
            out_scale = (row_max - row_min) / 255
            out_offset = - (row_max + row_min) / (2 * out_scale)

            input_x = input_x.astype(np.float32)
            input_x = input_x / out_scale
            input_x = input_x + out_offset
            input_x = np.clip(input_x, -128, 127)
            out_x = np.round(input_x)
            return [torch.from_numpy(out_x).to(torch.int8),
                    torch.from_numpy(out_scale.squeeze(axis=-1)).to(torch.float32),
                    torch.from_numpy(out_offset.squeeze(axis=-1)).to(torch.float32)]
        else:
            if "outTensorType" in json_data.keys():
                outDtype = json_data['outTensorType']
            else:
                outDtype = 2
            dmax = np.float32(0.0)
            if outDtype == 2:
                dmax = np.float32(127.0)
            elif outDtype == 34:
                dmax = np.float32(32768.0)
            elif outDtype == 35:
                dmax = np.float32(57344.0)
            elif outDtype == 36:
                dmax = np.float32(448.0)

            input_abs = np.abs(input_x)
            input_max = np.max(input_abs, axis=-1, keepdims=True)
            scale = input_max * (np.float32(1.0) / dmax)
            input_scaled = input_x / scale

            round_data = input_scaled if (outDtype) in (34, 35, 36) else np.round(input_scaled, 0)
            
            if outDtype == 2:
                round_data = round_data.astype(np.int8, copy=False)
                out = torch.from_numpy(round_data)
            elif outDtype == 34:
                out = round_data.astype(hifloat8, copy=False).view(np.int8)
                out = torch.from_numpy(out)
            elif outDtype == 35:
                out = torch.from_numpy(round_data).to(torch.float8_e5m2)
            elif outDtype == 36:
                out = torch.from_numpy(round_data).to(torch.float8_e4m3fn)

            return [out, torch.from_numpy(scale.squeeze(axis=-1)).to(torch.float32)]

    def elewiseTanh(in_tensors, op_params):
        golden_result = torch.tanh(in_tensors[0].float())
        return [golden_result]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        elewiseType = json_data["elewiseType"]
        if elewiseType == 1:
            return ElewiseOperation.elewiseCast(in_tensors, op_params)
        elif elewiseType == 2:
            return ElewiseOperation.elewiseMuls(in_tensors, op_params)
        elif elewiseType == 3:
            return ElewiseOperation.elewiseCos(in_tensors, op_params)
        elif elewiseType == 4:
            return ElewiseOperation.elewiseSin(in_tensors, op_params)
        elif elewiseType == 5:
            return ElewiseOperation.elewiseNeg(in_tensors, op_params)
        elif elewiseType == 6:
            return ElewiseOperation.elewiseQuant(in_tensors, op_params)
        elif elewiseType == 7:
            return ElewiseOperation.elewiseLogicalNot(in_tensors, op_params)
        elif elewiseType == 8:
            return ElewiseOperation.elewiseAdd(in_tensors, op_params)
        elif elewiseType == 9:
            return ElewiseOperation.elewiseMul(in_tensors, op_params)
        elif elewiseType == 10:
            return ElewiseOperation.elewiseRealdiv(in_tensors, op_params)
        elif elewiseType == 11:
            return ElewiseOperation.elewiseLogicalAnd(in_tensors, op_params)
        elif elewiseType == 12:
            return ElewiseOperation.elewiseLogicalOr(in_tensors, op_params)
        elif elewiseType == 13:
            return ElewiseOperation.elewiseLess(in_tensors, op_params)
        elif elewiseType == 14:
            return ElewiseOperation.elewiseGreater(in_tensors, op_params)
        elif elewiseType == 15:
            return ElewiseOperation.elewiseSub(in_tensors, op_params)
        elif elewiseType == 16:
            return ElewiseOperation.elewiseEqual(in_tensors, op_params)
        elif elewiseType == 17:
            return ElewiseOperation.elewiseQuantPerChannel(in_tensors, op_params)
        elif elewiseType == 18:
            return ElewiseOperation.elewiseDequantPerChannel(in_tensors, op_params)
        elif elewiseType == 19:
            return ElewiseOperation.elewiseDynamicQuant(in_tensors, op_params)
        elif elewiseType == 20:
            return ElewiseOperation.elewiseTanh(in_tensors, op_params)

    @staticmethod
    def get_op_type(op_params):
        json_data = json.loads(op_params)
        if json_data["elewiseType"] in [
            ElewiseOperation.ELEWISE_ADD,
            ElewiseOperation.ELEWISE_COS,
            ElewiseOperation.ELEWISE_MUL,
            ElewiseOperation.ELEWISE_MULS,
            ElewiseOperation.ELEWISE_NEG,
            ElewiseOperation.ELEWISE_REALDIV,
            ElewiseOperation.ELEWISE_SIN,
            ElewiseOperation.ELEWISE_SUB
            ]:
            return OpTypes.COMPUTE_FLOAT
        elif json_data["elewiseType"] in [
            ElewiseOperation.ELEWISE_CAST
        ]:
            return OpTypes.CAST
        elif json_data["elewiseType"] in [
            ElewiseOperation.ELEWISE_EQUAL,
            ElewiseOperation.ELEWISE_GREATER,
            ElewiseOperation.ELEWISE_LESS,
            ElewiseOperation.ELEWISE_LOGICAL_AND,
            ElewiseOperation.ELEWISE_LOGICAL_NOT,
            ElewiseOperation.ELEWISE_LOGICAL_OR
        ]:
            return OpTypes.COMPUTE_INTEGER
        else:
            return OpTypes.COMPUTE_QUANT


class TransposeOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        perm = json_data["perm"]
        golden_result = in_tensors[0].permute(perm).cpu()
        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CAST


class KvCacheOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            return KvCacheOperation.ret_data[i]
        layer = 28
        layer_id = 0
        batch = 16
        max_seqlen = 384
        hidden_size = 1024

        seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
        token_offset = seqlen
        ntokens = np.sum(seqlen)
        newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.float16)
        cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        layer_id = np.array([layer_id], dtype=np.int32)
        cache_out = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.float16)
        prefix_ntokens = 0
        for i in range(batch):
            for j in range(seqlen[i]):
                cache_out[layer_id[0]][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]
        raw_data = newkv, layer_id, cache_in, token_offset, seqlen, cache_out
        in_tensors = [torch.from_numpy(tensor) for tensor in raw_data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        KvCacheOperation.ret_data = in_tensors
        return in_tensors[0]


    def kv_other(shapes, i, idatatype, format, data_gen_ranges, op_params):
        return KvCacheOperation.ret_data[i]

    @staticmethod
    def golden(in_tensors, op_params):
        return [KvCacheOperation.ret_data[0].cpu(), KvCacheOperation.ret_data[1].cpu(), KvCacheOperation.ret_data[5].cpu(), \
            KvCacheOperation.ret_data[3].cpu(), KvCacheOperation.ret_data[4].cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class PagedAttentionOperation(DataGen):
    @staticmethod
    def custom_normalize(tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        normalized = (tensor - min_val) / (max_val - min_val)  # 先归一化到[0,1]
        scaled = normalized * (1 - (-1)) + (-1)  # 再映射到[-1,1]
        return scaled

    @staticmethod
    def load_tensor_from_file(intensor_file, i, op_params) -> torch.Tensor:
        bin = TensorBinFile(intensor_file)
        tensor = bin.get_tensor()
        needScale = i == 1 or i == 5 # kcache, mask
        if i == 2: # vcache
            json_data = json.loads(op_params)
            if "mlaVHeadSize" in json_data and json_data["mlaVHeadSize"] > 0:
                needScale = False
            else:
                needScale = True
        if needScale:
            tensor = PagedAttentionOperation.custom_normalize(tensor.cpu()).npu()
        return tensor

    @staticmethod
    def shape_nd_to_nz(shape, dtype='float16'):
        assert len(shape) >= 2
        batch = shape[:-2]   # 最后两维nd->nz
        a, b = shape[-2], shape[-1]
        a0, b0 = 16, 16
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    @staticmethod
    def gen_axes_for_transpose(offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    @staticmethod
    def convert_nd_to_nz(x):
        array_trans = PagedAttentionOperation.gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
        x_shape = PagedAttentionOperation.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans) # x原始需要对齐，才能reshape

    @staticmethod
    def trans_dtype(dtype):
        if dtype == "float16":
            return torch.float16
        if dtype == "float" or dtype == "float32":
            return torch.float32
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "int8":
            return torch.int8
        if dtype == "int32":
            return torch.int32
        if dtype == "int64":
            return torch.int64
        if dtype == "float8_e5m2":
            return torch.float8_e5m2
        if dtype == "float8_e4m3fn":
            return torch.float8_e4m3fn

    @staticmethod
    def process_deq_scale(deq_scale) -> np.ndarray:
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
        return new_deq_scale.astype(np.uint64)

    def get_former_head(kvsplit, block_num, num_tokens, num_heads):
        headLimit = 128
        if kvsplit > 1:
            return 1
        else:
            coreNumPerBatch = int((block_num + num_tokens - 1)) // num_tokens
            headsplit = int((num_heads + coreNumPerBatch- 1)) // coreNumPerBatch
            headsplit = headLimit if headsplit > headLimit else headsplit
            if num_tokens == 16 and num_heads == 32:
                headsplit = 8
        return headsplit

    def calc_head_split_nd(embeddingSize, embeddingSizeV, former_head):
        if former_head <= 64:
            embedQKSplit = 256 if embeddingSize > 256 else embeddingSize
            embedVOSplit = 256 if embeddingSizeV > 256 else embeddingSizeV
        else:
            embedQKSplit = 512 if embeddingSize > 512 else embeddingSize
            embedVOSplit = 256 if embeddingSizeV > 256 else embeddingSizeV
        return embedQKSplit, embedVOSplit

    def get_blockszie_calc(max_context_len, block_size, embeddingSize, embeddingSizeV, is_int8_flag, is_quant_flag, kv_split_per_core, head_num_move, former_head):
        if former_head <= 64:
            embedQKSplit = 256 if embeddingSize > 256 else embeddingSize
            embedVOSplit = 256 if embeddingSizeV > 256 else embeddingSizeV
        else:
            embedQKSplit = 512 if embeddingSize > 512 else embeddingSize
            embedVOSplit = 256 if embeddingSizeV > 256 else embeddingSizeV
        BLOCK_LIMIT = 128 * 128
        KV_SEQLEN_SLICE = 128
        KV_SEQLEN_SLICE_256 = 256
        KV_SEQLEN_SLICE_512 = 512
        BLOCK_LIMIT_NO_PINGPONG = 128 * 256
        BLOCK_LIMIT_NO_PINGPONG_UINT8 = 128 * 256 * 2
        block_size_calc = block_size
        headdimMax =  np.maximum(embedQKSplit, embedVOSplit)
        if is_quant_flag:
            tBlockAlign = 32
        else:
            tBlockAlign = 16
        l0Limit = tBlockAlign * BLOCK_LIMIT_NO_PINGPONG_UINT8 / 32
        if block_size <= KV_SEQLEN_SLICE / 2 and \
            block_size * 2 * embedQKSplit <= BLOCK_LIMIT and \
            block_size * 2 * embedVOSplit <= BLOCK_LIMIT:
            block_size_calc =  block_size * 2
        if not is_int8_flag and \
            max_context_len >= KV_SEQLEN_SLICE_256 and \
            kv_split_per_core >= KV_SEQLEN_SLICE_256 and \
            former_head <= BLOCK_LIMIT  / KV_SEQLEN_SLICE_256 and \
            KV_SEQLEN_SLICE_256 * embedQKSplit  <= l0Limit and \
            KV_SEQLEN_SLICE_256 * embedVOSplit <= l0Limit and \
            (block_size == KV_SEQLEN_SLICE_256 // 4 or block_size ==  KV_SEQLEN_SLICE_256 // 2):
            block_size_calc = 256
        if is_quant_flag and \
            max_context_len >= KV_SEQLEN_SLICE_512 and \
            kv_split_per_core >= KV_SEQLEN_SLICE_512 and \
            KV_SEQLEN_SLICE_512 * embedQKSplit  <= BLOCK_LIMIT_NO_PINGPONG * 2 and \
            KV_SEQLEN_SLICE_512 * embedVOSplit <= BLOCK_LIMIT_NO_PINGPONG * 2 and \
            (block_size == KV_SEQLEN_SLICE_256 // 4 or block_size ==  KV_SEQLEN_SLICE_256 // 2) and \
            head_num_move < 4:
            block_size_calc = KV_SEQLEN_SLICE_512
        return block_size_calc

    def getkvsplit(num_tokens, num_heads, max_context_len, block_size, blocknum, isLongSeq):
        if isLongSeq:
            kvSeqklenMaxAlign = (max_context_len + block_size - 1) // block_size * block_size
            kvSeqBlockNum = int(kvSeqklenMaxAlign / block_size)
            kvBlockPreCore = int((kvSeqBlockNum + blocknum - 1)) // blocknum
            kvSplitPerCore = int(kvBlockPreCore * block_size)
            kvSplitCoreNum = int(kvSeqklenMaxAlign + kvSplitPerCore - 1) // kvSplitPerCore
            headSplit = int((num_heads + kvSplitCoreNum - 1) // kvSplitCoreNum)
        else:
            coreNumPerBatch  = int((blocknum + num_tokens - 1) // num_tokens)
            kvSeqklenMaxAlign = (max_context_len + block_size - 1) // block_size * block_size
            kvSeqBlockNum = int(kvSeqklenMaxAlign / block_size)
            kvBlockPreCore = int((kvSeqBlockNum + coreNumPerBatch - 1)) // coreNumPerBatch
            kvSplitPerCore = int(kvBlockPreCore * block_size)
            kvSplitCoreNum = int(kvSeqklenMaxAlign + kvSplitPerCore - 1) // kvSplitPerCore
            headSplit = int((num_heads + kvSplitCoreNum - 1) // kvSplitCoreNum)
        return kvSplitCoreNum, kvSplitPerCore

    def get_head_num_move(num_heads, kvhead, embeddingSize, embeddingSizeV):
        if embeddingSize % 32 == 0 and embeddingSizeV % 32 == 0 and embeddingSize <= 128 and embeddingSizeV <= 128 and num_heads == kvhead:
            head_num_move = 4
        else:
            head_num_move = 1
        return head_num_move

    @staticmethod
    def group_mm_torch_quant(heads, group_num, A, B, is_k = False, k_descale=None, de_scalev = None):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            group_score_int32 = torch.matmul(A[i*group_head: (i + 1)*group_head, :, :].to(torch.int32),
                B[i: (i+1), :, :].to(torch.int32)).to(torch.int32)
            if is_k:
                group_score_high = group_score_int32.to(torch.float32) * k_descale[(i * group_head): (i + 1) * group_head].reshape(group_head, 1, 1).to(torch.float32)
            else:
                group_score_high = group_score_int32.to(torch.float32) * de_scalev[(i * group_head): (i + 1) * group_head].reshape(group_head, 1, 1).to(torch.float32)
            if score_high is None:
                score_high = group_score_high
            else:
                score_high = torch.cat((score_high, group_score_high), 0)
        return score_high

    @staticmethod
    def group_mm_torch(heads, group_num, A, B, razor_mod=0, is_k=False, k_descale=None, k_offset=None,\
                       v_descale=None, v_offset=None, is_int8_flag=False, has_bias=False):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            if is_int8_flag:
                int8_B = B[i: (i+1), :, :, ]
                head_dim = int8_B.shape[2]
                float32_B = int8_B.to(torch.float32)
                if is_k:
                    if has_bias:
                        float32_B = float32_B + k_offset[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim].to(torch.float32)
                    fp32_B = float32_B.to(torch.float32) * k_descale[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if has_bias:
                        float32_B = float32_B + v_offset[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = float32_B.to(torch.float32) * v_descale[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                            fp32_B)
            else:
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                           B[i:(i + 1), :, :].to(torch.float32))
            if score_high is None:
                score_high = group_score_high
            else:
                score_high = torch.cat((score_high, group_score_high), 0)
        return score_high

    def softmax_numpy(sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def softmax_quant_numpy(sim, is_first, is_quant_offline, v_descale, p_scale, gm):
        lm = np.max(sim, axis=-1, keepdims=True)
        if is_first:
            hm = lm
            dm = 0
        else:
            hm = np.maximum(gm, lm)
            dm = gm - hm
        gm = hm
        sim_sub = sim - hm
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        row_maxp = np.max(sim_sub, axis=-1, keepdims=True)
        if not is_quant_offline:
            scale = row_maxp.astype("float32") / 127.0
            sim_int8 = sim_sub / scale
            soft_res = sim_int8.astype("float16")
            soft_res = np.rint(soft_res).astype("int8")
            de_scalev = v_descale * row_maxp[:,0,0] / 127
        else:
            soft_res = sim_sub * p_scale.reshape(p_scale.shape[0], 1, 1).numpy()
            soft_res = soft_res.astype("float16")
            soft_res = np.rint(soft_res).astype("int8")
            de_scalev = v_descale
        return soft_res, row_sum, de_scalev, hm, dm, gm

    def softmax_quant_numpy_online(sim, heads, kv_head, value, num_heads, block_size,\
                                   block_size_calc, kvsplit, kv_split_per_core, k_descale, v_descale, is_quant_offline, p_scale, gm):
        group_head = heads // kv_head
        score_high = None
        # (kv_heads, context_len, head_size)
        kv_seqlen = value.shape[1]
        cur_kv_seqlen = kv_seqlen
        n_loop = (cur_kv_seqlen + block_size_calc - 1) // block_size_calc
        qk_n = block_size_calc
        tmp_l_list = []
        tmp_o_list = []
        for cur_nIndx in range(kvsplit):
            kv_seqlen_align =  (kv_seqlen + block_size - 1) // block_size  * block_size
            start_kv = cur_nIndx * kv_split_per_core
            cur_kv_seqlen = kv_split_per_core
            kv_loop = (kv_seqlen_align + kv_split_per_core - 1) // kv_split_per_core
            if cur_nIndx >= kv_loop:
                continue
            if cur_nIndx == (kv_loop - 1):
                cur_kv_seqlen = kv_seqlen - cur_nIndx * kv_split_per_core
            n_loop = (cur_kv_seqlen + block_size_calc - 1) // block_size_calc
            qk_n = block_size_calc
            end_kv = start_kv
            for n_idx in range(n_loop):
                is_first = (n_idx == 0)
                if n_idx == n_loop - 1:
                    qk_n = cur_kv_seqlen - n_idx * block_size_calc
                end_kv = end_kv + qk_n
                sim_block = sim[:, :, start_kv : end_kv]
                p_block, ll, de_scalev, hm, dm, gm = PagedAttentionOperation.softmax_quant_numpy(sim_block, is_first, is_quant_offline, v_descale, p_scale, gm)
                value_block = value[:, start_kv : end_kv, :]
                lo = PagedAttentionOperation.group_mm_torch_quant(heads, kv_head, torch.from_numpy(p_block), value_block, 0, k_descale, de_scalev)
                lo = lo.cpu().numpy()
                if n_idx == 0:
                    gl = ll
                    go = lo
                else:
                    dm = np.exp(dm)
                    gl = gl * dm
                    gl = gl + ll
                    go = go * dm
                    go = go + lo
                start_kv = start_kv + qk_n
            go = go / gl
            tmp_o_list.append(go.reshape([1, num_heads, 1, value.shape[2]]))
            ls = np.log(gl) + gm
            tmp_l_list.append(ls.reshape([1, num_heads]))
        if kvsplit > 1:
            l = np.concatenate(tmp_l_list, 0)
            o = np.concatenate(tmp_o_list, 0)
            l = np.transpose(l, (1, 0))
            lse_max = np.max(l, axis=1, keepdims=True)
            l_tmp = np.exp(l - lse_max)
            lse_sum = np.sum(l_tmp, axis=1, keepdims=True)
            lse_logsum = np.log(lse_sum) + lse_max
            scale = np.exp(l - lse_logsum)
            o = o * scale.transpose(1, 0)[:,:,np.newaxis,np.newaxis]
            go = np.sum(o, axis=0, keepdims=True)
            go = np.squeeze(go, axis=0)
        return torch.from_numpy(go)

    @staticmethod
    def ref_masked_attention(
            query,  # (1, num_heads, head_size)
            key,  # (context_len, kv_heads, head_size)
            value,
            scale,
            mask,
            compressType = 0,
            razor_mod = 0,
            razor_offset_list = None,
            k_descale = None,
            k_offset = None,
            v_descale = None,
            v_offset = None,
            p_scale = None,
            is_int8_flag = False,
            has_bias = False,
            is_quant = False,
            is_quant_offline = False,
            num_heads=0,
            block_size=0,
            block_size_calc=0,
            kvsplit=0,
            kv_split_per_core=0
        ):
        # Q * K.T
        query = torch.permute(query, (1, 0, 2))
        if is_int8_flag:
            key = torch.permute(key, (1, 0, 2))
        else:
            key = torch.permute(key, (1, 2, 0))
        if is_quant:
            sim = PagedAttentionOperation.group_mm_torch_quant(query.shape[0], key.shape[0], query, key, 1, k_descale)
        else:
            sim = PagedAttentionOperation.group_mm_torch(query.shape[0], key.shape[0], query, key, razor_mod, 1,
                                                    k_descale, k_offset, v_descale, v_offset, is_int8_flag, has_bias)
        if compressType == 2:
            razor_offset_list = razor_offset_list.view(1, 1, razor_offset_list.shape[0])
            sim = sim.to(torch.float32) + razor_offset_list
        sim = sim.to(torch.float32) * np.float32(scale)
        if compressType == 1:
            sim = sim + mask[:sim.shape[-2], :sim.shape[-1]]
        elif mask.shape[0] != 0:
            sim = sim + mask.to(torch.float32)

        # softmax
        if is_quant:
            gm = np.full([query.shape[0] , 1, 1],  np.finfo(np.float32).min)
            p, row_sum, de_scalev, _, _, _= PagedAttentionOperation.softmax_quant_numpy(sim.numpy(), 1, is_quant_offline, v_descale, p_scale, gm)
            value = torch.permute(value, (1, 0, 2))
            out = PagedAttentionOperation.group_mm_torch_quant(query.shape[0], key.shape[0], torch.from_numpy(p), value, 0, k_descale, de_scalev)
            out = out / row_sum
            out = torch.permute(out, (1, 0, 2))
            s_qk = sim.numpy()
            out = PagedAttentionOperation.softmax_quant_numpy_online(s_qk, query.shape[0], key.shape[0], value, num_heads, block_size,
                                                                     block_size_calc, kvsplit, kv_split_per_core, k_descale, v_descale,
                                                                     is_quant_offline, p_scale, gm)
        else:
            p = PagedAttentionOperation.softmax_numpy(sim)
            p = torch.from_numpy(p).to(torch.float32)
            # P * V
            value = torch.permute(value, (1, 0, 2))
            out = PagedAttentionOperation.group_mm_torch(query.shape[0], key.shape[0], p, value, razor_mod, 0,
                                                        k_descale, k_offset, v_descale, v_offset, is_int8_flag, has_bias)
            out = torch.permute(out, (1, 0, 2))
        return out

    @staticmethod
    def ref_single_query_cached_kv_attention(
            op_params,
            output,
            query,
            key_cache,  # (num_blocks, block_size, num_heads, head_size)
            value_cache,  # (num_blocks, block_size, num_heads, head_size)
            block_tables,
            context_lens,
            mask,
            k_descale,
            k_offset,
            v_descale,
            v_offset,
            q_seqlens,
            razor_offset,
            p_scale,
            logN
        ) -> None:
        json_data = json.loads(op_params)
        maskType = 0
        if "maskType" in json_data:
            maskType = json_data["maskType"]
        compressType = 0
        if "compressType" in json_data:
            compressType = json_data["compressType"]
        is_int8 = "quantType" in json_data and json_data["quantType"] == 1
        is_quant = "quantType" in json_data and json_data["quantType"] >= 2
        is_quant_offline = "quantType" in json_data and json_data["quantType"] == 2
        is_lookahead = "calcType" in json_data and json_data["calcType"] == 1
        is_compresshead = "compressType" in json_data and json_data["compressType"] != 0
        is_int8_offset = "hasQuantOffset" in json_data and json_data["hasQuantOffset"]
        is_logN = "scaleType" in json_data and json_data["scaleType"] == 1
        is_kv_combined = "mlaVHeadSize" in json_data and json_data["mlaVHeadSize"] > 0

        qkscale = json_data["qkScale"]

        num_tokens = query.shape[0]
        num_heads = query.shape[1]
        head_size_qk = query.shape[2]
        head_size_vo = value_cache.shape[3]
        num_blocks = value_cache.shape[0]
        block_size = value_cache.shape[1]
        max_context_len = context_lens[0]
        if is_compresshead:
            kv_heads = int(block_tables.shape[0] // num_tokens)


        mask_index_coff = 1
        if is_compresshead:
            query = query.view(int(num_tokens * kv_heads), int(num_heads // kv_heads), head_size_qk)
            output = output.view(int(num_tokens * kv_heads), int(num_heads // kv_heads), head_size_vo)
            if maskType == 2:
                mask = mask.view(int(mask.shape[0] * kv_heads), int(num_heads // kv_heads), 1, -1)
            else:
                mask_index_coff = kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]

        if get_soc_version() == "Ascend950":
            if key_cache.dtype == torch.float8_e5m2 or key_cache.dtype == torch.float8_e4m3fn:
                key_cache = key_cache.to(torch.float16)
                value_cache = value_cache.to(torch.float16)

        index = 0
        cu_seqlen = 0
        razor_mod = 0
        for i in range(len(context_lens)):
            block_table = block_tables[i]
            if is_lookahead:
                q_seqlen = int(q_seqlens[i])
            context_len = int(context_lens[i])
            if context_len == 0:
                continue
            q = query[index].view(1, num_heads, head_size_qk)
            if is_lookahead:
                q = query[cu_seqlen : cu_seqlen + q_seqlen, :, :]
            keys = []
            values = []
            razor_offset_list = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)

                if compressType == 2:
                    offset = razor_offset[block_number, block_offset]
                    razor_offset_list.append(offset)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            if compressType == 2:
                razor_offset_list = torch.stack(razor_offset_list, axis=0)
            if is_compresshead:
                razor_mod = i % kv_heads

            scale = qkscale
            if is_logN:
                scale *= logN[i]
            kv_split_per_core = 0
            head_num_move = 0
            block_size_calc = 0
            kvsplit = 0
            former_head = 0
            if is_quant:
                # isLongSeq = max_context_len > num_blocks * 128 * 2 and num_tokens < num_blocks * 0.8
                # if num_tokens * num_heads < 0.8 * num_blocks or isLongSeq:
                #     kvsplit, kv_split_per_core = PagedAttentionOperation.getkvsplit(num_tokens, num_blocks, max_context_len, block_size, num_blocks, isLongSeq)
                # else:
                kvsplit = 1
                kv_split_per_core = max_context_len
                head_num_move = 1
                former_head = PagedAttentionOperation.get_former_head(kvsplit, num_blocks, num_tokens, num_heads)
                # former_head = PagedAttentionOperation.get_head_num_move(num_heads, kv_heads, head_size_qk, head_size_vo)
                block_size_calc = PagedAttentionOperation.get_blockszie_calc(max_context_len, block_size, head_size_qk, head_size_vo, is_int8,
                                                                             is_quant, kv_split_per_core, head_num_move, former_head)
            out = []
            if maskType == 0:
                if is_lookahead:
                    mask_slice = mask[context_len - q_seqlen : context_len, :context_len]
                    out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask[i])
                else:
                    out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask, compressType, razor_mod, razor_offset_list,\
                                                                       k_descale, k_offset, v_descale, v_offset, p_scale, is_int8, is_int8_offset, is_quant, is_quant_offline,
                                                                       num_heads, block_size, block_size_calc, kvsplit, kv_split_per_core)
            elif maskType == 2:
                out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len], compressType, razor_mod, razor_offset_list,\
                                                                   k_descale, k_offset, v_descale, v_offset, p_scale, is_int8, is_int8_offset, is_quant, is_quant_offline,
                                                                   num_heads, block_size, block_size_calc, kvsplit, kv_split_per_core)
            elif maskType == 1:
                if is_lookahead:
                    mask_slice = mask[context_len - q_seqlen : context_len, :context_len]
                    out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask_slice)
                else:
                    out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask[i // mask_index_coff, :, :context_len], compressType, razor_mod, razor_offset_list,\
                                                                       k_descale, k_offset, v_descale, v_offset, p_scale, is_int8, is_int8_offset, is_quant, is_quant_offline,
                                                                       num_heads, block_size, block_size_calc, kvsplit, kv_split_per_core)
            elif maskType == 3:
                mask_slice = mask[cu_seqlen : (cu_seqlen + q_seqlen), :context_len]
                out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask_slice)
            elif maskType == 4:
                mask_slice = mask[cu_seqlen : (cu_seqlen + q_seqlen), :context_len]
                out = PagedAttentionOperation.ref_masked_attention(q, keys, values, scale, mask)

            if is_lookahead:
                out = out.reshape(-1, num_heads, head_size_vo)
                output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
                cu_seqlen += q_seqlens[i]
            else:
                out = out.reshape(num_heads, head_size_vo)
                output[index] = out
                index = index + 1

    def get_alibi_slopes(num_head):
        n = 2 ** math.floor(math.log2(num_head))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < num_head:
            m1 = 2.0 ** ( -4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (num_head - n), 2))
            slopes = torch.cat([slopes, mm])
        return slopes

    @staticmethod
    def create_lookahead_mask(num_tokens, batch, max_seq_len, q_seqlen_list, mask_type):
        mask = np.array([])
        # SPEC mask
        if mask_type == 3:
            mask = np.zeros((num_tokens, max_seq_len), dtype=np.float32)
            pre_qseqlen = 0
            for i in range(batch):
                qseqlen = q_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= -10000.0
                mask[pre_qseqlen:(pre_qseqlen + qseqlen), -qseqlen:] = tri
                pre_qseqlen += qseqlen
        # normal mask
        elif mask_type == 1:
            mask = np.ones(shape=(max_seq_len, max_seq_len)).astype(np.float32)
            mask = np.triu(mask, 1)
            mask *= -10000.0
        elif mask_type == 0:
            mask = np.zeros((batch, max_seq_len, max_seq_len), dtype=np.float32)
            for i in range(batch):
                qseq = q_seqlen_list[i]
                tri = np.zeros((qseq, qseq), dtype=np.float32)
                mask[i][-qseq:, -qseq:] = tri
        return mask

    @staticmethod
    def create_mask(num_tokens, batch, num_head, max_seq_len, context_lens, mask_type):
        mask = np.array([])
        if mask_type == 2:
            mask = np.zeros((batch, num_head, 1, max_seq_len), dtype=np.float16)
            alibi_slopes = PagedAttentionOperation.get_alibi_slopes(num_head)
            for i, context_len in enumerate(context_lens):
                if context_len == 0:
                    continue
                position_ids = np.arange(context_len).astype(np.int32)
                alibi_bias = (position_ids - context_len + 1).astype(np.float16)
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
                mask[i, :, :, :context_len] = alibi_bias
        # normal mask batch, 1, max_seq_len
        elif mask_type == 1:
            mask = np.zeros((batch, 1, max_seq_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
        return mask

    @staticmethod
    def create_seq_lens(num_tokens, batch):
        seq_lens = random.sample(range(1, num_tokens), k = batch - 1)
        seq_lens.append(0)
        seq_lens.append(num_tokens)
        seq_lens = sorted(seq_lens)
        seq_lens_list = [seq_lens[i] - seq_lens[i-1] for i in range(1, len(seq_lens))]
        return seq_lens_list

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            tensor_count = 0
            for j in range(len(PagedAttentionOperation.in_tensors)):
                if PagedAttentionOperation.in_tensors[j].shape[0] != 0:
                    tensor_count += 1
                    if tensor_count == i + 1:
                        if get_soc_version() == "Ascend950":
                            cpu_tensor = PagedAttentionOperation.in_tensors[j].cpu().to(PagedAttentionOperation.trans_dtype(datatype))
                            PagedAttentionOperation.in_tensors[j] = cpu_tensor.npu()
                            PagedAttentionOperation.golden_tensors[j] = cpu_tensor.npu()
                        else:
                            PagedAttentionOperation.in_tensors[j] = \
                                PagedAttentionOperation.in_tensors[j].to(PagedAttentionOperation.trans_dtype(datatype))
                        return torch_npu.npu_format_cast(PagedAttentionOperation.in_tensors[j], format_dict[format])
        soc_version = get_soc_version()
        json_data = json.loads(op_params)
        if soc_version == "Ascend910B" or soc_version == "Ascend950":
            is_NZ = False
        else:
            is_NZ = True
        maskType = 0
        if "maskType" in json_data:
            maskType = json_data["maskType"]
        is_mask = maskType != 0
        quantType = 0
        if "quantType" in json_data:
            quantType = json_data["quantType"]
        is_mask = maskType != 0
        is_dynamic = "batchRunStatusEnable" in json_data and json_data["batchRunStatusEnable"]
        is_lookahead = "calcType" in json_data and json_data["calcType"] == 1
        is_compresshead = "compressType" in json_data and json_data["compressType"] != 0
        is_logN = "scaleType" in json_data and json_data["scaleType"] == 1
        is_razor_rope = "compressType" in json_data and json_data["compressType"] == 2
        is_int8 = quantType == 1
        is_quant = quantType >= 2
        is_quant_offline = quantType == 2
        is_int8_offset = "hasQuantOffset" in json_data and json_data["hasQuantOffset"]
        is_kv_combined = "mlaVHeadSize" in json_data and json_data["mlaVHeadSize"] > 0
        is_bnsd = "inputLayout" in json_data and json_data["inputLayout"] == 1
        # 0 query [num_tokens, num_head, head_size]
        # 1 keyCache [num_blocks, block_size, kv_head_num, head_size_k]
        # 2 valueCache [num_blocks, block_size, kv_head_num, head_size_v]
        # 3 blockTables [num_tokens, max_num_blocks_per_query]
        # 4 contextLens [batch]
        if is_bnsd:
            num_tokens = shapes[0][0]
            num_head = shapes[0][1]
            head_size = shapes[0][2]
            num_blocks = shapes[1][0]
            if is_NZ:
                block_size = shapes[1][2]
                kv_head_num = num_head
                head_size_k = head_size
            else:
                block_size = shapes[1][2]
                kv_head_num = shapes[1][1]
                head_size_k = shapes[1][3]
        else:
            num_tokens = shapes[0][0]
            num_head = shapes[0][1]
            head_size = shapes[0][2]
            num_blocks = shapes[1][0]
            block_size = shapes[1][2] if is_NZ else shapes[1][1]
            kv_head_num = num_head if is_NZ else shapes[1][2]
            head_size_k = head_size if is_NZ else shapes[1][3]
        if not is_kv_combined:
            head_size_v = head_size if is_NZ else shapes[2][3]
            max_num_blocks_per_query = shapes[3][1]
            batch = shapes[4][0]
        else:
            head_size_v = json_data["mlaVHeadSize"]
            max_num_blocks_per_query = shapes[2][1]
            batch = shapes[3][0]
        if soc_version == "Ascend950":
            max_seq_len = 32
        else:
            max_seq_len = 256

        if is_compresshead:
            # 0 query [num_tokens, num_head, head_size]
            # 1 keyCache [num_blocks * kv_head_num, block_size, 1, head_size]
            # 2 valueCache [num_blocks * kv_head_num, block_size, 1, head_size]
            # 3 blockTables [num_tokens * kv_head_num, max_num_blocks_per_query]
            # 4 contextLens [batch * num_head]
            num_head = json_data["headNum"]
            kv_head_num = json_data["kvHeadNum"]
            batch = shapes[4][0] // num_head

        if is_mask:
            max_seq_len = (shapes[5][1] * 16) if is_NZ else shapes[5][-1]
            if maskType == 4:
                k_seq_len = np.array(json_data["k_seq_len"]) 
                max_seq_len = max(k_seq_len)
        k_seq_len = max_seq_len
        dtype = datatype
        if datatype == 'bf16':
            dtype = 'float32'
        data = []
        golden_data = []
        # inTensor list
        query = np.array([])
        key_cache = np.array([])
        value_cache = np.array([])
        block_tables = []
        context_lens = []
        mask = np.array([])
        batch_run_status = np.array([]).astype(np.int32)
        k_descale = np.array([])
        k_descale_int64 = np.array([])
        k_offset = np.array([])
        v_descale = np.array([])
        v_descale_int64 = np.array([])
        v_offset = np.array([])
        q_seq_lens = np.array([]).astype(np.int32)
        razor_offset = np.array([])
        p_scale = np.array([])
        logN = np.array([])
        q_range = 1.0
        kv_range = 1.0
        kv_dtype = dtype
        if quantType >= 2:
            q_range = 5
            kv_range = 5
            dtype = np.int8
            kv_dtype = np.int8
        if quantType == 1:
            q_range = 4
            kv_range = 4
            if soc_version == "Ascend950":
                if kv_dtype == "float8_e4m3fn" or kv_dtype == "float8_e5m2":
                    kv_dtype = np.float16
            else:
                kv_dtype = np.int8
        # create q,k,v
        if is_compresshead:
            query = np.random.uniform(-q_range, q_range, size=(num_tokens, num_head, head_size)).astype(dtype)
            key_cache = np.random.uniform(-kv_range, kv_range, size=(num_blocks * kv_head_num, block_size, 1, head_size_k)).astype(kv_dtype)
            if is_kv_combined:
                value_cache = key_cache[:, :, :, :head_size_v]
            else:
                value_cache = np.random.uniform(-kv_range, kv_range, size=(num_blocks * kv_head_num, block_size, 1, head_size_v)).astype(kv_dtype)
        else:
            query = np.random.uniform(-q_range, q_range, size=(num_tokens, num_head, head_size)).astype(dtype)
            key_cache = np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_head_num, head_size_k)).astype(kv_dtype)
            if is_kv_combined:
                value_cache = key_cache[:, :, :, :head_size_v]
            else:
                value_cache = np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_head_num, head_size_v)).astype(kv_dtype)

        # create context_lens
        context_lens = [random.randint(max_seq_len, max_seq_len) for _ in range(batch)]

        max_context_len = max_seq_len
        max_num_blocks_per_query = (max_context_len + block_size - 1) // block_size
        # create q_seq_lens,mask
        if is_lookahead:
            q_seq_lens = PagedAttentionOperation.create_seq_lens(num_tokens, batch)
            q_seq_lens = np.array(q_seq_lens).astype(np.int32)
            mask = PagedAttentionOperation.create_lookahead_mask(num_tokens, batch, max_seq_len, q_seq_lens, maskType)
        elif is_mask:
            mask = PagedAttentionOperation.create_mask(num_tokens, batch, num_head, max_seq_len, context_lens, maskType)
        # correct context_lens
        if is_compresshead:
            context_lens = [val for val in context_lens for _ in range(num_head)]

        batch = len(context_lens)
        # create block_tables
        for _ in range(batch):
            block_table = [random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_query)]
            block_tables.append(block_table)
        context_lens = np.array(context_lens).astype(np.int32)
        block_tables = np.array(block_tables).astype(np.int32)
        # create batch_run_status
        if is_dynamic:
            batch_run_status = [random.randint(0, 1) for _ in range(batch)]
            batch_run_status[0] = 1 # 防止全0
            batch_run_status = np.array(batch_run_status).astype(np.int32)
            context_lens = context_lens * batch_run_status
        # create k_descale,k_offset,v_descale,v_offset,p_scale
        if is_int8:
            k_descale = np.random.randint(-1, 2, size=(kv_head_num * head_size)).astype(np.float32)
            k_descale_int64 = PagedAttentionOperation.process_deq_scale(k_descale)
            v_descale =  np.random.randint(-1, 2, size=(kv_head_num * head_size)).astype(np.float32)
            v_descale_int64 = PagedAttentionOperation.process_deq_scale(v_descale)
            if is_int8_offset:
                k_offset = np.random.randint(-20, 20, size=(kv_head_num * head_size)).astype(np.int32)
                v_offset = np.random.randint(-20, 20, size=(kv_head_num * head_size)).astype(np.int32)
        if is_quant:
            k_descale = np.random.uniform(-5/127, 5/127, size=(num_head)).astype(np.float32)
            v_descale =  np.random.uniform(-5/127, 5/127, size=(num_head)).astype(np.float32)
            p_scale = np.random.uniform(0, 127, size=(num_head)).astype(np.float32)
            # isLongSeq = max_context_len > num_blocks * 128 * 2 and num_tokens < num_blocks * 0.8
            # if num_tokens * num_head < 0.8 * num_blocks or isLongSeq:
            #     kvsplit, kv_split_per_core = PagedAttentionOperation.getkvsplit(num_tokens, num_blocks, max_context_len, block_size, num_blocks, isLongSeq)
            # else:
            #     kvsplit = 1
            #     kv_split_per_core = max_context_len
            # head_num_move = PagedAttentionOperation.get_head_num_move(num_head, kv_head_num, head_size, head_size)
            # block_size_calc = PagedAttentionOperation.get_blockszie_calc(max_context_len, block_size, head_size, head_size, is_int8, is_quant, kv_split_per_core, head_num_move)

        # create razor_offset
        if is_razor_rope:
            razor_offset = np.zeros((num_blocks * kv_head_num, block_size), dtype = np.float32)
            mask_temp = np.random.choice([False, True], size = num_blocks * kv_head_num, p = [0.2, 0.8])
            random_indices = np.random.randint(0, block_size, size = np.sum(mask_temp))
            random_values = np.random.uniform(0, 20, size = np.sum(mask_temp))
            active_rows = np.where(mask_temp)[0]
            razor_offset[active_rows, random_indices] = torch.from_numpy(random_values).to(torch.float32)
        # create logN
        if is_logN:
            logN = np.random.uniform(1.0, 2.0, size=(batch)).astype(np.float32)
        # convert nd to nz
        if is_NZ:
            key_cache_nz = key_cache.reshape(num_blocks, block_size, -1)
            key_cache_nz = PagedAttentionOperation.convert_nd_to_nz(key_cache_nz)
            key_cache_nz = key_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float16)
            key_cache_nz = np.ascontiguousarray(key_cache_nz)

            value_cache_nz = value_cache.reshape(num_blocks, block_size, -1)
            value_cache_nz = PagedAttentionOperation.convert_nd_to_nz(value_cache_nz)
            value_cache_nz = value_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float16)
            value_cache_nz = np.ascontiguousarray(value_cache_nz)

            mask_nz = np.array([])
            max_context_len_pad = (max_context_len + 15) // 16 * 16
            if maskType == 1:
                mask_pad = np.zeros((num_tokens, 16, max_context_len_pad))
                mask_pad[:, :1, :max_context_len] = mask
                mask_nz = PagedAttentionOperation.convert_nd_to_nz(mask_pad)
                mask_nz = mask_nz.reshape(num_tokens, -1, 16, 16).astype(np.float16)
                mask_nz = np.ascontiguousarray(mask_nz)
            elif maskType == 2:
                mask_pad = np.zeros((num_tokens, num_head, 16, max_context_len_pad))
                mask_pad[:, :, :1, :max_context_len] = mask
                mask_nz = PagedAttentionOperation.convert_nd_to_nz(mask_pad)
                mask_nz = mask_nz.reshape(num_tokens * num_head, -1, 16, 16).astype(np.float16)
                mask_nz = np.ascontiguousarray(mask_nz)
            elif maskType == 3:
                num_tokens_pad = (num_tokens + 15) // 16 * 16
                mask_pad = np.zeros((1, num_tokens_pad, max_context_len_pad))
                mask_pad[:1, :num_tokens, :max_context_len] = mask
                mask_nz = PagedAttentionOperation.convert_nd_to_nz(mask_pad)
                mask_nz = mask_nz.reshape(1, -1, num_tokens_pad, 16).astype(np.float16)
                mask_nz = np.ascontiguousarray(mask_nz)
            elif maskType == 4:
                context_lens = np.array(json_data["k_seq_len"]) 
                q_seq_lens = np.array(json_data["q_seq_len"]) 

                seq_len = 128
                # 创建一个矩阵，初始为全 0
                mask = np.zeros((seq_len, seq_len), dtype=np.float16)
                # 构造严格上三角（不包括对角线）的 mask，赋值为 -inf
                mask[np.triu_indices(seq_len, k=1)] = 1
                mask *= -60000
                mask_nz = PagedAttentionOperation.convert_nd_to_nz(mask)
                mask_nz = mask_nz.reshape(1, 8, 128, 16).astype(np.float16)
                mask_nz = np.ascontiguousarray(mask_nz)

                max_k_seqlen = context_lens[0]
                mask = torch.zeros(size=(q_seq_lens.sum(), max(context_lens)))
                prev_qseq = 0
                for i in range(len(context_lens)):
                    qseq = q_seq_lens[i]
                    kseq = context_lens[i]
                    start = kseq - qseq
                    tri = torch.ones((qseq, qseq)).half()
                    tri = torch.triu(tri, 1)
                    tri *= -60000
                    mask[prev_qseq:(prev_qseq + qseq), start:kseq] = tri
                    prev_qseq += qseq
                mask = mask.numpy()

            data.extend([query, key_cache_nz, value_cache_nz, block_tables, context_lens])
            data.extend([mask_nz, batch_run_status, k_descale, k_offset, v_descale, v_offset, q_seq_lens, razor_offset, p_scale, logN])
        else:
            if is_kv_combined:
                data.extend([query, key_cache, block_tables, context_lens])
            elif is_bnsd:
                key_cache_bnsd = key_cache.transpose(0,2,1,3)
                value_cache_bnsd = value_cache.transpose(0,2,1,3)
                data.extend([query, key_cache_bnsd, value_cache_bnsd, block_tables, context_lens])
            else:
                data.extend([query, key_cache, value_cache, block_tables, context_lens])
            data.extend([mask, batch_run_status, k_descale, k_offset, v_descale, v_offset, q_seq_lens, razor_offset, p_scale, logN])

        golden_data.extend([query, key_cache, value_cache, block_tables, context_lens])
        golden_data.extend([mask, batch_run_status, k_descale, k_offset, v_descale, v_offset, q_seq_lens, razor_offset, p_scale, logN])

        in_tensors = [torch.from_numpy(tensor).npu() for tensor in data]
        
        golden_tensors = [torch.from_numpy(tensor).npu() for tensor in golden_data]
        PagedAttentionOperation.golden_tensors = golden_tensors
        PagedAttentionOperation.in_tensors = in_tensors

        if soc_version == "Ascend950":
            in_tensor0_cpu = PagedAttentionOperation.in_tensors[0].cpu().to(PagedAttentionOperation.trans_dtype(datatype))
            PagedAttentionOperation.in_tensors[0] = in_tensor0_cpu.npu()
        else:
            PagedAttentionOperation.in_tensors[0] = \
                PagedAttentionOperation.in_tensors[0].to(PagedAttentionOperation.trans_dtype(datatype))

        return torch_npu.npu_format_cast(PagedAttentionOperation.in_tensors[0], format_dict[format])

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        json_data = json.loads(op_params)
        host_tensor_dict = {}
        contextLens_index = 4
        batchRunStatus_index = 5
        qSeqLens_index = 5

        if "mlaVHeadSize" in json_data and json_data["mlaVHeadSize"] > 0:
            host_tensor_dict["mlaVHeadSize"] = json_data["mlaVHeadSize"]
            contextLens_index -= 1
            batchRunStatus_index -= 1
            qSeqLens_index -= 1
        if "maskType" in json_data and json_data["maskType"] != 0:
            host_tensor_dict["maskType"] = json_data["maskType"]
            batchRunStatus_index += 1
            qSeqLens_index += 1

        if "batchRunStatusEnable" in json_data and json_data["batchRunStatusEnable"]:
            host_tensor_dict["batchRunStatus"] = input_tensor_list[batchRunStatus_index].tolist()
        if "calcType" in json_data and json_data["calcType"] == 1:
            host_tensor_dict["qLens"] = input_tensor_list[qSeqLens_index].tolist()
        host_tensor_dict["contextLens"] = input_tensor_list[contextLens_index].tolist()

        run_param = json.dumps(host_tensor_dict)
        operation.set_varaintpack_param(run_param)

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if get_soc_version() == "Ascend910B" or get_soc_version() == "Ascend950":
            is_NZ = False
        else:
            is_NZ = True
        head_size_o = 0
        if "mlaVHeadSize" in json_data and json_data["mlaVHeadSize"] > 0:
            head_size_o = json_data["mlaVHeadSize"]
        else:
            head_size_o = in_tensors[0].shape[2] if is_NZ else in_tensors[2].shape[3]
        out_shape = (in_tensors[0].shape[0], json_data["headNum"], head_size_o)
        dtype = None
        if "outDataType" in json_data and json_data["outDataType"] == 1:
             dtype = torch.float16
        if "outDataType" in json_data and json_data["outDataType"] == 27:
             dtype = torch.bfloat16
        if dtype == None:
            dtype = in_tensors[0].dtype
        ref_output = torch.zeros(out_shape, dtype=dtype)
        maskType = 0
        if "maskType" in json_data:
            maskType = json_data["maskType"]
        # 非customize方法载入数据
        if not hasattr(PagedAttentionOperation, 'golden_tensors'):
            golden_tensors = []
            for i in range(15):
                golden_tensors.append(np.array([]))
            golden_tensors = [torch.from_numpy(tensor).npu() for tensor in golden_tensors]
            for i in range(len(in_tensors)):
                golden_tensors[i] = in_tensors[i].npu()
            if is_NZ:
                query = golden_tensors[0]
                key_cache = golden_tensors[1]
                key_cache = key_cache.permute(0, 2, 1, 3)
                dim0 = key_cache.shape[0]
                dim1 = key_cache.shape[1]
                dim2 = key_cache.shape[2]
                dim3 = key_cache.shape[3]
                key_cache = key_cache.contiguous().view(dim0, dim1, dim2 * dim3)
                embedding_size = query.shape[-1]
                key_cache = key_cache[:, :, : embedding_size * json_data["kvHeadNum"]]
                key_cache = key_cache.contiguous().view(dim0, dim1, json_data["kvHeadNum"], embedding_size)
                golden_tensors[1] = key_cache

                value_cache = golden_tensors[2]
                value_cache = value_cache.permute(0, 2, 1, 3)
                dim0 = value_cache.shape[0]
                dim1 = value_cache.shape[1]
                dim2 = value_cache.shape[2]
                dim3 = value_cache.shape[3]
                value_cache = value_cache.contiguous().view(dim0, dim1, dim2*dim3)
                embedding_size = query.shape[-1]
                value_cache = value_cache[:, :, : embedding_size * json_data["kvHeadNum"]]
                value_cache = value_cache.contiguous().view(dim0, dim1, json_data["kvHeadNum"], embedding_size)
                golden_tensors[2] = value_cache
                if maskType != 0:
                    # mask nz to nd
                    mask = golden_tensors[5]
                    mask = mask.permute(0, 2, 1, 3)
                    dim0 = mask.shape[0]
                    dim1 = mask.shape[1]
                    dim2 = mask.shape[2]
                    dim3 = mask.shape[3]
                    mask = mask.contiguous().view(dim0, dim1, dim2*dim3)
                    if maskType == 2:
                        batch = golden_tensors[4].shape[0]
                        if dim0 != json_data["headNum"]:
                            mask = mask.contiguous().view(batch, json_data["headNum"], dim1, dim2*dim3)
                        else:
                            mask = mask.contiguous().view(1, json_data["headNum"], dim1, dim2*dim3)
                    golden_tensors[5] = mask[:,:,:1,:]
                    if maskType == 4 :
                        num_tokens = in_tensors[6].sum().item()
                        max_k_seqlen = max(in_tensors[4]).item()
                        batch_size = in_tensors[4].shape[0]
                        # 转numpy
                        mask = torch.zeros(size=(num_tokens, max_k_seqlen))
                        prev_qseq = 0
                        for i in range(batch_size):
                            qseq = in_tensors[6][i]
                            kseq = in_tensors[4][i]
                            start = kseq - qseq
                            tri = torch.ones((qseq, qseq)).half()
                            tri = torch.triu(tri, 1)
                            tri *= -60000
                            mask[prev_qseq:(prev_qseq + qseq), start:kseq] = tri
                            prev_qseq += qseq
                        golden_tensors[5] = mask

            PagedAttentionOperation.golden_tensors = golden_tensors
        PagedAttentionOperation.ref_single_query_cached_kv_attention(
            op_params,
            ref_output,
            PagedAttentionOperation.golden_tensors[0].cpu(),
            PagedAttentionOperation.golden_tensors[1].cpu(),
            PagedAttentionOperation.golden_tensors[2].cpu(),
            PagedAttentionOperation.golden_tensors[3].cpu(),
            PagedAttentionOperation.golden_tensors[4].cpu(),
            PagedAttentionOperation.golden_tensors[5].cpu(),
            PagedAttentionOperation.golden_tensors[7].cpu(),
            PagedAttentionOperation.golden_tensors[8].cpu(),
            PagedAttentionOperation.golden_tensors[9].cpu(),
            PagedAttentionOperation.golden_tensors[10].cpu(),
            PagedAttentionOperation.golden_tensors[11].cpu(),
            PagedAttentionOperation.golden_tensors[12].cpu(),
            PagedAttentionOperation.golden_tensors[13].cpu(),
            PagedAttentionOperation.golden_tensors[14].cpu()
        )
        if hasattr(PagedAttentionOperation, 'in_tensors'):
            delattr(PagedAttentionOperation, 'in_tensors')
        if hasattr(PagedAttentionOperation, 'golden_tensors'):
            delattr(PagedAttentionOperation, 'golden_tensors')
        return [ref_output.cpu()]

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        if get_soc_version() == "Ascend950":
            if hasattr(PagedAttentionOperation, 'in_tensors'):
                delattr(PagedAttentionOperation, 'in_tensors')
            if hasattr(PagedAttentionOperation, 'golden_tensors'):
                delattr(PagedAttentionOperation, 'golden_tensors')

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION


class SelfAttentionOperation(DataGen):
    class GoldenMaskType(IntEnum):
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
        MASK_TYPE_RAZOR_FUSION = 11

    class ATBMaskType(IntEnum):
        MASK_TYPE_UNDEFINED = 0
        MASK_TYPE_NORM = 1
        MASK_TYPE_ALIBI = 2
        MASK_TYPE_NORM_COMPRESS = 3

    def gen_mask(batch, heads, max_seq,data_type, mask_type,is_decoder=False,is_triu_mask=False,is_alibi=False,dynamic_batch=False,long_seq=False):
        import random
        q_max_seq = max_seq
        kv_max_seq = max_seq

        mask_type_dict = {
            # 不加mask
            __class__.GoldenMaskType.MASK_TYPE_NO_MASK : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: 0)),
            __class__.GoldenMaskType.MASK_TYPE_NO_HEAD : ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 四维的alibi mask
            __class__.GoldenMaskType.MASK_TYPE_ALIBI_WITH_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            # 三维的alibi mask
            __class__.GoldenMaskType.MASK_TYPE_ALIBI_NO_BATCH : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            __class__.GoldenMaskType.MASK_TYPE_NO_HEAD_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            __class__.GoldenMaskType.MASK_TYPE_NO_BATCH : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
        }
        
        # kernel中mask的系数
        if data_type == torch.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        elif data_type == torch.bfloat16 and is_alibi:
            post_mask_coff = 1
            pre_mask_coff = -float("inf")
        elif data_type == torch.float32 and is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        if data_type == torch.float16:
            if is_alibi or long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if is_alibi:
                select_zero = False
            elif dynamic_batch or is_decoder:
                select_zero = True
            else:
                select_zero = False
        else:
            if is_alibi or is_decoder:
                select_zero = True
            else:
                select_zero = False
        if is_triu_mask:
            select_zero = False

        mask_info = mask_type_dict[mask_type]
        mask = np.ones(shape=mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(max_seq), k = 300)
        if select_zero:
            mask.flat[zero_indice] = 0
        mask = torch.from_numpy(mask).to(torch.float32)
        SelfAttentionOperation.mask_info = mask_info
        SelfAttentionOperation.post_mask_coff = post_mask_coff
        SelfAttentionOperation.pre_mask_coff = pre_mask_coff
        return mask,post_mask_coff

    def gen_seq_len(batch, max_seq, variate_seq=False):
        if variate_seq:
            num = max_seq // 16
            seqlen_aligned_arange = np.arange(1, num) * 16
            if batch > num:
                seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
                seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
                seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
            else:
                seqlen_aligned = seqlen_aligned_arange
            sp_list = np.random.randint(0, 15, size=(num - 1))
            seqlen = seqlen_aligned - sp_list
            seqlen = seqlen[-batch:]
            seqlen_aligned = seqlen_aligned[-batch:]
            logging.debug(seqlen)
        else:
            max_seq_aligned = (max_seq + 15) // 16 * 16
            sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
            sp_list = sp_list.astype(np.int32)
            seqlen = np.ones((batch,)) * max_seq
            seqlen = seqlen.astype(np.int32)
            logging.debug(seqlen)
            seqlen_aligned = np.ones((batch,)) * max_seq_aligned
            seqlen_aligned = seqlen_aligned.astype(np.int32)

        ntokens = seqlen.sum()
        logging.debug(f"ntokens: {ntokens}")
        return seqlen, seqlen_aligned, ntokens

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

    def get_shape(container, indices):
        current = container
        for idx in indices:
            if not isinstance(current, (list, tuple)) or len(current) <= idx:
                raise ValueError(f"Invalid shape access at index {idx}, container: {current}")
            current = current[idx]
        if not isinstance(current, (int)):
            raise TypeError(f"Shape value must be int, got {type(current)} at {indices}")
        return int(current)

    def calc_expect_func_950(batch, seqlen, heads, embed, headDims_v, group_num=32, pseShiftFlag=False, is_mask=False,
                             datatype=torch.float16, mask_type=0, op_param=None, q_seqlen=None, shapes=None, atb_mask_type=0):
        logging.debug(f"batch {batch}, seqlen {seqlen}, heads {heads}, embed {embed}, headDims_v {headDims_v}, group_num {group_num}, pseShiftFlag {pseShiftFlag}, is_mask {is_mask}, mask_type {mask_type}")
        variate_seq = False
        is_decoder = False
        src_type = 'float16'
        mask_dtype = torch.int8
        fp32 = True
        logging.debug(f"group_num: {group_num}")
        logging.debug("q_seq is:")
        if q_seqlen:
            q_seqlen = np.array(q_seqlen)
            q_ntokens = q_seqlen.sum()
            kv_seqlen = q_seqlen
            kv_ntokens = q_ntokens
        elif is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = SelfAttentionOperation.gen_seq_len(batch, 1, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = SelfAttentionOperation.gen_seq_len(batch, seqlen, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = SelfAttentionOperation.gen_seq_len(batch, seqlen, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens   # crossattention时，q_seqlen != k_seqlen

        max_s = np.max(q_seqlen)
        if atb_mask_type == __class__.ATBMaskType.MASK_TYPE_NORM_COMPRESS:
            max_s = max(2048, max_s)

        ntokens2 = (q_seqlen * kv_seqlen).sum()

        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)))
        tor = np.float32(1.0 / math.sqrt(1.0 * embed))
        q = (q * tor).to(datatype)
        k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed))).to(datatype)
        if headDims_v == 0:
            headDims_v = embed
        v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * headDims_v))).to(datatype)
        kv_head = group_num
        
        mask = np.ones(shape=(1, max_s, max_s)).astype(np.int8)  # 使用当前最大seqlen生成mask
        mask = np.triu(mask, 1)
        # mask *= -10000.0
        logging.debug(mask)
        
        mask = torch.from_numpy(mask)

        is_alibi = mask_type in (__class__.GoldenMaskType.MASK_TYPE_ALIBI_NO_BATCH, __class__.GoldenMaskType.MASK_TYPE_ALIBI_WITH_BATCH)

        if is_alibi:
            is_triu_mask = True
            logging.info(f"before gen_mask mask_type {mask_type}")
            pseshift, post_mask_coff = __class__.gen_mask(batch, heads, max_s, datatype, mask_type, is_decoder, is_triu_mask, is_alibi)
            mask_shape = pseshift.shape
            logging.debug(f"mask shape: {mask_shape}")
            # logging.debug(pseshift)
            pseShiftFlag = True
        else:
            pseshift = np.ones(shape=(1, max_s, max_s)).astype(np.float16)  # 使用当前最大seqlen生成pseshift
            pseshift = np.triu(pseshift, 1)
            pseshift *= -10000.0
            pseshift = torch.from_numpy(pseshift)
            logging.debug(pseshift)

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
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice = torch.permute(k_slice, (1, 0, 2))
            k_slice_t =torch.permute(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, headDims_v)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            score = SelfAttentionOperation.group_mm_torch_encoder(heads, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.reshape([-1, ])), 0)
            score = score * tor
            if is_alibi:
                score = score + __class__.mask_info[1](pseshift, idx, q_s, kv_s) * __class__.post_mask_coff
            if is_mask:
                mask_pb = mask[:, :q_s, :kv_s]
                mask_pb = torch.repeat_interleave(mask_pb, repeats=heads, dim=0)
                score[mask_pb.to(torch.bool)] = -1.7 * 10 ** 38

            score_max, _ = torch.max(score, dim=-1)
            score = score - score_max.view((heads, q_s, 1))
            score_exp = torch.exp(score)

            if not fp32:
                score_sum = torch.sum(score_exp.to(datatype), dim=-1)
                if _p is None:
                    _p = score_exp.to(datatype).view([-1, ])
                else:
                    _p = torch.cat((_p, score_exp.view([-1, ])), 0)
                p = score_exp.to(datatype) / score_sum.view((head_num, q_s, 1)).to(datatype)

                out_sub = SelfAttentionOperation.group_mm_torch_encoder(heads, group_num, p, v_slice).to(datatype)
            else:
                score_sum = torch.sum(score_exp, dim=-1)
                if _p is None:
                    _p = score_exp.to(datatype).view([-1, ])
                else:
                    _p = torch.cat((_p, score_exp.to(datatype).view([-1, ])), 0)
                p = score_exp.to(datatype)
                out_sub = SelfAttentionOperation.group_mm_torch_encoder(heads, group_num, p, v_slice)
                out_sub = out_sub / score_sum.view((heads, q_s, 1)).to(datatype)

            out_sub = out_sub.view(heads, q_s, headDims_v)
            out_sub = torch.permute(out_sub, (1, 0, 2))
            out_sub = out_sub.contiguous()
            
            logging.debug(f"out_sub shape {out_sub.shape}")
            if out is None:
                out = out_sub
            else:
                out = torch.cat((out, out_sub), dim=0)
            logging.debug(f"out shape {out.shape}")
            

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        logging.debug("==> data generate finished!")

        q = q.to(datatype).view(-1, heads, embed)
        k = k.to(datatype).view(-1, group_num, embed)
        v = v.to(datatype).view(-1, group_num, headDims_v)
        # pseshift = pseshift.to(src_type).view(1, 1, max_s, max_s)
        # pseshift = torch.repeat_interleave(pseshift, repeats=heads, dim=1)
        if is_alibi:
            pseshift = pseshift.to(datatype).view(mask_shape)
        mask = mask.to(mask_dtype).view(max_s, max_s)
        
        q_len = torch.Tensor(q_seqlen).to(torch.int32)
        # prefix_sum_q_len = torch.cumsum(q_len, dim=0)
        out = out.to(datatype).view(-1, heads, headDims_v)

        ret_data_list = [q, k, v]
        if atb_mask_type == __class__.ATBMaskType.MASK_TYPE_NORM_COMPRESS:
            mask = mask[:2048,:2048]
        if is_mask:
            ret_data_list.append(mask)
        if is_alibi:
            ret_data_list.append(pseshift)
        ret_data_list.append(q_len)
        ret_data_list.append(out)
        ret_data = tuple(ret_data_list)
        
        for tensor in ret_data:
            logging.info(f"calc_expect_func_950 tensor shape {tensor.shape}")
        return ret_data

    def calc_expect_func(batch, seqlen, heads, embed, group_num=32, is_mask=False, datatype=torch.float16, mask_type=0,
                        op_param=None, q_seqlen=None, shapes=None, embed_v=0):
        variate_seq = False
        is_decoder = False
        fp32 = True
        if q_seqlen:
            q_seqlen = np.array(q_seqlen)
            q_ntokens = q_seqlen.sum()
            kv_seqlen = q_seqlen
            kv_ntokens = q_ntokens
        elif is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = SelfAttentionOperation.gen_seq_len(batch, 1, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = SelfAttentionOperation.gen_seq_len(batch, seqlen, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = SelfAttentionOperation.gen_seq_len(batch, seqlen, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens   # crossattention时，q_seqlen != k_seqlen

        max_s = np.max(q_seqlen)
        ntokens2 = (q_seqlen * kv_seqlen).sum()

        layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)))
        tor = np.float32(1.0 / math.sqrt(1.0 * embed))
        
        q = (q * tor).to(datatype)
        k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed))).to(datatype)
        
        if embed_v == 0:
            embed_v = embed
        
        v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed_v))).to(datatype)
        
        is_triu_mask = False
        is_alibi = False
        is_decoder=False
        
        if mask_type in (__class__.GoldenMaskType.MASK_TYPE_ALIBI_NO_BATCH, __class__.GoldenMaskType.MASK_TYPE_ALIBI_WITH_BATCH):
            is_triu_mask = True
            is_alibi = True

        mask = None
        if is_mask:
            mask, post_mask_coff = __class__.gen_mask(batch, heads, max_s, datatype, mask_type, is_decoder, is_triu_mask, is_alibi)
            mask_shape = (max_s, max_s)
            if is_alibi:
                mask_shape = mask.shape
            logging.debug(f"mask shape: {mask.shape}")
            logging.debug(mask)

        asdops_param = __class__.get_asdops_param(q, k.view(kv_ntokens, group_num * embed), v.view(kv_ntokens, group_num * embed_v), mask, q_seqlen, op_param)
        embeddim_v = asdops_param["v_embeddim"]

        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {q.shape}")
        logging.debug(f"k shape: {k.shape}")
        logging.debug(f"v shape: {v.shape}")
        logging.debug(f"layer_id shape: {layer_id.shape}")
        if is_mask:
            logging.debug(f"mask shape: {mask.shape}")

        q_offset = 0
        k_offset = 0
        v_offset = 0

        s = None
        _p = None
        out = None
        
        head_num = heads
        kv_head = group_num
        
        
        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, head_num, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice = torch.permute(k_slice, (1, 0, 2))
            k_slice_t =torch.permute(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embeddim_v)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            score = SelfAttentionOperation.group_mm_torch_encoder(head_num, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.reshape([-1, ])), 0)

            score = score * tor
            if is_mask:
                if is_alibi:
                    score = score + __class__.mask_info[1](mask, idx, q_s, kv_s) * __class__.post_mask_coff
                else:
                    score = score + mask[:, :q_s, :kv_s]
        

            score_max, _ = torch.max(score, axis=-1)
            score = score - score_max.view((head_num, q_s, 1))
            score_exp = torch.exp(score)

            score_sum = torch.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.view([-1, ])
            else:
                _p = torch.cat((_p, score_exp.view([-1, ])), 0)
            p = score_exp / score_sum.view((head_num, q_s, 1))
            out_sub = SelfAttentionOperation.group_mm_torch_encoder(head_num, kv_head, p, v_slice)

            out_sub = out_sub.view(head_num, q_s, embeddim_v)
            out_sub = torch.permute(out_sub, (1, 0, 2)).contiguous()
            # out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = torch.cat((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        logging.debug("==> data generate finished!")

        q = q.to(datatype).view(-1, heads, embed)
        k = k.to(datatype).view(-1, group_num, embed)
        v = v.to(datatype).view(-1, group_num, embeddim_v)
        q_len = torch.from_numpy(q_seqlen.astype(np.int32))
        out = out.to(datatype).view(-1, heads, embeddim_v)

        if is_mask:
            mask = mask.to(datatype).view(mask_shape)
            ret_data = [q, k, v, mask, q_len, out]
        else:
            ret_data = [q, k, v, q_len, out]
        if shapes:
            for i in range(len(shapes)):
                ret_data[i] = ret_data[i].view(shapes[i])
            if len(shapes[0]) == 2:
                ret_data[-1] = ret_data[-1].reshape(q_ntokens, heads * embeddim_v)
        return ret_data

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        json_data = json.loads(op_params)
        run_param = None
        if json_data["calcType"] == 3:
            seqLen_id = 4
            if json_data["maskType"] == 0:
                seqLen_id -= 1
            if "mlaVHeadSize" in json_data and json_data["mlaVHeadSize"] > 0:
                seqLen_id -= 1
            run_param = json.dumps({"seqLen": input_tensor_list[seqLen_id].tolist()})
        elif json_data["calcType"] == 4:
            if json_data["maskType"] == 9:
                run_param = json.dumps({"tokenOffset": input_tensor_list[4].tolist(), "seqLen": input_tensor_list[5].tolist()})
            else:
                run_param = json.dumps({"tokenOffset": input_tensor_list[5].tolist(), "seqLen": input_tensor_list[6].tolist()})
        elif 'kvcacheCfg' in json_data.keys() and json_data['kvcacheCfg'] == 1:
            run_param = json.dumps({"tokenOffset": input_tensor_list[4].tolist(), "seqLen": input_tensor_list[5].tolist()})
        else:
            run_param = json.dumps({"tokenOffset": input_tensor_list[6].tolist(), "seqLen": input_tensor_list[7].tolist()})
        if 'kvCacheWithParam' in json_data.keys() and json_data['kvCacheWithParam'] == 1:
            run_param = json.loads(run_param)
            run_param["kvCacheWithParam"] = True
            run_param = json.dumps(run_param)
        operation.set_varaintpack_param(run_param)

    def gen_swa_mask(max_seq, window_size, pre_mask_coff, mask_info, cache_type=0):
        swa_mask = np.ones(shape=mask_info[0] * pre_mask_coff)
        if window_size < max_seq:
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -window_size)
            swa_mask = triu_mask + tril_mask
        return swa_mask

    def gen_mla_swa_mask(batch, heads, data_type, mask_type, window_size):
        import random
        q_max_seq = 512
        kv_max_seq = 512
        max_seq = 512
        post_mask_coff = 1
        pre_mask_coff = -10000.0

        if window_size > 0:
            select_zero =False

        mask_info = ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s])))
        mask = np.ones(shape=mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(max_seq), k = 300)
        if window_size > 0:
            mask = SelfAttentionOperation.gen_swa_mask(max_seq, window_size, pre_mask_coff, mask_info)
        if select_zero:
            mask.flat[zero_indice] = 0
        
        mask = torch.from_numpy(mask).to(torch.float16)
        post_mask_coff = post_mask_coff
        pre_mask_coff = pre_mask_coff

        return mask.reshape(max_seq, max_seq)

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        json_data = json.loads(op_params)
        logging.info(f"customize json: {json_data}")
        q_seqlen = None
        if "seqLen" in json_data:
            q_seqlen = json_data["seqLen"]
        torch_data_type = dtype_dict[datatype]

        if get_soc_version() == "Ascend950":
            if i == 0:
                try:
                    pseShiftFlag = False
                    is_mask = False
                    golden_mask_type = 0
                    atb_mask_type = 0

                    if "pseShiftFlag" in json_data and json_data["pseShiftFlag"]:
                        pseShiftFlag = True
                    if "maskType" in json_data and json_data["maskType"] != 0:
                        seqlen_index = 4
                        if  json_data["maskType"] in (__class__.ATBMaskType.MASK_TYPE_NORM, __class__.ATBMaskType.MASK_TYPE_NORM_COMPRESS): # 1: norm mask, 3: norm compress mask
                            is_mask = True
                            if json_data["maskType"] == __class__.ATBMaskType.MASK_TYPE_NORM_COMPRESS:
                                golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_NO_BATCH
                            atb_mask_type = json_data["maskType"]
                        elif json_data["maskType"] == __class__.ATBMaskType.MASK_TYPE_ALIBI:
                            pseShiftFlag = True
                            if len(shapes[3]) == 3: # index3: mask, 3: headnum, max_s, max_s
                                golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_ALIBI_NO_BATCH # value: 4
                            elif len(shapes[3]) == 4: # index3: mask, 4: batch, headnum, max_s, max_s
                                golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_ALIBI_WITH_BATCH # value: 3
                    else:
                        seqlen_index = 3
                    logging.info(f" seqlen_index {seqlen_index}")
                    kv_head = SelfAttentionOperation.get_shape(shapes, [1, 1])  # shapes[1][1]
                    qhead = SelfAttentionOperation.get_shape(shapes, [0, 1])    # shapes[0][1]
                    headDims = SelfAttentionOperation.get_shape(shapes, [0, 2]) # shapes[0][2]
                    headDims_v = SelfAttentionOperation.get_shape(shapes, [2, 2]) # shapes[2][2]
                    batch = SelfAttentionOperation.get_shape(shapes, [seqlen_index, 0])    # shapes[3][0]
                    
                    bs = SelfAttentionOperation.get_shape(shapes, [0, 0])
                    if batch == 0:
                        raise ZeroDivisionError("Batch size cannot be zero.")
                    seqlen = bs // batch

                except (ValueError, ZeroDivisionError) as e:
                    logging.error(f"Shape validation failed: {str(e)}")
                    raise

                data = SelfAttentionOperation.calc_expect_func_950(batch=batch, seqlen=seqlen, heads=qhead, embed=headDims, headDims_v=headDims_v,
                                                                  group_num=kv_head, pseShiftFlag = pseShiftFlag, is_mask=is_mask,
                                                                    datatype=torch_data_type, mask_type=golden_mask_type, op_param=op_params,
                                                                   q_seqlen=q_seqlen, shapes=shapes, atb_mask_type=atb_mask_type)

                logging.info(f"seqlen_index!!!: {seqlen_index=}")
                for item in data:
                    if type(item) is torch.Tensor:
                        logging.info(f"tensor shape!!!: {item.shape=}")
                    elif type(item) is list:
                        logging.info(f"list len !!!: {len(item)=}")
                    else:
                        logging.info(f"type(item) !!!: {type(item)=}")
                
                param_seqlen = data[seqlen_index].tolist()
                in_tensors = list(data)
                if datatype == "bf16":
                    in_tensors[0] = in_tensors[0].to(torch.bfloat16)
                    in_tensors[1] = in_tensors[1].to(torch.bfloat16)
                    in_tensors[2] = in_tensors[2].to(torch.bfloat16)
                    in_tensors[-1] = in_tensors[-1].to(torch.bfloat16)
                SelfAttentionOperation.in_tensors_encoder = [tensor.npu() for tensor in in_tensors]
                for i, tensor in enumerate(in_tensors):
                    logging.debug(f"customize, i {i} dtype {tensor.dtype} shape {tensor.shape}")
                return SelfAttentionOperation.in_tensors_encoder[0]
            else:
                return SelfAttentionOperation.in_tensors_encoder[i]
        if 'kvcacheCfg' in json_data.keys() and json_data['kvcacheCfg'] == 1: 
            MASK_TYPE_NO_HEAD_DECODER = 5
            mask_type =MASK_TYPE_NO_HEAD_DECODER
            data_type = torch.float16
            batch = 8
            kv_head = 32        # kv_head num
            SelfAttentionOperation.is_decoder = 1       # prefill or decoder
            heads = 32          # llama7b  hidden_size 4096
            embeddim = 128
            max_seq = 256
            tor = 1
            SelfAttentionOperation.dynamic_batch = False
            kv_seqLen = [114] * batch
            is_clamp = 0
            clamp_min = 0
            clamp_max = 0
            q_seqlen = [1] * batch
            q_ntokens = sum(q_seqlen)
            kv_ntokens =  sum(kv_seqLen)
            layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32).npu()
            q_max_seq = np.max(q_seqlen)
            kv_max_seq = np.max(kv_seqLen)
            q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embeddim)))
            tor = np.float32(1.0 / math.sqrt(1.0 * embeddim))
            q = q.to(data_type).npu()
            k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(layer_id[0] + 1, batch, max_seq, kv_head * embeddim))).to(data_type).npu()
            v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(layer_id[0] + 1, batch, max_seq, kv_head * embeddim))).to(data_type).npu()
            attention_mask,post_mask_coff = SelfAttentionOperation.gen_mask(batch, heads, max_seq,data_type,mask_type,True)
            attention_mask[0] = attention_mask[1]
            SelfAttentionOperation.q_scale = 1
            SelfAttentionOperation.qk_scale = tor
            SelfAttentionOperation.batch = batch
            SelfAttentionOperation.heads = heads
            SelfAttentionOperation.embeddim = embeddim
            SelfAttentionOperation.max_seq = max_seq
            SelfAttentionOperation.clamp_min = clamp_min
            SelfAttentionOperation.clamp_max = clamp_max
            SelfAttentionOperation.in_tensors = [q,k,v,attention_mask.to(data_type).npu(),torch.tensor(kv_seqLen).to(torch.int32).npu(),torch.tensor(q_seqlen).to(torch.int32).npu(),layer_id]
            return SelfAttentionOperation.in_tensors[i]

        if json_data["calcType"] == 3:
            if i == 0:
                qhead = json_data["headNum"]
                kv_head = json_data["kvHeadNum"] if "kvHeadNum" in json_data else json_data["headNum"]
                
                # golden mask type
                golden_mask_type = 0
                if "maskType" in json_data:
                    seqlen_index = 3
                    is_mask = False
                    if json_data["maskType"] != 0:
                        seqlen_index = 4
                        is_mask = True
                    if json_data["maskType"] == 2: #2: alibi mask
                        if len(shapes[3]) == 3: # index3: mask, 3: headnum, max_s, max_s
                            golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_ALIBI_NO_BATCH
                        elif len(shapes[3]) == 4: # index3: mask, 4: batch, headnum, max_s, max_s
                            golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_ALIBI_WITH_BATCH
                else:
                    seqlen_index = 3
                    is_mask = False
                batch = __class__.get_shape(shapes, [seqlen_index, 0])    # shapes[3/4][0]

                headDims = 0
                if len(shapes[0]) == 2: # index 0: q, 2: bs, nd
                    headDims = __class__.get_shape(shapes, [0, 1]) // qhead
                elif len(shapes[0]) == 3: # 3: bs, n, d
                    headDims = __class__.get_shape(shapes, [0, 2]) # shapes[0][2]

                seqlen = __class__.get_shape(shapes, [0, 0]) // batch # shapes[0][0] / batch
                
                # index 2: v, 3: bs, n, d
                #   len(shapes[2]) == 3
                embeddim_v = __class__.get_shape(shapes, [2, -1])
                if len(shapes[2]) == 2: # 2: bs, nd, nd合轴
                    embeddim_v //= kv_head

                logging.debug(f"seqlen_index {seqlen_index} batch {batch}, seqlen {seqlen}, qhead {qhead}, headDims {headDims}, kv_head {kv_head}, q_seqlen {q_seqlen}")
                logging.debug(f"embeddim_v = {embeddim_v}")
                data = SelfAttentionOperation.calc_expect_func(batch=batch, seqlen=seqlen, heads=qhead, embed=headDims, group_num=kv_head, is_mask=is_mask,
                                                               datatype=torch_data_type, mask_type=golden_mask_type, op_param=op_params, q_seqlen=q_seqlen,
                                                               shapes=shapes, embed_v=embeddim_v)

                param_seqlen = data[seqlen_index].tolist()
                in_tensors = data
                SelfAttentionOperation.in_tensors_encoder = [tensor.npu() for tensor in in_tensors]
                for i, tensor in enumerate(in_tensors):
                    logging.debug(f"customize, i {i} dtype {tensor.dtype} shape {tensor.shape}")
                return SelfAttentionOperation.in_tensors_encoder[0]
            else:
                return SelfAttentionOperation.in_tensors_encoder[i]
        elif json_data["clampType"] == 1:
            if i != 0:
                return SelfAttentionOperation.in_tensors_clamp[i]
            min_seqlen = 1
            max_seqlen = 5
            SelfAttentionOperation.batch = 3
            SelfAttentionOperation.layer = 4
            seqlen = torch.randint(min_seqlen, max_seqlen, (SelfAttentionOperation.batch,), dtype=torch.int32).npu()
            min_token_offset_start = 0
            max_token_offset_start = 5
            token_offset_start = torch.randint(min_token_offset_start, max_token_offset_start, (SelfAttentionOperation.batch,), dtype=torch.int32).npu()
            token_offset = token_offset_start + seqlen
            total_seqlen = max_token_offset_start + max_seqlen
            ntokens = int(seqlen.sum())
            SelfAttentionOperation.head_num = json_data["headNum"]
            SelfAttentionOperation.head_size = 8
            hidden_size = SelfAttentionOperation.head_num * SelfAttentionOperation.head_size
            mixed_q = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
            mixed_k = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
            mixed_v = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
            cache_k = torch.rand(SelfAttentionOperation.layer, SelfAttentionOperation.batch, total_seqlen, hidden_size, dtype=torch.float16).npu()
            cache_v = torch.rand(SelfAttentionOperation.layer, SelfAttentionOperation.batch, total_seqlen, hidden_size, dtype=torch.float16).npu()
            attention_mask = torch.zeros(SelfAttentionOperation.batch, total_seqlen, total_seqlen, dtype=torch.float16).npu()
            layerid = torch.randint(SelfAttentionOperation.layer, (1,), dtype=torch.int32).npu()

            SelfAttentionOperation.q_scale = json_data["qScale"]
            SelfAttentionOperation.qk_scale = json_data["qkScale"]
            SelfAttentionOperation.clampMin = json_data["clampMin"]
            SelfAttentionOperation.clampMax = json_data["clampMax"]
            SelfAttentionOperation.isClamp = 1
            SelfAttentionOperation.param_seqlen = seqlen.tolist()
            SelfAttentionOperation.param_token_offset = token_offset.tolist()

            SelfAttentionOperation.in_tensors_clamp = mixed_q, mixed_k, mixed_v, cache_k, cache_v, attention_mask, token_offset, seqlen, layerid

            return mixed_q
        else:
            if i != 0:
                return SelfAttentionOperation.in_tensors[i]
            min_seqlen = 1
            max_seqlen = 5
            SelfAttentionOperation.batch = 3
            SelfAttentionOperation.layer = 4
            seqlen = torch.randint(min_seqlen, max_seqlen, (SelfAttentionOperation.batch,), dtype=torch.int32).npu()
            min_token_offset_start = 0
            max_token_offset_start = 5
            token_offset_start = torch.randint(min_token_offset_start, max_token_offset_start, (SelfAttentionOperation.batch,), dtype=torch.int32).npu()
            token_offset = token_offset_start + seqlen
            total_seqlen = max_token_offset_start + max_seqlen
            ntokens = int(seqlen.sum())
            SelfAttentionOperation.head_num = json_data["headNum"]
            SelfAttentionOperation.head_size = 8
            hidden_size = SelfAttentionOperation.head_num * SelfAttentionOperation.head_size
            mixed_q = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
            mixed_k = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
            mixed_v = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
            cache_k = torch.rand(SelfAttentionOperation.layer, SelfAttentionOperation.batch, total_seqlen, hidden_size, dtype=torch.float16).npu()
            cache_v = torch.rand(SelfAttentionOperation.layer, SelfAttentionOperation.batch, total_seqlen, hidden_size, dtype=torch.float16).npu()
            attention_mask = torch.zeros(SelfAttentionOperation.batch, total_seqlen, total_seqlen, dtype=torch.float16).npu()
            layerid = torch.randint(SelfAttentionOperation.layer, (1,), dtype=torch.int32).npu()

            SelfAttentionOperation.q_scale = json_data["qScale"]
            SelfAttentionOperation.qk_scale = json_data["qkScale"]

            SelfAttentionOperation.param_seqlen = seqlen.tolist()
            SelfAttentionOperation.param_token_offset = token_offset.tolist()

            SelfAttentionOperation.in_tensors = mixed_q, mixed_k, mixed_v, cache_k, cache_v, attention_mask, token_offset, seqlen, layerid
            return mixed_q

    @staticmethod
    def zero(shape, datatype, format, data_gen_ranges, op_params):
        try:
            json_data = json.loads(op_params)
            supported_mask_sofar950 = set([__class__.ATBMaskType.MASK_TYPE_NORM, __class__.ATBMaskType.MASK_TYPE_ALIBI, __class__.ATBMaskType.MASK_TYPE_NORM_COMPRESS])
            if get_soc_version() == "Ascend950":
                out_index = 4
                if "maskType" in json_data and json_data["maskType"] in supported_mask_sofar950: # 1: norm mask 2: alibi mask, 3: norm compress mask
                    out_index += 1
                if "pseShiftFlag" in json_data and json_data["pseShiftFlag"]:
                    out_index += 1
                shape = SelfAttentionOperation.in_tensors_encoder[out_index].shape
                logging.info(f"zero index {out_index} shape {shape}")
                data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
                return torch_npu.npu_format_cast(data, format_dict[format])
            elif json_data["calcType"] == 3:
                out_index = 4 # 4: q, k, v, seqlen, out
                if "maskType" in json_data and json_data["maskType"] in supported_mask_sofar950:
                    out_index += 1 # 1: mask
                shape = SelfAttentionOperation.in_tensors_encoder[out_index].shape
                data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
                return torch_npu.npu_format_cast(data, format_dict[format])
            elif json_data["clampType"] == 1:
                shape = SelfAttentionOperation.in_tensors_clamp[0].shape
                data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
                return torch_npu.npu_format_cast(data, format_dict[format])
            else:
                shape = SelfAttentionOperation.in_tensors[0].shape
                data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
                return torch_npu.npu_format_cast(data, format_dict[format])
        except:
            data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
            return torch_npu.npu_format_cast(data, format_dict[format])


    @staticmethod
    def get_asdops_param(q, k, v, mask, seq_len, op_params):
        json_data = json_data = json.loads(op_params)
        asdops_param = {}
        asdops_param["head_num"] = json_data["headNum"]
        asdops_param["is_decoder"] = False
        asdops_param["embeddim"] = int(q.shape[1] / json_data["headNum"])
        kvHeadNum = json_data["kvHeadNum"] if json_data["kvHeadNum"] != 0 else json_data["headNum"]
        asdops_param["kv_head"] = kvHeadNum
        asdops_param["is_mask"] = (json_data["maskType"] != 0)
        asdops_param["qk_scale"] = json_data["qkScale"]
        asdops_param["post_mask_coff"] = -3e38
        if "kernelType" in json_data and json_data["kernelType"] == 1:
            asdops_param["post_mask_coff"] = 1
        if "pseShiftFlag" in json_data:
            asdops_param["pseShiftFlag"] = json_data["pseShiftFlag"]
            asdops_param["is_alibi"] = True
            asdops_param["is_mask"] = False
        
        asdops_param["data_type"] = q.dtype
        asdops_param["q_ntokens"] = q.shape[0]
        asdops_param["kv_ntokens"] = k.shape[0]
        
        # len(v.shape) == 3: bs, n, d
        v_embeddim = v.shape[-1]
        
        if len(v.shape) == 2: # shape dim_size 2: bs, nd
            v_embeddim //= kvHeadNum
        asdops_param["v_embeddim"] = v_embeddim
        asdops_param["q_seqlen"] = seq_len.tolist()
        asdops_param["maskType"] = json_data["maskType"]

        golden_mask_type = 0
        asdops_param["is_alibi"] = False
        if "maskType" in json_data and json_data["maskType"] == 2:
            if json_data["maskType"] == 2: # 2: alibi mask
                if len(mask.shape) == 3: # 3: headnum, max_s, max_s
                    golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_ALIBI_NO_BATCH
                elif len(mask.shape) == 4: # 4: batch, headnum, max_s, max_s
                    golden_mask_type = __class__.GoldenMaskType.MASK_TYPE_ALIBI_WITH_BATCH
                asdops_param["is_alibi"] = True
                if get_soc_version() == "Ascend950":
                    asdops_param["is_mask"] = False
                    asdops_param["pseShiftFlag"] = True
        asdops_param["golden_mask_type"] = golden_mask_type

        MASK_TYPE_NO_MASK = 0
        MASK_TYPE_NO_HEAD = 1
        MASK_TYPE_NO_BATCH = 2
        MASK_TYPE_ALIBI_WITH_BATCH = 3
        MASK_TYPE_ALIBI_NO_BATCH = 4
        MASK_TYPE_NO_HEAD_DECODER = 5

        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH : ((lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH : ((lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD : ((lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER : ((lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH : ((lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_NO_MASK : ((lambda mask, idx, q_s, kv_s: 0))
        }
        asdops_mask_type = 0
        asdops_param["mask_info"] = mask_type_dict[asdops_mask_type]
        return asdops_param

    @staticmethod
    def group_mm_torch_encoder(heads, group_num, A, B):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32), B[i:(i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    @staticmethod
    def calc_golden_encoder_950(q, k, v, pseshift_in, mask_in, seq_len, asdops_param):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = len(seq_len)
        logging.info(f"asdops_param {asdops_param}")
        
        head_num = asdops_param["head_num"]
        is_decoder = asdops_param["is_decoder"]
        embed = asdops_param["embeddim"]
        embed_v = asdops_param["v_embeddim"]
        kv_head = asdops_param["kv_head"]
        is_mask = asdops_param["is_mask"]

        qk_scale = asdops_param["qk_scale"]
        post_mask_coff = asdops_param["post_mask_coff"]
        mask_info = asdops_param["mask_info"]
        data_type = asdops_param["data_type"]
        q_ntokens = asdops_param["q_ntokens"]
        kv_ntokens = asdops_param["kv_ntokens"]
        q_seqlen = asdops_param["q_seqlen"]
        is_alibi = asdops_param["is_alibi"]
        if "pseShiftFlag" in asdops_param:
            pseShiftFlag = asdops_param["pseShiftFlag"]
        else:
            pseShiftFlag = False
        if is_alibi:
            pseShiftFlag = True
        kv_seqlen = q_seqlen

        logging.debug(f"asdops params: {asdops_param}")
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        pseshift = None
        if pseShiftFlag:
            pseshift = pseshift_in
        mask = None
        if is_mask:
            mask = mask_in
            if len(mask.shape) == 2:
                dim0, dim1 = mask.shape
                mask = mask.view(1, dim0, dim1)
        s = None
        _p = None
        out = None

        max_seq_len = max(q_seqlen)
        
        
        logging.info(f"calc_golden_encoder_950 {is_mask}")

        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, head_num, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice = torch.permute(k_slice, (1, 0, 2))
            k_slice_t =torch.permute(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embed_v)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            score = SelfAttentionOperation.group_mm_torch_encoder(head_num, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.reshape([-1, ])), 0)

            tor = qk_scale
            score = score * tor
            if is_alibi:
                score = score + __class__.mask_info[1](pseshift, idx, q_s, kv_s) * __class__.post_mask_coff
            if is_mask:
                if (asdops_param["maskType"] == 1 or asdops_param["maskType"] == 3) and q_s > mask.shape[1]:
                    # 压缩norm mask
                    no_compress_mask = np.ones(shape=(1, max_seq_len, max_seq_len)).astype(np.float16)  # 使用当前最大seqlen生成mask
                    no_compress_mask = np.triu(no_compress_mask, 1)
                    no_compress_mask *= -10000.0
                    score = score + no_compress_mask[:, :q_s, :kv_s]
                else:
                    mask_pb = mask[:, :q_s, :kv_s]
                    mask_pb = np.repeat(mask_pb.cpu(), repeats=head_num, axis=0)
                    score[mask_pb.to(bool)] = -1.7 * 10 ** 38
            score_max, _ = torch.max(score, axis=-1)
            score = score - score_max.view((head_num, q_s, 1))
            score_exp = torch.exp(score)

            score_sum = torch.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.view([-1, ])
            else:
                _p = torch.cat((_p, score_exp.view([-1, ])), 0)
            p = score_exp / score_sum.view((head_num, q_s, 1))
            out_sub = SelfAttentionOperation.group_mm_torch_encoder(head_num, kv_head, p, v_slice)

            out_sub = out_sub.view(head_num, q_s, embed_v)
            out_sub = torch.permute(out_sub, (1, 0, 2)).contiguous()
            # out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = torch.cat((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s
        
        # golden data
        # out = torch.from_numpy(out)
        out = out.view(q_ntokens, head_num, embed_v)
        return out.to(data_type)

    @staticmethod
    def calc_golden_encoder(q, k, v, mask_in, seq_len, asdops_param):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = len(seq_len)
        head_num = asdops_param["head_num"]
        is_decoder = asdops_param["is_decoder"]
        embed = asdops_param["embeddim"]
        embeddim_v = asdops_param["v_embeddim"]
        kv_head = asdops_param["kv_head"]
        is_mask = asdops_param["is_mask"]
        qk_scale = asdops_param["qk_scale"]
        post_mask_coff = asdops_param["post_mask_coff"]
        mask_info = asdops_param["mask_info"]
        data_type = asdops_param["data_type"]
        q_ntokens = asdops_param["q_ntokens"]
        kv_ntokens = asdops_param["kv_ntokens"]
        q_seqlen = asdops_param["q_seqlen"]
        is_alibi = asdops_param["is_alibi"]
        kv_seqlen = q_seqlen

        logging.debug(f"asdops params: {asdops_param}")
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        mask = None
        if is_mask:
            # mask = mask_in.numpy()
            mask = mask_in
            if len(mask.shape) == 2:
                dim0, dim1 = mask.shape
                mask = mask.view(1, dim0, dim1)
        s = None
        _p = None
        out = None

        max_seq_len = max(q_seqlen)

        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, head_num, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice = torch.permute(k_slice, (1, 0, 2))
            k_slice_t =torch.permute(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embeddim_v)
            v_slice = torch.permute(v_slice, (1, 0, 2))
            score = SelfAttentionOperation.group_mm_torch_encoder(head_num, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.reshape([-1, ])), 0)

            tor = qk_scale
            score = score * tor
            if is_mask:
                if (asdops_param["maskType"] == 1 or asdops_param["maskType"] == 3) and q_s > mask.shape[1]:
                    # 压缩norm mask
                    no_compress_mask = np.ones(shape=(1, max_seq_len, max_seq_len)).astype(np.float16)  # 使用当前最大seqlen生成mask
                    no_compress_mask = np.triu(no_compress_mask, 1)
                    no_compress_mask *= -10000.0
                    score = score + no_compress_mask[:, :q_s, :kv_s]
                elif is_alibi:
                    score = score + __class__.mask_info[1](mask, idx, q_s, kv_s) * __class__.post_mask_coff
                else:
                    score = score + mask[:, :q_s, :kv_s]
            score_max, _ = torch.max(score, axis=-1)
            score = score - score_max.view((head_num, q_s, 1))
            score_exp = torch.exp(score)

            score_sum = torch.sum(score_exp, axis=-1)
            if _p is None:
                _p = score_exp.view([-1, ])
            else:
                _p = torch.cat((_p, score_exp.view([-1, ])), 0)
            p = score_exp / score_sum.view((head_num, q_s, 1))
            out_sub = SelfAttentionOperation.group_mm_torch_encoder(head_num, kv_head, p, v_slice)

            out_sub = out_sub.view(head_num, q_s, embeddim_v)
            out_sub = torch.permute(out_sub, (1, 0, 2)).contiguous()
            # out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = torch.cat((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        # golden data
        out = out.view(q_ntokens, head_num, embeddim_v)
        return out.to(data_type).cpu()

    @staticmethod
    def softmax1(
        qk_result,
        is_first,
        gm
    ):
        sim = qk_result.cpu().numpy()
        
        # sim = qk_result.numpy()
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
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True) # ll
        return torch.from_numpy(sim_sub), row_sum, dm, gm #返回P~ 、P~的rowsum和rowmax的差值 都是局部的结果

    @staticmethod
    def qkMM1(
        query,
        key
    ):
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

    @staticmethod
    def calc_golden_encoder_new(heads,kv_head,in_tensors,batch,kv_seqLen,mask_in, asdops_param,op_params):
        q = in_tensors[0]
        k = in_tensors[1]
        v = in_tensors[2]
        qk_scale = asdops_param["qk_scale"]
        is_mask = asdops_param["is_mask"]
        is_alibi = asdops_param["is_alibi"]
        
        mask = None
        if is_mask:
            mask = in_tensors[3]

        scale = qk_scale
        if len(q.shape) == 2:
            dim0, dim1 = q.shape
            embeddim = dim1 // heads 
            q = q.contiguous().view(dim0, heads, embeddim)
        elif len(q.shape) == 3:
            embeddim = q.shape[2]
        if len(k.shape) == 2:
            dim0, dim1 = k.shape
            embeddim = dim1 // kv_head 
            k = k.contiguous().view(dim0,  kv_head, embeddim)
        embeddimv = v.shape[-1]
        if len(v.shape) == 2:
            dim0, dim1 = v.shape
            embeddimv = dim1 // kv_head
            v = v.contiguous().view(dim0,  kv_head, embeddimv)
        q_ntokens = q.shape[0]
        ref_output = torch.zeros(q_ntokens, heads, embeddimv)
        q_seqlen = kv_seqLen
        q_offset = 0
        k_offset = 0
        v_offset = 0
        mask_compress = 0
        if is_mask and (asdops_param["maskType"] == 1 or asdops_param["maskType"] == 3):
            mask_compress = 1
            no_compress_mask = None
            if q.dtype == torch.bfloat16:
                no_compress_mask = np.ones(shape=(heads, 128, 128)).astype(np.float32)
                no_compress_mask = np.triu(no_compress_mask, 1)
                no_compress_mask *= -3e38
            elif q.dtype == torch.float16:
                no_compress_mask = np.ones(shape=(heads, 128, 128)).astype(np.float16)
                no_compress_mask = np.triu(no_compress_mask, 1)
                no_compress_mask *= -10000.0
            no_compress_mask = torch.Tensor(no_compress_mask)
        context_size = 128
        for idx in range(batch):
            q_s = q_seqlen[idx]
            for q_start in range(0, q_s - 1, context_size):
                q_id = q_start // context_size
                sub_q_len = context_size
                if q_start + context_size > q_s:
                    sub_q_len = q_s - q_start
                context_len = kv_s = kv_seqLen[idx]
                query = q[q_offset : q_offset + sub_q_len,:,:]
                
                key = k[k_offset:k_offset + kv_s,:,:]
                value = v[v_offset:v_offset + kv_s,:,:]
                query = torch.permute(query, (1, 0, 2)) #转成head seqlen和headdim顺序
                key = torch.permute(key, (1, 2, 0)) #转成head headdim和seqlen顺序
                value = torch.permute(value, (1, 0, 2)) #同query
                group_num = heads // kv_head
                if group_num != 1:
                    key = key.repeat_interleave(group_num, dim=0)
                    value = value.repeat_interleave(group_num, dim=0)
                for kv_start in range(0, context_len - 1, context_size):
                    kv_id = kv_start // context_size
                    if kv_id > q_id:
                        break
                    sub_len = context_size
                    if kv_start + context_size > context_len:
                        sub_len = context_len - kv_start
                    sub_key = key[:, :, kv_start : kv_start + sub_len] # [14, 128, 128]
                    sub_value = value[:, kv_start : kv_start + sub_len, :]
                    if q.dtype == torch.bfloat16:
                        qk_result = SelfAttentionOperation.qkMM1(query.cpu(), sub_key.cpu()) 
                    else:
                        qk_result = SelfAttentionOperation.qkMM1(query.cpu().to(torch.float16), sub_key.cpu().to(torch.float16)) 
                    if q.dtype == torch.bfloat16:
                        qk_result = qk_result.to(torch.float32) * scale
                    else:
                        qk_result = qk_result.to(torch.float16) * scale
                    if is_mask:
                        if mask_compress:
                            if q_start // context_size == kv_start // context_size:
                                qk_result = qk_result.cpu() + no_compress_mask[:,:sub_q_len,:sub_len].cpu()
                        elif is_alibi:
                            qk_result = qk_result.cpu() + __class__.mask_info[1](mask, idx, q_s, kv_s).cpu() * __class__.post_mask_coff
                    if kv_start == 0:
                        gm = None
                    p_result, row_sum, dm, gm = SelfAttentionOperation.softmax1(qk_result, kv_start == 0, gm)
                    if kv_start == 0:
                        gm_high = None
                    if q.dtype == torch.bfloat16 or (q.dtype == torch.float16 and is_mask):
                        lo = torch.matmul(p_result.cpu().to(q.dtype).to(torch.float32), sub_value.cpu().to(torch.float32))
                        lo = lo.to(torch.float32)
                    else:
                        lo = torch.matmul(p_result.cpu().to(torch.float32), sub_value.cpu().to(torch.float32))
                        lo = lo.to(torch.float16)
                    lo = lo.numpy()
                    if kv_start == 0:
                        gl = row_sum
                        go = lo
                    else:  #更新
                        dm = np.exp(dm)
                        gl = gl * dm
                        gl = gl + row_sum

                        go = go * dm
                        go = go + lo
                go = go / gl
                go = np.transpose(go, (1, 0, 2))
                out = torch.from_numpy(go).to(q.dtype)
                out = out.reshape(sub_q_len, heads, embeddimv)
                ref_output[q_offset:q_offset + sub_q_len][:] = out
                q_offset += sub_q_len
            k_offset += kv_s
            v_offset += kv_s
            logging.info(f"ref_output dtype {ref_output.dtype}")
        return ref_output


    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        if 'kvcacheCfg' in json_data.keys() and json_data['kvcacheCfg'] == 1:
            q_offset = 0
            k_offset = 0
            v_offset = 0
            isdecoder = 1
            is_clamp = 0
            batch = SelfAttentionOperation.batch
            heads = SelfAttentionOperation.heads
            embed = SelfAttentionOperation.embeddim
            max_seq = SelfAttentionOperation.max_seq
            q_seqlen = in_tensors[5].tolist()
            kv_seqlen = in_tensors[4].tolist()
            kv_head = SelfAttentionOperation.heads
            mask = in_tensors[3]
            is_mask = True
            q = in_tensors[0]
            k = in_tensors[1]
            v = in_tensors[2]
            q_ntokens = sum(q_seqlen)
            kv_ntokens = sum(kv_seqlen)
            layer_id = in_tensors[6][0]
            is_multi_layer = True
            s = None
            _p = None
            out = None

            for idx in range(batch):
                q_s = q_seqlen[idx]
                kv_s = kv_seqlen[idx]
                q_slice = q[q_offset:q_offset + q_s][:]
                q_slice = q_slice.view(q_s, heads, embed)
                q_slice = torch.permute(q_slice, (1, 0, 2))
                k_slice = k[layer_id][idx][:kv_s][:]
                k_slice = k_slice.view(kv_s, kv_head, embed)
                k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
                v_slice = v[layer_id][idx][:kv_s][:]
                v_slice = v_slice.view(kv_s, kv_head, embed)
                v_slice = torch.permute(v_slice, (1, 0, 2))

                score = torch.from_numpy(SelfAttentionOperation.group_matmul(heads, kv_head, q_slice.numpy(), k_slice_t.numpy()))

                if s is None:
                    s = score.view([-1, ])
                else:
                    s = torch.cat((s, score.view([-1, ])), 0)

                scale = 1
                tor = np.float32(1.0 / math.sqrt(1.0 * SelfAttentionOperation.embeddim))
                if not is_multi_layer:
                    # 当前scale和tor保持一致，模型侧可能传入scale = np.float32(layer_id + 1)
                    scale = np.float32(layer_id + 1)
                score = score * tor

                if is_clamp == 1:
                    clamp_min_brc = np.ones((score.shape)) * SelfAttentionOperation.clamp_min
                    clamp_max_brc = np.ones((score.shape)) * SelfAttentionOperation.clamp_max
                    score = np.float16(np.maximum(score, clamp_min_brc))
                    score = torch.from_numpy(np.float16(np.minimum(score, clamp_max_brc)))
                if is_mask:
                    #score = score + self.mask_info[1](self.mask, idx, q_s, kv_s)
                    score = score + mask[idx, :q_s, :kv_s] * SelfAttentionOperation.post_mask_coff
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

                p = (score_exp / score_sum.reshape((heads, q_s, 1)))
                #p = torch.from_numpy(p).to(torch.bfloat16)
                o = torch.from_numpy(SelfAttentionOperation.group_matmul(heads, kv_head, p, v_slice.numpy()))
                o = o.view(heads, q_s, embed)
                o = torch.permute(o, (1, 0, 2)).contiguous()
                if out is None:
                    out = o
                else:
                    out = torch.cat((out, o), 0)

                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq

            # golden data
            out = out.view(q_ntokens, heads * embed)
            golden_out = out.to(q.dtype)
            return [golden_out]
        if "calcType" in json_data and json_data["calcType"] == 3:
            q = in_tensors[0]
            k = in_tensors[1]
            v = in_tensors[2]
            if len(q.shape) == 3:
                dim0, dim1, dim2 = q.shape
                q = q.contiguous().view(dim0, dim1*dim2)
            elif len(q.shape) == 4:
                dim0, dim1, dim2, dim3 = q.shape
                q = q.contiguous().view(dim0*dim1, dim2*dim3)
            if len(k.shape) == 3:
                dim0, dim1, dim2 = k.shape
                k = k.contiguous().view(dim0, dim1*dim2)
            elif len(k.shape) == 4:
                dim0, dim1, dim2, dim3 = k.shape
                k = k.contiguous().view(dim0*dim1, dim2, dim3)
            if len(v.shape) == 3:
                dim0, dim1, dim2 = v.shape
                v = v.contiguous().view(dim0, dim1*dim2)
            elif len(v.shape) == 4:
                dim0, dim1, dim2, dim3 = v.shape
                v = v.contiguous().view(dim0*dim1, dim2, dim3)
            mask = None
            seq_len = None
            kv_seqLen_tesnor =None
            pseShift = None
            if json_data["maskType"] != 0:
                mask = in_tensors[3]
                seq_len = in_tensors[4]
                batch = len(in_tensors[4])
                kv_seqLen_tesnor = in_tensors[4]
                if "pseShiftFlag" in json_data and json_data["pseShiftFlag"]:
                    pseShift = in_tensors[5]
                if json_data["maskType"] == 2: # 2: alibi mask
                    pseShift = in_tensors[3]
            else:
                seq_len = in_tensors[3]
                batch = len(in_tensors[3])
                kv_seqLen_tesnor = in_tensors[3]
                if "pseShiftFlag" in json_data and json_data["pseShiftFlag"]:
                    pseShift = in_tensors[4]
            kv_seqLen = kv_seqLen_tesnor.cpu().numpy()
            
            max_seq = np.max(kv_seqLen)
            heads = json_data['headNum']
            kv_head = json_data['kvHeadNum'] if 'kvHeadNum' in json_data else json_data['headNum']
            if mask is not None:
                if get_soc_version() != "Ascend910B" and get_soc_version() != "Ascend950":
                    mask = mask.contiguous().permute(0, 2, 1, 3)
                    dim0, dim1, dim2, dim3 = mask.shape
                    mask = mask.contiguous().view(dim0, dim1, dim2*dim3)
                    mask = mask[:, :, : dim1]
                    batch =len(seq_len)
                    if dim0 == 1:
                        mask = mask.contiguous().view(dim1, dim1)
                    elif dim0 != batch:
                        if dim1 > dim2 * dim3:
                            # alibi mask compress
                            step = dim2 * dim3
                            mask_padded = -10000 * torch.ones(dim0, dim1 * 2, dim1)
                            for i in range(dim1 // step):
                                dim1_offset = i * step
                                dim2_offset = i * step
                                mask_padded[:, dim1_offset : dim1_offset + dim1, dim2_offset : dim2_offset + step ] = mask
                            mask = mask_padded[:, : dim1, : dim1]
            asdops_params = SelfAttentionOperation.get_asdops_param(q, k, v, mask, seq_len, op_params)
            if get_soc_version() == "Ascend950":
                return [SelfAttentionOperation.calc_golden_encoder_950(q, k, v, pseShift, mask, seq_len, asdops_params).cpu()]
            out = SelfAttentionOperation.calc_golden_encoder(q, k, v, mask, seq_len, asdops_params)
            ref_output = SelfAttentionOperation.calc_golden_encoder_new(heads, kv_head, in_tensors,
                            batch, kv_seqLen, mask, asdops_params, op_params)
            if len(in_tensors[0].shape) == 2: # q shape: bs, nd -> out shape: bs, nd
                dim0, dim1, dim2 = out.shape
                out = out.view(dim0, dim1 * dim2)
                ref_output = ref_output.view(dim0, dim1 * dim2)
            return [out, ref_output]
        elif "clampType" in json_data and json_data["clampType"] == 1:
            layerid = int(in_tensors[8][0])
            mixed_q = in_tensors[0]
            mixed_k = in_tensors[1]
            mixed_v = in_tensors[2]
            if len(mixed_q.shape) == 4:
                dim0 = mixed_q.shape[0]
                dim1 = mixed_q.shape[1]
                dim2 = mixed_q.shape[2]
                dim3 = mixed_q.shape[3]
                mixed_q = mixed_q.contiguous().view(dim0*dim1, dim2*dim3)
            if len(mixed_k.shape) == 4:
                dim0 = mixed_k.shape[0]
                dim1 = mixed_k.shape[1]
                dim2 = mixed_k.shape[2]
                dim3 = mixed_k.shape[3]
                mixed_k = mixed_k.contiguous().view(dim0*dim1, dim2*dim3)
            if len(mixed_v.shape) == 4:
                dim0 = mixed_v.shape[0]
                dim1 = mixed_v.shape[1]
                dim2 = mixed_v.shape[2]
                dim3 = mixed_v.shape[3]
                mixed_v = mixed_v.contiguous().view(dim0*dim1, dim2*dim3)
            cache_k = in_tensors[3]
            cache_v = in_tensors[4]
            attention_mask = in_tensors[5]
            soc_version = get_soc_version()
            if soc_version != 'Ascend910B':
                cache_k = cache_k.permute(0, 1, 3, 2, 4)
                dim0 = cache_k.shape[0]
                dim1 = cache_k.shape[1]
                dim2 = cache_k.shape[2]
                dim3 = cache_k.shape[3]
                dim4 = cache_k.shape[4]
                cache_k = cache_k.contiguous().view(dim0, dim1, dim2, dim3*dim4)
                k_hidden_size = mixed_k.shape[-1]
                cache_k = cache_k[:, :, : , : k_hidden_size]

                cache_v = cache_v.permute(0, 1, 3, 2, 4)
                dim0 = cache_v.shape[0]
                dim1 = cache_v.shape[1]
                dim2 = cache_v.shape[2]
                dim3 = cache_v.shape[3]
                dim4 = cache_v.shape[4]
                cache_v = cache_v.contiguous().view(dim0, dim1, dim2, dim3*dim4)
                v_hidden_size = mixed_v.shape[-1]
                cache_v = cache_v[:, :, : , : v_hidden_size]

                # if torch_npu.get_npu_format(attention_mask) == 29:
                attention_mask = attention_mask.contiguous().permute(0, 2, 1, 3)
                dim0, dim1, dim2, dim3 = attention_mask.shape
                attention_mask = attention_mask.contiguous().view(dim0, dim1, dim2*dim3)
                attention_mask = attention_mask[:, :, : dim1]
                batch =len(in_tensors[7])
                if dim0 == 1:
                    attention_mask = attention_mask.contiguous().view(dim1, dim1)
                elif dim0 != batch:
                    attention_mask = attention_mask.contiguous().view(batch, dim0 // batch, dim1, dim1)


            param_token_offset = in_tensors[6].tolist()
            param_seqlen = in_tensors[7].tolist()
            offset = 0
            context_list = []
            batch = cache_k.shape[1]
            q_scale = json_data["qScale"]
            head_num = json_data["headNum"]
            head_size = int(mixed_q.shape[1] / head_num)
            clampMin = json_data["clampMin"]
            clampMax = json_data["clampMax"]
            qk_scale = json_data["qkScale"]
            for i, _ in enumerate(range(batch)):
                cur_seqlen = param_seqlen[i]
                cur_token_offset = param_token_offset[i]
                cur_token_offset_start = cur_token_offset - cur_seqlen
                next_offset = offset + cur_seqlen
                cur_q = mixed_q[offset:next_offset]
                cur_k = mixed_k[offset:next_offset]
                cur_v = mixed_v[offset:next_offset]
                if cur_token_offset_start > 0:
                    past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                    past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                    cur_k = torch.concat([past_k, cur_k], dim=0)
                    cur_v = torch.concat([past_v, cur_v], dim=0)
                cur_q = (cur_q * q_scale).view(cur_seqlen, head_num, head_size).transpose(0, 1)
                cur_k = cur_k.view(cur_token_offset, head_num, head_size).permute(1, 2, 0)
                cur_qk = torch.bmm(cur_q.float(), cur_k.float()) # [head_num, seqlen, token_offset]
                cur_qk = torch.clamp(cur_qk.npu(), clampMin, clampMax).cpu()
                if attention_mask.ndim == 3: # masked_fill
                    cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
                else:
                    cur_qk = cur_qk + attention_mask[:cur_seqlen, :cur_token_offset]
                cur_qk = cur_qk * qk_scale
                cur_qk = torch.nn.functional.softmax(cur_qk.type(torch.float32), dim=-1).type(torch.float16)

                cur_v = cur_v.view(cur_token_offset, head_num, head_size).transpose(0, 1)
                cur_context = torch.bmm(cur_qk.float(), cur_v.float()).transpose(0, 1).contiguous().view(cur_seqlen, head_num * head_size)
                context_list.append(cur_context)

                offset = next_offset

            context = torch.concat(context_list, dim=0)
            return [context]
        else:

            layerid = int(in_tensors[8][0])
            mixed_q = in_tensors[0]
            mixed_k = in_tensors[1]
            mixed_v = in_tensors[2]
            if len(mixed_q.shape) == 4:
                dim0 = mixed_q.shape[0]
                dim1 = mixed_q.shape[1]
                dim2 = mixed_q.shape[2]
                dim3 = mixed_q.shape[3]
                mixed_q = mixed_q.contiguous().view(dim0*dim1, dim2*dim3)
            if len(mixed_k.shape) == 4:
                dim0 = mixed_k.shape[0]
                dim1 = mixed_k.shape[1]
                dim2 = mixed_k.shape[2]
                dim3 = mixed_k.shape[3]
                mixed_k = mixed_k.contiguous().view(dim0*dim1, dim2*dim3)
            if len(mixed_v.shape) == 4:
                dim0 = mixed_v.shape[0]
                dim1 = mixed_v.shape[1]
                dim2 = mixed_v.shape[2]
                dim3 = mixed_v.shape[3]
                mixed_v = mixed_v.contiguous().view(dim0*dim1, dim2*dim3)
            cache_k = in_tensors[3]
            cache_v = in_tensors[4]
            attention_mask = in_tensors[5]
            soc_version = get_soc_version()
            if soc_version != 'Ascend910B':
                cache_k = cache_k.permute(0, 1, 3, 2, 4)
                dim0 = cache_k.shape[0]
                dim1 = cache_k.shape[1]
                dim2 = cache_k.shape[2]
                dim3 = cache_k.shape[3]
                dim4 = cache_k.shape[4]
                cache_k = cache_k.contiguous().view(dim0, dim1, dim2, dim3*dim4)
                k_hidden_size = mixed_k.shape[-1]
                cache_k = cache_k[:, :, : , : k_hidden_size]

                cache_v = cache_v.permute(0, 1, 3, 2, 4)
                dim0 = cache_v.shape[0]
                dim1 = cache_v.shape[1]
                dim2 = cache_v.shape[2]
                dim3 = cache_v.shape[3]
                dim4 = cache_v.shape[4]
                cache_v = cache_v.contiguous().view(dim0, dim1, dim2, dim3*dim4)
                v_hidden_size = mixed_v.shape[-1]
                cache_v = cache_v[:, :, : , : v_hidden_size]

                # if torch_npu.get_npu_format(attention_mask) == 29:
                attention_mask = attention_mask.contiguous().permute(0, 2, 1, 3)
                dim0, dim1, dim2, dim3 = attention_mask.shape
                attention_mask = attention_mask.contiguous().view(dim0, dim1, dim2*dim3)
                attention_mask = attention_mask[:, :, : dim1]
                batch =len(in_tensors[7])
                if dim0 == 1:
                    attention_mask = attention_mask.contiguous().view(dim1, dim1)
                elif dim0 != batch:
                    attention_mask = attention_mask.contiguous().view(batch, dim0 // batch, dim1, dim1)

            param_token_offset = in_tensors[6].tolist()
            param_seqlen = in_tensors[7].tolist()
            offset = 0
            context_list = []
            batch = cache_k.shape[1]
            q_scale = json_data["qScale"]
            head_num = json_data["headNum"]
            head_size = int(mixed_q.shape[1] / head_num)
            qk_scale = json_data["qkScale"]
            for i, _ in enumerate(range(batch)):
                cur_seqlen = param_seqlen[i]
                cur_token_offset = param_token_offset[i]
                cur_token_offset_start = cur_token_offset - cur_seqlen
                next_offset = offset + cur_seqlen
                cur_q = mixed_q[offset:next_offset]
                cur_k = mixed_k[offset:next_offset]
                cur_v = mixed_v[offset:next_offset]
                if cur_token_offset_start > 0:
                    past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                    past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                    cur_k = torch.concat([past_k, cur_k], dim=0)
                    cur_v = torch.concat([past_v, cur_v], dim=0)
                cur_q = (cur_q * q_scale).view(cur_seqlen, head_num, head_size).transpose(0, 1)
                cur_k = cur_k.view(cur_token_offset, head_num, head_size).permute(1, 2, 0)
                cur_qk = torch.bmm(cur_q.float(), cur_k.float()) # [head_num, seqlen, token_offset]
                if attention_mask.ndim == 3: # masked_fill
                    cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
                else:
                    cur_qk = cur_qk + attention_mask[:cur_seqlen, :cur_token_offset]
                cur_qk = cur_qk * qk_scale
                cur_qk = torch.nn.functional.softmax(cur_qk.type(torch.float32), dim=-1).type(torch.float16)

                cur_v = cur_v.view(cur_token_offset, head_num, head_size).transpose(0, 1)


                cur_context = torch.bmm(cur_qk.float(), cur_v.float()).transpose(0, 1).contiguous().view(cur_seqlen, head_num * head_size)
                context_list.append(cur_context)

                offset = next_offset
            context = torch.concat(context_list, dim=0)
            logging.debug(context.shape)
            return [context]

    @staticmethod
    def truth_value(in_tensors, op_params):
        logging.info("begin truth_value")
        ret = []
        json_data = json.loads(op_params)
        if "calcType" in json_data and json_data["calcType"] == 3:
            q = in_tensors[0]
            k = in_tensors[1]
            v = in_tensors[2]
            if len(q.shape) == 3:
                dim0, dim1, dim2 = q.shape
                q = q.contiguous().view(dim0, dim1*dim2)
            elif len(q.shape) == 4:
                dim0, dim1, dim2, dim3 = q.shape
                q = q.contiguous().view(dim0*dim1, dim2*dim3)
            if len(k.shape) == 3:
                dim0, dim1, dim2 = k.shape
                k = k.contiguous().view(dim0, dim1*dim2)
            elif len(k.shape) == 4:
                dim0, dim1, dim2, dim3 = k.shape
                k = k.contiguous().view(dim0*dim1, dim2, dim3)
            if len(v.shape) == 3:
                dim0, dim1, dim2 = v.shape
                v = v.contiguous().view(dim0, dim1*dim2)
            elif len(v.shape) == 4:
                dim0, dim1, dim2, dim3 = v.shape
                v = v.contiguous().view(dim0*dim1, dim2, dim3)
            mask = None
            seq_len = None
            kv_seqLen_tesnor =None
            pseShift = None
            if json_data["maskType"] != 0:
                mask = in_tensors[3]
                seq_len = in_tensors[4]
                batch = len(in_tensors[4])
                kv_seqLen_tesnor = in_tensors[4]
                if "pseShiftFlag" in json_data and json_data["pseShiftFlag"]:
                    pseShift = in_tensors[5]
            else:
                seq_len = in_tensors[3]
                batch = len(in_tensors[3])
                kv_seqLen_tesnor = in_tensors[3]
                if "pseShiftFlag" in json_data and json_data["pseShiftFlag"]:
                    pseShift = in_tensors[4]
            # kv_seqLen = kv_seqLen_tesnor.numpy()
            kv_seqLen = kv_seqLen_tesnor.cpu().numpy()
            
            max_seq = np.max(kv_seqLen)
            heads = json_data['headNum']
            kv_head = json_data['kvHeadNum'] if 'kvHeadNum' in json_data else json_data['headNum']
            if mask is not None:
                if get_soc_version() != "Ascend910B" and get_soc_version() != "Ascend950":
                    mask = mask.contiguous().permute(0, 2, 1, 3)
                    dim0, dim1, dim2, dim3 = mask.shape
                    mask = mask.contiguous().view(dim0, dim1, dim2*dim3)
                    mask = mask[:, :, : dim1]
                    batch =len(seq_len)
                    if dim0 == 1:
                        mask = mask.contiguous().view(dim1, dim1)
                    elif dim0 != batch:
                        if dim1 > dim2 * dim3:
                            # alibi mask compress
                            step = dim2 * dim3
                            mask_padded = -10000 * torch.ones(dim0, dim1 * 2, dim1)
                            for i in range(dim1 // step):
                                dim1_offset = i * step
                                dim2_offset = i * step
                                mask_padded[:, dim1_offset : dim1_offset + dim1, dim2_offset : dim2_offset + step ] = mask
                            mask = mask_padded[:, : dim1, : dim1]
            asdops_params = SelfAttentionOperation.get_asdops_param(q, k, v, mask, seq_len, op_params)
            if get_soc_version() == "Ascend950":
                return [SelfAttentionOperation.calc_golden_encoder_950(q, k, v, pseShift, mask, seq_len, asdops_params).cpu()]
            # out = SelfAttentionOperation.calc_golden_encoder(q, k, v, mask, seq_len, asdops_params)
            ref_output = SelfAttentionOperation.calc_golden_encoder_new(heads, kv_head, in_tensors,
                            batch, kv_seqLen, mask, asdops_params, op_params)
            if len(in_tensors[0].shape) == 2: # q shape: bs, nd -> out shape: bs, nd
                dim0, dim1, dim2 = ref_output.shape
                # out = out.view(dim0, dim1 * dim2)
                ref_output = ref_output.view(dim0, dim1 * dim2)
            ret = [ref_output]
        logging.info("finish truth_value")
        return ret

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION


class SelfAttentionOperationDumpTensor(DataGen):
    golden_generator = None
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i == 0:
            op_params = json.loads(op_params)
            if not __class__.golden_generator:
                __class__.golden_generator = SelfAttentionGolden()
            __class__.golden_generator.is_910b == get_soc_version() == "Ascend910B"
        return __class__.golden_generator.tensor_customize(shapes[i], dtype_dict[datatype], format, data_gen_ranges)

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        json_data = json.loads(op_params)
        run_param = None
        new_item = {"maskType": int(json_data["maskType"])}
        if json_data["calcType"] == 3:
            new_item = new_item | {"calcType": 3}
            seqlen_id  = 4
            if json_data["maskType"] == 0:
                seqlen_id -= 1
            if ('mlaVHeadSize' in json_data.keys() and json_data['mlaVHeadSize'] > 0):
                seqlen_id -= 1
            run_param = {"seqLen": input_tensor_list[seqlen_id].tolist()}


        else:
            token_offset_id = 6
            if 'kvcacheCfg' in json_data.keys() and json_data['kvcacheCfg'] == 1:
                new_item = new_item | {"kvcacheCfg": 1}
                token_offset_id -= 2
            if json_data["maskType"] == 0:
                token_offset_id -= 1
            if 'batchRunStatusEnable' in json_data.keys() and json_data['batchRunStatusEnable']:
                run_param = {"tokenOffset": input_tensor_list[token_offset_id].tolist(),
                            "seqLen": input_tensor_list[token_offset_id+1].tolist(),
                            "batchStatus": input_tensor_list[token_offset_id+3].tolist()}
            else:
                run_param = {"tokenOffset": input_tensor_list[token_offset_id].tolist(),
                            "seqLen": input_tensor_list[token_offset_id+1].tolist()}
        run_param = new_item | run_param
        operation.set_varaintpack_param(json.dumps(run_param))

    @staticmethod
    def zero(shape, datatype, format, data_gen_ranges, op_params):
        data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
        return torch_npu.npu_format_cast(data, format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        golden_generator = SelfAttentionGolden()
        golden_generator.data_type = in_tensors[0].dtype
        golden_generator.is_910b == get_soc_version() == "Ascend910B"
        # 先从tensor shape 和 param里面获取需要的参数
        json_data = json.loads(op_params)
        golden_generator.load_from_op_params(json_data)
        golden_generator.prepare_in_tensors(in_tensors)
        # golden计算，包括kvcache
        return golden_generator.gen_out_tensor()

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION


class StridedBatchMatmulOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        A = in_tensors[0].flatten()
        B = in_tensors[1].flatten()

        batchStartA = 0
        batchStartB = 0
        batchStartC = 0

        C = torch.empty(sum([json_data["m"][i] * json_data["n"][i] for i in range(json_data["batch"])]) * json_data["headNum"], dtype=torch.float16, device=A.device)

        for i in range(json_data["batch"]):
            for j in range(json_data["headNum"]):
                listA = []
                listB = []
                rowA = json_data["m"][i] if not json_data["transA"] else json_data["k"][i]
                colA = json_data["k"][i] if not json_data["transA"] else json_data["m"][i]

                for t in range(rowA):
                    startA = json_data["lda"][i] * t + json_data["strideA"][i] * j + batchStartA
                    endA = startA + colA
                    listA.append(A[startA:endA])
                rowB = json_data["k"][i] if not json_data["transB"] else json_data["n"][i]
                colB = json_data["n"][i] if not json_data["transB"] else json_data["k"][i]
                for t in range(rowB):
                    startB = json_data["ldb"][i] * t +  json_data["strideB"][i] * j + batchStartB
                    endB = startB + colB
                    listB.append(B[startB:endB])

                matA = torch.stack(listA)
                matB = torch.stack(listB)
                matA = torch.transpose(matA, 0, 1) if json_data["transA"] else matA
                matB = torch.transpose(matB, 0, 1) if json_data["transB"] else matB
                matC = torch.matmul(matA.float(), matB.float()).half()
                for t in range(matC.shape[0]):
                    startC = json_data["ldc"][i] * t + json_data["strideC"][i] * j + batchStartC
                    endC = startC + matC.shape[1]
                    C[startC:endC] = matC[t, :]
            batchStartA += json_data["m"][i] * json_data["k"][i] * json_data["headNum"]
            batchStartB += json_data["n"][i] * json_data["k"][i] * json_data["headNum"]
            batchStartC += json_data["m"][i] * json_data["n"][i] * json_data["headNum"]
        return [C]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class GenAttentionMaskOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        out = []
        for i, s in enumerate(json_data['seqLen']):
            for j in range(json_data["headNum"]):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class RopeGradOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        cos_list = [in_tensors[2][:x, :] for x in json_data['qSeqLen']]
        sin_list = [in_tensors[3][:x, :] for x in json_data['qSeqLen']]
        cos = torch.cat(cos_list, dim=0)
        sin = torch.cat(sin_list, dim=0)
        sin1 = sin[:,:64]
        sin2 = sin[:,64:]
        rohqgsin = torch.cat((sin2, -sin1), dim=-1)
        q_grad = torch.zeros_like(in_tensors[0])
        bs = int(in_tensors[0].shape[1] / 128)
        for i in range(bs):
            q_grad[:, i * 128:(i + 1) * 128] = in_tensors[0][:, i * 128:(i + 1) * 128] * (cos + rohqgsin)

        k_grad = torch.zeros_like(in_tensors[1])
        for i in range(bs):
            k_grad[:,i * 128:(i + 1) * 128] = in_tensors[1][:, i * 128:(i + 1) * 128] *(cos + rohqgsin)
        return [q_grad, k_grad]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class SortOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        num = json_data['num']
        values, indices = torch.topk(in_tensors[0].npu(), k=num[0], largest=True)
        return [values.cpu(),indices.int().cpu()]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class UnpadWithHiddenStateOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        seq_len_list = json_data['qSeqLen']
        data_input = in_tensors[0]
        hidden_size_imm = data_input.shape[-1]

        golden_result = torch.empty(size=[sum(seq_len_list), hidden_size_imm], dtype=torch.float16)
        start = 0
        for i, sample_seq_len in enumerate(seq_len_list):
            golden_result[start:start + sample_seq_len] = data_input[i][:sample_seq_len]
            start = start + sample_seq_len

        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE


class PadWithHiddenStateOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        seq_len_list = json_data['qSeqLen']
        max_seq_len_imm = json_data['maxSeqLen']
        data_input = in_tensors[0]
        hidden_size_imm = data_input.shape[-1]

        golden_result = torch.empty(size=[len(seq_len_list), max_seq_len_imm, hidden_size_imm], dtype=torch.float16)
        start = 0
        for i, sample_seq_len in enumerate(seq_len_list):
            golden_result[i][:sample_seq_len] = data_input[start:start + sample_seq_len]
            golden_result[i][seq_len_list[i]:] = 0
            start = start + sample_seq_len

        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class FastSoftMaxOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        seq_len_list = json_data['qSeqLen']
        head_num_imm = json_data['headNum']
        data_input = in_tensors[0]
        golden_result = torch.empty_like(data_input)

        start = 0
        for seq_len in seq_len_list:
            end = start + head_num_imm * seq_len * seq_len
            sample_data_input = data_input[start:end].reshape(-1, seq_len)
            sample_golden = torch.softmax(sample_data_input.to(torch.float32), dim=-1).to(torch.float16)
            golden_result[start:end] = sample_golden.reshape(-1)
            start = end

        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class FastSoftMaxGradOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        seq_len_list = json_data['qSeqLen']
        head_num_imm = json_data['headNum']
        y_input = in_tensors[0]
        y_grad = in_tensors[1]
        golden_result = torch.empty_like(y_input)

        start = 0
        for seq_len in seq_len_list:
            end = start + head_num_imm * seq_len * seq_len
            sample_y_input = y_input[start:end].reshape(-1, seq_len).to(torch.float32)
            sample_y_grad = y_grad[start:end].reshape(-1, seq_len).to(torch.float32)
            sample_x_grad = torch.empty(size=(head_num_imm * seq_len, seq_len), dtype=torch.float16)
            for i in range(head_num_imm * seq_len):
                grad_matrix = torch.diag(sample_y_input[i]) - torch.ger(sample_y_input[i], sample_y_input[i])
                sample_x_grad[i] = torch.mv(grad_matrix, sample_y_grad[i]).to(torch.float16)
            golden_result[start:end] = sample_x_grad.reshape(-1)
            start = end

        return [golden_result]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT


class GatingOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        json_data = json.loads(op_params)
        topk_expert_num = json_data['topkExpertNum']
        cum_sum_num = json_data['cumSumNum']
        if cum_sum_num == 0:
            cum_sum_num = random.randint(64, 128)
        if i == 0:
            in_tensor_topk = torch.zeros(shapes[0][0], dtype=dtype_dict[datatype])
            unused_expert = set(range(cum_sum_num))
            selected_expert_num = 0
            for i in range(shapes[0][0]):
                cur_expert = random.choice(list(unused_expert))
                in_tensor_topk[i] = cur_expert
                unused_expert.remove(cur_expert)
                selected_expert_num += 1
                if selected_expert_num == topk_expert_num:
                    unused_expert = set(range(cum_sum_num))
                    selected_expert_num = 0
            return torch_npu.npu_format_cast(in_tensor_topk.npu(), format_dict[format])
        if i == 1:
            index_num = shapes[1][0]
            index_array = torch.arange(0, index_num, dtype=dtype_dict[datatype])
            return torch_npu.npu_format_cast(index_array.npu(), format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        topk_size = json_data["topkExpertNum"] if "topkExpertNum" in json_data else 0
        cum_sum_num = json_data["cumSumNum"] if "cumSumNum" in json_data else 0
        device_expert_index = json_data["deviceExpert"] if "deviceExpert" in json_data else []
        is_cumsum_int64 = json_data["cumSumInt64"] if "cumSumInt64" in json_data else False
        is_ep = len(device_expert_index) > 0
        if cum_sum_num == 0:
            expert_size = 1
        elif is_ep:
            expert_size = cum_sum_num
            device_expert_index.sort()
        else:
            expert_size = cum_sum_num
        input_topk = in_tensors[0].clone().numpy()
        input_index = in_tensors[1].clone().numpy()
        used_expert_num = expert_size
        if expert_size > 0 and is_ep:
            used_expert_num = len(device_expert_index)
            device_expert_index = np.sort(device_expert_index)

        input_topk_view = input_topk.view()

        inputLen = len(input_topk)
        token_index_golden = np.zeros(inputLen, dtype=np.int32)
        cumsum_pre = np.zeros(expert_size, dtype=np.int32)

        if cum_sum_num > 0:
            unique, counts = np.unique(input_topk_view, return_counts=True)
            cumsum_pre[unique] = counts
            cumsum_full = np.cumsum(cumsum_pre, dtype=np.int32)

            if is_ep:
                cumsum_golden = np.cumsum(cumsum_pre[device_expert_index],
                                          dtype=np.int64 if is_cumsum_int64 else np.int32)
            else:
                cumsum_golden = np.cumsum(cumsum_pre,
                                          dtype=np.int64 if is_cumsum_int64 else np.int32)
        else:
            cumsum_golden = np.zeros(used_expert_num, dtype=np.int64 if is_cumsum_int64 else np.int32)
        sort_indices = np.argsort(input_topk_view, kind='mergesort')
        input_topk_sorted = input_topk_view[sort_indices]
        original_index_golden = input_index[sort_indices]

        if cum_sum_num > 0:
            token_index_golden = np.floor_divide(original_index_golden, topk_size, dtype=np.int32)

        if expert_size > 0 and is_ep:
            valid_index_golden = np.zeros(1, dtype=np.int32)
            valid_index_golden[0] = np.sum(cumsum_pre[device_expert_index])
            p = 0

            for _, expert_idx in enumerate(device_expert_index):
                left = cumsum_full[expert_idx - 1] if expert_idx > 0 else 0
                right = cumsum_full[expert_idx]

                slice_length = right - left
                original_index_golden[p:p + slice_length] = original_index_golden[left:right]
                token_index_golden[p:p + slice_length] = token_index_golden[left:right]
                p += slice_length
            token_index_golden[valid_index_golden.item():] = 0
            original_index_golden[valid_index_golden.item():] = 0
        else:
            valid_index_golden = np.array([]).astype(np.int32)
        if is_ep:
            return torch.from_numpy(token_index_golden), torch.from_numpy(cumsum_golden), torch.from_numpy(
                original_index_golden), torch.from_numpy(valid_index_golden)
        return torch.from_numpy(token_index_golden), torch.from_numpy(cumsum_golden), torch.from_numpy(
            original_index_golden)

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_INTEGER


class LaserAttentionOperation(DataGen):
    atten_mask_value = -65536

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        shape = shapes[i]
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        if i == 0 or i == 1 or i == 2:
            if low >= -100 and high <= 100:
                customize_tensor = torch.empty(shape).uniform_(low, high).to(dtype_dict[datatype])
            else:
                mean = np.random.uniform(-100, 100)
                std = np.random.uniform(1, 25)
                customize_tensor = torch.from_numpy(np.random.normal(mean, std, shape)).to(dtype_dict[datatype])
            return torch_npu.npu_format_cast(customize_tensor.npu(), format_dict[format])
        if i == 6:
            return torch_npu.npu_format_cast(torch.from_numpy(1 - np.tri(shapes[i][0])).to(dtype_dict[datatype]).npu(),
                                             format_dict[format])

    @staticmethod
    def torch_batch_dot(a, b):
        batch = a.shape[0]
        n = a.shape[1]
        h = a.shape[2]
        w = b.shape[3]
        a = a.to(torch.float32)
        b = b.to(torch.float32)
        res = torch.zeros((batch, n, h, w), dtype=torch.float32)
        for i in range(batch):
            for j in range(n):
                res[i, j] = torch.from_numpy(np.matmul(a[i, j].numpy(), b[i, j].numpy()))
        return res

    @staticmethod
    def golden_base(op_params, query, key, value, atten_mask):
        json_data = json.loads(op_params)
        pre_tokens = json_data["preTokens"] if "preTokens" in json_data else 2147483647
        seq_size = query.shape[2]
        head_dim = query.shape[3]
        scale_value = json_data["scaleValue"] if "scaleValue" in json_data else 0.08838834764831843
        bmm1_res = LaserAttentionOperation.torch_batch_dot(query, torch.transpose(key[:,:,:,:head_dim], 2, 3))
        if atten_mask.numel() == 0:
            bmm1_res = bmm1_res * scale_value
        else:
            sparse_mask = torch.from_numpy(np.tri(seq_size - pre_tokens, k=-1)).to(torch.float32)
            bmm1_res = bmm1_res * scale_value + (
                    atten_mask * LaserAttentionOperation.atten_mask_value)
            if pre_tokens < seq_size:
                bmm1_res[:, :, pre_tokens:, 0:seq_size - pre_tokens] = bmm1_res[:, :, pre_tokens:,
                                                                       0:seq_size - pre_tokens] + sparse_mask * LaserAttentionOperation.atten_mask_value
        softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
        softmax_sub = bmm1_res - softmax_max
        softmax_exp = torch.exp(softmax_sub)
        softmax_sum = torch.sum(softmax_exp, -1, keepdim=True)
        softmax_out = (softmax_exp / softmax_sum).to(torch.float32)
        attention_out = LaserAttentionOperation.torch_batch_dot(softmax_out, value)
        softmax_max = softmax_max.squeeze(3)
        softmax_sum = softmax_sum.squeeze(3)
        return softmax_max, softmax_sum, attention_out

    @staticmethod
    def golden(in_tensors, op_params):
        query = in_tensors[0].to(torch.float32)
        key = in_tensors[1].to(torch.float32)
        value = in_tensors[2].to(torch.float32)
        atten_mask = in_tensors[6].to(torch.float32)
        q_num_head = query.shape[1]
        kv_num_head = key.shape[1]
        if q_num_head == kv_num_head:
            softmax_max, softmax_sum, attention_out = LaserAttentionOperation.golden_base(op_params, query, key, value,
                                                                                          atten_mask)
            return [softmax_max, softmax_sum, torch.tensor([0]), attention_out]
        else:
            head_num_per_group = q_num_head // kv_num_head
            for i in range(q_num_head):
                group_index = i // head_num_per_group
                query_per_group = query[:, i:i + 1, :, :].to(torch.float32)
                key_per_group = key[:, group_index:group_index + 1, :, :].to(torch.float32)
                value_per_group = value[:, group_index:group_index + 1, :, :].to(torch.float32)
                softmax_max, softmax_sum, attention_out = LaserAttentionOperation.golden_base(op_params,
                                                                                              query_per_group,
                                                                                              key_per_group,
                                                                                              value_per_group,
                                                                                              atten_mask)
                if i == 0:
                    total_softmax_max = softmax_max
                    total_softmax_sum = softmax_sum
                    total_attention_out = attention_out
                else:
                    total_softmax_max = torch.cat((total_softmax_max, softmax_max), dim=1)
                    total_softmax_sum = torch.cat((total_softmax_sum, softmax_sum), dim=1)
                    total_attention_out = torch.cat((total_attention_out, attention_out), dim=1)
            return [total_softmax_max, total_softmax_sum, torch.tensor([0]), total_attention_out]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION


class LaserAttentionGradOperation(DataGen):
    atten_mask_value = -65536

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        shape = shapes[i]
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        if i == 0 or i == 1 or i == 2:
            if low >= -100 and high <= 100:
                customize_tensor = torch.empty(shape).uniform_(low, high).to(dtype_dict[datatype])
            else:
                mean = np.random.uniform(-100, 100)
                std = np.random.uniform(1, 25)
                customize_tensor = torch.from_numpy(np.random.normal(mean, std, shape)).to(dtype_dict[datatype])
            return torch_npu.npu_format_cast(customize_tensor.npu(), format_dict[format])
        if i == 3:
            customize_tensor = torch.rand(shape).to(dtype_dict[datatype]) - 0.5
            return torch_npu.npu_format_cast(customize_tensor.npu(), format_dict[format])
        if i == 7:
            return torch_npu.npu_format_cast(torch.from_numpy(1 - np.tri(shapes[i][0])).to(dtype_dict[datatype]).npu(),
                                             format_dict[format])
        if i == 8 or i == 9 or i == 10 or i == 11:
            data = torch.zeros(shape, dtype=dtype_dict[datatype]).npu()
            return torch_npu.npu_format_cast(data, format_dict[format])

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        removed_tensor_index = [3, 8, 9, 10, 11]
        in_tensors = [in_tensor.cpu() for i, in_tensor in enumerate(input_tensor_list) if i not in removed_tensor_index]
        softmax_max, softmax_sum, _, attention_in = LaserAttentionOperation.golden(in_tensors, op_params)
        input_tensor_list[8] = softmax_max.to(input_tensor_list[8].dtype).npu()
        input_tensor_list[9] = softmax_sum.to(input_tensor_list[9].dtype).npu()
        input_tensor_list[11] = attention_in.to(input_tensor_list[11].dtype).npu()

    @staticmethod
    def golden_base(op_params, query, key, value, attention_out_grad, atten_mask, attention_in, softmax_log_max_sum,
                    seq_size):
        json_data = json.loads(op_params)
        pre_tokens = json_data["preTokens"] if "preTokens" in json_data else 2147483647
        batch, q_num_head, q_seq_len, head_dim = query.shape
        softmax_log_max_sum = softmax_log_max_sum.view((batch, q_num_head, q_seq_len, 1))
        softmax_out_sum = (attention_in * attention_out_grad).sum(dim=-1, keepdim=True).reshape(
            (batch, q_num_head, q_seq_len, 1))

        scale_value = json_data["scaleValue"] if "scaleValue" in json_data else 0.08838834764831843
        query_key_mul = LaserAttentionOperation.torch_batch_dot(query, torch.transpose(key[:, :, :, :head_dim], 2, 3))
        softmax_grad = LaserAttentionOperation.torch_batch_dot(attention_out_grad, torch.transpose(value, 2, 3))
        if atten_mask.numel() == 0:
            bmm1_res_drop = query_key_mul * scale_value
        else:
            sparse_mask = torch.from_numpy(np.tri(seq_size - pre_tokens, k=-1)).to(torch.float32)
            bmm1_res_drop = query_key_mul * scale_value + (atten_mask * LaserAttentionGradOperation.atten_mask_value)
            if pre_tokens < seq_size:
                bmm1_res_drop[:, :, pre_tokens:, 0:seq_size - pre_tokens] = bmm1_res_drop[:, :, pre_tokens:,
                                                                            0:seq_size - pre_tokens] + sparse_mask * LaserAttentionGradOperation.atten_mask_value
        softmax_out = torch.exp(bmm1_res_drop - softmax_log_max_sum)
        bmm1_res_drop_grad_flash = (softmax_grad - softmax_out_sum) * softmax_out
        bmm1_res_grad = bmm1_res_drop_grad_flash * scale_value
        query_grad = LaserAttentionOperation.torch_batch_dot(bmm1_res_grad.to(torch.float32), key)
        key_grad = LaserAttentionOperation.torch_batch_dot(torch.transpose(bmm1_res_grad, 2, 3).to(torch.float32),
                                                           query)
        value_grad = LaserAttentionOperation.torch_batch_dot(torch.transpose(softmax_out, 2, 3).to(torch.float32),
                                                             attention_out_grad)
        pad_len = key.shape[3] - head_dim
        key_grad = torch.nn.functional.pad(key_grad, (0, pad_len), mode='constant', value=0)
        return query_grad[:, :, :, :192], key_grad, value_grad

    @staticmethod
    def golden(in_tensors, op_params):
        query = in_tensors[0].to(torch.float32)
        key = in_tensors[1].to(torch.float32)
        value = in_tensors[2].to(torch.float32)
        attention_out_grad = in_tensors[3].to(torch.float32)
        atten_mask = in_tensors[7].to(torch.float32)
        softmax_max = in_tensors[8].to(torch.float32)
        softmax_sum = in_tensors[9].to(torch.float32)
        attention_in = in_tensors[11].to(torch.float32)
        q_num_head = query.shape[1]
        seq_size = query.shape[2]
        kv_num_head = key.shape[1]
        softmax_log_max_sum = softmax_max + torch.log(softmax_sum)
        if q_num_head == kv_num_head:
            query_grad, key_grad, value_grad = LaserAttentionGradOperation.golden_base(op_params, query, key, value,
                                                                                       attention_out_grad, atten_mask,
                                                                                       attention_in,
                                                                                       softmax_log_max_sum, seq_size)
        else:
            query_grad = torch.zeros_like(query)
            key_grad = torch.zeros_like(key)
            value_grad = torch.zeros_like(value)
            head_num_per_group = q_num_head // kv_num_head
            for i in range(q_num_head):
                group_index = i // head_num_per_group
                query_per_group = query[:, i:i + 1, :, :]
                key_per_group = key[:, group_index:group_index + 1, :, :]
                value_per_group = value[:, group_index:group_index + 1, :, :]
                attention_out_grad_per_group = attention_out_grad[:, i:i + 1, :, :]
                attention_in_per_group = attention_in[:, i:i + 1, :, :]
                softmax_log_max_sum_per_group = softmax_log_max_sum[:, i:i + 1, :]
                query_grad_per_group, key_grad_per_group, value_grad_per_group = LaserAttentionGradOperation.golden_base(
                    op_params, query_per_group, key_per_group, value_per_group, attention_out_grad_per_group,
                    atten_mask, attention_in_per_group, softmax_log_max_sum_per_group, seq_size)
                query_grad[:, i:i + 1, :, :] = query_grad_per_group
                key_grad[:, group_index:group_index + 1, :, :] += key_grad_per_group
                value_grad[:, group_index:group_index + 1, :, :] += value_grad_per_group

        return [query_grad, key_grad, value_grad, torch.tensor([0]).to(torch.float32)]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION

class GroupTopkOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params) -> torch.Tensor:
        json_data = json.loads(op_params)
        group_num = json_data['groupNum'] if "groupNum" in json_data else 1
        k = json_data['k'] if "k" in json_data else 0
        if i == 0:
            input0_np = np.random.random(shapes[0]).astype(np.float32)
            input0 = torch.from_numpy(input0_np).type(torch.float16).to(dtype_dict[datatype])
            input0_return = torch_npu.npu_format_cast(input0.npu(), format_dict[format])
            return input0_return
        if i == 1:
            total_size = reduce(lambda x, y: x * y, shapes[1])
            input1 = torch.arange(total_size, dtype=torch.int32)
            input1 = torch.reshape(input1, shapes[1])
            input1_return = torch_npu.npu_format_cast(input1.npu(), format_dict[format])
            return input1_return

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        group_num: int = json_data["groupNum"] if "groupNum" in json_data else 1
        k: int = json_data["k"] if "k" in json_data else 0
        groupMultiFlag: int = json_data["groupMultiFlag"] if "groupMultiFlag" in json_data else 0
        n: int = json_data["n"] if "n" in json_data else 1
        if groupMultiFlag == 0:
            n = 1
        input0 = in_tensors[0]
        token_num, expert_num = input0.shape
        input0 = torch.reshape(input0, (token_num, group_num, expert_num // group_num))
        output = input0.clone()
        input0 = input0.to(torch.float)
        group_tensor = torch.topk(input0, n).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable'))
        cols_to_use = torch.arange(k, group_num, dtype=torch.long)
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        output[row_indices, col_indices] = 0
        output0_golden = [torch.reshape(output, (token_num, expert_num))]
        return output0_golden

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        output_tensor_list[0] = input_tensor_list[0]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class GroupedMatmulWithRoutingOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        json_data = json.loads(op_params)
        groupedMatmulType = json_data['groupedMatmulType']
        topK = json_data['topK']
        outDataType = json_data['outDataType']
        num_tokens = shapes[0][0]
        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        if groupedMatmulType == 1:
            num_tokens = num_tokens // topK
        if i == 0:
            if outDataType == -1:
                activation = np.random.uniform(low, high, shapes[0])
            else:
                activation = np.random.randint(int(low), int(high), shapes[0])
            return torch_npu.npu_format_cast(torch.tensor(activation).to(dtype_dict[datatype]).npu(),format_dict[format])
        if i == 1:
            if outDataType == -1:
                weight = np.random.uniform(low, high, shapes[1])
            else:
                weight = np.random.randint(int(low), int(high), shapes[1])
            return torch_npu.npu_format_cast(torch.tensor(weight).to(dtype_dict[datatype]).npu(),format_dict[format])
        if i == 2:
            experts_map = np.array([np.random.permutation(shapes[i][0])[:topK] for _ in range(num_tokens)])
            experts_index = [np.where(experts_map == e)[0].tolist() for e in range(shapes[i][0])]
            experts_count = [len(each) for each in experts_index]
            experts_count = torch_npu.npu_format_cast(torch.tensor(experts_count).npu().to(torch.int32), format_dict[format])
            GroupedMatmulWithRoutingOperation.expertindex = torch.tensor([item for sublist in experts_index for item in sublist])
            GroupedMatmulWithRoutingOperation.experts_count = experts_count
            return experts_count.cumsum(-1).int().npu()
        if i == 3:
            return torch_npu.npu_format_cast(GroupedMatmulWithRoutingOperation.expertindex.npu().to(torch.int32), format_dict[format])
        if i == 4 or i == 5:
            return torch_npu.npu_format_cast(torch.tensor(np.random.uniform(low, high, size=shapes[i])).float().npu(), format_dict[format])


    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        groupedMatmulType = json_data['groupedMatmulType']
        transposeB = json_data['transposeB']
        topK = json_data['topK']
        outDataType = json_data['outDataType']
        if outDataType != -1:
            toDataType = torch.float16 if outDataType else torch.bfloat16
        activation = in_tensors[0]
        weight = in_tensors[1]
        experts_count = GroupedMatmulWithRoutingOperation.experts_count
        experts_index = in_tensors[3]
        num_experts = weight.shape[0]
        num_tokens = activation.shape[0]
        if groupedMatmulType == 1:
            num_tokens //= topK
        hidden_size_out = weight.shape[1] if transposeB else weight.shape[2]
        if groupedMatmulType == 0:
            if outDataType == -1:
                ref = []
                e_start = 0
                for e in range(num_experts):
                    a = activation[experts_index[e_start : e_start + experts_count[e]]]
                    if a.shape[0] == 0:
                        continue
                    b = weight[e]
                    if transposeB:
                        b = b.transpose_(0, 1)
                    ref.append(torch.mm(a.float(), b.float()).numpy())
                    e_start += experts_count[e]
                ref = np.concatenate(ref)
                return [torch.tensor(ref).to(activation.dtype)]
            else:
                nscale = in_tensors[4]
                mscale = in_tensors[5]
                ref = []
                e_start = 0
                for e in range(num_experts):
                    a = activation[experts_index[e_start : e_start + experts_count[e]]]
                    if a.shape[0] == 0:
                        continue
                    b = weight[e]
                    if transposeB:
                        b = b.transpose_(0, 1)
                    c = torch.mm(a.int(), b.int())
                    c_quant = c * mscale[experts_index[e_start : e_start + experts_count[e]]].reshape(-1,1) * nscale[e].reshape(1,-1)
                    ref.append(c_quant)
                    e_start += experts_count[e]
                ref = torch.cat(ref)
                return [torch.tensor(ref).to(toDataType)]
        else:
            if outDataType == -1:
                ref = torch.zeros(num_tokens, hidden_size_out, dtype=torch.float32)
                e_start = 0
                for e in range(num_experts):
                    a = activation[e_start : e_start + experts_count[e]]
                    if a.shape[0] == 0:
                        continue
                    b = weight[e]
                    if transposeB:
                        b = b.transpose_(0, 1)
                    c = torch.mm(a.float(), b.float())
                    ref[experts_index[e_start : e_start + experts_count[e]]] += c
                    e_start += experts_count[e]
                return [torch.tensor(ref).to(activation.dtype)]
            else:
                nscale = in_tensors[4]
                mscale = in_tensors[5]
                ref = torch.zeros(num_tokens, hidden_size_out, dtype=torch.float32)
                e_start = 0
                for e in range(num_experts):
                    a = activation[e_start : e_start + experts_count[e]]
                    if a.shape[0] == 0:
                        continue
                    b = weight[e]
                    if transposeB:
                        b = b.transpose_(0, 1)
                    c = torch.mm(a.int(), b.int())
                    c_quant = (c * mscale[e_start : e_start + experts_count[e]].reshape(-1,1)) * nscale[e].reshape(1,-1)
                    ref[experts_index[e_start : e_start + experts_count[e]]] += c_quant
                    e_start += experts_count[e]
                return [torch.tensor(ref).to(toDataType)]

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.COMPUTE_FLOAT

class CohereLayerNormOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def golden(in_tensors, op_params):
        epsilon = json.loads(op_params).get("epsilon", 1e-5)
        input0 = in_tensors[0]
        isbf16 = input0.dtype == torch.bfloat16
        input0 = input0.to(float)
        input1 = in_tensors[1].to(float)
        mean = input0.mean(-1, keepdim=True)
        variance = (input0 - mean).pow(2).mean(-1, keepdim=True)
        ret = input1 * (input0 - mean) * torch.rsqrt(variance + epsilon)
        if isbf16:
            ret = ret.to(torch.bfloat16)
        return [ret]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class GatherPreRmsNormOperation(DataGen):
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        soc_version = get_soc_version()
        if soc_version == 'Ascend910B':
            # if i != 0:
            #     return GatherPreRmsNormOperation.ret_data[i]
            res_tokens = shapes[1][0]
            ind_tokens = shapes[0][0]
            hidden_size = shapes[0][1]
            if i == 0:
                low = float(data_gen_ranges.split(',')[0])
                high = float(data_gen_ranges.split(',')[1])
                input0 = torch.empty((shapes[0][0], shapes[0][1]), dtype=dtype_dict[datatype]).uniform_(low, high).npu()
                return input0
            if i == 1:
                low = float(data_gen_ranges.split(',')[0])
                high = float(data_gen_ranges.split(',')[1])
                input1 = torch.empty((shapes[1][0], shapes[1][1]), dtype=dtype_dict[datatype]).uniform_(low, high).npu()
                return input1
            if i == 2:
                low = float(data_gen_ranges.split(',')[0])
                high = float(data_gen_ranges.split(',')[1])
                input2 = torch.randint(0, res_tokens, (shapes[2][0],), dtype=dtype_dict[datatype]).npu()
                return input2

            low = float(data_gen_ranges.split(',')[0])
            high = float(data_gen_ranges.split(',')[1])
            input3 = torch.empty((1, hidden_size), dtype=dtype_dict[datatype]).uniform_(low, high).npu()
            return input3


    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        epsilon = json_data['epsilon']

        data_type = in_tensors[0].dtype
        input0 = in_tensors[0].clone().detach().float().cpu()
        input1 = in_tensors[1].clone().detach().float().cpu()
        input2 = in_tensors[2].clone().detach().to(torch.int64).cpu()
        input3 = in_tensors[3].clone().detach().float().cpu()
        indices_brcb = input2.unsqueeze(1)
        index = indices_brcb.repeat(1, input1.shape[1])
        try:
            res_in_after_gather = input1.gather(0, index)
        except RuntimeError as e:
            print("some index values are out of range", e)
        # rmsnorm compute
        input_sum = input0 + res_in_after_gather
        square_sum = torch.sum(torch.square(input_sum), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / input_sum.shape[-1] + epsilon)
        output = input_sum * factor * input3
        return [output.clone().detach().to(float), input_sum.clone().detach().to(data_type)]

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.COMPUTE_FLOAT

class NormRopeReshapeOperation(DataGen):
    random_idx = 0
    def __init__(self):
        pass

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        soc_version = get_soc_version()
        if soc_version == 'Ascend910B':
            if i == 0:
                input0 = torch.rand((shapes[0][0], shapes[0][1], shapes[0][2])).to(dtype_dict[datatype]).npu()
                return input0
            if i == 1:
                input1 = torch.rand((shapes[1][0],)).to(dtype_dict[datatype]).npu()
                return input1
            if i == 2:
                input2 = torch.rand((shapes[2][0], shapes[2][1])).to(dtype_dict[datatype]).npu()
                return input2
            if i == 3:
                input3 = torch.rand((shapes[3][0], shapes[3][1])).to(dtype_dict[datatype]).npu()
                return input3
            if i == 4:
                input4 = torch.rand((shapes[4][0], shapes[4][1])).to(dtype_dict[datatype]).npu()
                return input4
            if i == 5:
                input5 = torch.tensor(np.random.choice(100, shapes[5][0], replace=False)).to(dtype_dict[datatype]).npu()
                return input5

            input6 = torch.zeros((shapes[6][0], shapes[6][1], shapes[6][2], shapes[6][3])).to(dtype_dict[datatype]).npu()
            return input6

    @staticmethod
    def random(shape, datatype, format, data_gen_ranges, op_params):
        NormRopeReshapeOperation.random_idx = NormRopeReshapeOperation.random_idx + 1
        if NormRopeReshapeOperation.random_idx > 7:
            NormRopeReshapeOperation.random_idx = 1

        soc_version = get_soc_version()
        if soc_version == 'Ascend910B':
            if NormRopeReshapeOperation.random_idx == 1:
                input0 = torch.rand((shape[0], shape[1], shape[2])).to(dtype_dict[datatype]).npu()
                return input0
            if NormRopeReshapeOperation.random_idx == 2:
                input1 = torch.rand((shape[0],)).to(dtype_dict[datatype]).npu()
                return input1
            if NormRopeReshapeOperation.random_idx == 3:
                input2 = torch.rand((shape[0], shape[1])).to(dtype_dict[datatype]).npu()
                return input2
            if NormRopeReshapeOperation.random_idx == 4:
                input3 = torch.rand((shape[0], shape[1])).to(dtype_dict[datatype]).npu()
                return input3
            if NormRopeReshapeOperation.random_idx == 5:
                input4 = torch.rand((shape[0], shape[1])).to(dtype_dict[datatype]).npu()
                return input4
            if NormRopeReshapeOperation.random_idx == 6:
                input5 = torch.randperm(100)[:shape[0]].to(dtype=dtype_dict[datatype]).npu()
                return input5
            if NormRopeReshapeOperation.random_idx == 7:
                input6 = torch.zeros((shape[0], shape[1], shape[2], shape[3])).to(dtype_dict[datatype]).npu()
                return input6

            return DataGen.random(shape, datatype, format, data_gen_ranges, op_params)

        print("NormRopeReshap only supports on Atalas 800I2")
        return 0

    @staticmethod
    def rmsnorm_golden(x, gamma, epsilon):
        x_float32 = x.astype(np.float32)
        square_sum = np.sum(np.square(x_float32), axis=-1, keepdims=True)
        rms = 1.0 / np.sqrt(square_sum / x_float32.shape[-1] + epsilon)
        gamma_float32 = gamma.astype(np.float32)
        rms_norm = rms * x_float32 * gamma_float32
        result = rms_norm.astype(np.float16)
        return result
    @staticmethod
    def rotate_half(k_temp):
        first_half, second_half = np.split(k_temp, 2, axis=1)
        processed_k_split = np.concatenate((-second_half, first_half), axis=1)
        return processed_k_split
    @staticmethod
    def rope_golden(key_rope, sin, cos):
        res = key_rope*cos + NormRopeReshapeOperation.rotate_half(key_rope)*sin
        return res
    @staticmethod
    def rac_golden(block_size, key_rac, slot_mapping, keycacheout_golden):
        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size
            token_key = key_rac[i]

            keycacheout_golden[block_index][block_offset] = token_key
        return keycacheout_golden

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        epsilon = json_data['epsilon']

        data_type = in_tensors[6].dtype

        input0 = np.array(in_tensors[0].cpu()).astype(np.float16)  # x
        input1 = np.array(in_tensors[1].cpu()).astype(np.float16)  # gamma
        input2 = np.array(in_tensors[2].cpu()).astype(np.float16)  # keyRope
        input3 = np.array(in_tensors[3].cpu()).astype(np.float16)  # cos
        input4 = np.array(in_tensors[4].cpu()).astype(np.float16)  # sin
        input5 = np.array(in_tensors[5].cpu()).astype(np.int32)    # slotMapping
        input6 = np.array(in_tensors[6].cpu()).astype(np.float16)  # keycachein

        keycacheout_golden = np.zeros_like(input6)                 # keycacheout

        rmsnorm_output = NormRopeReshapeOperation.rmsnorm_golden(input0, input1, epsilon)
        rope_output = NormRopeReshapeOperation.rope_golden(input2, input4, input3)
        rope_reshape = rope_output.reshape(input4.shape[0], -1, input4.shape[-1])
        key_rac = np.concatenate((rmsnorm_output, rope_reshape), axis=-1)
        output = NormRopeReshapeOperation.rac_golden(input6.shape[1], key_rac, input5, keycacheout_golden)
        return [torch.tensor(output).to(data_type)]

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.COMPUTE_FLOAT

class FusedAddTopkDivOperation(DataGen):

    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        torch.manual_seed(42)
        return DataGen.random(shapes[i], datatype, format, data_gen_ranges, op_params)

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        group_num = json_data['groupNum']
        group_topk = json_data['groupTopk']
        is_norm = json_data['isNorm']
        n = json_data['n']
        k = json_data['k']
        scale = json_data['scale']

        x = in_tensors[0]
        add_num = in_tensors[1]
        # sigmoid
        sigmoid = torch.nn.Sigmoid()
        sigmoid_output = sigmoid(x.to(torch.float32))
        # add
        add_output = torch.add(sigmoid_output, add_num.to(torch.float32))
        token_num, expert_num = add_output.shape
        group_eles = expert_num // group_num
        add_output = torch.reshape(add_output, (token_num, group_num, group_eles))
        add_output_copy = add_output.clone()
        # group topk
        group_tensor = torch.topk(add_output, n).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable'))
        cols_to_use = torch.arange(group_topk, group_num, dtype=torch.long)
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        add_output_copy[row_indices, col_indices] = float(0)
        group_topk_output = torch.reshape(add_output_copy, (token_num, expert_num))

        sort_output = torch.sort(group_topk_output, descending=True, stable=True)
        # gather
        gather_output = torch.gather(sigmoid_output, -1, sort_output.indices[:, 0:k])
        # reduce sum
        if is_norm:
            sum_output = torch.sum(gather_output, -1, keepdim=True)
            y = torch.div(gather_output, sum_output)
            y = y * torch.tensor(scale, dtype=torch.float32)
        else:
            y = gather_output
        out_indices = sort_output.indices.to(torch.int32)
        if json_data['enableExpertMapping'] if "enableExpertMapping" in json_data else False:
            offset = 7
            prime = 100000007
            mapping_num = in_tensors[2]
            mapping_table = in_tensors[3]
            out_indices_clone = out_indices.clone().detach()
            for bs in range(token_num):
                for ki in range(k):
                    expert_id = out_indices_clone[bs][ki]
                    redundantOffset = bs % max(1, mapping_num[expert_id])
                    out_indices[bs][ki] = mapping_table[expert_id][redundantOffset]

        return y, out_indices[:, 0:k]

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.VECTOR_FUSION

class ReshapeAndCacheOmniOperation(DataGen):


    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        output_tensor_list[0] = input_tensor_list[2]
        output_tensor_list[1] = input_tensor_list[3]

    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        key = in_tensors[0]
        value = in_tensors[1]
        key_expect = in_tensors[2]
        value_expect = in_tensors[3]
        slot_mapping = in_tensors[4]
        wins = in_tensors[5]
        seqLen = in_tensors[6]
        offsetIndex = in_tensors[7]

        num_tokens, num_heads, head_size = key.shape
        block_size = key_expect.shape[1]


        key_expect_fp32 = key_expect.clone().to(torch.float32)
        value_expect_fp32 = value_expect.clone().to(torch.float32)

        new_seq = seqLen.clone()
        new_seq[0] = seqLen[0]
        for n in range(1, seqLen.shape[0]):
            new_seq[n] = seqLen[n] + new_seq[n-1]
        new_seq = torch.cat((torch.zeros(1, dtype=torch.int32), new_seq), dim=0)

        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            win = wins[i].clone()
            curIdxOffset = offsetIndex[i]

            bsID = i // num_heads
            headID = i % num_heads
            headStartIdx = new_seq[bsID]
            headEndIdx = curIdxOffset + wins[i]
            bs = new_seq[bsID]

            sum_key = torch.zeros(head_size, dtype = torch.float32)
            sum_value = torch.zeros(head_size, dtype = torch.float32)
            for j in range(seqLen[bsID]):
                block_index = torch.div(slot, block_size, rounding_mode='trunc')
                block_offset = slot % block_size

                token_key = key[bs+j][headID]
                token_v = value[bs+j][headID]
                if curIdxOffset == -1 or (j < curIdxOffset and curIdxOffset != -1):
                    key_expect_fp32[block_index][block_offset] = token_key
                    value_expect_fp32[block_index][block_offset] = token_v
                    slot+=1

                if j >= headEndIdx and curIdxOffset != -1:
                    key_expect_fp32[block_index][block_offset] = token_key.to(torch.float32)
                    value_expect_fp32[block_index][block_offset] = token_v.to(torch.float32)
                    slot+=1

        return key_expect_fp32, value_expect_fp32


    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class RazorFusionAttentionOperation(DataGen):
    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.VECTOR_FUSION

class FaUpdateOperation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        all_lse = in_tensors[0]
        all_out = in_tensors[1]
        sp = all_out.shape[0]
        hd = all_out.shape[-1]

        all_lse = all_lse.transpose(0, 1)
        all_out = all_out.permute(1,2,0).reshape(-1, sp*hd)

        # (b * s * hc, sp)
        lse_exp = torch.exp(all_lse)
        # (b * s * hc, 1)
        sum_lse_exp = torch.sum(lse_exp, dim=-1, keepdim=True)
        # (b * s * hc, sp)
        sum_lse_exp = sum_lse_exp.repeat(1, sp)
        lse_exp = lse_exp/sum_lse_exp

        # oi = lse_exp*oi (b * s * hc, hd, sp)*(b*s*hc, hd, sp)
        lse_exp = lse_exp.unsqueeze(1)
        lse_exp = lse_exp.repeat(1, hd, 1)
        all_out = all_out.reshape(-1, hd, sp)
        all_out = all_out * lse_exp

        # o = sum(oi) (b*s*hc, hd)
        all_out = torch.sum(all_out, dim=-1)
        return [all_out]

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.COMPUTE_FLOAT

class MlaPreprocessOperation(DataGen):
    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.COMPUTE_FLOAT

class MultiLatentAttentionOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            if i < len(MultiLatentAttentionOperation.in_tensors):
                t = MultiLatentAttentionOperation.in_tensors[i]
                if t.numel() > 0:
                    return torch_npu.npu_format_cast(t, format_dict[format])
        json_data = json.loads(op_params)
        mask_type = json_data.get("maskType", 0)
        num_tokens, num_heads, head_dim_q = shapes[0]
        rope_dim = shapes[1][2]
        num_blocks, block_size, kv_heads, kv_head_dim = shapes[2]
        batch, max_blocks = shapes[4]
        rope_head_dim = shapes[3][3]

        low = float(data_gen_ranges.split(',')[0])
        high = float(data_gen_ranges.split(',')[1])
        query_np = np.random.uniform(low, high, size=(num_tokens, num_heads, head_dim_q)).astype(np.float32)
        query_rope_np = np.random.uniform(low, high, size=(num_tokens, num_heads, rope_dim)).astype(np.float32)
        kv_np = np.random.uniform(low, high, size=(num_blocks, block_size, kv_heads, kv_head_dim)).astype(np.float32)
        kv_rope_np = np.random.uniform(low, high, size=(num_blocks, block_size, kv_heads, rope_head_dim)).astype(np.float32)
        bt_np = np.zeros((batch, max_blocks), dtype=np.int32)
        for b in range(batch):
            for j in range(max_blocks):
                idx = b * max_blocks + j
                bt_np[b, j] = idx if idx < num_blocks else 0
        ctx_np = np.full((batch,), block_size, dtype=np.int32)
        qsl_np = np.ones((batch,), dtype=np.int32)

        base_shape_idx = 6
        has_mask = mask_type != 0
        if has_mask:
            m_shapes = shapes[base_shape_idx]
            m_np = np.random.uniform(low, high, size=(m_shapes[0], m_shapes[1])).astype(np.float32)
            base_shape_idx = 7

        tensor_list = [
            torch.from_numpy(query_np).to(dtype_dict[datatype]).npu(),
            torch.from_numpy(query_rope_np).to(dtype_dict[datatype]).npu(),
            torch.from_numpy(kv_np).to(dtype_dict[datatype]).npu(),
            torch.from_numpy(kv_rope_np).to(dtype_dict[datatype]).npu(),
            torch.from_numpy(bt_np).npu(),
            torch.from_numpy(ctx_np).npu(),
        ]
        if has_mask:
            tensor_list.append(torch.from_numpy(m_np).to(dtype_dict[datatype]).npu())
        tensor_list.append(torch.from_numpy(qsl_np).npu())

        MultiLatentAttentionOperation.in_tensors = tensor_list
        return tensor_list[0]

    @staticmethod
    def get_op_type(op_params) -> OpTypes:
        return OpTypes.CV_FUSION

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        json_data = json.loads(op_params)
        host_dict = {}
        host_dict["contextLens"] = input_tensor_list[5].tolist()
        calc_type = json_data.get("calcType", 0)
        if calc_type in [1, 3]:
            mask_type = json_data.get("maskType", 0)
            qseq_idx = 6 if mask_type == 0 else 7
            host_dict["qSeqlen"] = input_tensor_list[qseq_idx].tolist()
        if "maskType" in json_data:
            host_dict["maskType"] = json_data["maskType"]
        if "cacheMode" in json_data and json_data["cacheMode"] == 2:
            host_dict["cacheType"] = 1
        run_param = json.dumps(host_dict)
        operation.set_varaintpack_param(run_param)

class PagedCacheLoadOperation(DataGen):
    @staticmethod
    def customize(shapes, i, datatype, format, data_gen_ranges, op_params):
        if i != 0:
            PagedCacheLoadOperation.in_tensors[i] = torch_npu.npu_dtype_cast(PagedCacheLoadOperation.in_tensors[i].npu(), dtype_dict[datatype])
            return torch_npu.npu_format_cast(PagedCacheLoadOperation.in_tensors[i], format_dict[format])
        datatype_dict = {"float16": "float16", "bf16": "float32", "int8": "int8"}
        dtype = datatype_dict[datatype]
        soc_version = get_soc_version()
        MAX_SEQ_LEN = 1024
        json_data = json.loads(op_params)
        # ND
        if "kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 1:
            seq_len = 1
            num_blocks_k = shapes[0][0]
            num_blocks_v = shapes[1][0]
            block_size_k = shapes[0][1]
            block_size_v = shapes[1][1]
            num_heads_k = shapes[0][2]
            num_heads_v = shapes[1][2]
            head_size_k = shapes[0][3]
            head_size_v = shapes[1][3]
            batch = shapes[2][0]
            num_tokens = seq_len * batch
            context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
            num_tokens = len(context_lens)
            
            key_cache = np.random.randint(1, 11, size=(num_blocks_k, block_size_k, num_heads_k, head_size_k)).astype(
            dtype)
            value_cache = np.random.randint(1, 11, size=(num_blocks_v, block_size_v, num_heads_v, head_size_v)).astype(
            dtype)
            
            sum_context_lens = context_lens[-1]
            max_context_len = max(context_lens)
            
            cu_context_lens = [0]
            for elem in context_lens:
                cu_context_lens.append(cu_context_lens[-1] + elem)
                
            max_num_blocks_per_req = (max_context_len + block_size_k -1) // block_size_k + 4
            block_tables = [] #[num_tokens, max_num_blocks_per_seq]
            for _ in range(num_tokens):
                block_table = [
                    random.randint(0, num_blocks_k -1) for _ in range(max_num_blocks_per_req)
                ]
                block_tables.append(block_table)
            seq_starts = [random.randint(0, 4) * block_size_k for _ in range(num_tokens)]
            if "isSeqLensCumsumMode" in json_data and json_data["isSeqLensCumsumMode"]:
                context_lens = np.array(cu_context_lens).astype(np.int32)
            else:
                context_lens = np.array(context_lens).astype(np.int32)

            block_tables = np.array(block_tables).astype(np.int32)
            seq_starts = np.array(seq_starts).astype(np.int32)
            
            key = np.zeros((sum_context_lens, num_heads_k, head_size_k)).astype(dtype)
            value = np.zeros((sum_context_lens, num_heads_v, head_size_v)).astype(dtype)
            
            ret_data = key_cache, value_cache, block_tables, context_lens, key, value, seq_starts
            in_tensors = [torch.from_numpy(tensor) for tensor in ret_data]
            in_tensors = [tensor.npu() for tensor in in_tensors]
            PagedCacheLoadOperation.in_tensors = in_tensors
            return torch_npu.npu_format_cast(PagedCacheLoadOperation.in_tensors[i], format_dict[format])
        # NZ
        batch = shapes[3][0]
        context_lens = [random.randint(128, 128) for _ in range(batch)]
        num_tokens = len(context_lens)
        max_context_len = max(context_lens)
        block_size = shapes[0][2]
        num_blocks = shapes[0][0]
        num_blocksv = shapes[1][0]
        elenum_alignedk = shapes[0][3]
        elenum_alignedv = shapes[1][3]
        num_heads_sizek = shapes[0][1]
        num_heads_sizev = shapes[1][1]
        num_heads_sizek16 = shapes[4][1]
        num_heads_sizev16 = shapes[5][1]
        sumcontext = shapes[4][0]
        key_cache = np.random.randint(1, 11, size=(num_blocks, num_heads_sizek, block_size, elenum_alignedk)).astype(
            dtype)
        value_cache = np.random.randint(1, 11, size=(num_blocksv, num_heads_sizev, block_size, elenum_alignedv)).astype(
            dtype)

        max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size
        block_tables = []  # [num_tokens, max_num_blocks_per_seq]
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
            ]
            block_tables.append(block_table)

        context_lens = np.array(context_lens).astype(np.int32)
        block_tables = np.array(block_tables).astype(np.int32)

        key = np.zeros((sumcontext, num_heads_sizek16)).astype(dtype)
        value = np.zeros((sumcontext, num_heads_sizev16)).astype(dtype)

        ret_data = key_cache, value_cache, block_tables, context_lens, key, value
        in_tensors = [torch.from_numpy(tensor) for tensor in ret_data]
        in_tensors = [tensor.npu() for tensor in in_tensors]
        PagedCacheLoadOperation.in_tensors = in_tensors

        return torch_npu.npu_format_cast(PagedCacheLoadOperation.in_tensors[i], format_dict[format])

    @staticmethod
    def golden(in_tensors, op_params):
        MAX_SEQ_LEN = 1024
        data_type = in_tensors[0].dtype
        datatype_dictkv = {"torch.float16": "float16", "torch.bfloat16": "float32", "torch.int8": "int8"}
        json_data = json.loads(op_params)
        if "kvCacheCfg" in json_data and  json_data["kvCacheCfg"] == 1:
            seq_len = 1
            num_heads_k =  in_tensors[0][2]
            num_heads_v =  in_tensors[1][2]
            head_size_k =  in_tensors[0][3]
            head_size_v =  in_tensors[1][3]
            block_size = in_tensors[0][1]
            batch = in_tensors[2][0]
            num_tokens = seq_len * batch
            
            block_tables = in_tensors[2].numpy()  # [num_tokens, max_num_blocks_per_seq]
            context_lens = in_tensors[3].numpy()
            seq_starts = in_tensors[6].numpy()
            
            key_expect = np.zeros((sum_context_lens, num_heads_k, head_size_k)).astype(datatype_dictkv[str(data_type)])
            value_expect = np.zeros((sum_context_lens, num_heads_v, head_size_v)).astype(datatype_dictkv[str(data_type)])
            
            if "hasSeqStarts" in json_data and  json_data["hasSeqStarts"]:
                kv_rslt_id = 0
                context_start = 0
                for i in range(num_tokens):
                    block_table = block_tables[i]
                    context_end = int(context_lens[i+1])
                    context_len = context_end - context_start
                    context_start = context_end
                    block_table_offset = seq_starts[i] // block_size
                    for j in range(context_len):
                        block_id = int(block_table[block_table_offset + j // block_size])
                        block_offset = j % block_size
                        
                        if block_id < 0:
                            continue
                        
                        temp_k = key_cache[block_id][block_offset]
                        temp_v = value_cache[block_id][block_offset]
                        
                        key_expect[kv_rslt_id] = temp_k
                        value_expect[kv_rslt_id] = temp_v
                        kv_rslt_id += 1
                        
                res_data = key_expect, value_expect
                out_tensors = [torch.from_numpy(tensor) for tensor in res_data]
                return out_tensors
            
            kv_rslt_id = 0
            for i in range(num_tokens):
                block_table = block_tables[i]
                context_len = int(context_lens[i])

                for j in range(context_len):
                    block_id = int(block_table[j // block_size])
                    block_offset = j % block_size

                    if block_id < 0:
                        continue

                    temp_k = key_cache[block_id][block_offset]
                    temp_v = value_cache[block_id][block_offset]

                    key_expect[kv_rslt_id] = temp_k
                    value_expect[kv_rslt_id] = temp_v
                    kv_rslt_id += 1
                    
            res_data = key_expect, value_expect
            out_tensors = [torch.from_numpy(tensor) for tensor in res_data]
            return out_tensors
        context_lens = in_tensors[3].numpy()
        sum_context_lens = sum(context_lens)
        max_context_len = max(context_lens)
        num_tokens = len(context_lens)
        num_blocks, num_heads_sizek_aligend, block_size, elenum_alignedk = in_tensors[0].shape
        num_blocks, num_heads_sizev_aligend, block_size, elenum_alignedv = in_tensors[1].shape
        num_heads_sizek = num_heads_sizek_aligend * elenum_alignedk
        num_heads_sizev = num_heads_sizev_aligend * elenum_alignedv
        data_type = in_tensors[0].dtype
        datatype_dictkv = {"torch.float16": "float16", "torch.bfloat16": "float32", "torch.int8": "int8"}
        dtypekv = datatype_dictkv[str(data_type)]
        key_expect = np.zeros((sum_context_lens, num_heads_sizek)).astype(dtypekv)
        value_expect = np.zeros((sum_context_lens, num_heads_sizev)).astype(dtypekv)

        key_cache = in_tensors[0].numpy()
        if in_tensors[1].dtype == torch.bfloat16:
            value_cache = in_tensors[1].to(torch.float32).numpy()
        else:
            value_cache = in_tensors[1].numpy()
        block_tables = in_tensors[2].numpy()  # [num_tokens, max_num_blocks_per_seq]
        kv_rslt_id = 0
        for i in range(num_tokens):
            block_table = block_tables[i]
            context_len = int(context_lens[i])

            for j in range(context_len):
                block_id = int(block_table[j // block_size])
                block_offset = j % block_size

                if block_id < 0:
                    continue

                temp_k = np.zeros((num_heads_sizek))
                temp_v = np.zeros((num_heads_sizev))

                for k in range(num_heads_sizek // elenum_alignedk):
                    temp_k[k * elenum_alignedk: k * elenum_alignedk + elenum_alignedk] = key_cache[block_id][k][
                                                                                          block_offset][:]

                for k in range(num_heads_sizev // elenum_alignedv):
                    temp_v[k * elenum_alignedv: k * elenum_alignedv + elenum_alignedv] = value_cache[block_id][k][
                                                                                          block_offset][:]

                key_expect[kv_rslt_id] = temp_k
                value_expect[kv_rslt_id] = temp_v
                kv_rslt_id += 1
        res_data = key_expect, value_expect
        out_tensors = [torch.from_numpy(tensor) for tensor in res_data]
        return out_tensors

    @staticmethod
    def case_postprocess(op_params, operation, input_tensor_list, output_tensor_list):
        output_tensor_list[0] = input_tensor_list[4]
        output_tensor_list[1] = input_tensor_list[5]

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.MOVE

class ScatterElementsV2Operation(DataGen):
    @staticmethod
    def golden(in_tensors, op_params):
        json_data = json.loads(op_params)
        input_tensor = ScatterElementsV2Operation.origin_input_tensor
        indice_tensor = in_tensors[1]
        update_tensor = in_tensors[2]

        axis = json_data["axis"]
        reduction = "add" if "reduction" in json_data and json_data[ "reduction"] == 1 else None

        if reduction:
            input_tensor.scatter_(axis, indice_tensor.long(), update_tensor, reduce=reduction)
        else:
            input_tensor.scatter_(axis, indice_tensor.long(), update_tensor)
        return [input_tensor,indice_tensor,update_tensor]

    @staticmethod
    def case_preprocess(op_params, operation, input_tensor_list):
        # print(input_tensor_list[0])
        ScatterElementsV2Operation.origin_input_tensor = input_tensor_list[0].cpu().clone()

    @staticmethod
    def get_op_type(op_params):
        return OpTypes.COMPUTE_FLOAT

class GmmDeqSwigluQuantGmmDeqOperation(DataGen):
    # only for counterexample
    pass

class MmDeqSwigluQuantMmDeqOperation(DataGen):
    # only for counterexample
    pass

class RingMLAOperation(DataGen):
    @staticmethod
    def get_op_type(op_params):
        return OpTypes.CV_FUSION
