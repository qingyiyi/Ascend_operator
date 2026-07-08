#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import pandas as pd
import numpy as np
import logging
import sys
import json
import argparse
import torch
import torch_npu
import data_generation
import torch.multiprocessing as mp
import random
import re
import shutil
import scipy.stats
import pickle
import ml_dtypes

from en_dtypes import hifloat8
from multiprocessing.connection import Client

dtype_enum_dict = {-1: "undefined", 0: "float", 1: "float16", 2: "int8", 3: "int32", 4: "uint8",
                    6: "int16", 7: "uint16", 8: "uint32", 9: "int64", 10: "uint64",
                    11: "double", 12: "bool", 13: "string", 16: "complex64", 17: "complex128", 27: "bf16",
                    34: "hifloat8", 35: "float8_e5m2", 36: "float8_e4m3fn"}

format_enum_dict = {-1: "undefined", 0: "nchw", 1: "nhwc", 2: "nd", 3: "nc1hwc0",
                4: "fractal_z", 12: "nc1hwc0_c04", 16: "hwcn", 27: "ndhwc",
                29: "fractal_nz", 30: "ncdhw", 32: "ndc1hwc0", 33: "fractal_z_3d"}

err_enum_dict = {0: "NO_ERROR", 1: "ERROR_INVALID_PARAM", 2: "ERROR_INVALID_GRAPH", 3: "ERROR_INTERNAL_ERROR", 4: "ERROR_RT_FAIL", 
                 5: "ERROR_INVALID_IN_TENSOR_NUM", 6: "ERROR_INVALID_TENSOR_DTYPE", 7: "ERROR_INVALID_TENSOR_FORMAT", 8: "ERROR_INVALID_TENSOR_DIM", 
                 9: "ERROR_INVALID_TENSOR_SIZE", 10: "ERROR_OPERATION_NULL_RUNNER", 11: "ERROR_GRAPH_INFERSHAPE_FUNC_FAIL", 12: "ERROR_CANN_ERROR",
                 13: "ERROR_INVALID_TENSOR_INI_MATCH", 14: "ERROR_INVALID_TENSOR_ADDR", 15: "ERROR_INVALID_TENSOR_NUM", 16: "ERROR_INVALID_TENSOR_DIM_NUM",
                 17: "ERROR_INVALID_SINGLE_OPERATION_PARAM", 18: "ERROR_GRAPH_NODE_RESHAPE_FUNC_FAIL", 19: "ERROR_INVALID_GRAPH_NODE_CHUNK",
                 20: "ERROR_INVALID_CONTEXT_ADDR", 21: "ERROR_INVALID_STREAM", 22: "ERROR_INVALID_WORKSPACE_SIZE", 23: "ERROR_INVALID_WORKSPACE_ADDR",
                 24: "ERROR_INVALID_OPERATION_ADDR", 25: "ERROR_HCCL_FAIL", 26: "ERROR_OUT_OF_DEVICE_MEMORY", 27: "ERROR_OUT_OF_HOST_MEMORY",
                 28: "ERROR_COMM_EMPTY", 29: "ERROR_COPY_HOST_MEMORY_FAIL"}

MIN_ERR = 1e-7

class RPCProxy:
    def __init__(self, connection):
        self._connection = connection
    def __getattr__(self, name):
        def do_rpc(*args, **kwargs):
            self._connection.send(pickle.dumps((name, args, kwargs)))
            result = pickle.loads(self._connection.recv())
            if isinstance(result, Exception):
                raise result
            return result
        return do_rpc

class CsvOpsTest():
    DEFAULT_VALUE = {'DataGenType': 'customize', 'DataGenRange': '-5,5'}
    def __init__(self, operation_info_file, args):
        self.args = args
        self.operation_name = ''
        self.op_param_str = ''
        self.file_data = pd.read_csv(operation_info_file, sep = '|', skipinitialspace = True, dtype={'InShape': str, 'OutShape': str, 'TestType': str})
        self.case_num = self.file_data.loc[:, 'CaseNum']
        self.case_name = self.file_data.loc[:, 'CaseName']
        self.op_name = self.file_data.loc[:, 'OpName']
        self.op_param = self.file_data.loc[:, 'OpParam']
        self.in_num = self.file_data.loc[:, 'InNum']
        self.in_dtype = self.file_data.loc[:, 'InDType']
        self.in_format = self.file_data.loc[:, 'InFormat']
        self.in_shape = self.file_data.loc[:, 'InShape']
        self.out_num = self.file_data.loc[:, 'OutNum']
        self.out_dtype = self.file_data.loc[:, 'OutDType']
        self.out_format = self.file_data.loc[:, 'OutFormat']
        self.out_shape = self.file_data.loc[:, 'OutShape']
        self.data_gen_type = self.file_data.loc[:, 'DataGenType']
        self.data_gen_range = self.file_data.loc[:, 'DataGenRange']
        self.intensor_file = self.file_data.loc[:, 'InTensorFile']
        self.outtensor_file = self.file_data.loc[:, 'OutTensorFile']
        self.test_type = self.file_data.loc[:, 'TestType']
        self.test_level = self.file_data.loc[:, 'TestLevel']
        self.from_model = self.file_data.loc[:, 'FromModel']
        self.soc_version = self.file_data.loc[:, 'SocVersion']
        self.expected_error = self.file_data.loc[:, 'ExpectedError']
        self.file_data.loc[:, 'ActualError'] = ''
        self.file_data.loc[:, 'Result'] = ''
        self.file_data.loc[:, 'CasePassed'] = ''
        self.file_data.loc[:, 'SetupTime(us)'] = ''
        self.file_data.loc[:, 'ExecuteTime(us)'] = ''
        self.file_data.loc[:, 'SyncTime(us)'] = ''
        self.file_data.loc[:, 'TotalTime(us)'] = ''
        self.file_data.loc[:, 'Error0.1‰'] = ''
        self.file_data.loc[:, 'Error0.5‰'] = ''
        self.file_data.loc[:, 'Error1‰'] = ''
        self.file_data.loc[:, 'Error4‰'] = ''
        self.file_data.loc[:, 'Error5‰'] = ''
        self.file_data.loc[:, 'Error+/-1'] = ''
        self.file_data.loc[:, 'PrecisionPercent'] = ''
        self.file_data.loc[:, 'EBPercent'] = ''
        self.compute_num = -1

    def need_to_run_case(self, case_index, args, card_type):
        op_name = self.op_name.loc[case_index].split('_')[0]
        if args.operation_name != '' and not re.search(op_name, args.operation_name, re.I):
            logging.warning("CaseNum %d, Case %s will not run because OpName %s differs from input OperationName %s",
                            self.case_num.loc[case_index], self.case_name.loc[case_index], self.op_name.loc[case_index], args.operation_name)
            logging.info("")
            return False
        if card_type == 'single_card' and op_name in ['AllGatherOperation', 'AllGatherVOperation', 'AllReduceOperation', 'BroadcastOperation', 'ReduceScatterOperation', 'ReduceScatterVOperation', 'LinearParallelOperation','AllToAllVOperation', 'AllToAllVV2Operation', 'AllToAllOperation']:
            return False
        if card_type == 'multi_card':
            if op_name not in ['AllGatherOperation', 'AllGatherVOperation', 'AllReduceOperation', 'BroadcastOperation', 'ReduceScatterOperation', 'ReduceScatterVOperation', 'LinearParallelOperation','AllToAllVOperation', 'AllToAllVV2Operation', 'AllToAllOperation']:
                return False
            elif self.args.world_size != json.loads(self.op_param.loc[case_index])['rankSize']:
                logging.warning("CaseNum %d, Case %s will not run because args.world_size %s differs from testcase's rank_size %s",
                                self.case_num.loc[case_index], self.case_name.loc[case_index], self.args.world_size, json.loads(self.op_param.loc[case_index])['rankSize'])
                logging.info("")
                return False
        if args.test_type != '' and self.test_type.loc[case_index] != args.test_type:
            logging.warning("CaseNum %d, Case %s will not run because TestType %s differs from input TestType %s",
                            self.case_num.loc[case_index], self.case_name.loc[case_index], self.test_type.loc[case_index], args.test_type)
            logging.info("")
            return False
        if args.test_level != '' and not pd.isnull(self.test_level.loc[case_index]) and self.test_level.loc[case_index] != args.test_level:
            logging.warning("CaseNum %d, Case %s will not run because TestLevel %s differs from input TestLevel %s",
                            self.case_num.loc[case_index], self.case_name.loc[case_index], self.test_level.loc[case_index], args.test_level)
            logging.info("")
            return False
        if args.model != '' and self.from_model.loc[case_index] != args.model:
            logging.warning("CaseNum %d, Case %s will not run because FromModel %s differs from input FromModel %s",
                            self.case_num.loc[case_index], self.case_name.loc[case_index], self.from_model.loc[case_index], args.model)
            logging.info("")
            return False
        if args.soc_version != '' and not pd.isnull(self.soc_version.loc[case_index]):
            soc_version_list = self.soc_version.loc[case_index].split(',')
            if not args.soc_version in soc_version_list:
                logging.warning("CaseNum %d, Case %s will not run because SocVersion %s differs from input SocVersion %s",
                                self.case_num.loc[case_index], self.case_name.loc[case_index], self.soc_version.loc[case_index], args.soc_version)
                logging.info("")
                return False
        #CSV中soc_version为空的用例不在910A/310B上执行
        if pd.isnull(self.soc_version.loc[case_index]) and (args.soc_version == "Ascend910A" or args.soc_version == "Ascend310B"):
            logging.warning("CaseNum %d, Case %s will not run because SocVersion %s differs from input SocVersion %s",
                            self.case_num.loc[case_index], self.case_name.loc[case_index], self.soc_version.loc[case_index], args.soc_version)
            logging.info("")
            return False
        return True

    def api_data_reset(self, output_file):
        self.file_data.loc[self.index, 'ActualError'] = ''
        self.file_data.loc[self.index, 'Result'] = ''
        self.file_data.loc[self.index, 'SetupTime(us)'] = ''
        self.file_data.loc[self.index, 'ExecuteTime(us)'] = ''
        self.file_data.loc[self.index, 'SyncTime(us)'] = ''
        self.file_data.loc[self.index, 'TotalTime(us)'] = ''
        self.file_data.loc[self.index, 'PrecisionPercent'] = ''
        self.file_data.loc[self.index, 'EBPercent'] = ''
        self.file_data.loc[self.index, 'Error0.1‰'] = ''
        self.file_data.loc[self.index, 'Error0.5‰'] = ''
        self.file_data.loc[self.index, 'Error1‰'] = ''
        self.file_data.loc[self.index, 'Error4‰'] = ''
        self.file_data.loc[self.index, 'Error5‰'] = ''
        self.file_data.loc[self.index, 'Error+/-1'] = ''
        self.input_tensor_list = []
        self.output_tensor_list = []
        self.golden_output_tensor_list = []
        self.gpu_golden_output_tensor_list = []
        if self.args.save_tensor:
            output_file_path = os.path.dirname(os.path.abspath(output_file))
            self.data_save_path = output_file_path + '/csvopstest/'
            if os.path.exists(self.data_save_path):
                shutil.rmtree(self.data_save_path)
            if not os.path.exists(self.data_save_path):
                os.makedirs(self.data_save_path)

    def __dump_tensor(self, tensor, tensor_data_type, index, info):
        if tensor.dtype == torch.float8_e5m2 or tensor.dtype == torch.float8_e4m3fn or tensor.dtype == torch.bits8:
            tensor = tensor.view(torch.int8)
        if self.args.save_tensor:
            dump_path = self.data_save_path + self.case_name.loc[self.index] + '_' + \
                str(self.case_num.loc[self.index]) + '_deviceid_' + str(torch.npu.current_device()) + '_' + tensor_data_type + '_index_' + str(index)
            dump_path_pt = dump_path + '.pt'
            dump_path_txt = dump_path + '.txt'
            types =['input','golden','output']
            if tensor_data_type in types:
                torch.save(tensor, dump_path_pt)
                logging.info(f"the {tensor_data_type} tensor {index} is saved in {dump_path}.pt")
            with open(dump_path_txt, mode='a') as f:
                temp = sys.stdout
                sys.stdout = f
                f.write(info)
                f.write('\n')
                f.write(f'dtype:{tensor.dtype}, shape:{tensor.shape}')
                f.write('\n')
                print(tensor)
                f.write('\n')
                sys.stdout = temp

    def torch_operation_setup(self):
        self.operation_name = self.op_name.loc[self.index].split('_')[0]
        self.operation = torch.classes.OperationTorch.OperationTorch(self.operation_name)
        self.op_param_str = ''
        op_param_str = self.op_param.loc[self.index]
        if isinstance(op_param_str, dict):
                op_param_str = json.dumps(op_param_str)
        self.op_param_str = op_param_str
        return self.operation.set_param(self.op_param_str)

    def generate_input_tensors(self):
        if self.in_num.loc[self.index] == 0:
            logging.debug("input_tensor_list is empty")
            return
        input_dtypes_list = self.in_dtype.loc[self.index].split(';')
        input_shapes_list = self.in_shape.loc[self.index].split(';')
        input_formats_list = self.in_format.loc[self.index].split(';')
        if not pd.isnull(self.data_gen_type.loc[self.index]):
            data_gen_type_list = self.data_gen_type.loc[self.index].split(';')
        else:
            data_gen_type_list = [CsvOpsTest.DEFAULT_VALUE['DataGenType']] * self.in_num.loc[self.index]
        if not pd.isnull(self.data_gen_range.loc[self.index]):
            data_gen_ranges_list = self.data_gen_range.loc[self.index].split(';')
        else:
            data_gen_ranges_list = [CsvOpsTest.DEFAULT_VALUE['DataGenRange']] * self.in_num.loc[self.index]
        if not self.args.no_file and not pd.isnull(self.intensor_file.loc[self.index]):
            intensor_file_list = self.intensor_file.loc[self.index].split(';')
        else:
            intensor_file_list = [''] * self.in_num.loc[self.index]
        shapes = [[int(s) for s in shape_str.split(",")] for shape_str in input_shapes_list]
        for i in range(self.in_num.loc[self.index]):
            if intensor_file_list[i] != '':
                tensor_gen_func = 'data_generation.' + self.operation_name + '.load_tensor_from_file'
                input_tensor = eval(tensor_gen_func)(intensor_file_list[i], i, self.op_param_str)
            else:
                dtype = input_dtypes_list[i]
                tensor_gen_func = 'data_generation.' + self.operation_name + '.' + data_gen_type_list[i]
                if data_gen_type_list[i] in ['one', 'zero', 'random']:
                    input_tensor = eval(tensor_gen_func)(shapes[i], dtype, input_formats_list[i], data_gen_ranges_list[i], self.op_param_str)
                elif data_gen_type_list[i] == 'customize':
                    input_tensor = eval(tensor_gen_func)(shapes, i, dtype, input_formats_list[i], data_gen_ranges_list[i], self.op_param_str)
                else:
                    logging.error("DataGenType only supports one/zero/random/customize but for this case it is %s", data_gen_type_list[i])
                    quit(1)
            if self.args.soc_version == "Ascend950" and torch_npu.get_npu_format(input_tensor).name == "FRACTAL_NZ":
                pass
            else:
                self.__dump_tensor(input_tensor.cpu(), 'input', i, 'index: {}'.format(i))
            self.input_tensor_list.append(input_tensor)
        if self.operation_name == "LinearOperation":
            json_data = json.loads(self.op_param_str)
            transpose_a = json_data["transposeA"] if "transposeA" in json_data else False
            self.compute_num = shapes[0][-2] if transpose_a else shapes[0][-1]
        if self.operation_name == "LinearParallelOperation":
            json_data = json.loads(self.op_param_str)
            transpose_a = json_data["transposeA"] if "transposeA" in json_data else False
            self.compute_num = shapes[0][-2] * 2 if transpose_a else shapes[0][-1] * 2

    def generate_output_tensors(self, infershape_result):
        output_num = 0
        output_dtypes_list = []
        output_shapes_list = []
        output_formats_list = []
        if self.test_type.loc[self.index] == 'Generalization':
            out_tensor_data = json.loads(infershape_result)
            output_num = out_tensor_data["num"]
            self.file_data.loc[self.index, 'OutNum'] = output_num
            if output_num == 0:
                logging.debug("output_tensor_list is empty")
                return
            output_dtypes_list = [dtype_enum_dict[dtype] for dtype in out_tensor_data['dtype']]
            output_shapes_list = [','.join(str(s) for s in shape) for shape in out_tensor_data['shape']]
            output_formats_list = [format_enum_dict[format] for format in out_tensor_data['format']]
            self.file_data.loc[self.index, 'OutDType'] = ";".join(output_dtypes_list)
            self.file_data.loc[self.index, 'OutFormat'] = ";".join(output_formats_list)
            self.file_data.loc[self.index, 'OutShape'] = ";".join(output_shapes_list)
        else:
            output_num = self.out_num.loc[self.index]
            if output_num == 0:
                logging.debug("output_tensor_list is empty")
                return
            output_dtypes_list = self.out_dtype.loc[self.index].split(';')
            output_shapes_list = self.out_shape.loc[self.index].split(';')
            output_formats_list = self.out_format.loc[self.index].split(';')
        for i in range(output_num):
            dtype = output_dtypes_list[i]
            shape_str = output_shapes_list[i]
            shape = [int(s) for s in shape_str.split(",")]
            tensor_gen_func = 'data_generation.' + self.operation_name + '.zero'
            output_tensor = eval(tensor_gen_func)(shape, dtype, output_formats_list[i], "", self.op_param_str)
            self.output_tensor_list.append(output_tensor)

    def analyse_result(self):
        get_op_type_func = 'data_generation.' + self.operation_name + '.get_op_type'
        self.op_type = eval(get_op_type_func)(self.op_param_str)
        if self.args.precision_standard == 'new' and self.op_type == data_generation.OpTypes.NA:
            logging.error(f"{self.operation_name}'s precision_standard is 'new', so its op_type(from get_op_type method) cannot be 'OpTypes.NA'!")
            exit(1)
        if self.args.precision_standard == 'new' and self.op_type == data_generation.OpTypes.CV_FUSION:
            if self.args.gpu_info == '':
                self.op_type = data_generation.OpTypes.VECTOR_FUSION
                logging.debug("gpu info is not provided, use VECTOR_FUSION standard instead of CV_FUSION standard")
        logging.debug(f"{self.operation_name}'s precision_standard is: {self.args.precision_standard}")
        self.generate_golden_output_tensors()
        self.precision_performance_analysis()

    def get_json_result(self, json_result, running_stage, times):
        json_data = json.loads(json_result)
        result_flag = True
        self.file_data.loc[self.index, 'ActualError'] = err_enum_dict[json_data['result']]
        if self.file_data.loc[self.index, 'ActualError'] != "NO_ERROR":
            result_flag = False
        if self.expected_error.loc[self.index] != "NO_ERROR" and self.expected_error.loc[self.index].startswith(running_stage):
            result_flag = False
        if result_flag:
            self.file_data.loc[self.index, 'Result'] = 'succ'
            if running_stage == "S":
                self.file_data.loc[self.index, 'SetupTime(us)'] = json_data['setup_time']
                self.workspace_size = json_data['workspace_size']
            elif running_stage == "E":
                self.file_data.loc[self.index, 'ExecuteTime(us)'] = json_data['execute_time']
                self.file_data.loc[self.index, 'SyncTime(us)'] = json_data['sync_time']
        else:
            self.file_data.loc[self.index, 'Result'] = 'fail'
            result_flag = False
            self.file_data.loc[self.index, 'ActualError'] = running_stage + ':' + err_enum_dict[json_data['result']]
            logging.info("CaseNum " + str(self.case_num.loc[self.index]) + ", Case " + self.case_name.loc[self.index] + " times " + str(times) + " end, "
                "result: " + self.file_data.loc[self.index, 'Result'])
        logging.debug("Stage " + running_stage + " end, ActualError: " + self.file_data.loc[self.index, 'ActualError'] + ", ExpectedError: " + self.expected_error.loc[self.index])
        return result_flag

    def run_one_case(self, index, times, output_file):
        logging.info("CaseNum " + str(self.case_num.loc[index]) + ", Case " + self.case_name.loc[index] + " times " + str(times) + " start...")
        self.index = index
        if self.file_data.loc[self.index, 'Result'] != 'succ':
            self.create_op_result = self.torch_operation_setup()
            logging.debug("create_op_result:%s", self.create_op_result)
            if (not self.get_json_result(self.create_op_result, "C", times)):
                return
        self.api_data_reset(output_file)

        self.generate_input_tensors()
        case_preprocess_func = 'data_generation.' + self.operation_name + '.case_preprocess'
        eval(case_preprocess_func)(self.op_param_str, self.operation, self.input_tensor_list)

        infershape_result = self.operation.infer_shape(self.input_tensor_list)
        logging.debug("infershape_result:%s", infershape_result)
        if (not self.get_json_result(infershape_result, "I", times)):
            return

        self.generate_output_tensors(infershape_result)
        setup_result = self.operation.setup(self.input_tensor_list, self.output_tensor_list)
        logging.debug("setup_result:%s", setup_result)
        if (not self.get_json_result(setup_result, "S", times)):
            return

        case_postprocess_func = 'data_generation.' + self.operation_name + '.case_postprocess'
        eval(case_postprocess_func)(self.op_param_str, self.operation, self.input_tensor_list, self.output_tensor_list)

        execute_result = self.operation.execute_sync(self.input_tensor_list, self.output_tensor_list, self.workspace_size)
        logging.debug("execute_result:%s", execute_result)
        if (not self.get_json_result(execute_result, "E", times)):
            return
        if not self.args.skip_verify:
            self.analyse_result()
        logging.info("CaseNum " + str(self.case_num.loc[index]) + ", Case " + self.case_name.loc[index] + " times " + str(times) + " end, "
            "result: " + self.file_data.loc[index, 'Result'])

    def generate_golden_output_tensors(self):
        golden_input_tensors = []
        for tensor in self.input_tensor_list:
            if self.args.soc_version == "Ascend950" and torch_npu.get_npu_format(tensor).name == "FRACTAL_NZ":
                pass
            else:
                dtype = tensor.dtype
                golden_input_tensors.append(tensor.cpu().to(dtype))
        golden_tensor_gen_func = 'data_generation.' + self.operation_name + '.golden'
        golden_output_tensors = eval(golden_tensor_gen_func)(golden_input_tensors, self.op_param_str)
        for i in range(len(golden_output_tensors)):
            self.golden_output_tensor_list.append(golden_output_tensors[i])
        if self.operation_name == "SelfAttentionOperation" and len(golden_output_tensors) == 2:
            self.gpu_golden_output_tensor_list = [golden_output_tensors[1]]
            self.golden_output_tensor_list.pop()
            self.op_type = data_generation.OpTypes.CV_FUSION
            json_data = json.loads(self.op_param_str)
            if json_data["maskType"] != 0:
                kv_seqLen_tesnor = golden_input_tensors[4]
            else:
                kv_seqLen_tesnor = golden_input_tensors[3]
            kv_seqLen = kv_seqLen_tesnor.numpy()
            self.max_seq = np.max(kv_seqLen)
            self.heads = json_data['headNum']
        elif self.args.precision_standard == 'new' and self.op_type == data_generation.OpTypes.CV_FUSION:
            ip, port = self.args.gpu_info.split(':')
            logging.debug("start to get gpu golden result")
            client = Client((ip, int(port)))
            proxy = RPCProxy(client)
            self.gpu_golden_output_tensor_list = eval('proxy.' + self.operation_name)(golden_input_tensors, self.op_param_str)

    def __error_percent(self, i, actual_output, golden_output, atol: float, rtol: float):
        actual_output = actual_output if actual_output.dtype != torch.bool else actual_output.long()
        golden_output = golden_output if golden_output.dtype != torch.bool else golden_output.long()
        actual_output = torch.where(torch.isnan(actual_output), torch.full_like(actual_output, 0), actual_output)
        actual_output = torch.where(torch.isinf(actual_output), torch.full_like(actual_output, 0), actual_output)
        golden_output = torch.where(torch.isnan(golden_output), torch.full_like(golden_output, 0), golden_output)
        golden_output = torch.where(torch.isinf(golden_output), torch.full_like(golden_output, 0), golden_output)
        diff = torch.subtract(actual_output, golden_output)
        self.__dump_tensor(diff, 'diff', i, 'index: {}, atol: {}, rtol: {}'.format(i, atol, rtol))
        tolerance = torch.subtract(torch.abs(diff), atol + rtol * torch.abs(golden_output))
        self.__dump_tensor(tolerance, 'tolerance', i, 'index {}, atol: {}, rtol: {}'.format(i, atol, rtol))
        return str(torch.sum(tolerance <= 0).numpy() / torch.numel(tolerance) * 100)[:5]

    def __precision_eb_percent(self, i, actual_output, golden_output, precision_threshold, eb_threshold):
        if actual_output.dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
            actual_output = actual_output.view(torch.int8)
            actual_output = actual_output.flatten()
            golden_output = golden_output.view(torch.int8)
            golden_output = golden_output.flatten()
            diff_results = torch.abs(torch.subtract(actual_output, golden_output))
            diff_indices = torch.where(diff_results > 1)[0]
            del diff_results
            precision = (golden_output.size()[0] - diff_indices.size()[0]) / golden_output.size()[0]
            eb = eb_threshold
            if eb_threshold != 0:
                eb = torch.abs(torch.mean(torch.div(diff, tensor_max)))
                self.__dump_tensor(eb, 'eb', i, 'index {}, eb_threshold: {}'.format(i, eb_threshold))
            eb_percent = '0' if eb == 0 else str(torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100)[:5]
            return str(precision*100)[:5], eb_percent

        elif actual_output.dtype in [torch.bits8]:
            golden_output = np.float64(golden_output.detach().numpy().view(hifloat8))
            golden_output = golden_output.flatten()
            actual_output = actual_output.view(torch.int8)
            actual_output = np.float64(actual_output.detach().numpy().view(hifloat8))
            actual_output = actual_output.flatten()
            EX = np.log2(abs(golden_output) + 2**(-1000))
            EX[EX < -22] = -22
            E = np.floor(EX)
            Eabs = np.abs(E)
            Wm = np.zeros_like(golden_output)
            Wm[Eabs <= 15] = 1
            Wm[Eabs <= 7 ] = 2
            Wm[Eabs <= 3 ] = 3
            ulp_err = (actual_output - golden_output) * 2 ** (-E + Wm)
            S_EX = EX * np.where(actual_output >= 0, 1, -1)
            EH = np.log2(abs(golden_output) + 2**(-1000))
            S_EH = EH * np.where(golden_output >= 0, 1, -1)
            ulp_err1 = S_EH - S_EX
            ulp_err[Wm == 0] = ulp_err1[Wm == 0]
            diff_indices = np.where( ulp_err > 1)[0]
            precision = (len(golden_output) - len(diff_indices)) / len(golden_output)
            eb = eb_threshold
            if eb_threshold != 0:
                eb = torch.abs(torch.mean(torch.div(diff, tensor_max)))
                self.__dump_tensor(eb, 'eb', i, 'index {}, eb_threshold: {}'.format(i, eb_threshold))
            eb_percent = '0' if eb == 0 else str(torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100)[:5]
            return str(precision * 100), eb_percent
        actual_output = actual_output if actual_output.dtype != torch.bool else actual_output.long()
        golden_output = golden_output if golden_output.dtype != torch.bool else golden_output.long()
        if self.op_type in [data_generation.OpTypes.COMPUTE_FLOAT, data_generation.OpTypes.COMPUTE_FLOAT_HIGH_PRECISION, data_generation.OpTypes.VECTOR_FUSION] and actual_output.dtype in [torch.float16, torch.bfloat16]:
            actual_output = actual_output.to(torch.float32)
            golden_output = golden_output.to(torch.float32)
        actual_output = torch.where(torch.isnan(actual_output), torch.full_like(actual_output, 0), actual_output)
        actual_output = torch.where(torch.isinf(actual_output), torch.full_like(actual_output, 0), actual_output)
        golden_output = torch.where(torch.isnan(golden_output), torch.full_like(golden_output, 0), golden_output)
        golden_output = torch.where(torch.isinf(golden_output), torch.full_like(golden_output, 0), golden_output)
        if self.op_type == data_generation.OpTypes.RAND:
            alpha = 0.01
            t_statistic, p_value = scipy.stats.ks_2samp(actual_output, golden_output)
            logging.debug(f"p_value: {p_value}")
            precision_percent = '100' if p_value > alpha else '0'
            eb_percent = '0'
            return precision_percent, eb_percent
        diff = torch.subtract(actual_output, golden_output)
        tensor_max = torch.maximum(torch.ones(golden_output.shape, dtype=golden_output.dtype), torch.abs(golden_output))
        if self.op_type in [data_generation.OpTypes.CV_FUSION]:
            cv_err = {torch.float16: 2**(-11), torch.bfloat16: 2**(-8), torch.float32: 2**(-14)}
            gpu_golden_output = self.gpu_golden_output_tensor_list[i]
            if actual_output.dtype in [torch.float16, torch.bfloat16]:
                gpu_golden_output = torch.where(torch.isnan(gpu_golden_output), torch.full_like(gpu_golden_output, 0), gpu_golden_output)
                gpu_golden_output = torch.where(torch.isinf(gpu_golden_output), torch.full_like(gpu_golden_output, 0), gpu_golden_output)
            def get_eb_threshold(dtype:torch.dtype):
                eb_threshold = 0
                if dtype in [torch.bfloat16]:
                    eb_threshold = 2**(-7)
                if dtype in [torch.float16]:
                    eb_threshold = 2**(-10)
                if dtype in [torch.float32]:
                    eb_threshold = 2**(-14)
                return eb_threshold

            def get_err_threshold(op_type, dtype:torch.dtype):
                err_threshold = 0
                if op_type in [data_generation.OpTypes.MOVE, data_generation.OpTypes.RAND, data_generation.OpTypes.CAST, data_generation.OpTypes.COMPUTE_INTEGER]:
                    pass
                if op_type in [data_generation.OpTypes.COMPUTE_QUANT, data_generation.OpTypes.COMPUTE_FLOAT]:
                    if dtype in [torch.bfloat16]:
                        err_threshold = 2**(-7)
                    if dtype in [torch.float16]:
                        err_threshold = 2**(-8)
                    if dtype in [torch.float32]:
                        err_threshold = 2**(-11)
                if op_type in [data_generation.OpTypes.CV_FUSION]:
                    if dtype in [torch.bfloat16]:
                        err_threshold = 2**(-8)
                    if dtype in [torch.float16]:
                        err_threshold = 2**(-11)
                    if dtype in [torch.float32]:
                        err_threshold = 2**(-14)
                return err_threshold
            #误差均衡性（EB）
            def get_eb(golden:torch.Tensor, actual:torch.Tensor):
                golden = golden.to(torch.float32)
                golden_nmax = torch.clamp(torch.abs(golden), min = 1)
                actual_error = actual.to(torch.float32) - golden
                EB = torch.mean(actual_error / golden_nmax)
                return EB
            #最大相对误差：max relative error，MARE
            def get_mare(golden:torch.Tensor, actual:torch.Tensor):
                golden = golden.to(torch.float32)
                abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
                mare = torch.max(abs_error.flatten())
                return mare

            #平均相对误差：mean relative error，MERE
            def get_mere(golden:torch.Tensor, actual:torch.Tensor):
                golden = golden.to(torch.float32)
                abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
                mere = torch.mean(abs_error)
                return mere

            #均方根误差:Root Mean Squared Error，RMSE
            def get_rmse(golden:torch.Tensor, actual:torch.Tensor):
                golden = golden.to(torch.float32)
                sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
                rmse = torch.sqrt(torch.mean(sqr_err))
                return rmse
            # actual_output为npu对应ATB的输出，golden_output为cpu对应的真值，gpu_golden_output为gpu对应的实现输出
            eb_threshold = get_eb_threshold(actual_output.dtype)
            err_threshold = get_err_threshold(self.op_type, actual_output.dtype)
            logging.info(f"err_threshold:{err_threshold} eb_threshold:{eb_threshold}")
            mare_npu = get_mare(golden_output, actual_output)
            mare_gpu = get_mare(golden_output, gpu_golden_output)

            mere_npu = get_mere(golden_output, actual_output)
            mere_gpu = get_mere(golden_output, gpu_golden_output)

            rmse_npu = get_rmse(golden_output, actual_output)
            rmse_gpu = get_rmse(golden_output, gpu_golden_output)

            mare_rate = mare_npu / max(mare_gpu, err_threshold)
            mere_rate = mere_npu / max(mere_gpu, err_threshold)
            rmse_rate = rmse_npu / max(rmse_gpu, err_threshold)
            EB = get_eb(gpu_golden_output, actual_output)
            def compare_output_data(out, golden, ratios):
                error_count = 0
                strict_error_count = 0
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
                if self.op_type == torch.bfloat16:
                    logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[2])
                else:
                    logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[0])
                calc_times = self.heads * self.max_seq + 4
                if self.op_type == torch.bfloat16:
                    if calc_times < 2048:
                        error = 2**(-7)
                    else :
                        error = 2**(-6)
                    error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
                    res = (diff <= error_threshold).all().item()
                    logging.debug("accuracy is correct in new standard: %r", res)
                    return res
                elif self.op_type == torch.float16:
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
            result = ((mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2)) or (compare_output_data(actual_output.half(), golden_output.half(), [0.001, 0.001, 0.005, 0.005]))
            logging.info(f"mare_npu:{mare_npu} mare_gpu:{mare_gpu}")
            logging.info(f"mere_npu:{mere_npu} mere_gpu:{mere_gpu}")
            logging.info(f"rmse_npu:{rmse_npu} rmse_gpu:{rmse_gpu}")
            logging.info(f"MARE:{mare_rate} MERE:{mere_rate} RMSE:{rmse_rate} EB:{EB}")
            logging.info(f"new golden cv result:{result}")
            precision_percent = "100.0" if result else "99.9"
        else:
            self.__dump_tensor(diff, 'diff', i, 'index: {}, precision_threshold: {}'.format(i, precision_threshold))
            if precision_threshold == 1:
                tolerance = torch.subtract(torch.abs(diff), torch.ones(diff.shape, dtype=diff.dtype))
            else:
                tolerance = torch.subtract(torch.abs(diff), precision_threshold * tensor_max)
            self.__dump_tensor(tolerance, 'tolerance', i, 'index {}, precision_threshold: {}'.format(i, precision_threshold))
            precision_percent = str(torch.sum(tolerance <= 0).numpy() / torch.numel(tolerance) * 100)[:5]
        # calculate EB
        eb = eb_threshold
        if eb_threshold != 0:
            eb = torch.abs(torch.mean(torch.div(diff, tensor_max)))
            self.__dump_tensor(eb, 'eb', i, 'index {}, eb_threshold: {}'.format(i, eb_threshold))
        eb_percent = '0' if eb == 0 else str(torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100)[:5]
        return precision_percent, eb_percent
        
    def precision_performance_analysis(self):
        if (len(self.output_tensor_list) == 0):
            self.output_tensor_list = self.input_tensor_list
        for i in range(len(self.golden_output_tensor_list)):
            actual_output = self.output_tensor_list[i].cpu()
            self.__dump_tensor(actual_output, 'output', i, 'index: {}'.format(i))
            golden_output = self.golden_output_tensor_list[i]
            self.__dump_tensor(golden_output, 'golden', i, 'index: {}'.format(i))
            if self.args.precision_standard == 'new':
                if self.op_type in [data_generation.OpTypes.CV_FUSION]:
                    gpu_golden_output = self.gpu_golden_output_tensor_list[i]
                    self.__dump_tensor(gpu_golden_output, 'gpu_golden', i, 'index: {}'.format(i))
                precision_threshold, eb_threshold = data_generation.get_precision_and_eb_threshold(self.op_type, actual_output.dtype, self.compute_num)
                precision, eb = self.__precision_eb_percent(i, actual_output, golden_output, precision_threshold, eb_threshold)
                self.file_data.loc[self.index, 'PrecisionPercent'] += precision + ';'
                self.file_data.loc[self.index, 'EBPercent'] += eb + ';'
            else:
                self.file_data.loc[self.index, 'Error0.1‰'] += self.__error_percent(i, actual_output, golden_output, 0.0001, 0.0001) + ';'
                self.file_data.loc[self.index, 'Error0.5‰'] += self.__error_percent(i, actual_output, golden_output, 0.0005, 0.0005) + ';'
                self.file_data.loc[self.index, 'Error1‰'] += self.__error_percent(i, actual_output, golden_output, 0.001, 0.001) + ';'
                self.file_data.loc[self.index, 'Error4‰'] += self.__error_percent(i, actual_output, golden_output, 0.004, 0.004) + ';'
                self.file_data.loc[self.index, 'Error5‰'] += self.__error_percent(i, actual_output, golden_output, 0.005, 0.005) + ';'
                self.file_data.loc[self.index, 'Error+/-1'] += self.__error_percent(i, actual_output, golden_output, 1, 0) + ';'

        if self.args.precision_standard == 'new':
            self.file_data.loc[self.index, 'PrecisionPercent'] = self.file_data.loc[self.index, 'PrecisionPercent'][:-1]
            self.file_data.loc[self.index, 'EBPercent'] = self.file_data.loc[self.index, 'EBPercent'][:-1]
        else:
            self.file_data.loc[self.index, 'Error0.1‰'] = self.file_data.loc[self.index, 'Error0.1‰'][:-1]
            self.file_data.loc[self.index, 'Error0.5‰'] = self.file_data.loc[self.index, 'Error0.5‰'][:-1]
            self.file_data.loc[self.index, 'Error1‰'] = self.file_data.loc[self.index, 'Error1‰'][:-1]
            self.file_data.loc[self.index, 'Error4‰'] = self.file_data.loc[self.index, 'Error4‰'][:-1]
            self.file_data.loc[self.index, 'Error5‰'] = self.file_data.loc[self.index, 'Error5‰'][:-1]
            self.file_data.loc[self.index, 'Error+/-1'] = self.file_data.loc[self.index, 'Error+/-1'][:-1]
        if self.args.save_tensor:
            logging.info("tensor data is saved to " + self.data_save_path)

class CsvOpsResult():
    SKIP_PERFORMANCE_CASE_NUM = 200
    precision_standard = {'double': ['Error0.1‰', 99.99], 'uint32': ['Error0.1‰', 99.99], 'int64': ['Error0.1‰', 99.99],
                          'float': ['Error0.1‰', 99.99], 'int32': ['Error0.1‰', 99.99], 'uint64': ['Error0.1‰', 99.99],
                          'float16': ['Error1‰', 99.9], 'bf16': ['Error4‰', 99.6], 'int8': ['Error+/-1', 99.9],
                          'uint8': ['Error+/-1', 99], 'int16': ['Error+/-1', 99.9], 'uint16': ['Error+/-1', 99.9],
                          'bool': ['Error0.1‰', 100]}

    def __init__(self, opsTest):
        self.opsTest = opsTest

    def reset(self, index):
            self.result = {"succ": 0, "fail": 0, "SetupTime(us)": 0, "ExecuteTime(us)": 0, "SyncTime(us)": 0,
                           "Error0.1‰": '', "Error0.5‰": '', "Error1‰": '', "Error4‰": '', "Error5‰": '', "Error+/-1": '', 
                           "PrecisionPercent": '', "EBPercent": '', "FirstErrType": self.opsTest.file_data.loc[index, 'ExpectedError']}

    def add_list(self, list_index, list_key, list_str):
        add_str = ''
        list1_num = self.result[list_key].split(';')
        list2_num = list_str.split(';')
        for i in range(self.compare_tensor_num):
            add_str += str(float(list1_num[i]) + float(list2_num[i])) + ';'
        self.result[list_key] = add_str[:-1]

    def div_list(self, list_index, list_key, div_num):
        div_str = ''
        list_num = self.result[list_key].split(';')
        for i in range(self.compare_tensor_num):
            div_str += str(round((float(list_num[i]) / div_num), 2)) + '%;'
        self.opsTest.file_data.loc[list_index, list_key] = div_str[:-1]

    def add_result(self, opsTest):
        if (opsTest.file_data.loc[opsTest.index, 'Result'] == 'succ'):
            if self.result['succ'] == 0 and not opsTest.args.skip_verify:
                self.compare_tensor_num = opsTest.out_num.loc[opsTest.index] if opsTest.out_num.loc[opsTest.index] > 0 else opsTest.in_num.loc[opsTest.index]
                self.compare_dtypes = opsTest.out_dtype.loc[opsTest.index] if opsTest.out_num.loc[opsTest.index] > 0 else opsTest.in_dtype.loc[opsTest.index]
                for i in range(self.compare_tensor_num):
                    if self.opsTest.args.precision_standard == 'new':
                        self.result['PrecisionPercent'] += '0' + ';'
                        self.result['EBPercent'] += '0' + ';'
                    else:
                        self.result['Error0.1‰'] += '0' + ';'
                        self.result['Error0.5‰'] += '0' + ';'
                        self.result['Error1‰'] += '0' + ';'
                        self.result['Error4‰'] += '0' + ';'
                        self.result['Error5‰'] += '0' + ';'
                        self.result['Error+/-1'] += '0' + ';'
                if self.opsTest.args.precision_standard == 'new':
                    self.result['PrecisionPercent'] = self.result['PrecisionPercent'][:-1]
                    self.result['EBPercent'] = self.result['EBPercent'][:-1]
                else:
                    self.result['Error0.1‰'] = self.result['Error0.1‰'][:-1]
                    self.result['Error0.5‰'] = self.result['Error0.5‰'][:-1]
                    self.result['Error1‰'] = self.result['Error1‰'][:-1]
                    self.result['Error4‰'] = self.result['Error4‰'][:-1]
                    self.result['Error5‰'] = self.result['Error5‰'][:-1]
                    self.result['Error+/-1'] = self.result['Error+/-1'][:-1]
            self.result['succ'] += 1
            if opsTest.args.skip_verify:
                return
            if self.opsTest.test_type.loc[opsTest.index] == "Performance" and self.result['succ'] <= CsvOpsResult.SKIP_PERFORMANCE_CASE_NUM:
                return
            self.result['SetupTime(us)'] += opsTest.file_data.loc[opsTest.index, 'SetupTime(us)']
            self.result['ExecuteTime(us)'] += opsTest.file_data.loc[opsTest.index, 'ExecuteTime(us)']
            self.result['SyncTime(us)'] += opsTest.file_data.loc[opsTest.index, 'SyncTime(us)']
            if self.opsTest.args.precision_standard == 'new':
                self.add_list(opsTest.index, 'PrecisionPercent', opsTest.file_data.loc[opsTest.index, 'PrecisionPercent'])
                self.add_list(opsTest.index, 'EBPercent', opsTest.file_data.loc[opsTest.index, 'EBPercent'])
            else:
                self.add_list(opsTest.index, 'Error0.1‰', opsTest.file_data.loc[opsTest.index, 'Error0.1‰'])
                self.add_list(opsTest.index, 'Error0.5‰', opsTest.file_data.loc[opsTest.index, 'Error0.5‰'])
                self.add_list(opsTest.index, 'Error1‰', opsTest.file_data.loc[opsTest.index, 'Error1‰'])
                self.add_list(opsTest.index, 'Error4‰', opsTest.file_data.loc[opsTest.index, 'Error4‰'])
                self.add_list(opsTest.index, 'Error5‰', opsTest.file_data.loc[opsTest.index, 'Error5‰'])
                self.add_list(opsTest.index, 'Error+/-1', opsTest.file_data.loc[opsTest.index, 'Error+/-1'])
        elif (opsTest.file_data.loc[opsTest.index, 'Result'] == 'fail'):
            self.result['fail'] += 1
            if (self.result['FirstErrType'] == self.opsTest.file_data.loc[opsTest.index, 'ExpectedError']):
                self.result['FirstErrType'] = opsTest.file_data.loc[opsTest.index, 'ActualError']

    def __precision_check(self, case_sum):
        compare_dtype_list = self.compare_dtypes.split(";")
        for i in range(len(compare_dtype_list)):
            if self.opsTest.args.precision_standard == 'new':
                precision_percent = [(float(percent[:-1])) for percent in self.opsTest.file_data.loc[self.opsTest.index, 'PrecisionPercent'].split(";")]
                eb_percent = [(float(percent[:-1])) for percent in self.opsTest.file_data.loc[self.opsTest.index, 'EBPercent'].split(";")]
                if self.opsTest.op_type == data_generation.OpTypes.RAND:
                    alpha = 0.01
                    z = -3.0902
                    case_pass_percent = ((1 - alpha) + z * np.sqrt(alpha * (1 - alpha) / case_sum)) * 100
                    if precision_percent[i] < case_pass_percent:
                        self.opsTest.file_data.loc[self.opsTest.index, 'CasePassed'] = 'No'
                        self.opsTest.file_data.loc[self.opsTest.index, 'ActualError'] = 'precision_percent {}% under case_pass_percent {}%, op type: {}'.format(precision_percent[i], str(case_pass_percent)[:5], self.opsTest.op_type)
                        return False
                elif precision_percent[i] < 100:
                    self.opsTest.file_data.loc[self.opsTest.index, 'CasePassed'] = 'No'
                    self.opsTest.file_data.loc[self.opsTest.index, 'ActualError'] = 'precision under 100%, dtype: {}, op type: {}'.format(compare_dtype_list[i], self.opsTest.op_type)
                    return False
                if eb_percent[i] > 100:
                    self.opsTest.file_data.loc[self.opsTest.index, 'CasePassed'] = 'No'
                    self.opsTest.file_data.loc[self.opsTest.index, 'ActualError'] = 'eb over 100%, dtype: {}, op type: {}'.format(compare_dtype_list[i], self.opsTest.op_type)
                    return False
            else:
                error_percent = [(float(percent[:-1])) for percent in self.opsTest.file_data.loc[self.opsTest.index, CsvOpsResult.precision_standard[compare_dtype_list[i]][0]].split(";")]
                if error_percent[i] < CsvOpsResult.precision_standard[compare_dtype_list[i]][1]:
                    self.opsTest.file_data.loc[self.opsTest.index, 'CasePassed'] = 'No'
                    self.opsTest.file_data.loc[self.opsTest.index, 'ActualError'] = 'precision check fail, dtype: {}, standard: {}'.format(compare_dtype_list[i], CsvOpsResult.precision_standard[compare_dtype_list[i]])
                    return False
        return True

    def __performance_check(self):
        performance_threshold = eval('data_generation.' + self.opsTest.operation_name + '.performance_threshold')(self.opsTest.op_param.loc[self.opsTest.index])
        for key, thre in performance_threshold.items():
            value = self.opsTest.file_data.loc[self.opsTest.index, key]
            if int(value) > thre:
                self.opsTest.file_data.loc[self.opsTest.index, 'CasePassed'] = 'No'
                self.opsTest.file_data.loc[self.opsTest.index, 'ActualError'] = 'performance check fail, stage: {}, value: {}, threshold: {}'.format(key, value, thre)
                return False
        return True

    def compute_result(self, case_index):
        self.opsTest.file_data.loc[case_index, 'Result'] = \
            "succ:" + str(self.result['succ']) + \
            " fail:" + str(self.result['fail'])
        if (self.opsTest.file_data.loc[case_index, 'ExpectedError'] == 'NO_ERROR' and self.result['fail'] > 0):
            self.opsTest.file_data.loc[case_index, 'CasePassed'] = 'No'
        elif (self.opsTest.file_data.loc[case_index, 'ExpectedError'] != 'NO_ERROR' and (self.result['succ'] > 0
            or self.result['FirstErrType'] != self.opsTest.file_data.loc[case_index, 'ExpectedError'])):
            self.opsTest.file_data.loc[case_index, 'CasePassed'] = 'No'
        else:
            self.opsTest.file_data.loc[case_index, 'CasePassed'] = 'Yes'
        self.opsTest.file_data.loc[case_index, 'ActualError'] = self.result['FirstErrType']
        case_sum = self.result['succ']
        if self.opsTest.test_type.loc[self.opsTest.index] == "Performance":
            case_sum -= CsvOpsResult.SKIP_PERFORMANCE_CASE_NUM
        if (case_sum > 0 and not self.opsTest.args.skip_verify):
            self.opsTest.file_data.loc[case_index, 'SetupTime(us)'] = str(int(self.result['SetupTime(us)'] / case_sum))
            self.opsTest.file_data.loc[case_index, 'ExecuteTime(us)'] = str(int(self.result['ExecuteTime(us)'] / case_sum))
            self.opsTest.file_data.loc[case_index, 'SyncTime(us)'] = str(int(self.result['SyncTime(us)'] / case_sum))
            self.opsTest.file_data.loc[case_index, 'TotalTime(us)'] = str((int(self.result['SetupTime(us)']) + int(self.result['ExecuteTime(us)']) + int(self.result['SyncTime(us)'])) / case_sum)
            if self.opsTest.args.precision_standard == 'new':
                self.div_list(case_index, 'PrecisionPercent', case_sum)
                self.div_list(case_index, 'EBPercent', case_sum)
            else:
                self.div_list(case_index, 'Error0.1‰', case_sum)
                self.div_list(case_index, 'Error0.5‰', case_sum)
                self.div_list(case_index, 'Error1‰', case_sum)
                self.div_list(case_index, 'Error4‰', case_sum)
                self.div_list(case_index, 'Error5‰', case_sum)
                self.div_list(case_index, 'Error+/-1', case_sum)
            if self.__precision_check(case_sum):
                if self.opsTest.test_type.loc[self.opsTest.index] == "Performance":
                    self.__performance_check()
        logging.info("CaseNum " + str(self.opsTest.case_num.loc[case_index]) + ", Case " + self.opsTest.case_name.loc[case_index] + " end, "
            "Case Passed: " + self.opsTest.file_data.loc[case_index, 'CasePassed'])
        if self.opsTest.file_data.loc[case_index, 'CasePassed'] == 'No':
            logging.error("CaseNum " + str(self.opsTest.case_num.loc[case_index]) + ", Case " + self.opsTest.case_name.loc[case_index] +
                " ActualError: " + self.opsTest.file_data.loc[case_index, 'ActualError'])

    def __need_to_save_case(self, data, index, args, card_type):
        op_name = data.loc[index, "OpName"].split('_')[0]
        if args.operation_name != '' and not re.search(op_name, args.operation_name, re.I):
            return False
        if card_type == 'single_card' and op_name in ['AllGatherOperation', 'AllGatherVOperation', 'AllReduceOperation', 'BroadcastOperation', 'ReduceScatterOperation', 'ReduceScatterVOperation', 'LinearParallelOperation','AllToAllVOperation','AllToAllOperation', 'AllToAllVV2Operation']:
            return False
        if card_type == 'multi_card' and op_name not in ['AllGatherOperation', 'AllGatherVOperation', 'AllReduceOperation', 'BroadcastOperation', 'ReduceScatterOperation', 'ReduceScatterVOperation', 'LinearParallelOperation','AllToAllVOperation','AllToAllOperation', 'AllToAllVV2Operation']:
            return False
        if args.test_type != '' and data.loc[index, "TestType"] != args.test_type:
            return False
        if args.test_level != '' and not pd.isnull(data.loc[index, "TestLevel"]) and data.loc[index, "TestLevel"] != args.test_level:
            return False
        if args.model != '' and data.loc[index, "FromModel"] != args.model:
            return False
        if args.soc_version != '' and not pd.isnull(data.loc[index, "SocVersion"]):
            soc_version_list = data.loc[index, "SocVersion"].split(',')
            if not args.soc_version in soc_version_list:
                return False
        #CSV中soc_version为空并且芯片是910A
        if pd.isnull(data.loc[index, "SocVersion"]) and args.soc_version == "Ascend910A":
            return False
        return True

    def save_result_to_csv(self, actual_output_file, from_case_index, to_case_index, args, card_type):
        data = self.opsTest.file_data.loc[from_case_index:to_case_index]
        self.data_to_export = pd.DataFrame(columns=list(data))
        for i in range(from_case_index, to_case_index + 1):
            if self.__need_to_save_case(data, i, args, card_type):
                self.data_to_export.loc[len(self.data_to_export)] = data.loc[i]
        max_widths = [max([len(str(row[i])) for row in self.data_to_export.values] + [len(str(self.data_to_export.columns[i]))]) for i in range(len(self.data_to_export.columns))]
        with open(actual_output_file, 'w') as f:
            header = [str(col).ljust(max_widths[i]) for i, col in enumerate(self.data_to_export.columns)]
            f.write('|'.join(header) + '\n')
            for row in self.data_to_export.values:
                line = [str(col).ljust(max_widths[i]) for i, col in enumerate(row)]
                f.write('|'.join(line) + '\n')
        case_pass_summary = self.data_to_export['CasePassed'].value_counts()
        case_passed_num = 0 if not 'Yes' in case_pass_summary else case_pass_summary.loc['Yes']
        case_failed_num = 0 if not 'No' in case_pass_summary else case_pass_summary.loc['No']
        logging.info("Case pass result summary: Yes:{} No:{}".format(case_passed_num, case_failed_num))
        case_no_pass_list = []
        for index, row in self.data_to_export.iterrows():
            if row['CasePassed'] == 'No':
                case_no_pass_list.append(row['CaseNum'])
        if case_no_pass_list:
            logging.info("Failed cases' CaseNum are: {}".format(case_no_pass_list))
        logging.info("Csv output file is saved in " + actual_output_file)
        return False if case_failed_num > 0 else True

    def save_result_to_excel(self, actual_output_file):
        (excel_with_path, ext) = os.path.splitext(actual_output_file)
        actual_output_file_excel = excel_with_path + '.xlsx'
        self.data_to_export.to_excel(actual_output_file_excel, index=False)
        logging.info("excel output file is saved in " + actual_output_file_excel)

    def save_result_to_html(self, actual_output_file):
        (html_with_path, ext) = os.path.splitext(actual_output_file)
        actual_output_file_html = html_with_path + '.html'
        self.data_to_export.to_html(actual_output_file_html, index=False)
        logging.info("html output file is saved in " + actual_output_file_html)

class CsvOpsTestUtil:
    PERFORMANCE_RUNNING_TIMES = 400 # run 400 times and compute the average value of result; the result of first 200 time is ignored(due to initial overhead)
    def rm_space_from_input_file(input_file, output_path):
        (file_name_with_path, ext) = os.path.splitext(input_file)
        file_name = os.path.basename(input_file).split('.')[0]
        no_space_file = file_name_with_path + '_csvopstest_tmp' + ext
        if (output_path == ''):
            actual_output_file = file_name_with_path + '_csvopstest_result' + ext
        else:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            actual_output_file = output_path + '/' + file_name + '_csvopstest_result' + ext
        #eliminate spaces and blank line in the file
        lines = open(input_file, 'r').readlines()
        file_handle = open(no_space_file, 'w')
        actual_lines_len = 0
        for i in range(len(lines)):
            if lines[i].strip():
                actual_lines_len += 1
                file_handle.write(lines[i].replace(" ", ""))
        file_handle.close()
        return no_space_file, actual_output_file, actual_lines_len

    def get_case_range(number, max_case_number):
        case_number = [int(num) for num in number.split(":")]
        from_case_num = case_number[0]
        if (len(case_number) == 1):    
            to_case_num = case_number[0]
        else:
            to_case_num = case_number[1]
        if (from_case_num < 0 or to_case_num > max_case_number):
            logging.error("case number is out of case range!")
            quit(1)
        # default is run all cases
        if (from_case_num == 0):
            from_case_num = 1
        if (to_case_num == 0):
            to_case_num = max_case_number
        return from_case_num - 1, to_case_num - 1

    def main(args, input_file, card_type):
        no_space_file, actual_output_file, lines = CsvOpsTestUtil.rm_space_from_input_file(input_file, args.output)
        from_case_index, to_case_index = CsvOpsTestUtil.get_case_range(args.number, lines - 1)

        # test process
        logging.info('')
        logging.info(f"-------------------------------------AtbCsvopstest Begins, op type: {card_type}-------------------------------------")
        logging.info("Cases are running from file: " + input_file)
        obj = CsvOpsTest(no_space_file, args)
        os.remove(no_space_file)
        result = CsvOpsResult(obj)
        for case_index in range(from_case_index, to_case_index + 1):
            if pd.isnull(obj.test_type.loc[case_index]):
                obj.file_data.loc[case_index, 'TestType'] = 'Function'
            if not obj.need_to_run_case(case_index, args, card_type):
                continue
            result.reset(case_index)
            if obj.test_type.loc[case_index] == "Performance":
                total_times = CsvOpsTestUtil.PERFORMANCE_RUNNING_TIMES if args.times < CsvOpsTestUtil.PERFORMANCE_RUNNING_TIMES else args.times + 1
            else:
                total_times = args.times
            for times in range(total_times):
                obj.run_one_case(case_index, times + 1, actual_output_file)
                result.add_result(obj)
            result.compute_result(case_index)
            logging.info("")
        cases_passed = result.save_result_to_csv(actual_output_file, from_case_index, to_case_index, args, card_type)
        if (args.format == 'excel'):
            result.save_result_to_excel(actual_output_file)
        if (args.format == 'html'):
            result.save_result_to_html(actual_output_file)
        logging.info(f"--------------------------------------AtbCsvopstest Ends, op_type: {card_type}--------------------------------------")
        return cases_passed

    def main_worker(rank, rank_root, args, input_file, output_file):
        set_log_level(args.log_level)
        torch_npu.npu.set_device(rank)
        logging.info(f'Process {rank} started, using device npu:{rank}.')
        (file_name, ext) = os.path.splitext(os.path.basename(input_file))
        output_file_path = os.path.dirname(os.path.abspath(output_file))
        rank_file = '{}/{}_{}{}'.format(output_file_path, file_name, rank, ext)
        command = 'cp {} {}'.format(input_file, rank_file)
        logging.debug(command)
        os.system(command)
        command = "sed -i 's/\"rank\":[0-9]\+,/\"rank\":{},/g' {}".format(rank, rank_file)
        logging.debug(command)
        os.system(command)
        if args.test_type == 'Generalization':
            command = "sed -i 's/\"rankSize\":[0-9]\+,/\"rankSize\":{},/g' {}".format(args.world_size, rank_file)
            logging.debug(command)
            os.system(command)
            command = "sed -i 's/\"rankRoot\":[0-9]\+,/\"rankRoot\":{},/g' {}".format(rank_root, rank_file)
            logging.debug(command)
            os.system(command)
        multi_card_cases_passed = CsvOpsTestUtil.main(args, rank_file, 'multi_card')
        if not args.exit_after and not multi_card_cases_passed:
            exit(1)

    def run_one_file(args, input_file):
        single_card_cases_passed = CsvOpsTestUtil.main(args, input_file, 'single_card')
        if not args.exit_after and not single_card_cases_passed:
            return single_card_cases_passed
        with open(input_file) as f:
            content = f.read()
            if re.search("AllGatherOperation|AllGatherVOperation|AllReduceOperation|BroadcastOperation|ReduceScatterOperation|ReduceScatterVOperation|LinearParallelOperation|AllToAllVOperation|AllToAllOperation|AllToAllVV2Operation", content, re.I):
                if args.world_size <= 1:
                    args.world_size = 2
                if soc_version == "Ascend310B":
                    args.world_size = 1
                no_space_file, actual_output_file, lines = CsvOpsTestUtil.rm_space_from_input_file(input_file, args.output)
                rank_root = random.randint(0, args.world_size - 1)
                os.environ["random_seed"]= str(random.randint(1,1000))
                mp.spawn(CsvOpsTestUtil.main_worker, nprocs = args.world_size, args = (rank_root, args, no_space_file, actual_output_file))
                os.remove(no_space_file)
        return True

def init_args():
    parser = argparse.ArgumentParser(description="ascend-transformer-boost operation api test by csv. \n \
                                     if you encounter 'ModuleNotFoundError', run 'pip3 install -r requirements.txt'.")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('-i', '--input', default='../../../apitest/opstest/csv/testcase_example.csv', help='the input path or csv file to be tested, default: testcase_example.csv')
    parser.add_argument('-o', '--output', default='', help='output path to save the test result csv file, default: input csv file path')
    parser.add_argument('-n', '--number', default='0', help='case numbers of test cases to be run in the input csv file'
                        'format: case_num(single case) or from_case_num:to_case_num(multiple cases), default: 1:max_case_num')
    parser.add_argument('-t', '--times', default=1, type=int, help='times that testcases should be run, default: 1(performance case\'s default is 101)')
    parser.add_argument('-ll', '--log_level', default='info', choices=['info', 'debug'], help='log level, default: info')
    parser.add_argument('-f', '--format', default='csv', choices=['csv', 'excel', 'html'], help='export result to output file in addtional format, default: csv')
    parser.add_argument('-tt', '--test_type', default='', choices=['Function', 'Performance', 'Generalization'], help='run testcases of specific test type, default: any test type')
    parser.add_argument('-tl', '--test_level', default='', choices=['Level0', 'Level1', 'Level2'], help='run testcases of specific test level, default: any test level')
    parser.add_argument('-m', '--model', default='', choices=['LLaMA-65B', 'ChatGLM-6B', 'LLaMA-Adapter', 'Baichuan2-13B', 'ChatGLM2-6B',
                                                             'LLaMA2-13B', 'LLaMA2-7B', 'GPT-NeoX-20B', 'ChatGLM-130B', 'BLOOM-7B'], help='run testcases from specific model, default: any model')
    parser.add_argument('-s', '--soc_version', default='', choices=['Ascend310P', 'Ascend910B'], help='run testcases of specific ascend soc platform, default: local soc platform')
    parser.add_argument('-op', '--operation_name', default='', help='run testcases of specific operations, default: any operation name')
    parser.add_argument('-ws', '--world_size', default=2, type=int, help='the number of cards used for multicard op test, default: 2')
    parser.add_argument('-st', '--save_tensor', action='store_true', help='save tensor data to "output path"/csvopstest')
    parser.add_argument('-ps', '--precision_standard', default='new', choices=['old', 'new'], help='choose precision standard version')
    parser.add_argument('-sv', '--skip_verify', action='store_true', help='skip verifying precision and preformance for op tests')
    parser.add_argument('-gpu', '--gpu_info', default='', help='gpu info to run golden on gpu, format: ip:port')
    parser.add_argument('-ea', '--exit_after', action='store_true', help='exit after running all testcase files rather than exit immediately when running testcase file fails')
    parser.add_argument('-nf', '--no_file', action='store_true', help='not load intensors from paths in column InTensorFile and OutTensorFile')
    return parser

def torch_load_framework():
    ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
    if ATB_HOME_PATH is None:
        raise RuntimeError("env ATB_HOME_PATH not exist, source set_env.sh")
    torch.classes.load_library(os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so"))

def get_device_properties():
    device_name = torch.npu.get_device_name()
    if (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10):
        soc_version = "Ascend910B"
    elif re.search("Ascend910_93", device_name, re.I):
        torch.npu.config.allow_internal_format = True
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
    elif re.search("Ascend310B", device_name, re.I):
        soc_version = "Ascend310B"
    elif re.search("Ascend950", device_name, re.I):
        soc_version = "Ascend950"
    else:
        logging.error("device_name %s is not supported", device_name)
        quit(1)
    device_count = torch.npu.device_count()
    DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
    if DEVICE_ID is not None:
        torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))
    current_device = torch.npu.current_device()
    logging.info("Device Properties: device_name: %s, soc_version: %s, device_count: %d, current_device: %d",
                 device_name, soc_version, device_count, current_device)
    return device_name, soc_version, device_count, current_device

def set_log_level(level_param):
    if (level_param == 'info'):
        logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    elif (level_param == 'debug'):
        logging.basicConfig(level=logging.DEBUG,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def get_abs_path_files_from_dir(input_dir):
    abs_path_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            (file_name, ext) = os.path.splitext(file)
            if not re.search("_csvopstest_", file_name) and ext == '.csv':
                abs_path_files.append(os.path.join(root, file))
    abs_path_files.sort()
    return abs_path_files

torch_load_framework()

if __name__ == '__main__':
    parser = init_args()
    args = parser.parse_args()
    set_log_level(args.log_level)
    device_name, soc_version, device_count, current_device = get_device_properties()
    if args.soc_version == '':
        args.soc_version = soc_version
    if os.path.exists(args.input):
        if os.path.isdir(args.input):
            files = get_abs_path_files_from_dir(args.input)
            if not files:
                logging.error("input path {} has no file to be processed".format(args.input))
                exit(1)
            else:
                logging.info("*******************************************")
                logging.info("the input files to be processed are: %s", files)
                logging.info("*******************************************")
                test_result = 'Succ'
                for file in files:
                   if not CsvOpsTestUtil.run_one_file(args, file):
                       logging.error("Test failed for file %s", file)
                       test_result = 'Fail'
                       if not args.exit_after:
                           exit(1)
                if test_result == 'Fail':
                    logging.error("Test failed for path %s", args.input)
                    exit(1)
        else:
            if not CsvOpsTestUtil.run_one_file(args, args.input):
                logging.error("Test failed for file %s", args.input)
                exit(1)
    else:
        logging.error("input path {} is invalid".format(args.input))
        exit(0)
