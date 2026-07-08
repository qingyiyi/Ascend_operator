#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import json
import unittest
import sys
import socket
import random
import threading
from time import sleep
import numpy as np
import math
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import  Process, set_start_method
from typing import List
from functools import reduce

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

import logging
from precision_calcu import compare_cv
logging.basicConfig(level=logging.INFO)

# usage:
# export HCCL_WHITELIST_DISABLE=1
# python3 -m unittest test_all_to_allv_operation.py
# Attention: only supports hccl

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)

def main_worker(rank, world_size, in_tensor_dtypes, random_seed, sendcount, senddisp, recvout, recvdis, tensorlist):
    # init process group
    torch_npu.npu.set_device(rank)
    logging.info(f'Process {rank} started, using device npu:{rank}.')
    # init all_to_allv_operation
    all_to_allvv2_operation = torch.classes.OperationTorch.OperationTorch(
        "AllToAllVV2Operation")
    torch.manual_seed(random_seed)
    gold_outtensors = []
    
    # 多维的intensor处理为一维
    tensorafters = np.array(tensorlist).flatten().reshape(world_size, -1).tolist()
    for j in range(len(recvout[rank])):
        logging.debug(f"tensorafters[j:{j}][senddisp[j:{j}][rank:{rank}]:senddisp[j][rank]+sendcount[j][rank]] \
                      {tensorafters[j][senddisp[j][rank]:sendcount[j][rank] + senddisp[j][rank]]}")
        gold_outtensors.append(tensorafters[j][senddisp[j][rank]:sendcount[j][rank] + senddisp[j][rank]])
    gold_outtensors = [i for arr in gold_outtensors for i in arr]
    for k in range(len(in_tensor_dtypes)):
        golden_out = (torch.tensor(gold_outtensors, dtype=in_tensor_dtypes[k]))
        logging.debug(f"golden_out {golden_out}")
        #计算inTensors
        inTensors = np.zeros([world_size, world_size]).tolist()
        for i in range(world_size):
            inTensors[i] = torch.tensor(tensorlist[i], dtype=in_tensor_dtypes[k]).npu()
    
        acl_param = json.dumps({"rank": rank, "rankSize": world_size,"rankRoot": 0, "backend": "hccl"})
        run_param = json.dumps({"sendCounts": sendcount[rank], "sdispls": senddisp[rank], "recvCounts": recvout[rank],
                                "rdispls": recvdis[rank]})
        host_list = [sendcount[rank], senddisp[rank], recvout[rank], recvdis[rank]]
        host_tensors = [torch.from_numpy(np.array(tensor)).to(torch.int64).npu() for tensor in host_list]
        logging.debug(f"host_tensors={host_tensors}")
        tensor_for_infer_shape = torch.zeros(sum(recvout[rank])).to(torch.int8).npu()
        all_to_allvv2_operation.set_param(acl_param)
        all_to_allvv2_operation.set_varaintpack_param(run_param)
        out_tensors = all_to_allvv2_operation.execute(
            [inTensors[rank], *host_tensors, tensor_for_infer_shape]
        )
        torch.npu.synchronize()
        
        assert golden_compare(out_tensors[0].cpu(), golden_out.cpu(), dtype=k)
        
def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001, dtype=torch.float16):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
 
    if not result:
        logging.info(f"out shape {out_tensor.shape}")
        logging.info(f"golden shape {golden_out_tensor.shape}")
        logging.info(f"out: {out_tensor}")
        logging.info(f"golden_oute_tensor: {golden_out_tensor}")
    return result

class all_to_allv_operationTest(operation_test.OperationTest):
    def set_data_params(self, world_size=2, shape=None):
        if shape is None:
            dim_num = random.randint(1, 4)
            # set max count < 1e8
            max_dim = 100
            shape = [random.randint(1, max_dim) for _ in range(dim_num)]
        rest_count = reduce(lambda x,y:x*y, shape, 1)
        total_numel = rest_count
        # 生成host intensors
        send_counts = []
        for _ in range(world_size):
            rest_count = total_numel
            send_count = [0] * world_size
            for i in range(world_size):
                send_count[i] = random.randint(0, rest_count)
                rest_count -= send_count[i]
            send_counts.append(send_count)
        
        senddisp = np.zeros([world_size, world_size]).tolist()
        recvout = np.zeros([world_size, world_size]).tolist()
        rdispls = np.zeros([world_size, world_size]).tolist()
        for i in range(len(send_counts)):
            for j in range(len(send_counts[i])):
                if j == 0:
                    senddisp[i][j] = 0
                else:
                    senddisp[i][j] = senddisp[i][j - 1] + send_counts[i][j - 1]
                recvout[i][j] = send_counts[j][i]
        for i in range(len(recvout)):
            for j in range(len(recvout[j])):
                if j == 0:
                    rdispls[i][j] = 0
                else:
                    rdispls[i][j] = recvout[i][j - 1] + rdispls[i][j - 1]

        logging.info(f"shape {shape}")
        logging.info(f"sendcount {send_counts}")
        logging.info(f"senddisp {senddisp}")
        logging.info(f"recvout {recvout}")
        logging.info(f"rdispls {rdispls}")
        # 生成intensors
        max_value = 100
        min_value = -100
        
        tensorlist = []
        for i in range(world_size):
            line = np.random.rand(*shape) * (max_value - min_value) + min_value
            # switch to int
            # line = np.floor(line).astype(np.int32)
            # line = np.arange(total_numel).reshape(shape).astype(np.int32)
            tensorlist.append(line.tolist())
        return send_counts, senddisp, recvout, rdispls, tensorlist

    def test_all_to_allv_operation(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        world_size = 2
        random_seed = 123
        np.random.seed(random_seed)
        # send_counts, senddisp, recvout, rdispls, intensorlist
        params = self.set_data_params(world_size)
        in_tensor_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float16, torch.bfloat16]
        set_start_method('spawn', force=True)
        process_list = []
        for i in range(world_size):
            p = Process(target=main_worker,args=(i, world_size, in_tensor_dtypes, random_seed, *params))
            p.start()
            process_list.append(p)
        for i in process_list:
            p.join()


if __name__ == '__main__':
    unittest.main()
