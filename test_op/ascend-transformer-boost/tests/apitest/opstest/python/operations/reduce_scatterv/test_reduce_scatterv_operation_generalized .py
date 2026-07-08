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
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Process, set_start_method

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test
import itertools

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# usage:
# export HCCL_WHITELIST_DISABLE=1
# python3 -m unittest test_all_reduce_operation.py
# Attention: when you use lccl backend, unset HCCL_MTE_ENABLE and copy lcal.o to current directory

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)


def main_worker(rank, world_size, reduceType, inTensors, sendCounts, sdispls, recvCount, y):
        torch_npu.npu.set_device(rank)
        print(f'Process {rank} started, using device npu:{rank}.')
        reduce_scatterv_operation = torch.classes.OperationTorch.OperationTorch("ReduceScatterVOperation")
        acl_param = json.dumps({"rank": rank, "rankSize": world_size, "rankRoot": 0, "backend": "hccl", "reduceType": reduceType,
                                "sendCounts": sendCounts, "sdispls": sdispls, "recvCount": recvCount[rank][0]})
        run_param = json.dumps({"sendCounts": sendCounts, "sdispls": sdispls, "recvCount": recvCount[rank][0]})
        reduce_scatterv_operation.set_param(acl_param)
        reduce_scatterv_operation.set_varaintpack_param(run_param)
        acl_out_tensor = reduce_scatterv_operation.execute([inTensors[rank].npu(), torch.tensor(sendCounts).npu(), torch.tensor(sdispls).npu(), torch.tensor(recvCount[rank]).npu(), y[rank].npu()])[0]
        torch.npu.synchronize()

        result = inTensors[0].clone()
        if reduceType == 'sum':
            for i in range(1, len(inTensors)):
                result += inTensors[i]

        elif reduceType == 'max':
            for i in range(1,len(inTensors)):
                result = torch.max(result,inTensors[i])

        elif reduceType == "min":
            for i in range(1,len(inTensors)):
                result = torch.min(result,inTensors[i])

        gold_outtensor = result.narrow(0, sdispls[rank] // inTensors[0].shape[1], sendCounts[rank] // inTensors[0].shape[1])
        assert golden_compare(acl_out_tensor.cpu(), gold_outtensor)


def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
            "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
            ", \ngolden_oute_tensor:", golden_out_tensor)
    return result


def log(out_tensor, golden_out_tensor, filename):
    # 把输出重定向到文件
    f = open(filename, 'w')
    # 之后使用print函数，都将内容打印到 screenshot.log 文件中
    sys.stdout = f
    print("diff:", out_tensor - golden_out_tensor)
    f.close()


class reduce_scatterv_operationTest(operation_test.OperationTest):
    def test_reduce_scatterv_operation(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # 指定shape
        shape = [13107, 256, 1]
        dims = list(itertools.combinations(shape, 2))
        # 指定卡数
        world_sizes = [8]
        # 指定reduceType
        reduceTypes = ['max', 'min', 'sum']
        low = 1
        high = 17
        # 指定intensor数据格式
        inTensorDtypes = [torch.int8, torch.float16, torch.bfloat16]
        
        for reduceType in reduceTypes:
            for inTensorDtype in inTensorDtypes:
                for world_size in world_sizes:
                    for dim in dims:
                        print("-----------------------------------------------" + f"reduceType:{reduceType}, inTensorDtype:{inTensorDtype}, world_size:{world_size}, dim:{dim}")
                        if dim[0] >= world_size:
                            # 生成inTensors
                            inTensor = torch.randint(low, high, dim, dtype=inTensorDtype)
                            inTensors = []
                            for _ in range(world_size):
                                inTensors.append(inTensor)
                            # 生成sendCounts
                            sendCounts = [dim[0] // world_size * dim[1]] * world_size
                            sendCounts[0] += dim[0] % world_size * dim[1]
                            # 生成sdispls
                            sdispls = [0] * world_size
                            for i in range(1, world_size):
                                sdispls[i] = sdispls[i - 1] + sendCounts[i - 1]
                            # 生成recvCount
                            recvCount = [[sendCounts[i]] for i in range(world_size)]
                            # 生成y
                            y = []
                            for i in range(world_size):
                                y.append(((high - low) * torch.rand(sendCounts[i] // dim[1]) + low).type(torch.float16))

                            print("sendCounts: \n", sendCounts)
                            print("sdispls: \n", sdispls)
                            print("recvCount: \n", recvCount)
                            print("y: \n", y)
                            set_start_method('spawn', force=True)
                            process_list = []
                            for i in range(world_size):
                                p = Process(target=main_worker, args=(i, world_size, reduceType, inTensors, sendCounts, sdispls, recvCount, y))
                                p.start()
                                process_list.append(p)

                            for _ in process_list:
                                p.join()


if __name__ == '__main__':
    unittest.main()
