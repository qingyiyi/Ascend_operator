#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "SliceOperation"
PARAM = {"offsets": [2, 8], "size": [10, 100]}


class TestSliceOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0][2:12, 8:108]]

    def test_2d_float(self):
        self.execute(OP_NAME, PARAM, [torch.randn(32, 128).npu().half()])

class TestSliceOperationBf16(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0][2:12, 8:108]]

    def test_2d_float(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        self.execute(OP_NAME, PARAM, [torch.randn(32, 128).bfloat16().npu()])

if __name__ == '__main__':
    unittest.main()