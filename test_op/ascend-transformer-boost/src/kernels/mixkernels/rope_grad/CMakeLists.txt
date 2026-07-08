# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(rope_grad_srcs
    ${CMAKE_CURRENT_LIST_DIR}/rope_grad_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rope_grad_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/rope_grad_tiling.cpp
)

add_operation(RopeGradOperation "${rope_grad_srcs}")

add_kernel(rope_grad ascend910b vector
        op_kernel/rope_grad.cpp
        RopeGradKernel)